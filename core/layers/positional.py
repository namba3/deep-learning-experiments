import math
import torch
import torch.nn as nn



def build_2d_sincos_pos_embed(
    embed_dim,
    height,
    width,
    *,
    normalize_coordinates=False,
):
    """2Dグリッド用の固定Sin-Cos位置埋め込みを生成する。

    Args:
        normalize_coordinates: Trueの場合、各軸の座標を0〜1に正規化する。
    """
    if embed_dim % 4 != 0:
        raise ValueError("embed_dim must be divisible by 4")
    y = torch.arange(height, dtype=torch.float32)
    x = torch.arange(width, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
    if normalize_coordinates:
        if height > 1:
            y_grid = y_grid / (height - 1)
        if width > 1:
            x_grid = x_grid / (width - 1)
    frequency = torch.arange(embed_dim // 4, dtype=torch.float32)
    frequency = 1.0 / (10000 ** (frequency / (embed_dim // 4)))
    y_phase = y_grid.flatten()[:, None] * frequency[None, :]
    x_phase = x_grid.flatten()[:, None] * frequency[None, :]
    embedding = torch.cat(
        (torch.sin(y_phase), torch.cos(y_phase),
         torch.sin(x_phase), torch.cos(x_phase)),
        dim=1,
    )
    return embedding.unsqueeze(0)

class PositionalEncoding2D_SineCosine(nn.Module):
    """
    2Dデータに対する正弦・余弦位置エンコーディング
    - 入力テンソルに直接加算する形で位置情報を付与
    - embed_dimは4の倍数である必要がある
    """
    def __init__(self, embed_dim, grid_size):
        super().__init__()
        if embed_dim % 4 != 0:
            raise ValueError("embed_dimは4の倍数が必要")
        pos_embed = build_2d_sincos_pos_embed(
            embed_dim,
            grid_size,
            grid_size,
            normalize_coordinates=True,
        )
        self.register_buffer("pos_embed", pos_embed.squeeze(0))

    def forward(self, x):
        return x + self.pos_embed.unsqueeze(0).to(x.device)

class DtypeAwareRMSNorm(nn.RMSNorm):
    """Run RMSNorm in parameter dtype and restore the input dtype.

    Mixed-precision linear layers in the text model intentionally return
    FP32 activations for numerical stability.  ``nn.RMSNorm`` warns (and may
    miss its fused path) when those activations meet BF16 affine weights.
    Keeping the cast at this boundary makes the dtype contract explicit while
    preserving the surrounding activation dtype.
    """

    def forward(self, input):
        input_dtype = input.dtype
        norm_input = input
        if self.weight is not None:
            norm_input = input.to(dtype=self.weight.dtype)
        output = super().forward(norm_input)
        return output.to(dtype=input_dtype)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def rotate_rope_pairs(x, cos, sin):
    """Apply RoPE to adjacent even/odd feature pairs.

    2D RoPE stores each spatial axis as repeated adjacent frequencies, so the
    half-split rotation in ``rotate_half`` would mix the Y and X axes.
    """
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return torch.stack(
        (even * cos[..., 0::2] - odd * sin[..., 0::2],
         even * sin[..., 0::2] + odd * cos[..., 0::2]),
        dim=-1,
    ).flatten(-2)

class RotaryEmbedding(nn.Module):
    """One-dimensional pair-wise RoPE for token sequences."""
    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be divisible by 2 for RoPE")
        if base <= 0:
            raise ValueError("base must be positive")
        self.head_dim = head_dim
        self.base = float(base)
        self._cache = {}

    def forward(self, x):
        if x.ndim != 4 or x.size(-1) != self.head_dim:
            raise ValueError(
                "RoPE input must have shape (B, heads, tokens, head_dim)"
            )
        seq_len = x.size(2)
        cache_key = (x.device.type, x.device.index, x.dtype, seq_len)
        cached = self._cache.get(cache_key)
        if cached is None:
            inverse_frequency = 1.0 / (
                self.base ** (
                    torch.arange(
                        0, self.head_dim, 2,
                        device=x.device, dtype=torch.float32,
                    ) / self.head_dim
                )
            )
            positions = torch.arange(
                seq_len, device=x.device, dtype=torch.float32,
            )
            phase = positions[:, None] * inverse_frequency[None, :]
            cos = phase.cos().repeat_interleave(2, dim=-1)
            sin = phase.sin().repeat_interleave(2, dim=-1)
            cached = (
                cos[None, None].to(dtype=x.dtype),
                sin[None, None].to(dtype=x.dtype),
            )
            self._cache[cache_key] = cached
        cos, sin = cached
        return rotate_rope_pairs(x, cos, sin)

class RotaryEmbedding2D(nn.Module):
    def __init__(self, head_dim, height, width, base=10000.0):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError("head_dim must be divisible by 4 for 2D RoPE")
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive for 2D RoPE")
        if base <= 0:
            raise ValueError("base must be positive for 2D RoPE")
        self.base = float(base)
        cos, sin = self._build_tables(
            head_dim, height, width, self.base,
        )
        # Keep these buffers for checkpoint compatibility.  Non-base grids
        # are rebuilt from integer coordinates in ``forward`` below.
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.head_dim = head_dim
        self.height = height
        self.width = width

    @staticmethod
    def _build_tables(head_dim, height, width, base, *, device=None, dtype=None):
        """Build exact pair-wise 2D RoPE tables for an integer grid."""
        quarter_dim = head_dim // 4
        inverse_frequency = 1.0 / (
            base ** (
                torch.arange(
                    quarter_dim, dtype=torch.float32, device=device,
                ) / quarter_dim
            )
        )
        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=device),
            torch.arange(width, dtype=torch.float32, device=device),
            indexing="ij",
        )
        y_phase = y.flatten()[:, None] * inverse_frequency[None, :]
        x_phase = x.flatten()[:, None] * inverse_frequency[None, :]
        phase = torch.cat(
            (y_phase.repeat_interleave(2, dim=-1),
             x_phase.repeat_interleave(2, dim=-1)),
            dim=-1,
        )
        cos = phase.cos().unsqueeze(0).unsqueeze(0)
        sin = phase.sin().unsqueeze(0).unsqueeze(0)
        if dtype is not None:
            cos = cos.to(dtype=dtype)
            sin = sin.to(dtype=dtype)
        return cos, sin

    def forward(self, query, key, grid_shape=None):
        if query.ndim != 4 or key.ndim != 4:
            raise ValueError("query and key must have shape (B, heads, tokens, head_dim)")
        if query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2]:
            raise ValueError("query and key must have matching batch and token axes")
        if query.size(-1) != self.head_dim or key.size(-1) != self.head_dim:
            raise ValueError(
                f"query and key head_dim must equal {self.head_dim}"
            )
        if query.device != key.device or query.dtype != key.dtype:
            raise ValueError("query and key must have matching device and dtype")
        cos = self.cos.to(device=query.device, dtype=query.dtype)
        sin = self.sin.to(device=query.device, dtype=query.dtype)
        if grid_shape is None:
            if query.size(2) != self.height * self.width:
                grid_size = int(math.sqrt(query.size(2)))
                if grid_size * grid_size != query.size(2):
                    raise ValueError(
                        "grid_shape is required for a non-square 2D RoPE grid"
                    )
                grid_shape = (grid_size, grid_size)
        grid_height, grid_width = grid_shape if grid_shape is not None else (
            self.height, self.width,
        )
        if query.size(2) != grid_height * grid_width:
            raise ValueError(
                "grid_shape token count does not match query: "
                f"grid={grid_height}x{grid_width}, tokens={query.size(2)}"
            )
        if grid_height <= 0 or grid_width <= 0:
            raise ValueError("grid_shape dimensions must be positive")
        if (grid_height, grid_width) != (self.height, self.width):
            cos, sin = self._build_tables(
                query.size(-1),
                grid_height,
                grid_width,
                self.base,
                device=query.device,
                dtype=query.dtype,
            )
        return (
            rotate_rope_pairs(query, cos, sin),
            rotate_rope_pairs(key, cos, sin),
        )
