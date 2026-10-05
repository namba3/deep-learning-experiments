import torch
import torch.nn as nn
from .transformer_blocks import (
    GatedGQATransformerBlock,
    InterpolatedTransformerBlock,
    KVTransformerBlock,
    MHLATransformerBlock,
    TransformerBlock,
)



def _causal_mask(seq_len, device, dtype):
    return torch.zeros(
        (seq_len, seq_len), device=device, dtype=dtype,
    ).masked_fill(
        torch.triu(
            torch.ones((seq_len, seq_len), device=device, dtype=torch.bool),
            diagonal=1,
        ),
        float("-inf"),
    )

class CausalTransformerDecoder(nn.Module):
    """Independent-weight pre-norm decoder used by the naive architecture."""
    def __init__(self, num_layers, embed_dim, num_heads, drop_out=0.1,
                 rope_base=10000.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, drop_out, rope_base=rope_base,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(embed_dim)

    def forward(self, x, attention_mask=None):
        attn_mask = _causal_mask(x.shape[1], x.device, x.dtype)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.to(
                device=x.device, dtype=torch.bool,
            )
        for layer in self.layers:
            x = layer(
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
        return self.norm(x)

class LoopedTransformerDecoder(nn.Module):
    """Pre-norm decoder that reuses a physical block stack across depth.

    ``num_layers`` is the total unrolled depth. Only ``looped_blocks``
    physical blocks are stored and that stack is repeated
    ``num_layers // looped_blocks`` times. Residual branches are scaled by
    the inverse square root of the loop count for repeated pre-norm updates.
    """
    def __init__(self, num_layers, embed_dim, num_heads, looped_blocks=1,
                 drop_out=0.1, rope_base=10000.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if not 1 <= looped_blocks <= num_layers:
            raise ValueError("looped_blocks must be in [1, num_layers]")
        if num_layers % looped_blocks != 0:
            raise ValueError("num_layers must be divisible by looped_blocks")
        self.num_layers = num_layers
        self.looped_blocks = looped_blocks
        self.num_loops = num_layers // looped_blocks
        residual_scale = self.num_loops ** -0.5
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_heads,
                drop_out,
                rope_base=rope_base,
                residual_scale=residual_scale,
            )
            for _ in range(looped_blocks)
        ])
        self.norm = nn.RMSNorm(embed_dim)

    def forward(self, x, attention_mask=None):
        attn_mask = _causal_mask(x.shape[1], x.device, x.dtype)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.to(
                device=x.device, dtype=torch.bool,
            )
        for _ in range(self.num_loops):
            for layer in self.layers:
                x = layer(
                    x,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                )
        return self.norm(x)

class HybridLoopedTransformerDecoder(nn.Module):
    """Independent prelude/coda around a recurrent-depth middle stack.

    The unrolled layout is ``prefix -> (loop stack x repeats) -> suffix``.
    Only the middle stack is reused; the prefix and suffix keep independent
    parameters. ``num_layers`` is the total unrolled depth.
    """
    def __init__(self, num_layers, embed_dim, num_heads,
                 prefix_layers=1, looped_blocks=1, looped_repeats=2,
                 suffix_layers=1, drop_out=0.1, rope_base=10000.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if prefix_layers < 0 or suffix_layers < 0:
            raise ValueError("prefix_layers and suffix_layers must be >= 0")
        if looped_blocks < 1 or looped_repeats < 1:
            raise ValueError("looped_blocks and looped_repeats must be >= 1")
        expected_layers = (
            prefix_layers + looped_blocks * looped_repeats + suffix_layers
        )
        if expected_layers != num_layers:
            raise ValueError(
                "num_layers must equal prefix_layers + "
                "looped_blocks * looped_repeats + suffix_layers"
            )
        self.num_layers = num_layers
        self.prefix_layers_count = prefix_layers
        self.looped_blocks = looped_blocks
        self.num_loops = looped_repeats
        self.suffix_layers_count = suffix_layers
        residual_scale = looped_repeats ** -0.5

        def make_layers(count, repeated=False):
            return nn.ModuleList([
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    drop_out,
                    rope_base=rope_base,
                    residual_scale=residual_scale if repeated else 1.0,
                )
                for _ in range(count)
            ])

        self.prefix_layers = make_layers(prefix_layers)
        self.looped_layers = make_layers(looped_blocks, repeated=True)
        self.suffix_layers = make_layers(suffix_layers)
        self.norm = nn.RMSNorm(embed_dim)

    def forward(self, x, attention_mask=None):
        attn_mask = _causal_mask(x.shape[1], x.device, x.dtype)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.to(
                device=x.device, dtype=torch.bool,
            )

        def apply_layers(layers):
            nonlocal x
            for layer in layers:
                x = layer(
                    x,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                )

        apply_layers(self.prefix_layers)
        for _ in range(self.num_loops):
            apply_layers(self.looped_layers)
        apply_layers(self.suffix_layers)
        return self.norm(x)

class GatedGQATransformerDecoder(CausalTransformerDecoder):
    """Independent-weight pre-norm decoder with query-gated GQA blocks."""
    def __init__(self, num_layers, embed_dim, num_heads, kv_heads,
                 drop_out=0.1, rope_base=10000.0):
        nn.Module.__init__(self)
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            GatedGQATransformerBlock(
                embed_dim, num_heads, kv_heads, drop_out,
                rope_base=rope_base,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(embed_dim)

class HybridMHLA3GQADecoder(nn.Module):
    """Repeat ``MHLA, MHLA, MHLA, Gated-GQA`` for ``num_cycles``."""
    def __init__(self, num_cycles, embed_dim, num_heads, kv_heads,
                 drop_out=0.1, rope_base=10000.0):
        super().__init__()
        if num_cycles < 1:
            raise ValueError("num_cycles must be >= 1")
        self.num_cycles = num_cycles
        self.num_layers = num_cycles * 4
        layers = []
        for _ in range(num_cycles):
            layers.extend([
                MHLATransformerBlock(
                    embed_dim, num_heads, kv_heads, drop_out,
                    rope_base=rope_base,
                )
                for _ in range(3)
            ])
            layers.append(
                GatedGQATransformerBlock(
                    embed_dim, num_heads, kv_heads, drop_out,
                    rope_base=rope_base,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.norm = nn.RMSNorm(embed_dim)

    def forward(self, x, attention_mask=None):
        attn_mask = _causal_mask(x.shape[1], x.device, x.dtype)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.to(
                device=x.device, dtype=torch.bool,
            )
        for layer in self.layers:
            if isinstance(layer, MHLATransformerBlock):
                x = layer(x, key_padding_mask=key_padding_mask)
            else:
                x = layer(
                    x,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                )
        return self.norm(x)

class HybridMHLA3GQALoopedDecoder(nn.Module):
    """MHLA/GQA cycles with a recurrent middle cycle.

    The unrolled layout is ``independent cycles x prefix -> one shared
    MHLA/MHLA/MHLA/GQA cycle x repeats -> independent cycles x suffix``.
    ``num_layers`` is the total unrolled block count, so it must be four times
    the total number of cycles.
    """
    def __init__(self, num_layers, embed_dim, num_heads, kv_heads,
                 prefix_cycles=1, looped_repeats=2, suffix_cycles=1,
                 drop_out=0.1, rope_base=10000.0):
        super().__init__()
        if num_layers < 1 or num_layers % 4 != 0:
            raise ValueError("num_layers must be a positive multiple of 4")
        if prefix_cycles < 0 or suffix_cycles < 0:
            raise ValueError("prefix_cycles and suffix_cycles must be >= 0")
        if looped_repeats < 1:
            raise ValueError("looped_repeats must be >= 1")
        total_cycles = num_layers // 4
        if prefix_cycles + looped_repeats + suffix_cycles != total_cycles:
            raise ValueError(
                "num_layers must equal 4 * "
                "(prefix_cycles + looped_repeats + suffix_cycles)"
            )
        if not 1 <= kv_heads <= num_heads or num_heads % kv_heads != 0:
            raise ValueError(
                "kv_heads must be positive, no greater than num_heads, "
                "and divide num_heads"
            )
        self.num_layers = num_layers
        self.num_cycles = total_cycles
        self.prefix_cycles = prefix_cycles
        self.looped_repeats = looped_repeats
        self.suffix_cycles = suffix_cycles
        self.num_loops = looped_repeats
        residual_scale = looped_repeats ** -0.5

        def make_cycle(repeated=False):
            scale = residual_scale if repeated else 1.0
            return [
                MHLATransformerBlock(
                    embed_dim, num_heads, kv_heads, drop_out,
                    rope_base=rope_base, residual_scale=scale,
                )
                for _ in range(3)
            ] + [
                GatedGQATransformerBlock(
                    embed_dim, num_heads, kv_heads, drop_out,
                    rope_base=rope_base, residual_scale=scale,
                )
            ]

        self.prefix_layers = nn.ModuleList([
            layer
            for _ in range(prefix_cycles)
            for layer in make_cycle()
        ])
        self.looped_layers = nn.ModuleList(make_cycle(repeated=True))
        self.suffix_layers = nn.ModuleList([
            layer
            for _ in range(suffix_cycles)
            for layer in make_cycle()
        ])
        self.norm = nn.RMSNorm(embed_dim)

    def forward(self, x, attention_mask=None):
        attn_mask = _causal_mask(x.shape[1], x.device, x.dtype)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.to(
                device=x.device, dtype=torch.bool,
            )

        def apply_layers(layers):
            nonlocal x
            for layer in layers:
                if isinstance(layer, MHLATransformerBlock):
                    x = layer(x, key_padding_mask=key_padding_mask)
                else:
                    x = layer(
                        x,
                        attn_mask=attn_mask,
                        key_padding_mask=key_padding_mask,
                    )

        apply_layers(self.prefix_layers)
        for _ in range(self.num_loops):
            apply_layers(self.looped_layers)
        apply_layers(self.suffix_layers)
        return self.norm(x)

class InterpolatedTransformerEncoder(nn.Module):
    """
    ブロック位置を 0.0～1.0 の depth_step に変換する Encoder。

    num_layers=1 の場合は depth_step=0.0 とします。外部から
    depth_steps を渡せば、ブロック位置以外の連続値も利用できます。
    """
    def __init__(self, num_layers, embed_dim, num_heads,
                 num_steps=4, drop_out=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            InterpolatedTransformerBlock(
                embed_dim, num_heads, num_steps=num_steps,
                drop_out=drop_out
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(embed_dim)

    def forward(self, x, depth_steps=None):
        if depth_steps is None:
            if self.num_layers == 1:
                depth_steps = [0.0]
            else:
                depth_steps = [
                    i / (self.num_layers - 1)
                    for i in range(self.num_layers)
                ]
        if len(depth_steps) != self.num_layers:
            raise ValueError("depth_steps length must equal num_layers")

        for layer, depth_step in zip(self.layers, depth_steps):
            x = layer(x, depth_step)
        return self.norm(x)

class TransformerEncoder(nn.Module):
    """
    Transformerエンコーダ
    - 複数のTransformerBlockを積み重ねる
    - 最終的にRMSNormを適用
    - BatchFirst前提
    - 入力xの形状: (B, N, D)
    - 出力の形状: (B, N, D)
    """
    def __init__(self, num_layers, embed_dim, num_heads, drop_out=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, drop_out) for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

class KVTransformerEncoder(nn.Module):
    """
    Qを廃止したKV Transformer Encoder
    - 複数のKVTransformerBlockを積み重ねる
    - 最終段にRMSNormを適用
    - BatchFirst前提
    - 入力/出力形状: (B, N, D)
    """
    def __init__(self, num_layers, embed_dim, num_heads, drop_out=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            KVTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                drop_out=drop_out
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.RMSNorm(embed_dim)

    def forward(self, x):
        # x: (B, N, D)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x
