import torch
import torch.nn as nn
import torch.nn.functional as F
from .positional import RotaryEmbedding
from .feedforward import GatedLinear



class DepthStepEmbedding(nn.Module):
    """[depth_step, step_interval] を学習可能なFFNで埋め込む。"""
    def __init__(self, embedding_dim=4):
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1")
        hidden_dim = max(32, embedding_dim * 2)
        self.embedding_dim = embedding_dim
        self.projection = nn.Sequential(
            GatedLinear(2, hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
        )

    @property
    def output_dim(self):
        return self.embedding_dim

    def forward(self, depth, step_interval=None):
        if step_interval is None:
            step_interval = torch.zeros_like(depth)
        inputs = torch.cat([depth, step_interval], dim=-1)
        return self.projection(inputs)

class DepthFactorGenerator(nn.Module):
    """条件ベクトルから変換因子を生成する通常/Gated MLP。"""
    def __init__(self, input_dim, hidden_dim, output_dim, gated=False):
        super().__init__()
        self.gated = gated
        if gated:
            self.input_proj = nn.Linear(input_dim, hidden_dim * 2)
            self.output_proj = nn.Linear(hidden_dim, output_dim)
        else:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        hidden = self.input_proj(x)
        if self.gated:
            value, gate = hidden.chunk(2, dim=-1)
            hidden = F.silu(value) * gate
        else:
            hidden = F.silu(hidden)
        return self.output_proj(hidden)

class DepthSuperLinear(nn.Module):
    """
    全 Transformer block で共有する、入力条件付きスーパー行列 Linear。

    depth_step から変換行列を生成し、次式で実効重みを作成します::

        W_effective = T_out(depth) @ W_super @ T_in(depth)

    T は ``I + U @ V.T`` という低ランク表現です。これにより、巨大な
    (out x out) / (in x in) 行列をハイパーネットワークから直接出力せずに、
    depth依存の変換行列を学習できます。
    """
    def __init__(self, in_features, out_features, condition_dim=64,
                 transform_rank=4, bias=False, depth_embedding=None,
                 compute_dtype=None,
                 ):
        super().__init__()
        if transform_rank < 1:
            raise ValueError("transform_rank must be >= 1")
        self.in_features = in_features
        self.out_features = out_features
        self.transform_rank = transform_rank
        self.compute_dtype = compute_dtype
        self._save_parameters_as_bf16 = compute_dtype == torch.bfloat16

        self.super_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = (
            nn.Parameter(torch.zeros(out_features)) if bias else None
        )

        # depth_step(1) -> 変換行列の因子
        factor_dim = transform_rank * (out_features * 2 + in_features * 2)
        hidden_dim = max(condition_dim, 32)
        self.depth_embedding = (
            depth_embedding
            if depth_embedding is not None
            else DepthStepEmbedding()
        )
        self.transform_generator = DepthFactorGenerator(
            self.depth_embedding.output_dim,
            hidden_dim,
            factor_dim,
            gated=False,
        )
        # 外側の model.apply(init_weights) から除外する。
        # この層はゼロ初期化して恒等変換 T=I から開始する必要がある。
        self.transform_generator.output_proj._preserve_init = True
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.super_weight)
        # 初期状態では T_in=T_out=I とし、通常のスーパー行列から開始する。
        nn.init.zeros_(self.transform_generator.output_proj.weight)
        nn.init.zeros_(self.transform_generator.output_proj.bias)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _condition(self, x, depth_step):
        if x.ndim < 2:
            raise ValueError("x must have a batch dimension")
        if (
            torch.is_tensor(depth_step)
            and depth_step.ndim == 2
            and depth_step.shape[-1] == self.depth_embedding.output_dim
        ):
            return depth_step.to(device=x.device, dtype=x.dtype)
        if not torch.is_tensor(depth_step):
            depth_step = x.new_tensor(depth_step)
        depth_step = depth_step.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)
        depth = depth_step.expand(x.shape[0], 1)
        return self.depth_embedding(depth)

    def _factors(self, x, depth_step):
        condition = self._condition(x, depth_step)
        # 因子を有界にして T=I+UV^T の急激な増幅を防ぐ。
        factors = 0.1 * torch.tanh(self.transform_generator(condition))
        rank = self.transform_rank
        offset = 0

        def take(rows):
            nonlocal offset
            size = rows * rank
            first = factors[:, offset:offset + size].view(-1, rows, rank)
            offset += size
            second = factors[:, offset:offset + size].view(-1, rows, rank)
            offset += size
            return first, second

        u_out, v_out = take(self.out_features)
        u_in, v_in = take(self.in_features)
        return u_out, v_out, u_in, v_in

    def effective_weight(self, x, depth_step):
        """デバッグ・検証用に完全な実効行列を生成する。"""
        u_out, v_out, u_in, v_in = self._factors(x, depth_step)

        eye_out = torch.eye(
            self.out_features, device=x.device, dtype=x.dtype
        ).expand(x.shape[0], -1, -1)
        eye_in = torch.eye(
            self.in_features, device=x.device, dtype=x.dtype
        ).expand(x.shape[0], -1, -1)
        t_out = eye_out + torch.bmm(u_out, v_out.transpose(1, 2))
        t_in = eye_in + torch.bmm(u_in, v_in.transpose(1, 2))

        super_weight = self.super_weight.to(dtype=x.dtype).unsqueeze(0)
        weight = torch.matmul(torch.matmul(t_out, super_weight), t_in)
        return weight

    def forward(self, x, depth_step):
        # W_effective = T_out @ W_super @ T_in を直接作らず、
        # x @ T_in.T -> W_super.T -> T_out.T の順に適用する。
        # T = I + U @ V.T なので、変換行列そのものは materialize しない。
        u_out, v_out, u_in, v_in = self._factors(x, depth_step)

        if x.ndim != 3:
            raise ValueError("DepthSuperLinear expects x with shape (B, N, D)")
        compute_dtype = self.compute_dtype or x.dtype
        compute_x = x.to(dtype=compute_dtype)
        u_out = u_out.to(dtype=compute_dtype)
        v_out = v_out.to(dtype=compute_dtype)
        u_in = u_in.to(dtype=compute_dtype)
        v_in = v_in.to(dtype=compute_dtype)
        # depth-only の因子は (1, ..., ...) で、バッチ全体に broadcast する。
        input_delta = torch.matmul(compute_x, v_in)
        transformed_x = compute_x + torch.matmul(
            input_delta, u_in.transpose(-1, -2)
        )

        super_weight = self.super_weight.to(dtype=compute_dtype)
        output = torch.matmul(transformed_x, super_weight.transpose(0, 1))

        output_delta = torch.matmul(output, v_out)
        output = output + torch.matmul(
            output_delta, u_out.transpose(-1, -2)
        )
        if self.bias is not None:
            output = output + self.bias.to(dtype=compute_dtype)
        return output

class DepthRMSNorm(nn.Module):
    """depth_step だけから RMSNorm のスケールを生成する RMSNorm。

    x は正規化の対象としては使うが、動的スケールの生成条件には使わない。
    """
    def __init__(self, dim, condition_dim=64, eps=1e-6,
                 depth_embedding=None):
        super().__init__()
        self.dim = dim
        self.eps = eps
        hidden_dim = max(condition_dim, 32)
        self.depth_embedding = (
            depth_embedding
            if depth_embedding is not None
            else DepthStepEmbedding()
        )
        self.scale_generator = nn.Sequential(
            nn.Linear(self.depth_embedding.output_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )
        # 初期状態を通常の RMSNorm (scale=1) にする。
        self.scale_generator[-1]._preserve_init = True
        nn.init.zeros_(self.scale_generator[-1].weight)
        nn.init.zeros_(self.scale_generator[-1].bias)

    def forward(self, x, depth_step):
        if x.ndim != 3:
            raise ValueError("DepthRMSNorm expects x with shape (B, N, D)")
        if (
            torch.is_tensor(depth_step)
            and depth_step.ndim == 2
            and depth_step.shape[-1] == self.depth_embedding.output_dim
        ):
            depth_features = depth_step.to(device=x.device, dtype=x.dtype)
        else:
            if not torch.is_tensor(depth_step):
                depth_step = x.new_tensor(depth_step)
            depth = depth_step.to(
                device=x.device, dtype=x.dtype
            ).clamp(0.0, 1.0).expand(x.shape[0], 1)
            depth_features = self.depth_embedding(depth)
        # 有界な動的 scale。1.0 からの変化を小さく開始する。
        scale = 1.0 + 0.1 * torch.tanh(self.scale_generator(depth_features))
        rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * scale[:, None, :]

class DepthHeadGate(nn.Module):
    """depth_step だけから Attention のヘッドゲートを生成する。"""
    def __init__(self, dim, num_heads, condition_dim=64,
                 depth_embedding=None):
        super().__init__()
        hidden_dim = max(condition_dim, 32)
        self.depth_embedding = (
            depth_embedding
            if depth_embedding is not None
            else DepthStepEmbedding()
        )
        self.generator = nn.Sequential(
            nn.Linear(self.depth_embedding.output_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads),
        )
        self.generator[-1]._preserve_init = True
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.zeros_(self.generator[-1].bias)

    def forward(self, x, depth_step):
        if (
            torch.is_tensor(depth_step)
            and depth_step.ndim == 2
            and depth_step.shape[-1] == self.depth_embedding.output_dim
        ):
            depth_features = depth_step.to(device=x.device, dtype=x.dtype)
        else:
            if not torch.is_tensor(depth_step):
                depth_step = x.new_tensor(depth_step)
            depth = depth_step.to(
                device=x.device, dtype=x.dtype
            ).clamp(0.0, 1.0).expand(x.shape[0], 1)
            depth_features = self.depth_embedding(depth)
        logits = 0.1 * torch.tanh(self.generator(depth_features))
        return torch.sigmoid(logits)

class DepthSuperGatedLinear(nn.Module):
    """DepthSuperLinear を使った depth step 条件付き GatedLinear。"""
    def __init__(self, dim_in, dim_out, condition_dim=64,
                 transform_rank=4, bias=False, depth_embedding=None,
                 compute_dtype=None,
                 ):
        super().__init__()
        self.proj = DepthSuperLinear(
            dim_in, dim_out * 2, condition_dim=condition_dim,
            transform_rank=transform_rank, bias=bias,
            depth_embedding=depth_embedding,
            compute_dtype=compute_dtype,
        )

    def forward(self, x, depth_step):
        x1, x2 = self.proj(x, depth_step).chunk(2, dim=-1)
        return F.silu(x1) * x2

class SharedDepthAttention(nn.Module):
    """Q/K/V/O のスーパー行列を共有する depth step 条件付き Attention。"""
    def __init__(self, embed_dim, num_heads, condition_dim=64,
                 transform_rank=4, dropout=0.0, bias=False,
                 depth_embedding=None, compute_dtype=None,
                 rope_base=10000.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base)
        # Q/K は同じ入力から生成されるため、射影を結合する。
        self.qk_proj = DepthSuperLinear(
            embed_dim, embed_dim * 2, condition_dim, transform_rank, bias,
            depth_embedding=depth_embedding,
            compute_dtype=compute_dtype,
        )
        self.v_proj = DepthSuperGatedLinear(
            embed_dim, embed_dim, condition_dim, transform_rank, bias,
            depth_embedding=depth_embedding,
            compute_dtype=compute_dtype,
        )
        self.out_proj = DepthSuperLinear(
            embed_dim, embed_dim, condition_dim, transform_rank, bias,
            depth_embedding=depth_embedding,
            compute_dtype=compute_dtype,
        )
        self.gate = DepthHeadGate(
            embed_dim, num_heads, condition_dim=condition_dim,
            depth_embedding=depth_embedding,
        )

    def forward(self, x, depth_step, key_padding_mask=None,
                attn_mask=None, need_weights=False):
        batch_size, seq_len, _ = x.shape
        qk = self.qk_proj(x, depth_step)
        q, k = qk.chunk(2, dim=-1)
        q = q.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = k.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        q = self.rope(q)
        k = self.rope(k)
        v = self.v_proj(x, depth_step).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        if need_weights:
            # Attention重みが必要な場合だけ手動計算する。
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            scores = torch.nan_to_num(
                scores, nan=0.0, posinf=30.0, neginf=-30.0
            )
            if attn_mask is not None:
                scores = scores + attn_mask
            if key_padding_mask is not None:
                scores = scores.masked_fill(
                    key_padding_mask[:, None, None, :], float("-inf")
                )
            probs = F.softmax(scores, dim=-1)
            probs = F.dropout(probs, p=self.dropout, training=self.training)
            output = torch.matmul(probs, v)
        else:
            # 通常のEncoder経路は fused / Flash Attention 対応のSDPAを使う。
            sdpa_mask = None
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    sdpa_mask = attn_mask
                else:
                    sdpa_mask = attn_mask.to(dtype=q.dtype)
            if key_padding_mask is not None:
                padding_mask = key_padding_mask[:, None, None, :]
                if sdpa_mask is None:
                    sdpa_mask = torch.zeros(
                        (batch_size, 1, seq_len, seq_len),
                        device=q.device,
                        dtype=q.dtype,
                    )
                elif sdpa_mask.dtype == torch.bool:
                    bool_mask = sdpa_mask
                    sdpa_mask = torch.zeros(
                        (batch_size, 1, seq_len, seq_len),
                        device=q.device,
                        dtype=q.dtype,
                    ).masked_fill(~bool_mask, float("-inf"))
                sdpa_mask = sdpa_mask.masked_fill(padding_mask, float("-inf"))
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=sdpa_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )

        output = output.transpose(1, 2)
        gate = self.gate(x, depth_step)
        output = output * gate[:, None, :, None].to(dtype=output.dtype)
        output = output.contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output, depth_step)
        if need_weights:
            return output, probs.mean(dim=1)
        return output

class SharedDepthTransformerBlock(nn.Module):
    """共有スーパー行列と共有Normを使う Transformer block。"""
    def __init__(self, attention, ffn_in, ffn_out, norm1, norm2,
                 drop_out=0.1):
        super().__init__()
        self.norm1 = norm1
        self.attn = attention
        self.attn_dropout = nn.Dropout(drop_out)
        self.norm2 = norm2
        self.ffn_in = ffn_in
        self.ffn_dropout_layer = nn.Dropout(drop_out)
        self.ffn_out = ffn_out
        self.ffn_dropout = nn.Dropout(drop_out)

    def forward(self, x, depth_features, attn_mask=None,
                key_padding_mask=None):
        x_norm = self.norm1(x, depth_features)
        x = x + self.attn_dropout(
            self.attn(
                x_norm,
                depth_features,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
        )
        x_norm = self.norm2(x, depth_features)
        x_ffn = self.ffn_in(x_norm, depth_features)
        x_ffn = self.ffn_dropout_layer(x_ffn)
        x_ffn = self.ffn_out(x_ffn, depth_features)
        return x + self.ffn_dropout(x_ffn)

class SharedDepthTransformerEncoder(nn.Module):
    """
    スーパー行列と変換行列生成器を全 block で共有する Encoder。

    各 block は同じ重みバンクを使い、block の depth_step に応じて
    実際に使われる重みを変換します。入力 x は重み生成条件には使いません。
    """
    def __init__(self, num_layers, embed_dim, num_heads, condition_dim=64,
                 transform_rank=4, drop_out=0.1,
                 depth_embedding_dim=None, compute_dtype=None,
                 rope_base=10000.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        hidden_dim = embed_dim * 3
        self.depth_embedding = DepthStepEmbedding(
            embedding_dim=4 if depth_embedding_dim is None else depth_embedding_dim
        )
        attention = SharedDepthAttention(
            embed_dim, num_heads, condition_dim, transform_rank, drop_out,
            depth_embedding=self.depth_embedding,
            compute_dtype=compute_dtype,
            rope_base=rope_base,
        )
        ffn_in = DepthSuperGatedLinear(
            embed_dim, hidden_dim, condition_dim, transform_rank, bias=False,
            depth_embedding=self.depth_embedding,
            compute_dtype=compute_dtype,
        )
        ffn_out = DepthSuperGatedLinear(
            hidden_dim, embed_dim, condition_dim, transform_rank, bias=False,
            depth_embedding=self.depth_embedding,
            compute_dtype=compute_dtype,
        )
        norm1 = DepthRMSNorm(
            embed_dim, condition_dim=condition_dim,
            depth_embedding=self.depth_embedding,
        )
        norm2 = DepthRMSNorm(
            embed_dim, condition_dim=condition_dim,
            depth_embedding=self.depth_embedding,
        )

        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            SharedDepthTransformerBlock(
                attention, ffn_in, ffn_out, norm1, norm2,
                drop_out=drop_out,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(embed_dim)

    def forward(self, x, depth=None, depth_steps=None, causal=False,
                attention_mask=None):
        """最大深度以下の任意の block 数で forward する。"""
        if depth is None:
            depth = self.num_layers
        if not isinstance(depth, int) or not 1 <= depth <= self.num_layers:
            raise ValueError(
                f"depth must be an integer in [1, {self.num_layers}]"
            )

        if depth_steps is None:
            depth_steps = [
                0.0 if depth == 1 else i / (depth - 1)
                for i in range(depth)
            ]
        if len(depth_steps) != depth:
            raise ValueError("depth_steps length must equal depth")

        attn_mask = None
        if causal:
            seq_len = x.shape[1]
            attn_mask = torch.zeros(
                (seq_len, seq_len), device=x.device, dtype=x.dtype
            ).masked_fill(
                torch.triu(
                    torch.ones(
                        (seq_len, seq_len), device=x.device, dtype=torch.bool
                    ),
                    diagonal=1,
                ),
                float("-inf"),
            )

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.to(
                device=x.device, dtype=torch.bool
            )

        for layer, depth_step in zip(self.layers[:depth], depth_steps):
            if not torch.is_tensor(depth_step):
                depth_step = x.new_tensor(depth_step)
            # depth step は実行する depth の範囲で常に 0.0～1.0 にする。
            # 最大 depth ではなく、今回実行する block 数だけに依存する。
            depth_value = depth_step.to(
                device=x.device, dtype=x.dtype
            ).clamp(0.0, 1.0).reshape(1, 1)
            if depth <= 1:
                step_interval = x.new_zeros((1, 1))
            else:
                step_interval = x.new_tensor(
                    1.0 / (depth - 1)
                ).reshape(1, 1)
            depth_features = self.depth_embedding(
                depth_value, step_interval
            )
            x = layer(
                x,
                depth_features,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
        return self.norm(x)

DynamicTransformerEncoder = SharedDepthTransformerEncoder

class SharedDepthTransformerDecoder(SharedDepthTransformerEncoder):
    def forward(self, x, depth=None, depth_steps=None, attention_mask=None):
        return super().forward(
            x,
            depth=depth,
            depth_steps=depth_steps,
            causal=True,
            attention_mask=attention_mask,
        )

SharedInputDepthAttention = SharedDepthAttention

SharedInputDepthTransformerBlock = SharedDepthTransformerBlock

SharedInputDepthTransformerEncoder = SharedDepthTransformerEncoder
