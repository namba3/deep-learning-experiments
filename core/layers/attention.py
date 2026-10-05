import torch
import torch.nn as nn
import torch.nn.functional as F
from .positional import RotaryEmbedding, rotate_rope_pairs
from .feedforward import GatedLinear, InterpolatedSuperLinear



class GroupedQueryAttention(nn.Module):
    """Multi-head attention with optional GQA and head-wise output gates.

    ``gate_mode`` controls the independent head output gate:

    * ``"none"``: regular MHA/GQA;
    * ``"static"``: one learned scalar per query head;
    * ``"query"``: a query-dependent gate for every token and head.

    The regular path uses fused scaled-dot-product attention.  Attention maps
    are materialized only when ``need_weights=True`` is requested.
    """
    def __init__(self, embed_dim, num_heads, kv_heads=None, dropout=0.0,
                 bias=False, batch_first=True, kdim=None, vdim=None,
                 gate_mode="query", static_gate_scale=2.0,
                 add_bias_kv=False, add_zero_attn=False, fuse_kv=True,
                 device=None, dtype=None, rope_base=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        if kv_heads is None:
            kv_heads = num_heads
        if num_heads <= 0 or kv_heads <= 0 or num_heads % kv_heads != 0:
            raise ValueError("num_heads must be divisible by positive kv_heads")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if gate_mode not in {"none", "static", "query"}:
            raise ValueError("gate_mode must be 'none', 'static', or 'query'")
        if static_gate_scale <= 0:
            raise ValueError("static_gate_scale must be positive")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.kdim = embed_dim if kdim is None else kdim
        self.vdim = embed_dim if vdim is None else vdim
        self.gate_mode = gate_mode
        self.static_gate_scale = static_gate_scale
        self.add_zero_attn = add_zero_attn
        self.rope = (
            RotaryEmbedding(self.head_dim, base=rope_base)
            if rope_base is not None else None
        )
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.kv_proj = None
        if fuse_kv and self.kdim == self.vdim:
            self.kv_proj = nn.Linear(
                self.kdim, 2 * kv_heads * self.head_dim, bias=bias,
                **factory_kwargs,
            )
            self.k_proj = self.v_proj = None
        else:
            self.k_proj = nn.Linear(
                self.kdim, kv_heads * self.head_dim, bias=bias,
                **factory_kwargs,
            )
            self.v_proj = nn.Linear(
                self.vdim, kv_heads * self.head_dim, bias=bias,
                **factory_kwargs,
            )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, **factory_kwargs,
        )
        self.bias_k = self.bias_v = None
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(
                1, 1, kv_heads * self.head_dim, **factory_kwargs,
            ))
            self.bias_v = nn.Parameter(torch.empty(
                1, 1, kv_heads * self.head_dim, **factory_kwargs,
            ))
        if gate_mode == "static":
            self.gate = nn.Parameter(torch.zeros(num_heads, **factory_kwargs))
            self.head_gate = None
        elif gate_mode == "query":
            self.gate = None
            self.head_gate = nn.Linear(
                embed_dim, num_heads, **factory_kwargs,
            )
            self.head_gate._preserve_init = True
            nn.init.zeros_(self.head_gate.weight)
            nn.init.zeros_(self.head_gate.bias)
        else:
            self.gate = self.head_gate = None
        self._reset_parameters()

    def _reset_parameters(self):
        for module in (self.q_proj, self.kv_proj, self.k_proj, self.v_proj, self.out_proj):
            if module is not None:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
            nn.init.xavier_normal_(self.bias_v)

    def _project(self, query, key, value):
        batch, query_tokens, _ = query.shape
        source_tokens = key.shape[1]
        q = self.q_proj(query).reshape(
            batch, query_tokens, self.num_heads, self.head_dim,
        ).transpose(1, 2)
        if self.kv_proj is not None:
            kv = self.kv_proj(key).reshape(
                batch, source_tokens, 2, self.kv_heads, self.head_dim,
            ).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
        else:
            if self.k_proj is None or self.v_proj is None:
                raise RuntimeError("separate K/V projections are not initialized")
            k = self.k_proj(key).reshape(
                batch, source_tokens, self.kv_heads, self.head_dim,
            ).transpose(1, 2)
            v = self.v_proj(value).reshape(
                batch, source_tokens, self.kv_heads, self.head_dim,
            ).transpose(1, 2)
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        if self.bias_k is not None:
            bias_k = self.bias_k.reshape(1, self.kv_heads, 1, self.head_dim)
            bias_v = self.bias_v.reshape(1, self.kv_heads, 1, self.head_dim)
            k = torch.cat((k, bias_k.expand(batch, -1, -1, -1)), dim=2)
            v = torch.cat((v, bias_v.expand(batch, -1, -1, -1)), dim=2)
        if self.add_zero_attn:
            zeros = torch.zeros(
                batch, self.kv_heads, 1, self.head_dim,
                device=k.device, dtype=k.dtype,
            )
            k = torch.cat((k, zeros), dim=2)
            v = torch.cat((v, zeros.to(dtype=v.dtype)), dim=2)
        return q, k, v

    @staticmethod
    def _merge_masks(attn_mask, key_padding_mask, batch, query_tokens, source_tokens,
                     device, dtype):
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(device=device, dtype=torch.bool)
            if key_padding_mask.shape != (batch, source_tokens):
                raise ValueError("key_padding_mask must have shape (batch, source_tokens)")
            valid = ~key_padding_mask[:, None, None, :]
            if attn_mask is None:
                attn_mask = valid
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.to(device=device) & valid
            else:
                padding = torch.zeros((), device=device, dtype=dtype).masked_fill(
                    ~valid, float("-inf"),
                )
                attn_mask = attn_mask.to(device=device, dtype=dtype) + padding
        elif attn_mask is not None:
            attn_mask = attn_mask.to(
                device=device,
                dtype=attn_mask.dtype if attn_mask.dtype == torch.bool else dtype,
            )
        return attn_mask

    def _apply_gate(self, attention, query):
        if self.gate_mode == "static":
            if self.gate is None:
                raise RuntimeError("static gate is not initialized")
            gate = self.static_gate_scale * torch.sigmoid(self.gate)
            return attention * gate[None, :, None, None]
        if self.gate_mode == "query":
            if self.head_gate is None:
                raise RuntimeError("query gate is not initialized")
            gate = 2.0 * torch.sigmoid(self.head_gate(query))
            return attention * gate.transpose(1, 2).unsqueeze(-1)
        return attention

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=False, attn_mask=None):
        if self.batch_first:
            query_batch, key_batch, value_batch = query, key, value
        else:
            query_batch = query.transpose(0, 1)
            key_batch = key.transpose(0, 1)
            value_batch = value.transpose(0, 1)
        batch, query_tokens, _ = query_batch.shape
        source_tokens = key_batch.shape[1]
        if value_batch.shape[1] != source_tokens:
            raise ValueError("key and value must have the same source length")
        q, k, v = self._project(query_batch, key_batch, value_batch)
        if self.bias_k is not None or self.add_zero_attn:
            # Bias/zero tokens are appended after the user-visible source mask.
            source_tokens_with_extra = k.shape[2]
            if key_padding_mask is not None:
                extra = source_tokens_with_extra - source_tokens
                key_padding_mask = F.pad(key_padding_mask, (0, extra), value=False)
            if attn_mask is not None:
                extra = source_tokens_with_extra - source_tokens
                attn_mask = F.pad(attn_mask, (0, extra), value=0)
            source_tokens = source_tokens_with_extra
        attn_mask = self._merge_masks(
            attn_mask, key_padding_mask, batch, query_tokens, source_tokens,
            q.device, q.dtype,
        )
        dropout_p = self.dropout if self.training else 0.0
        if need_weights:
            # Returning maps necessarily leaves the fused SDPA fast path.
            expanded_k = k.repeat_interleave(self.num_heads // self.kv_heads, dim=1)
            expanded_v = v.repeat_interleave(self.num_heads // self.kv_heads, dim=1)
            scores = torch.matmul(q, expanded_k.transpose(-2, -1))
            scores = scores / (self.head_dim ** 0.5)
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    scores = scores.masked_fill(~attn_mask, float("-inf"))
                else:
                    scores = scores + attn_mask
            valid_rows = torch.isfinite(scores).any(dim=-1, keepdim=True)
            safe_scores = torch.where(valid_rows, scores, torch.zeros_like(scores))
            attn_probs = F.softmax(safe_scores, dim=-1)
            attn_probs = torch.where(
                valid_rows, attn_probs, torch.zeros_like(attn_probs),
            )
            attn_probs = F.dropout(attn_probs, p=dropout_p, training=self.training)
            attention = torch.matmul(attn_probs, expanded_v)
            weights = attn_probs.mean(dim=1)
        else:
            attention = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                enable_gqa=self.num_heads != self.kv_heads,
            )
            weights = None
        attention = self._apply_gate(attention, query_batch)
        output = attention.transpose(1, 2).reshape(
            batch, query_tokens, self.embed_dim,
        )
        output = self.out_proj(output)
        if not self.batch_first:
            output = output.transpose(0, 1)
        if need_weights:
            return output, weights
        return output

class CausalMHLA(nn.Module):
    """Causal sequence MHLA for decoder-only token streams.

    ``Grid2DMHLA`` is bidirectional and requires a 2D grid, so it is not
    suitable for an autoregressive text model.  This text variant uses the
    same positive-feature linear-attention idea with cumulative K/V states;
    position ``t`` only consumes positions ``<= t``.

    Input/output shape is ``(B, T, D)``.  Q has ``num_heads`` heads and K/V
    have ``kv_heads`` heads.
    """
    def __init__(self, embed_dim, num_heads, kv_heads=None, dropout=0.0,
                 bias=False, rope_base=10000.0):
        super().__init__()
        if kv_heads is None:
            kv_heads = num_heads
        if num_heads <= 0 or kv_heads <= 0 or num_heads % kv_heads:
            raise ValueError("num_heads must be divisible by positive kv_heads")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.kv_proj = nn.Linear(
            embed_dim, 2 * kv_heads * self.head_dim, bias=bias,
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base)
        self.head_gate = nn.Linear(embed_dim, num_heads)
        self.head_gate._preserve_init = True
        nn.init.zeros_(self.head_gate.weight)
        nn.init.zeros_(self.head_gate.bias)
        self._reset_parameters()

    def _reset_parameters(self):
        for module in (self.q_proj, self.kv_proj, self.out_proj):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, key_padding_mask=None):
        if tokens.ndim != 3:
            raise ValueError("CausalMHLA expects tokens with shape (B, T, D)")
        batch, token_count, _ = tokens.shape
        query = self.q_proj(tokens).reshape(
            batch, token_count, self.num_heads, self.head_dim,
        ).transpose(1, 2)
        kv = self.kv_proj(tokens).reshape(
            batch, token_count, 2, self.kv_heads, self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        key, value = kv.unbind(0)
        query = self.rope(self.q_norm(query))
        key = self.rope(self.k_norm(key))
        query_features = F.elu(query) + 1.0
        key_features = F.elu(key) + 1.0
        if key_padding_mask is not None:
            valid = ~key_padding_mask.to(
                device=tokens.device, dtype=torch.bool,
            )
            key_features = key_features * valid[:, None, :, None]
            value = value * valid[:, None, :, None]

        # Prefix sums enforce the causal contract without materializing a
        # T-by-T attention matrix.
        kv_state = torch.einsum(
            "bhtd,bhte->bhtde", key_features, value,
        ).cumsum(dim=2)
        key_state = key_features.cumsum(dim=2)
        repeats = self.num_heads // self.kv_heads
        grouped_query = query_features.reshape(
            batch, self.kv_heads, repeats, token_count, self.head_dim,
        )
        numerator = torch.einsum(
            "bhgtd,bhtde->bhgte", grouped_query, kv_state,
        )
        denominator = torch.einsum(
            "bhgtd,bhtd->bhgt", grouped_query, key_state,
        ).clamp_min(1e-6).unsqueeze(-1)
        output = (numerator / denominator).reshape(
            batch, self.num_heads, token_count, self.head_dim,
        ).transpose(1, 2).reshape(
            batch, token_count, self.embed_dim,
        )
        gate = 2.0 * torch.sigmoid(self.head_gate(tokens))
        output = output.reshape(
            batch, token_count, self.num_heads, self.head_dim,
        ) * gate.unsqueeze(-1)
        output = output.reshape(batch, token_count, self.embed_dim)
        return self.out_proj(output)

class GatedMultiheadAttention(GroupedQueryAttention):
    """Compatibility wrapper for the former full MHA implementation.

    It keeps the old constructor and return contract while using the common
    GQA implementation with ``kv_heads=num_heads`` and a static head gate.
    ``need_weights=False`` still returns ``(output, None)`` as before.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=False,
                 add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None,
                 batch_first=False, device=None, dtype=None, rope_base=None):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kv_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            kdim=kdim,
            vdim=vdim,
            gate_mode="static",
            static_gate_scale=1.0,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            fuse_kv=False,
            device=device,
            dtype=dtype,
            rope_base=rope_base,
        )

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        output = super().forward(
            query, key, value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
        )
        return output if need_weights else (output, None)

class Grid2DMHLA(nn.Module):
    """Block-routed linear attention for a regular 2D token grid.

    Tokens are grouped into regular spatial blocks.  Each source block creates
    a linear-attention KV summary, while query-block similarities route and
    mix those summaries before token-level evaluation.  Dropout is applied by
    the enclosing residual block to the attention output, not inside the
    normalized linear-attention calculation.

    The implementation is vectorized and keeps the ``(B, heads, tokens,
    head_dim)`` convention used by the other attention modules in this file.
    """
    def __init__(self, dim, heads, kv_heads=None, block_size=8,
                 backend="auto", use_head_gate=True):
        super().__init__()
        if kv_heads is None:
            kv_heads = heads
        if heads <= 0 or kv_heads <= 0 or heads % kv_heads:
            raise ValueError("heads must be divisible by positive kv_heads")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if (dim // heads) % 4:
            raise ValueError("head_dim must be divisible by 4 for 2D RoPE")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if backend not in {"auto", "vectorized", "triton"}:
            raise ValueError(
                "Grid2DMHLA backend must be 'auto', 'vectorized', or 'triton'"
            )
        self.dim = dim
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim // heads
        self.block_size = int(block_size)
        self.backend = backend
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, 2 * kv_heads * self.head_dim)
        self.out_proj = nn.Linear(dim, dim)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.head_gate = nn.Linear(dim, heads) if use_head_gate else None
        if self.head_gate is not None:
            self.head_gate._preserve_init = True
            nn.init.zeros_(self.head_gate.weight)
            nn.init.zeros_(self.head_gate.bias)
        self._layout_cache = {}
        self._rope_cache = {}

    def _resolve_backend(self, query, key, value):
        if self.backend == "vectorized":
            return "vectorized"
        from ..mhla import triton_available

        available = triton_available(query, key, value)
        if self.backend == "triton" and not available:
            from ..mhla import triton_unavailable_reason

            raise RuntimeError(
                "Grid2DMHLA backend='triton' requires CUDA, Triton, and matching "
                "float16/bfloat16/float32 QKV tensors: "
                + triton_unavailable_reason(query, key, value)
            )
        return "triton" if available else "vectorized"

    def _layout(self, height, width, device):
        cache_key = (height, width, device.type, device.index, self.block_size)
        cached = self._layout_cache.get(cache_key)
        if cached is not None:
            return cached
        positions = torch.arange(height * width, device=device).reshape(height, width)
        blocks = []
        for row in range(0, height, self.block_size):
            for column in range(0, width, self.block_size):
                block = positions[
                    row:min(row + self.block_size, height),
                    column:min(column + self.block_size, width),
                ].flatten()
                blocks.append(block)
        max_tokens = self.block_size * self.block_size
        gather_index = torch.stack([
            F.pad(block, (0, max_tokens - block.numel()), value=0)
            for block in blocks
        ])
        valid = torch.stack([
            F.pad(
                torch.ones(block.numel(), device=device, dtype=torch.bool),
                (0, max_tokens - block.numel()),
                value=False,
            )
            for block in blocks
        ])
        cached = (gather_index, valid)
        self._layout_cache[cache_key] = cached
        return cached

    def _rope(self, height, width, device, dtype):
        cache_key = (height, width, device.type, device.index, dtype)
        cached = self._rope_cache.get(cache_key)
        if cached is not None:
            return cached
        quarter_dim = self.head_dim // 4
        inverse_frequency = 1.0 / (
            10000.0 ** (
                torch.arange(quarter_dim, device=device, dtype=torch.float32)
                / quarter_dim
            )
        )
        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        y_phase = y.flatten()[:, None] * inverse_frequency[None, :]
        x_phase = x.flatten()[:, None] * inverse_frequency[None, :]
        phase = torch.cat(
            (y_phase.repeat_interleave(2, dim=-1),
             x_phase.repeat_interleave(2, dim=-1)),
            dim=-1,
        )
        cached = (
            phase.cos().to(dtype=dtype)[None, None],
            phase.sin().to(dtype=dtype)[None, None],
        )
        self._rope_cache[cache_key] = cached
        return cached

    def forward(self, tokens, height, width):
        batch, token_count, _ = tokens.shape
        if token_count != height * width:
            raise ValueError("height * width must equal the token count")
        q = self.q_proj(tokens).reshape(
            batch, token_count, self.heads, self.head_dim,
        ).transpose(1, 2)
        kv = self.kv_proj(tokens).reshape(
            batch, token_count, 2, self.kv_heads, self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        key, value = kv.unbind(0)
        query = self.q_norm(q)
        key = self.k_norm(key)
        if query.dtype != value.dtype:
            query = query.to(dtype=value.dtype)
        if key.dtype != value.dtype:
            key = key.to(dtype=value.dtype)
        cos, sin = self._rope(height, width, query.device, query.dtype)
        query = rotate_rope_pairs(query, cos, sin)
        key = rotate_rope_pairs(key, cos, sin)

        gather_index, valid = self._layout(height, width, tokens.device)
        block_count, max_tokens = gather_index.shape
        backend = self._resolve_backend(query, key, value)
        if backend == "triton":
            from ..mhla import attention as triton_attention

            output = triton_attention(
                query,
                key,
                value,
                gather_index,
                valid,
                self.heads,
                self.kv_heads,
            )
            output = output.transpose(1, 2).reshape(
                batch, token_count, self.dim,
            )
            if self.head_gate is not None:
                gate = 2.0 * torch.sigmoid(self.head_gate(tokens))
                output = output.reshape(
                    batch, token_count, self.heads, self.head_dim,
                ) * gate.unsqueeze(-1)
                output = output.reshape(batch, token_count, self.dim)
            return self.out_proj(output)
        flat_index = gather_index.flatten()
        packed_query = query.index_select(2, flat_index).reshape(
            batch, self.heads, block_count, max_tokens, self.head_dim,
        )
        packed_key = key.index_select(2, flat_index).reshape(
            batch, self.kv_heads, block_count, max_tokens, self.head_dim,
        )
        packed_value = value.index_select(2, flat_index).reshape(
            batch, self.kv_heads, block_count, max_tokens, self.head_dim,
        )
        packed_valid = valid[None, None, :, :, None]
        packed_mask = packed_valid.to(dtype=query.dtype)
        query_for_mix = packed_query.reshape(
            batch, self.kv_heads, self.heads // self.kv_heads,
            block_count, max_tokens, self.head_dim,
        ).mean(dim=2)
        query_for_mix = query_for_mix * packed_mask
        key_for_mix = packed_key * packed_mask
        counts = packed_mask.sum(dim=3).clamp_min(1.0)
        query_blocks = query_for_mix.sum(dim=3) / counts
        key_blocks = key_for_mix.sum(dim=3) / counts
        mix_logits = torch.einsum(
            "bhmd,bhnd->bhmn", query_blocks, key_blocks,
        ) / (self.head_dim ** 0.5)
        block_has_tokens = valid.any(dim=1)
        mix_logits = mix_logits.masked_fill(
            ~block_has_tokens[None, None, None, :],
            torch.finfo(mix_logits.dtype).min,
        )
        mix = torch.softmax(mix_logits, dim=-1)

        feature_key = (F.elu(packed_key) + 1.0) * packed_mask
        packed_value = packed_value * packed_mask
        summaries = torch.einsum(
            "bhmld,bhmle->bhmde", feature_key, packed_value,
        )
        normalizers = feature_key.sum(dim=3)
        mixed_summaries = torch.einsum(
            "bhmn,bhnde->bhmde", mix, summaries,
        )
        mixed_normalizers = torch.einsum(
            "bhmn,bhnd->bhmd", mix, normalizers,
        )
        if self.heads != self.kv_heads:
            repeats = self.heads // self.kv_heads
            mixed_summaries = mixed_summaries.repeat_interleave(repeats, dim=1)
            mixed_normalizers = mixed_normalizers.repeat_interleave(repeats, dim=1)
        feature_query = F.elu(packed_query) + 1.0
        packed_output = torch.einsum(
            "bhmld,bhmde->bhmle", feature_query, mixed_summaries,
        )
        denominator = torch.einsum(
            "bhmld,bhmd->bhml", feature_query, mixed_normalizers,
        ).unsqueeze(-1)
        packed_output = (
            packed_output / denominator.clamp_min(1e-6)
        ) * packed_mask
        scatter_index = flat_index[None, None, :, None].expand(
            batch, self.heads, block_count * max_tokens, self.head_dim,
        )
        output = torch.zeros(
            batch, self.heads, token_count, self.head_dim,
            device=tokens.device, dtype=packed_output.dtype,
        ).scatter_add(
            2,
            scatter_index,
            packed_output.reshape(batch, self.heads, block_count * max_tokens, self.head_dim),
        )
        output = output.transpose(1, 2).reshape(batch, token_count, self.dim)
        if self.head_gate is not None:
            gate = 2.0 * torch.sigmoid(self.head_gate(tokens))
            output = output.reshape(
                batch, token_count, self.heads, self.head_dim,
            ) * gate.unsqueeze(-1)
            output = output.reshape(batch, token_count, self.dim)
        return self.out_proj(output)

class InterpolatedGatedMultiheadAttention(nn.Module):
    """Q/K/V と出力射影を depth_step で動的に生成する Attention。"""
    def __init__(self, embed_dim, num_heads, num_steps=4,
                 dropout=0.0, bias=False, batch_first=True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        self.q_proj = InterpolatedSuperLinear(
            embed_dim, embed_dim, num_steps=num_steps, bias=bias
        )
        self.k_proj = InterpolatedSuperLinear(
            embed_dim, embed_dim, num_steps=num_steps, bias=bias
        )
        self.v_proj = InterpolatedSuperLinear(
            embed_dim, embed_dim, num_steps=num_steps, bias=bias
        )
        self.out_proj = InterpolatedSuperLinear(
            embed_dim, embed_dim, num_steps=num_steps, bias=bias
        )
        self.gate = nn.Parameter(torch.zeros(num_heads))

    def forward(self, query, key, value, depth_step,
                key_padding_mask=None, attn_mask=None, need_weights=False):
        if not self.batch_first:
            query, key, value = (
                query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
            )

        batch_size, query_len, _ = query.shape
        source_len = key.shape[1]

        q = self.q_proj(query, depth_step).view(
            batch_size, query_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(key, depth_step).view(
            batch_size, source_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(value, depth_step).view(
            batch_size, source_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            scores = scores + attn_mask
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )

        probs = F.softmax(scores, dim=-1)
        probs = F.dropout(probs, p=self.dropout, training=self.training)
        output = torch.matmul(probs, v).transpose(1, 2)
        output = output * torch.sigmoid(self.gate)[None, None, :, None]
        output = output.contiguous().view(batch_size, query_len, self.embed_dim)
        output = self.out_proj(output, depth_step)

        if not self.batch_first:
            output = output.transpose(0, 1)
        if need_weights:
            return output, probs.mean(dim=1)
        return output

class KVSelfAttention(nn.Module):
    """
    Qを廃止し、K/V のみで構成された Self-Attention
    - K/V は GatedLinear により非線形射影
    - Attention は K の自己相関
    - 最後の output projection も GatedLinear
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=False,
        batch_first=False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        # --- Gated K / V projection ---
        self.k_proj = GatedLinear(embed_dim, embed_dim, bias=bias)
        self.v_proj = GatedLinear(embed_dim, embed_dim, bias=bias)

        # Output projection
        self.out_proj = GatedLinear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        x,
        key_padding_mask=None,
        attn_mask=None,
        need_weights=False,
    ):
        """
        x: (L, N, E) or (N, L, E) if batch_first
        """
        if self.batch_first:
            x = x.transpose(0, 1)  # (L, N, E)

        L, N, _ = x.size()

        # --- K / V ---
        k = self.k_proj(x)
        v = self.v_proj(x)

        # (L, N, E) -> (N, L, h, d)
        k = k.view(L, N, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(L, N, self.num_heads, self.head_dim).transpose(0, 1)

        # --- scaled dot-product (K K^T) ---
        # Scale the complete dot product once, as in standard attention.
        attn_weights = torch.einsum("nlhd,nshd->nhls", k, k)
        attn_weights = attn_weights / (self.head_dim ** 0.5)

        if attn_mask is not None:
            attn_weights += attn_mask.unsqueeze(0)

        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf")
            )

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        # --- Attention output ---
        attn_output = torch.einsum("nhls,nshd->nlhd", attn_probs, v)
        attn_output = attn_output.contiguous().view(N, L, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        if need_weights:
            # head 平均
            return attn_output, attn_probs.mean(dim=1)
        else:
            return attn_output
