"""Optional Triton kernels for Joint MHLA."""

import os

try:
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR = None
except Exception as error:  # Triton is optional; CPU and non-Triton installs use PyTorch.
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = repr(error)


if triton is not None:
    _MHLA_AUTOTUNE_ENABLED = os.environ.get("MHLA_AUTOTUNE", "1") != "0"
    _MHLA_AUTOTUNE_CONFIGS = [
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ]

    def _mhla_kernel(function):
        # Triton 3.x expects ``autotune`` to receive a JIT kernel.  Avoid
        # wrapping an already decorated kernel again: doing so makes Triton
        # inspect the JIT wrapper instead of the original Python function and
        # can fail with ``re.search(...).start()`` when its source has no
        # top-level ``def`` line.
        jitted = (
            function
            if hasattr(function, "arg_names")
            else triton.jit(function)
        )
        if not _MHLA_AUTOTUNE_ENABLED:
            return jitted
        return triton.autotune(
            configs=_MHLA_AUTOTUNE_CONFIGS,
            key=["head_dim", "block_count", "MAX_TOKENS"],
        )(jitted)

    @_mhla_kernel
    def _joint_mhla_block_means_kernel(
        query_ptr, key_ptr, valid_ptr, block_index_ptr, block_mask_ptr,
        query_mean_ptr, key_mean_ptr,
        batch_size, heads, kv_heads, token_count, head_dim, block_count,
        MAX_TOKENS: tl.constexpr, GROUP: tl.constexpr,
        TOKEN_TILE: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Compute Q/K means for every block without padded activations."""
        batch = tl.program_id(0)
        kv_head = tl.program_id(1)
        block = tl.program_id(2)
        d = tl.arange(0, BLOCK_D)
        d_mask = d < head_dim
        query_sum = tl.zeros((BLOCK_D,), dtype=tl.float32)
        key_sum = tl.zeros((BLOCK_D,), dtype=tl.float32)
        count = tl.zeros((), dtype=tl.float32)

        for token_start in range(0, MAX_TOKENS, TOKEN_TILE):
            token_offset = token_start + tl.arange(0, TOKEN_TILE)
            in_block = token_offset < MAX_TOKENS
            layout_offset = block * MAX_TOKENS + token_offset
            token_index = tl.load(
                block_index_ptr + layout_offset,
                mask=in_block,
                other=0,
            )
            valid = in_block & tl.load(
                block_mask_ptr + layout_offset,
                mask=in_block,
                other=0,
            ).to(tl.int1)
            valid = valid & tl.load(
                valid_ptr + batch * token_count + token_index,
                mask=valid,
                other=0,
            ).to(tl.int1)
            count += tl.sum(valid.to(tl.float32), axis=0)
            value_mask = valid[:, None] & d_mask[None, :]
            key_offsets = (
                ((batch * kv_heads + kv_head) * token_count + token_index[:, None])
                * head_dim + d[None, :]
            )
            key_sum += tl.sum(
                tl.load(key_ptr + key_offsets, mask=value_mask, other=0).to(tl.float32),
                axis=0,
            )
            for group in range(GROUP):
                query_head = kv_head * GROUP + group
                query_offsets = (
                    ((batch * heads + query_head) * token_count + token_index[:, None])
                    * head_dim + d[None, :]
                )
                query_sum += tl.sum(
                    tl.load(query_ptr + query_offsets, mask=value_mask, other=0).to(tl.float32),
                    axis=0,
                )

        safe_count = tl.maximum(count, 1.0)
        query_output_offset = (
            ((batch * kv_heads + kv_head) * block_count + block) * head_dim + d
        )
        key_output_offset = query_output_offset
        tl.store(
            query_mean_ptr + query_output_offset,
            query_sum / (safe_count * GROUP),
            mask=d_mask,
        )
        tl.store(
            key_mean_ptr + key_output_offset,
            key_sum / safe_count,
            mask=d_mask,
        )

    @_mhla_kernel
    def _joint_mhla_block_statistics_kernel(
        key_ptr, value_ptr, valid_ptr, block_index_ptr, block_mask_ptr,
        summary_ptr, normalizer_ptr,
        batch_size, kv_heads, token_count, head_dim, block_count,
        MAX_TOKENS: tl.constexpr, TOKEN_TILE: tl.constexpr,
        OUTPUT_TILE: tl.constexpr,
    ):
        """Compute phi(K)V and phi(K) summaries for each token block."""
        batch_kv_block = tl.program_id(0)
        block = batch_kv_block % block_count
        batch_kv = batch_kv_block // block_count
        batch = batch_kv // kv_heads
        kv_head = batch_kv % kv_heads
        d_tile = tl.program_id(1)
        e_tile = tl.program_id(2)
        d = d_tile * OUTPUT_TILE + tl.arange(0, OUTPUT_TILE)
        e = e_tile * OUTPUT_TILE + tl.arange(0, OUTPUT_TILE)
        d_mask = d < head_dim
        e_mask = e < head_dim
        normalizer = tl.zeros((OUTPUT_TILE,), dtype=tl.float32)
        summary = tl.zeros((OUTPUT_TILE, OUTPUT_TILE), dtype=tl.float32)

        for token_start in range(0, MAX_TOKENS, TOKEN_TILE):
            token_offset = token_start + tl.arange(0, TOKEN_TILE)
            in_block = token_offset < MAX_TOKENS
            layout_offset = block * MAX_TOKENS + token_offset
            token_index = tl.load(
                block_index_ptr + layout_offset,
                mask=in_block,
                other=0,
            )
            valid = in_block & tl.load(
                block_mask_ptr + layout_offset,
                mask=in_block,
                other=0,
            ).to(tl.int1)
            valid = valid & tl.load(
                valid_ptr + batch * token_count + token_index,
                mask=valid,
                other=0,
            ).to(tl.int1)
            key_offset = (
                ((batch * kv_heads + kv_head) * token_count
                 + token_index[:, None]) * head_dim + d[None, :]
            )
            value_offset = (
                ((batch * kv_heads + kv_head) * token_count
                 + token_index[:, None]) * head_dim + e[None, :]
            )
            key_value = tl.load(
                key_ptr + key_offset,
                mask=valid[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.float32)
            value_value = tl.load(
                value_ptr + value_offset,
                mask=valid[:, None] & e_mask[None, :],
                other=0,
            ).to(tl.float32)
            feature = tl.where(key_value > 0.0, key_value + 1.0, tl.exp(key_value))
            feature = tl.where(valid[:, None] & d_mask[None, :], feature, 0.0)
            value_value = tl.where(valid[:, None] & e_mask[None, :], value_value, 0.0)
            normalizer += tl.where(
                e_tile == 0,
                tl.sum(feature, axis=0),
                0.0,
            )
            if TOKEN_TILE >= 16:
                summary += tl.dot(tl.trans(feature), value_value)
            else:
                summary += tl.sum(
                    feature[:, :, None] * value_value[:, None, :], axis=0,
                )

        summary_offset = (
            (((batch * kv_heads + kv_head) * block_count + block) * head_dim
             + d[:, None]) * head_dim + e[None, :]
        )
        tl.store(
            summary_ptr + summary_offset,
            summary,
            mask=d_mask[:, None] & e_mask[None, :],
        )
        tl.store(
            normalizer_ptr + (
                ((batch * kv_heads + kv_head) * block_count + block) * head_dim + d
            ),
            normalizer,
            mask=(e_tile == 0) & d_mask,
        )

    @_mhla_kernel
    def _joint_mhla_output_kernel(
        query_ptr, query_mean_ptr, key_mean_ptr, summary_ptr, normalizer_ptr,
        valid_ptr, block_index_ptr, block_mask_ptr, block_valid_ptr,
        modality_ptr, bias_ptr, output_ptr,
        batch_size, heads, kv_heads, token_count, head_dim, block_count, scale,
        MAX_TOKENS: tl.constexpr, GROUP: tl.constexpr,
        TOKEN_TILE: tl.constexpr, BLOCK_D: tl.constexpr,
        OUTPUT_TILE: tl.constexpr, BLOCKS: tl.constexpr,
    ):
        """Apply block mixing and token-level linear attention."""
        batch = tl.program_id(0)
        head = tl.program_id(1)
        flat_output = tl.program_id(2)
        token_tiles = (MAX_TOKENS + TOKEN_TILE - 1) // TOKEN_TILE
        output_tiles = (head_dim + OUTPUT_TILE - 1) // OUTPUT_TILE
        query_block = flat_output // (token_tiles * output_tiles)
        remainder = flat_output % (token_tiles * output_tiles)
        token_tile = remainder // output_tiles
        output_tile = remainder % output_tiles
        token_offset = token_tile * TOKEN_TILE + tl.arange(0, TOKEN_TILE)
        output_offset = output_tile * OUTPUT_TILE + tl.arange(0, OUTPUT_TILE)
        in_block = token_offset < MAX_TOKENS
        output_mask = output_offset < head_dim
        layout_offset = query_block * MAX_TOKENS + token_offset
        token_index = tl.load(
            block_index_ptr + layout_offset,
            mask=in_block,
            other=0,
        )
        valid = in_block & tl.load(
            block_mask_ptr + layout_offset,
            mask=in_block,
            other=0,
        ).to(tl.int1)
        valid = valid & tl.load(
            valid_ptr + batch * token_count + token_index,
            mask=valid,
            other=0,
        ).to(tl.int1)
        feature_mask = valid[:, None] & (tl.arange(0, BLOCK_D)[None, :] < head_dim)
        query_offsets = (
            ((batch * heads + head) * token_count + token_index[:, None])
            * head_dim + tl.arange(0, BLOCK_D)[None, :]
        )
        query_value = tl.load(
            query_ptr + query_offsets,
            mask=feature_mask,
            other=0,
        ).to(tl.float32)
        feature_query = tl.where(
            feature_mask,
            tl.where(query_value > 0.0, query_value + 1.0, tl.exp(query_value)),
            0.0,
        )

        kv_head = head // GROUP
        query_mean_offset = (
            ((batch * kv_heads + kv_head) * block_count + query_block) * head_dim
            + tl.arange(0, BLOCK_D)
        )
        query_mean = tl.load(
            query_mean_ptr + query_mean_offset,
            mask=tl.arange(0, BLOCK_D) < head_dim,
            other=0,
        ).to(tl.float32)
        query_modality = tl.load(modality_ptr + query_block)
        max_logit = tl.full((), -1.0e30, tl.float32)
        for source_block in range(BLOCKS):
            key_mean_offset = (
                ((batch * kv_heads + kv_head) * block_count + source_block) * head_dim
                + tl.arange(0, BLOCK_D)
            )
            key_mean = tl.load(
                key_mean_ptr + key_mean_offset,
                mask=tl.arange(0, BLOCK_D) < head_dim,
                other=0,
            ).to(tl.float32)
            logit = tl.sum(query_mean * key_mean, axis=0) / scale
            source_modality = tl.load(modality_ptr + source_block)
            logit += tl.load(
                bias_ptr + query_modality * 3 + source_modality,
            ).to(tl.float32)
            source_valid = tl.load(
                block_valid_ptr + batch * block_count + source_block,
            ).to(tl.int1)
            max_logit = tl.maximum(
                max_logit,
                tl.where(source_valid, logit, -1.0e30),
            )

        weight_sum = tl.zeros((), dtype=tl.float32)
        for source_block in range(BLOCKS):
            key_mean_offset = (
                ((batch * kv_heads + kv_head) * block_count + source_block) * head_dim
                + tl.arange(0, BLOCK_D)
            )
            key_mean = tl.load(
                key_mean_ptr + key_mean_offset,
                mask=tl.arange(0, BLOCK_D) < head_dim,
                other=0,
            ).to(tl.float32)
            logit = tl.sum(query_mean * key_mean, axis=0) / scale
            source_modality = tl.load(modality_ptr + source_block)
            logit += tl.load(
                bias_ptr + query_modality * 3 + source_modality,
            ).to(tl.float32)
            source_valid = tl.load(
                block_valid_ptr + batch * block_count + source_block,
            ).to(tl.int1)
            weight_sum += tl.where(source_valid, tl.exp(logit - max_logit), 0.0)
        weight_sum = tl.maximum(weight_sum, 1.0e-6)

        output = tl.zeros((TOKEN_TILE, OUTPUT_TILE), dtype=tl.float32)
        denominator = tl.zeros((TOKEN_TILE,), dtype=tl.float32)
        for source_block in range(BLOCKS):
            key_mean_offset = (
                ((batch * kv_heads + kv_head) * block_count + source_block) * head_dim
                + tl.arange(0, BLOCK_D)
            )
            key_mean = tl.load(
                key_mean_ptr + key_mean_offset,
                mask=tl.arange(0, BLOCK_D) < head_dim,
                other=0,
            ).to(tl.float32)
            logit = tl.sum(query_mean * key_mean, axis=0) / scale
            source_modality = tl.load(modality_ptr + source_block)
            logit += tl.load(
                bias_ptr + query_modality * 3 + source_modality,
            ).to(tl.float32)
            source_valid = tl.load(
                block_valid_ptr + batch * block_count + source_block,
            ).to(tl.int1)
            weight = tl.where(
                source_valid,
                tl.exp(logit - max_logit) / weight_sum,
                0.0,
            )
            summary_offsets = (
                (((batch * kv_heads + kv_head) * block_count + source_block) * head_dim
                 + tl.arange(0, BLOCK_D)[:, None]) * head_dim
                + output_offset[None, :]
            )
            summary = tl.load(
                summary_ptr + summary_offsets,
                mask=(tl.arange(0, BLOCK_D)[:, None] < head_dim)
                & output_mask[None, :],
                other=0,
            ).to(tl.float32)
            if BLOCK_D < 16:
                output += weight * tl.sum(
                    feature_query[:, :, None] * summary[None, :, :], axis=1,
                )
            else:
                output += weight * tl.dot(feature_query, summary)
            normalizer_offsets = (
                ((batch * kv_heads + kv_head) * block_count + source_block) * head_dim
                + tl.arange(0, BLOCK_D)
            )
            normalizer = tl.load(
                normalizer_ptr + normalizer_offsets,
                mask=tl.arange(0, BLOCK_D) < head_dim,
                other=0,
            ).to(tl.float32)
            denominator += weight * tl.sum(
                feature_query * normalizer[None, :], axis=1,
            )

        output = output / tl.maximum(denominator[:, None], 1.0e-6)
        output_offsets = (
            ((batch * heads + head) * token_count + token_index[:, None])
            * head_dim + output_offset[None, :]
        )
        tl.store(
            output_ptr + output_offsets,
            output,
            mask=valid[:, None] & output_mask[None, :],
        )

    @_mhla_kernel
    def _joint_mhla_backward_mix_kernel(
        query_ptr, output_ptr, grad_output_ptr,
        summary_ptr, normalizer_ptr, mix_ptr,
        valid_ptr, block_index_ptr, block_mask_ptr,
        d_mix_ptr,
        batch_size, heads, kv_heads, token_count, head_dim, block_count,
        MAX_TOKENS: tl.constexpr, GROUP: tl.constexpr,
        TOKEN_TILE: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Compute d(mix) by reducing query-token output gradients."""
        batch_kv = tl.program_id(0)
        query_block = tl.program_id(1)
        source_block = tl.program_id(2)
        batch = batch_kv // kv_heads
        kv_head = batch_kv % kv_heads
        d_mix = tl.zeros((), dtype=tl.float32)

        for group in range(GROUP):
            head = kv_head * GROUP + group
            for token_start in range(0, MAX_TOKENS, TOKEN_TILE):
                token_offset = token_start + tl.arange(0, TOKEN_TILE)
                in_block = token_offset < MAX_TOKENS
                layout_offset = query_block * MAX_TOKENS + token_offset
                token_index = tl.load(
                    block_index_ptr + layout_offset,
                    mask=in_block,
                    other=0,
                )
                valid = in_block & tl.load(
                    block_mask_ptr + layout_offset,
                    mask=in_block,
                    other=0,
                ).to(tl.int1)
                valid = valid & tl.load(
                    valid_ptr + batch * token_count + token_index,
                    mask=valid,
                    other=0,
                ).to(tl.int1)
                d = tl.arange(0, BLOCK_D)
                token_mask = valid[:, None] & (d[None, :] < head_dim)
                query_offset = (
                    ((batch * heads + head) * token_count + token_index[:, None])
                    * head_dim + d[None, :]
                )
                query_value = tl.load(
                    query_ptr + query_offset, mask=token_mask, other=0,
                ).to(tl.float32)
                feature_query = tl.where(
                    token_mask,
                    tl.where(query_value > 0.0, query_value + 1.0, tl.exp(query_value)),
                    0.0,
                )
                denominator = tl.zeros((TOKEN_TILE,), dtype=tl.float32)
                for target_block in tl.range(0, block_count):
                    mix = tl.load(
                        mix_ptr + (
                            (batch * kv_heads + kv_head) * block_count
                            + query_block
                        ) * block_count + target_block,
                    ).to(tl.float32)
                    normalizer_offset = (
                        ((batch * kv_heads + kv_head) * block_count + target_block)
                        * head_dim + d
                    )
                    normalizer = tl.load(
                        normalizer_ptr + normalizer_offset,
                        mask=d < head_dim,
                        other=0,
                    ).to(tl.float32)
                    denominator += mix * tl.sum(
                        feature_query * normalizer[None, :], axis=1,
                    )
                denominator = tl.maximum(denominator, 1.0e-6)
                output_offset = (
                    ((batch * heads + head) * token_count + token_index[:, None])
                    * head_dim + d[None, :]
                )
                output_value = tl.load(
                    output_ptr + output_offset, mask=token_mask, other=0,
                ).to(tl.float32)
                grad_value = tl.load(
                    grad_output_ptr + output_offset, mask=token_mask, other=0,
                ).to(tl.float32)
                d_a = grad_value / denominator[:, None]
                d_r = -tl.sum(grad_value * output_value, axis=1) / denominator
                summary_offset = (
                    (((batch * kv_heads + kv_head) * block_count + source_block)
                     * head_dim + d[:, None]) * head_dim + d[None, :]
                )
                summary = tl.load(
                    summary_ptr + summary_offset,
                    mask=(d[:, None] < head_dim) & (d[None, :] < head_dim),
                    other=0,
                ).to(tl.float32)
                source_normalizer = tl.load(
                    normalizer_ptr + (
                        ((batch * kv_heads + kv_head) * block_count + source_block)
                        * head_dim + d
                    ),
                    mask=d < head_dim,
                    other=0,
                ).to(tl.float32)
                if BLOCK_D < 16:
                    summary_projection = tl.sum(
                        d_a[:, :, None] * summary[None, :, :], axis=2,
                    )
                else:
                    summary_projection = tl.dot(d_a, tl.trans(summary))
                source_value = tl.sum(
                    feature_query * summary_projection,
                    axis=1,
                )
                source_value += d_r * tl.sum(
                    feature_query * source_normalizer[None, :], axis=1,
                )
                d_mix += tl.sum(source_value, axis=0)

        tl.store(
            d_mix_ptr + (
                (batch * kv_heads + kv_head) * block_count + query_block
            ) * block_count + source_block,
            d_mix,
        )

    @_mhla_kernel
    def _joint_mhla_backward_summary_kernel(
        query_ptr, output_ptr, grad_output_ptr,
        summary_grad_ptr, normalizer_grad_ptr, normalizer_ptr, mix_ptr,
        valid_ptr, block_index_ptr, block_mask_ptr,
        batch_size, heads, kv_heads, token_count, head_dim, block_count,
        MAX_TOKENS: tl.constexpr, GROUP: tl.constexpr,
        TOKEN_TILE: tl.constexpr, BLOCK_D: tl.constexpr,
        OUTPUT_TILE: tl.constexpr,
    ):
        """Compute gradients of block KV summaries."""
        batch_kv_block = tl.program_id(0)
        source_block = batch_kv_block % block_count
        batch_kv = batch_kv_block // block_count
        batch = batch_kv // kv_heads
        kv_head = batch_kv % kv_heads
        d_tile = tl.program_id(1)
        e_tile = tl.program_id(2)
        d = d_tile * OUTPUT_TILE + tl.arange(0, OUTPUT_TILE)
        e = e_tile * OUTPUT_TILE + tl.arange(0, OUTPUT_TILE)
        d_mask = d < head_dim
        e_mask = e < head_dim
        summary_grad = tl.zeros((OUTPUT_TILE, OUTPUT_TILE), dtype=tl.float32)
        normalizer_grad = tl.zeros((OUTPUT_TILE,), dtype=tl.float32)
        feature_offset = tl.arange(0, BLOCK_D)

        for query_block in tl.range(0, block_count):
            mix = tl.load(
                mix_ptr + (
                    (batch * kv_heads + kv_head) * block_count + query_block
                ) * block_count + source_block,
            ).to(tl.float32)
            for group in range(GROUP):
                head = kv_head * GROUP + group
                for token_start in range(0, MAX_TOKENS, TOKEN_TILE):
                    token_offset = token_start + tl.arange(0, TOKEN_TILE)
                    in_block = token_offset < MAX_TOKENS
                    layout_offset = query_block * MAX_TOKENS + token_offset
                    token_index = tl.load(
                        block_index_ptr + layout_offset,
                        mask=in_block,
                        other=0,
                    )
                    valid = in_block & tl.load(
                        block_mask_ptr + layout_offset,
                        mask=in_block,
                        other=0,
                    ).to(tl.int1)
                    valid = valid & tl.load(
                        valid_ptr + batch * token_count + token_index,
                        mask=valid,
                        other=0,
                    ).to(tl.int1)
                    token_mask = valid[:, None] & (feature_offset[None, :] < head_dim)
                    query_offset = (
                        ((batch * heads + head) * token_count + token_index[:, None])
                        * head_dim + feature_offset[None, :]
                    )
                    query_value = tl.load(
                        query_ptr + query_offset, mask=token_mask, other=0,
                    ).to(tl.float32)
                    feature_query = tl.where(
                        token_mask,
                        tl.where(query_value > 0.0, query_value + 1.0, tl.exp(query_value)),
                        0.0,
                    )
                    denominator = tl.zeros((TOKEN_TILE,), dtype=tl.float32)
                    for target_block in tl.range(0, block_count):
                        target_mix = tl.load(
                            mix_ptr + (
                                (batch * kv_heads + kv_head) * block_count
                                + query_block
                            ) * block_count + target_block,
                        ).to(tl.float32)
                        target_normalizer = tl.load(
                            normalizer_ptr + (
                                ((batch * kv_heads + kv_head) * block_count
                                 + target_block) * head_dim + feature_offset
                            ),
                            mask=feature_offset < head_dim,
                            other=0,
                        ).to(tl.float32)
                        denominator += target_mix * tl.sum(
                            feature_query * target_normalizer[None, :], axis=1,
                        )
                    denominator = tl.maximum(denominator, 1.0e-6)
                    output_offset = (
                        ((batch * heads + head) * token_count + token_index[:, None])
                        * head_dim + feature_offset[None, :]
                    )
                    output_value = tl.load(
                        output_ptr + output_offset, mask=token_mask, other=0,
                    ).to(tl.float32)
                    grad_value = tl.load(
                        grad_output_ptr + output_offset, mask=token_mask, other=0,
                    ).to(tl.float32)
                    d_a = grad_value / denominator[:, None]
                    d_r = -tl.sum(grad_value * output_value, axis=1) / denominator
                    d_feature = tl.sum(
                        feature_query[:, None, :]
                        * (feature_offset[None, None, :] == d[None, :, None]),
                        axis=2,
                    )
                    d_output = tl.sum(
                        d_a[:, None, :]
                        * (feature_offset[None, None, :] == e[None, :, None]),
                        axis=2,
                    )
                    if TOKEN_TILE >= 16:
                        summary_grad += mix * tl.dot(
                            tl.trans(d_feature), d_output,
                        )
                    else:
                        summary_grad += tl.sum(
                            tl.where(
                                valid[:, None, None],
                                d_feature[:, :, None] * d_output[:, None, :],
                                0.0,
                            ),
                            axis=0,
                        ) * mix
                    normalizer_grad += tl.where(
                        e_tile == 0,
                        tl.sum(
                            tl.where(
                                valid[:, None], d_feature * d_r[:, None], 0.0,
                            ),
                            axis=0,
                        ) * mix,
                        0.0,
                    )

        summary_offset = (
            (((batch * kv_heads + kv_head) * block_count + source_block)
             * head_dim + d[:, None]) * head_dim + e[None, :]
        )
        tl.store(
            summary_grad_ptr + summary_offset,
            summary_grad,
            mask=d_mask[:, None] & e_mask[None, :],
        )
        tl.store(
            normalizer_grad_ptr + (
                ((batch * kv_heads + kv_head) * block_count + source_block)
                * head_dim + d
            ),
            normalizer_grad,
            mask=(e_tile == 0) & d_mask,
        )

    @_mhla_kernel
    def _joint_mhla_backward_query_kernel(
        query_ptr, output_ptr, grad_output_ptr,
        summary_ptr, normalizer_ptr, mix_ptr, query_mean_grad_ptr,
        valid_ptr, block_index_ptr, block_mask_ptr, query_grad_ptr,
        batch_size, heads, kv_heads, token_count, head_dim, block_count,
        MAX_TOKENS: tl.constexpr, GROUP: tl.constexpr,
        TOKEN_TILE: tl.constexpr, BLOCK_D: tl.constexpr,
        OUTPUT_TILE: tl.constexpr,
        TOKEN_TILES: tl.constexpr,
    ):
        """Compute query gradients, including the block-mean path."""
        batch_head = tl.program_id(0)
        head = batch_head % heads
        batch = batch_head // heads
        flat_tile = tl.program_id(1)
        query_block = flat_tile // TOKEN_TILES
        token_tile = flat_tile % TOKEN_TILES
        output_tile = tl.program_id(2)
        kv_head = head // GROUP
        feature_offset = tl.arange(0, BLOCK_D)
        output_offset = output_tile * OUTPUT_TILE + tl.arange(0, OUTPUT_TILE)
        feature_mask = feature_offset < head_dim
        output_mask = output_offset < head_dim
        token_offset = token_tile * TOKEN_TILE + tl.arange(0, TOKEN_TILE)
        in_block = token_offset < MAX_TOKENS
        layout_offset = query_block * MAX_TOKENS + token_offset
        token_index = tl.load(
            block_index_ptr + layout_offset,
            mask=in_block,
            other=0,
        )
        valid = in_block & tl.load(
            block_mask_ptr + layout_offset,
            mask=in_block,
            other=0,
        ).to(tl.int1)
        valid = valid & tl.load(
            valid_ptr + batch * token_count + token_index,
            mask=valid,
            other=0,
        ).to(tl.int1)
        token_mask = valid[:, None] & feature_mask[None, :]
        query_offset = (
            ((batch * heads + head) * token_count + token_index[:, None])
            * head_dim + feature_offset[None, :]
        )
        query_value = tl.load(
            query_ptr + query_offset, mask=token_mask, other=0,
        ).to(tl.float32)
        feature_query = tl.where(
            token_mask,
            tl.where(query_value > 0.0, query_value + 1.0, tl.exp(query_value)),
            0.0,
        )
        denominator = tl.zeros((TOKEN_TILE,), dtype=tl.float32)
        for source_block in tl.range(0, block_count):
            mix = tl.load(
                mix_ptr + (
                    (batch * kv_heads + kv_head) * block_count + query_block
                ) * block_count + source_block,
            ).to(tl.float32)
            normalizer = tl.load(
                normalizer_ptr + (
                    ((batch * kv_heads + kv_head) * block_count + source_block)
                    * head_dim + feature_offset
                ),
                mask=feature_mask,
                other=0,
            ).to(tl.float32)
            denominator += mix * tl.sum(
                feature_query * normalizer[None, :], axis=1,
            )
        denominator = tl.maximum(denominator, 1.0e-6)
        output_offset_full = (
            ((batch * heads + head) * token_count + token_index[:, None])
            * head_dim + feature_offset[None, :]
        )
        output_value = tl.load(
            output_ptr + output_offset_full, mask=token_mask, other=0,
        ).to(tl.float32)
        grad_value = tl.load(
            grad_output_ptr + output_offset_full, mask=token_mask, other=0,
        ).to(tl.float32)
        d_a = grad_value / denominator[:, None]
        d_r = -tl.sum(grad_value * output_value, axis=1) / denominator
        query_gradient = tl.zeros((TOKEN_TILE, OUTPUT_TILE), dtype=tl.float32)
        for source_block in tl.range(0, block_count):
            mix = tl.load(
                mix_ptr + (
                    (batch * kv_heads + kv_head) * block_count + query_block
                ) * block_count + source_block,
            ).to(tl.float32)
            summary_offset = (
                (((batch * kv_heads + kv_head) * block_count + source_block)
                 * head_dim + output_offset[:, None]) * head_dim
                + feature_offset[None, :]
            )
            summary = tl.load(
                summary_ptr + summary_offset,
                mask=output_mask[:, None] & feature_mask[None, :],
                other=0,
            ).to(tl.float32)
            normalizer = tl.load(
                normalizer_ptr + (
                    ((batch * kv_heads + kv_head) * block_count + source_block)
                    * head_dim + feature_offset
                ),
                mask=feature_mask,
                other=0,
            ).to(tl.float32)
            normalizer_output = tl.load(
                normalizer_ptr + (
                    ((batch * kv_heads + kv_head) * block_count + source_block)
                    * head_dim + output_offset
                ),
                mask=output_mask,
                other=0,
            ).to(tl.float32)
            if BLOCK_D < 16:
                source_gradient = tl.sum(
                    d_a[:, None, :] * summary[None, :, :], axis=2,
                )
            else:
                source_gradient = tl.dot(d_a, tl.trans(summary))
            query_gradient += mix * (
                source_gradient
                + d_r[:, None] * normalizer_output[None, :]
            )
        query_output_offset = (
            ((batch * heads + head) * token_count + token_index[:, None])
            * head_dim + output_offset[None, :]
        )
        query_output_value = tl.load(
            query_ptr + query_output_offset,
            mask=valid[:, None] & output_mask[None, :],
            other=0,
        ).to(tl.float32)
        query_gradient = query_gradient * tl.where(
            output_mask[None, :],
            tl.where(
                query_output_value > 0.0,
                1.0,
                tl.exp(query_output_value),
            ),
            0.0,
        )

        count = tl.zeros((), dtype=tl.float32)
        for count_start in range(0, MAX_TOKENS, TOKEN_TILE):
            count_offset = count_start + tl.arange(0, TOKEN_TILE)
            count_in_block = count_offset < MAX_TOKENS
            count_layout = query_block * MAX_TOKENS + count_offset
            count_index = tl.load(
                block_index_ptr + count_layout,
                mask=count_in_block,
                other=0,
            )
            count_valid = count_in_block & tl.load(
                block_mask_ptr + count_layout,
                mask=count_in_block,
                other=0,
            ).to(tl.int1)
            count_valid = count_valid & tl.load(
                valid_ptr + batch * token_count + count_index,
                mask=count_valid,
                other=0,
            ).to(tl.int1)
            count += tl.sum(count_valid.to(tl.float32), axis=0)
        query_mean_gradient = tl.load(
            query_mean_grad_ptr + (
                ((batch * kv_heads + kv_head) * block_count + query_block)
                * head_dim + output_offset
            ),
            mask=output_mask,
            other=0,
        ).to(tl.float32)
        query_gradient += (
            query_mean_gradient[None, :] / tl.maximum(count * GROUP, 1.0)
        )
        query_gradient = tl.where(valid[:, None] & output_mask[None, :], query_gradient, 0.0)
        query_output_offset = (
            ((batch * heads + head) * token_count + token_index[:, None])
            * head_dim + output_offset[None, :]
        )
        tl.store(
            query_grad_ptr + query_output_offset,
            query_gradient,
            mask=valid[:, None] & output_mask[None, :],
        )

    @_mhla_kernel
    def _joint_mhla_backward_kv_kernel(
        key_ptr, value_ptr, summary_grad_ptr, normalizer_grad_ptr,
        key_mean_grad_ptr, valid_ptr, block_index_ptr, block_mask_ptr,
        key_grad_ptr, value_grad_ptr,
        batch_size, kv_heads, token_count, head_dim, block_count,
        MAX_TOKENS: tl.constexpr, TOKEN_TILE: tl.constexpr,
        BLOCK_D: tl.constexpr, OUTPUT_TILE: tl.constexpr,
        TOKEN_TILES: tl.constexpr,
    ):
        """Compute K/V gradients from summary and block-mean gradients."""
        batch_kv = tl.program_id(0)
        kv_head = batch_kv % kv_heads
        batch = batch_kv // kv_heads
        flat_tile = tl.program_id(1)
        source_block = flat_tile // TOKEN_TILES
        token_tile = flat_tile % TOKEN_TILES
        feature_tile = tl.program_id(2)
        feature_offset = feature_tile * OUTPUT_TILE + tl.arange(0, OUTPUT_TILE)
        full_offset = tl.arange(0, BLOCK_D)
        feature_mask = feature_offset < head_dim
        full_mask = full_offset < head_dim
        token_offset = token_tile * TOKEN_TILE + tl.arange(0, TOKEN_TILE)
        in_block = token_offset < MAX_TOKENS
        layout_offset = source_block * MAX_TOKENS + token_offset
        token_index = tl.load(
            block_index_ptr + layout_offset,
            mask=in_block,
            other=0,
        )
        valid = in_block & tl.load(
            block_mask_ptr + layout_offset,
            mask=in_block,
            other=0,
        ).to(tl.int1)
        valid = valid & tl.load(
            valid_ptr + batch * token_count + token_index,
            mask=valid,
            other=0,
        ).to(tl.int1)
        full_token_mask = valid[:, None] & full_mask[None, :]
        key_offset = (
            ((batch * kv_heads + kv_head) * token_count + token_index[:, None])
            * head_dim + full_offset[None, :]
        )
        key_value = tl.load(
            key_ptr + key_offset, mask=full_token_mask, other=0,
        ).to(tl.float32)
        value_value = tl.load(
            value_ptr + key_offset, mask=full_token_mask, other=0,
        ).to(tl.float32)
        feature_key = tl.where(
            full_token_mask,
            tl.where(key_value > 0.0, key_value + 1.0, tl.exp(key_value)),
            0.0,
        )
        summary_k = tl.load(
            summary_grad_ptr + (
                (((batch * kv_heads + kv_head) * block_count + source_block)
                 * head_dim + feature_offset[:, None]) * head_dim
                + full_offset[None, :]
            ),
            mask=feature_mask[:, None] & full_mask[None, :],
            other=0,
        ).to(tl.float32)
        summary_v = tl.load(
            summary_grad_ptr + (
                (((batch * kv_heads + kv_head) * block_count + source_block)
                 * head_dim + full_offset[:, None]) * head_dim
                + feature_offset[None, :]
            ),
            mask=full_mask[:, None] & feature_mask[None, :],
            other=0,
        ).to(tl.float32)
        if BLOCK_D < 16:
            value_gradient = tl.sum(
                feature_key[:, :, None] * summary_v[None, :, :], axis=1,
            )
        else:
            value_gradient = tl.dot(feature_key, summary_v)
        value_output_offset = (
            ((batch * kv_heads + kv_head) * token_count + token_index[:, None])
            * head_dim + feature_offset[None, :]
        )
        tl.store(
            value_grad_ptr + value_output_offset,
            tl.where(valid[:, None] & feature_mask[None, :], value_gradient, 0.0),
            mask=valid[:, None] & feature_mask[None, :],
        )

        key_gradient = tl.sum(
            value_value[:, None, :] * summary_k[None, :, :], axis=2,
        )
        normalizer_gradient = tl.load(
            normalizer_grad_ptr + (
                ((batch * kv_heads + kv_head) * block_count + source_block)
                * head_dim + feature_offset
            ),
            mask=feature_mask,
            other=0,
        ).to(tl.float32)
        key_gradient += normalizer_gradient[None, :]
        key_tile_offset = (
            ((batch * kv_heads + kv_head) * token_count + token_index[:, None])
            * head_dim + feature_offset[None, :]
        )
        key_tile_value = tl.load(
            key_ptr + key_tile_offset,
            mask=valid[:, None] & feature_mask[None, :],
            other=0,
        ).to(tl.float32)
        key_tile_prime = tl.where(
            valid[:, None] & feature_mask[None, :],
            tl.where(key_tile_value > 0.0, 1.0, tl.exp(key_tile_value)),
            0.0,
        )
        key_gradient = key_gradient * key_tile_prime
        count = tl.zeros((), dtype=tl.float32)
        for count_start in range(0, MAX_TOKENS, TOKEN_TILE):
            count_offset = count_start + tl.arange(0, TOKEN_TILE)
            count_in_block = count_offset < MAX_TOKENS
            count_layout = source_block * MAX_TOKENS + count_offset
            count_index = tl.load(
                block_index_ptr + count_layout,
                mask=count_in_block,
                other=0,
            )
            count_valid = count_in_block & tl.load(
                block_mask_ptr + count_layout,
                mask=count_in_block,
                other=0,
            ).to(tl.int1)
            count_valid = count_valid & tl.load(
                valid_ptr + batch * token_count + count_index,
                mask=count_valid,
                other=0,
            ).to(tl.int1)
            count += tl.sum(count_valid.to(tl.float32), axis=0)
        key_mean_gradient = tl.load(
            key_mean_grad_ptr + (
                ((batch * kv_heads + kv_head) * block_count + source_block)
                * head_dim + feature_offset
            ),
            mask=feature_mask,
            other=0,
        ).to(tl.float32)
        key_gradient += key_mean_gradient[None, :] / tl.maximum(count, 1.0)
        key_gradient = tl.where(valid[:, None] & feature_mask[None, :], key_gradient, 0.0)
        key_output_offset = (
            ((batch * kv_heads + kv_head) * token_count + token_index[:, None])
            * head_dim + feature_offset[None, :]
        )
        tl.store(
            key_grad_ptr + key_output_offset,
            key_gradient,
            mask=valid[:, None] & feature_mask[None, :],
        )
