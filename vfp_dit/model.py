"""VFCB-free multimodal DiT with reusable bidirectional condition K/V."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def apply_multimodal_rope(
    value: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Rotate Q/K using (sequence-or-reference, y, x) coordinates.

    Args:
        value: ``(B, heads, tokens, head_dim)``.
        positions: integer or floating coordinates ``(B, tokens, 3)``.

    The three coordinate axes receive equal, even-sized subspaces. Text tokens
    use axis 0 as sequence position; reference-image hidden tokens and VAE
    latent tokens use the same reference id on axis 0 and spatial y/x on axes
    1/2. Target-image tokens use axis 0=0 and their latent-grid y/x.
    """
    if value.ndim != 4:
        raise ValueError("value must have shape (B, heads, tokens, head_dim)")
    if not value.is_floating_point():
        raise TypeError("value must use a floating-point dtype")
    batch, _heads, tokens, head_dim = value.shape
    if positions.shape != (batch, tokens, 3):
        raise ValueError("positions must have shape (B, tokens, 3)")
    if positions.device != value.device:
        raise ValueError("positions and value must be on the same device")
    if head_dim < 6 or head_dim % 2:
        raise ValueError("head_dim must be even and at least 6 for three-axis RoPE")
    pair_count = head_dim // 2
    base_pairs, extra_pairs = divmod(pair_count, 3)
    axis_dims = tuple(
        2 * (base_pairs + int(axis < extra_pairs)) for axis in range(3)
    )
    chunks = value.split(axis_dims, dim=-1)
    rotated = []
    for axis, chunk in enumerate(chunks):
        axis_dim = chunk.shape[-1]
        frequencies = 1.0 / (
            10000.0 ** (
                torch.arange(0, axis_dim, 2, device=value.device, dtype=torch.float32)
                / axis_dim
            )
        )
        phase = positions[..., axis].float()[..., None] * frequencies
        phase = phase[:, None, :, :]
        even, odd = chunk[..., 0::2], chunk[..., 1::2]
        cos, sin = phase.cos().to(chunk.dtype), phase.sin().to(chunk.dtype)
        rotated.append(torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2))
    return torch.cat(rotated, dim=-1)


def build_qwen_condition_positions(
    mm_token_type_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    *,
    spatial_merge_size: int,
    reference_id: int = 1,
) -> torch.Tensor:
    """Map Qwen3.5 text/image tokens to sequence/ref/y/x coordinates.

    Qwen image placeholders use ``mm_token_type_ids == 1``. Their hidden-token
    count must match each still-image grid after Qwen's spatial merge. Text
    tokens receive monotonic axis-0 positions; image tokens receive the shared
    reference id and row-major merged-grid y/x coordinates. The same
    ``reference_id`` should be passed to :func:`grid_positions` for the VAE
    latent tokens of that image.
    """
    if mm_token_type_ids.ndim != 2 or attention_mask.shape != mm_token_type_ids.shape:
        raise ValueError("Qwen modality types and attention mask must have shape (B, L)")
    if attention_mask.dtype != torch.bool:
        raise TypeError("attention_mask must be boolean")
    if spatial_merge_size <= 0 or reference_id <= 0:
        raise ValueError("spatial_merge_size and positive reference_id are required")
    if mm_token_type_ids.device != attention_mask.device:
        raise ValueError("Qwen modality types and attention mask must share a device")
    batch, tokens = mm_token_type_ids.shape
    positions = torch.zeros((batch, tokens, 3), device=mm_token_type_ids.device, dtype=torch.float32)
    grids = [] if image_grid_thw is None else image_grid_thw.reshape(-1, 3).tolist()
    grid_cursor = 0
    for row in range(batch):
        image_indices = torch.nonzero(
            (mm_token_type_ids[row] == 1) & attention_mask[row],
            as_tuple=False,
        ).flatten()
        text_position = 0
        image_lookup = {int(index): rank for rank, index in enumerate(image_indices.tolist())}
        image_coords = None
        if image_indices.numel():
            if grid_cursor >= len(grids):
                raise ValueError("Qwen image tokens require image_grid_thw metadata")
            temporal, grid_height, grid_width = (int(v) for v in grids[grid_cursor])
            grid_cursor += 1
            if temporal != 1:
                raise ValueError("reference inputs must be still images with temporal grid 1")
            if grid_height % spatial_merge_size or grid_width % spatial_merge_size:
                raise ValueError("Qwen image grid must divide evenly by spatial_merge_size")
            merged_height = grid_height // spatial_merge_size
            merged_width = grid_width // spatial_merge_size
            if image_indices.numel() != merged_height * merged_width:
                raise ValueError(
                    "Qwen image-token count does not match the merged image grid: "
                    f"tokens={image_indices.numel()}, grid={merged_height}x{merged_width}"
                )
            y, x = torch.meshgrid(
                torch.arange(merged_height, device=positions.device),
                torch.arange(merged_width, device=positions.device),
                indexing="ij",
            )
            image_coords = torch.stack((y, x), dim=-1).reshape(-1, 2)
        for token in torch.nonzero(attention_mask[row], as_tuple=False).flatten().tolist():
            if token in image_lookup:
                assert image_coords is not None
                positions[row, token, 0] = reference_id
                positions[row, token, 1:] = image_coords[image_lookup[token]]
            else:
                positions[row, token, 0] = text_position
            text_position += 1
    if grid_cursor != len(grids):
        raise ValueError("image_grid_thw contains images without Qwen image tokens")
    return positions


def grid_positions(
    batch: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    reference_id: int = 0,
    spatial_stride: int = 1,
    center_offset: float = 0.0,
) -> torch.Tensor:
    """Build row-major ``(reference_id, y, x)`` coordinates for a grid.

    ``spatial_stride`` and ``center_offset`` place compressed patch tokens at
    their centers in the original latent-grid coordinate system.
    """
    if batch <= 0 or height <= 0 or width <= 0 or reference_id < 0:
        raise ValueError("batch, grid dimensions, and reference_id must be valid")
    if spatial_stride <= 0 or center_offset < 0:
        raise ValueError("spatial_stride must be positive and center_offset non-negative")
    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    if spatial_stride == 1 and center_offset == 0:
        ref = torch.full_like(y, reference_id)
    else:
        y = y.to(torch.float32) * spatial_stride + center_offset
        x = x.to(torch.float32) * spatial_stride + center_offset
        ref = torch.full_like(y, float(reference_id))
    positions = torch.stack((ref, y, x), dim=-1).reshape(1, height * width, 3)
    return positions.expand(batch, -1, -1)


@dataclass(frozen=True)
class ConditionKVCache:
    """Per-layer condition keys/values prepared once for denoising."""

    keys: tuple[torch.Tensor, ...]  # each (B, Hkv, S, Dh), already RoPE'd
    values: tuple[torch.Tensor, ...]  # each (B, Hkv, S, Dh)
    mask: torch.Tensor  # bool (B, S), True for valid keys
    output_keys: tuple[torch.Tensor, ...] = ()
    output_values: tuple[torch.Tensor, ...] = ()

    def __post_init__(self) -> None:
        if not self.keys or len(self.keys) != len(self.values):
            raise ValueError("condition cache needs matching non-empty K/V layer tuples")
        if self.mask.ndim != 2 or self.mask.dtype != torch.bool:
            raise ValueError("condition cache mask must be bool (B, S)")
        if self.mask.shape[1] == 0 or not self.mask.any(-1).all():
            raise ValueError("every condition sequence must contain a valid token")
        for key, value in zip(self.keys, self.values, strict=True):
            if key.ndim != 4 or key.shape != value.shape:
                raise ValueError("cached K/V must have matching (B, Hkv, S, Dh) shapes")
            if key.shape[0] != self.mask.shape[0] or key.shape[2] != self.mask.shape[1]:
                raise ValueError("cached K/V batch and sequence dimensions must match mask")
            if key.device != self.mask.device or value.device != self.mask.device:
                raise ValueError("cached K/V and mask must share a device")
            if key.dtype != value.dtype:
                raise TypeError("cached K/V must use the same dtype")
        if len(self.output_keys) != len(self.output_values):
            raise ValueError("output-refinement condition K/V tuples must match")
        for key, value in zip(self.output_keys, self.output_values, strict=True):
            if key.ndim != 4 or key.shape != value.shape:
                raise ValueError("output condition K/V must have matching (B,Hkv,S,Dh) shapes")
            if key.shape[0] != self.mask.shape[0] or key.shape[2] != self.mask.shape[1]:
                raise ValueError("output condition K/V batch and sequence must match mask")
            if key.device != self.mask.device or value.device != self.mask.device:
                raise ValueError("output condition K/V and mask must share a device")
            if key.dtype != value.dtype:
                raise TypeError("output condition K/V must use the same dtype")

    def repeat_interleave(self, repeats: int) -> "ConditionKVCache":
        """Repeat each condition row to pair it with several target samples."""
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        if repeats == 1:
            return self
        return ConditionKVCache(
            keys=tuple(key.repeat_interleave(repeats, dim=0) for key in self.keys),
            values=tuple(value.repeat_interleave(repeats, dim=0) for value in self.values),
            mask=self.mask.repeat_interleave(repeats, dim=0),
            output_keys=tuple(
                key.repeat_interleave(repeats, dim=0) for key in self.output_keys
            ),
            output_values=tuple(
                value.repeat_interleave(repeats, dim=0) for value in self.output_values
            ),
        )


class _GatedFFNConditionBlock(nn.Module):
    """Per-token SwiGLU residual adapter block without token mixing."""

    def __init__(self, width: int, ff_mult: float):
        super().__init__()
        self.norm2 = nn.RMSNorm(width)
        hidden = max(1, int(width * ff_mult))
        self.ffn_in = nn.Linear(width, 2 * hidden, bias=False)
        self.ffn_out = nn.Linear(hidden, width, bias=False)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        gate, value = self.ffn_in(self.norm2(tokens)).chunk(2, dim=-1)
        tokens = tokens + self.ffn_out(F.silu(gate) * value)
        return tokens.masked_fill(~mask[..., None], 0)


class VLMAdapter(nn.Module):
    """Bidirectional adapter over Qwen hidden and reference-latent tokens.

    New fused checkpoints downsample the reference latent by two, align Qwen
    vision features to that grid, concatenate channels, and project each joint
    spatial token to model width. Full-resolution and legacy fusion paths remain
    available to reconstruct older checkpoints.
    """

    def __init__(
        self,
        qwen_dim: int,
        latent_channels: int,
        width: int,
        *,
        depth: int = 2,
        ff_mult: float = 2.0,
        latent_downsample_factor: int = 1,
        fuse_reference_latent_to_vision: bool = False,
        reference_latent_fusion_mode: str = "legacy_latent_to_qwen",
    ):
        super().__init__()
        if min(qwen_dim, latent_channels, width, depth) <= 0:
            raise ValueError("adapter dimensions and depth must be positive")
        if latent_downsample_factor not in (1, 2, 4):
            raise ValueError("latent_downsample_factor must be one of 1, 2, or 4")
        self.qwen_dim = qwen_dim
        self.latent_channels = latent_channels
        self.width = width
        # Kept in run metadata for checkpoint readability; only GatedFFN exists.
        self.adapter_type = "ffn"
        self.latent_downsample_factor = latent_downsample_factor
        self.fuse_reference_latent_to_vision = bool(fuse_reference_latent_to_vision)
        if reference_latent_fusion_mode not in (
            "legacy_latent_to_qwen", "qwen_to_full_latent", "qwen_to_half_latent",
            "qwen_to_half_latent_add",
        ):
            raise ValueError("unknown reference latent fusion mode")
        self.reference_latent_fusion_mode = reference_latent_fusion_mode
        self.qwen_in = nn.Linear(qwen_dim, width)
        if self.fuse_reference_latent_to_vision:
            if reference_latent_fusion_mode == "qwen_to_full_latent":
                # The spatial resize before this learned 2x upsampler handles
                # processor grids that do not exactly match half the latent size.
                self.reference_vision_upsample = nn.ConvTranspose2d(
                    qwen_dim, qwen_dim, kernel_size=4, stride=2, padding=1,
                    groups=qwen_dim, bias=False,
                )
                self._init_depthwise_bilinear_upsample()
            elif reference_latent_fusion_mode == "qwen_to_half_latent":
                self.reference_latent_in = nn.Conv2d(
                    latent_channels, latent_channels, kernel_size=2, stride=2,
                )
            elif reference_latent_fusion_mode == "qwen_to_half_latent_add":
                self.reference_qwen_in = nn.Linear(qwen_dim, width)
                self.reference_latent_in = nn.Conv2d(
                    latent_channels, width, kernel_size=2, stride=2,
                )
            else:
                # Legacy checkpoint path: reduce latent features to the Qwen grid.
                self.reference_latent_in = nn.Conv2d(
                    latent_channels, latent_channels, kernel_size=3, stride=2, padding=1,
                )
            if reference_latent_fusion_mode != "qwen_to_half_latent_add":
                self.reference_fusion = nn.Linear(qwen_dim + latent_channels, width)
        elif latent_downsample_factor == 1:
            self.reference_latent_in = nn.Linear(latent_channels, width)
        else:
            self.reference_latent_in = nn.Conv2d(
                latent_channels, width,
                kernel_size=latent_downsample_factor,
                stride=latent_downsample_factor,
            )
        self.qwen_type = nn.Parameter(torch.zeros(width))
        self.reference_latent_type = nn.Parameter(torch.zeros(width))
        self.blocks = nn.ModuleList(
            _GatedFFNConditionBlock(width, ff_mult) for _ in range(depth)
        )
        self.output_norm = nn.RMSNorm(width)

    @torch.no_grad()
    def _init_depthwise_bilinear_upsample(self) -> None:
        """Initialize the per-channel 2x transposed convolution as bilinear."""
        layer = self.reference_vision_upsample
        if layer.weight.device.type == "meta":
            return
        ramp = torch.tensor((0.25, 0.75, 0.75, 0.25), device=layer.weight.device,
                            dtype=layer.weight.dtype)
        kernel = ramp[:, None] * ramp[None, :]
        layer.weight[:, 0].copy_(kernel.expand_as(layer.weight[:, 0]))

    def gradient_norms(self) -> dict[str, torch.Tensor]:
        """Return global L2 gradient norms for adapter branches after backward."""
        groups: dict[str, list[nn.Parameter]] = {
            "input": [*self.qwen_in.parameters(), self.qwen_type],
            "reference": [self.reference_latent_type],
            "ffn": [],
            "output_norm": list(self.output_norm.parameters()),
        }
        reference_latent_in = getattr(self, "reference_latent_in", None)
        if reference_latent_in is not None:
            groups["reference"].extend(reference_latent_in.parameters())
        reference_qwen_in = getattr(self, "reference_qwen_in", None)
        if reference_qwen_in is not None:
            groups["reference"].extend(reference_qwen_in.parameters())
        reference_vision_upsample = getattr(self, "reference_vision_upsample", None)
        if reference_vision_upsample is not None:
            groups["reference"].extend(reference_vision_upsample.parameters())
        reference_fusion = getattr(self, "reference_fusion", None)
        if reference_fusion is not None:
            groups["reference"].extend(reference_fusion.parameters())
        for block in self.blocks:
            for name in ("norm2", "ffn_in", "ffn_out"):
                groups["ffn"].extend(getattr(block, name).parameters())
        zero = self.qwen_type.new_zeros((), dtype=torch.float32)
        result = {}
        for name, parameters in groups.items():
            norms = [parameter.grad.detach().float().norm() for parameter in parameters
                     if parameter.grad is not None]
            result[f"grad_adapter_{name}"] = (
                torch.stack(norms).norm() if norms else zero
            )
        return result

    def forward(
        self,
        qwen_hidden: torch.Tensor,
        qwen_mask: torch.Tensor,
        qwen_positions: torch.Tensor,
        reference_latent: torch.Tensor | None = None,
        reference_mask: torch.Tensor | None = None,
        reference_positions: torch.Tensor | None = None,
        qwen_vision_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if qwen_hidden.ndim != 3 or qwen_hidden.shape[-1] != self.qwen_dim:
            raise ValueError("qwen_hidden must have shape (B, L, qwen_dim)")
        batch, qwen_tokens, _ = qwen_hidden.shape
        if qwen_mask.shape != (batch, qwen_tokens) or qwen_mask.dtype != torch.bool:
            raise ValueError("qwen_mask must be bool (B, L)")
        if qwen_mask.device != qwen_hidden.device:
            raise ValueError("qwen_mask and qwen_hidden must share a device")
        if qwen_positions.shape != (batch, qwen_tokens, 3):
            raise ValueError("qwen_positions must have shape (B, L, 3)")
        if qwen_positions.device != qwen_hidden.device:
            raise ValueError("qwen_positions and qwen_hidden must share a device")
        if not qwen_mask.any(-1).all():
            raise ValueError("each sample needs at least one valid Qwen token")
        qwen = self.qwen_in(qwen_hidden.to(dtype=self.qwen_in.weight.dtype)) + self.qwen_type
        tokens, mask, positions = qwen, qwen_mask, qwen_positions
        if reference_latent is not None:
            if reference_latent.ndim != 4 or reference_latent.shape[:2] != (batch, self.latent_channels):
                raise ValueError("reference_latent must have shape (B, latent_channels, H, W)")
            if reference_latent.device != qwen_hidden.device:
                raise ValueError("reference_latent and qwen_hidden must share a device")
            ref_h, ref_w = reference_latent.shape[-2:]
            factor = self.latent_downsample_factor
            if (not self.fuse_reference_latent_to_vision or
                    self.reference_latent_fusion_mode != "qwen_to_full_latent") and (
                    ref_h % factor or ref_w % factor
            ):
                raise ValueError(
                    "reference latent height and width must be divisible by "
                    "latent_downsample_factor"
                )
            if (self.fuse_reference_latent_to_vision
                    and self.reference_latent_fusion_mode in (
                        "qwen_to_half_latent", "qwen_to_half_latent_add",
                    )
                    and (ref_h % 2 or ref_w % 2)):
                raise ValueError(
                    "reference latent height and width must be divisible by 2 "
                    "for qwen_to_half_latent fusion"
                )
            if reference_mask is None:
                reference_mask = torch.ones(
                    (batch, ref_h * ref_w), device=qwen.device, dtype=torch.bool,
                )
            if reference_mask.shape != (batch, ref_h * ref_w) or reference_mask.dtype != torch.bool:
                raise ValueError("reference_mask must be bool (B, H*W)")
            if reference_mask.device != qwen_hidden.device:
                raise ValueError("reference_mask and qwen_hidden must share a device")
            if reference_positions is None:
                reference_positions = grid_positions(
                    batch, ref_h, ref_w, device=qwen.device, reference_id=1,
                )
            if reference_positions.shape != (batch, ref_h * ref_w, 3):
                raise ValueError("reference_positions must have shape (B, H*W, 3)")
            if reference_positions.device != qwen_hidden.device:
                raise ValueError("reference_positions and qwen_hidden must share a device")
            if self.fuse_reference_latent_to_vision:
                if qwen_vision_mask is None:
                    raise ValueError(
                        "qwen_vision_mask is required when fusing reference latents "
                        "into Qwen vision tokens"
                    )
                if qwen_vision_mask.shape != (batch, qwen_tokens) or qwen_vision_mask.dtype != torch.bool:
                    raise ValueError("qwen_vision_mask must be bool (B, L)")
                if qwen_vision_mask.device != qwen_hidden.device:
                    raise ValueError("qwen_vision_mask and qwen_hidden must share a device")
                reference_present = reference_mask.reshape(batch, ref_h, ref_w).any(dim=(1, 2))
                if reference_present.any():
                    active_vision_mask = qwen_vision_mask & qwen_mask
                    expanded_rows: list[torch.Tensor] = []
                    expanded_positions: list[torch.Tensor] = []
                    expanded_masks: list[torch.Tensor] = []
                    for row in range(batch):
                        valid_indices = torch.nonzero(qwen_mask[row], as_tuple=False).flatten()
                        if not bool(reference_present[row]):
                            expanded_rows.append(qwen[row, valid_indices])
                            expanded_positions.append(qwen_positions[row, valid_indices])
                            expanded_masks.append(qwen_mask[row, valid_indices])
                            continue
                        vision_indices = torch.nonzero(
                            active_vision_mask[row], as_tuple=False,
                        ).flatten()
                        if vision_indices.numel() == 0:
                            raise ValueError(
                                "a reference latent is present without Qwen vision tokens"
                            )
                        vision_positions = qwen_positions[row, vision_indices, 1:]
                        vision_height = int(vision_positions[:, 0].max().item()) + 1
                        vision_width = int(vision_positions[:, 1].max().item()) + 1
                        vision_count = vision_indices.numel()
                        if vision_height * vision_width != vision_count:
                            raise ValueError(
                                "Qwen vision tokens must form a complete row-major grid"
                            )
                        if self.reference_latent_fusion_mode == "qwen_to_full_latent":
                            vision_grid = qwen_hidden[row, vision_indices].reshape(
                                vision_height, vision_width, self.qwen_dim,
                            ).permute(2, 0, 1).unsqueeze(0)
                            half_grid_size = (ref_h // 2, ref_w // 2)
                            if min(half_grid_size) <= 0:
                                raise ValueError("reference latent grid must be at least 2x2")
                            if vision_grid.shape[-2:] != half_grid_size:
                                vision_grid = F.interpolate(
                                    vision_grid, size=half_grid_size,
                                    mode="bilinear", align_corners=False,
                                )
                            vision_full = self.reference_vision_upsample(
                                vision_grid.to(dtype=self.reference_vision_upsample.weight.dtype),
                            )
                            if vision_full.shape[-2:] != (ref_h, ref_w):
                                raise RuntimeError(
                                    "Qwen vision upsampling did not match the reference latent grid"
                                )
                            latent_full = (
                                reference_latent[row:row + 1]
                                * reference_mask[row].reshape(1, 1, ref_h, ref_w).to(
                                    reference_latent.dtype,
                                )
                            ).to(dtype=self.reference_fusion.weight.dtype)
                            joint_tokens = torch.cat((
                                vision_full.flatten(2).transpose(1, 2).to(
                                    dtype=self.reference_fusion.weight.dtype,
                                ),
                                latent_full.flatten(2).transpose(1, 2),
                            ), dim=-1)[0]
                            fused_tokens = (
                                self.reference_fusion(joint_tokens)
                                + self.qwen_type + self.reference_latent_type
                            )
                            fused_mask = reference_mask[row]
                            fused_positions = reference_positions[row]
                        elif self.reference_latent_fusion_mode in (
                            "qwen_to_half_latent", "qwen_to_half_latent_add",
                        ):
                            half_grid_size = (ref_h // 2, ref_w // 2)
                            if min(half_grid_size) <= 0:
                                raise ValueError("reference latent grid must be at least 2x2")
                            vision_grid = qwen_hidden[row, vision_indices].reshape(
                                vision_height, vision_width, self.qwen_dim,
                            ).permute(2, 0, 1).unsqueeze(0)
                            if vision_grid.shape[-2:] != half_grid_size:
                                vision_grid = F.interpolate(
                                    vision_grid, size=half_grid_size,
                                    mode="bilinear", align_corners=False,
                                )
                            ref_mask_2d = reference_mask[row:row + 1].reshape(
                                1, 1, ref_h, ref_w,
                            )
                            latent_grid = self.reference_latent_in(
                                (reference_latent[row:row + 1] * ref_mask_2d.to(
                                    reference_latent.dtype,
                                )).to(dtype=self.reference_latent_in.weight.dtype),
                            )
                            if self.reference_latent_fusion_mode == "qwen_to_half_latent_add":
                                qwen_tokens = vision_grid.flatten(2).transpose(1, 2)[0].to(
                                    dtype=self.reference_qwen_in.weight.dtype,
                                )
                                latent_tokens = latent_grid.flatten(2).transpose(1, 2)[0]
                                fused_tokens = (
                                    self.reference_qwen_in(qwen_tokens)
                                    + latent_tokens.to(dtype=self.reference_qwen_in.weight.dtype)
                                    + self.qwen_type + self.reference_latent_type
                                )
                            else:
                                joint_grid = torch.cat((
                                    vision_grid.to(dtype=self.reference_fusion.weight.dtype),
                                    latent_grid.to(dtype=self.reference_fusion.weight.dtype),
                                ), dim=1)
                                fused_tokens = self.reference_fusion(
                                    joint_grid.flatten(2).transpose(1, 2)[0],
                                ) + self.qwen_type + self.reference_latent_type
                            fused_mask = F.max_pool2d(
                                ref_mask_2d.to(dtype=torch.float32), kernel_size=2, stride=2,
                            ).flatten(1)[0].bool()
                            fused_positions = reference_positions[row].to(
                                dtype=torch.float32,
                            ).reshape(ref_h, ref_w, 3).reshape(
                                ref_h // 2, 2, ref_w // 2, 2, 3,
                            ).mean(dim=(1, 3)).reshape(-1, 3)
                        else:
                            ref_mask_2d = reference_mask[row:row + 1].reshape(
                                1, 1, ref_h, ref_w,
                            )
                            masked_reference = reference_latent[row:row + 1] * ref_mask_2d.to(
                                reference_latent.dtype,
                            )
                            latent_features = self.reference_latent_in(
                                masked_reference.to(dtype=self.reference_latent_in.weight.dtype),
                            )
                            aligned_latent = F.adaptive_avg_pool2d(
                                latent_features, output_size=(vision_height, vision_width),
                            ).flatten(2).transpose(1, 2)[0]
                            joint_tokens = torch.cat((
                                qwen_hidden[row, vision_indices].to(
                                    dtype=self.reference_fusion.weight.dtype,
                                ),
                                aligned_latent.to(dtype=self.reference_fusion.weight.dtype),
                            ), dim=-1)
                            fused_tokens = (
                                self.reference_fusion(joint_tokens)
                                + self.qwen_type + self.reference_latent_type
                            )
                            fused_mask = torch.ones(
                                vision_count, dtype=torch.bool, device=qwen.device,
                            )
                            fused_positions = qwen_positions[row, vision_indices]

                        vision_start = int(vision_indices[0].item())
                        vision_end = int(vision_indices[-1].item()) + 1
                        expected_vision_indices = torch.arange(
                            vision_start, vision_end, device=vision_indices.device,
                        )
                        if not torch.equal(vision_indices, expected_vision_indices):
                            raise ValueError("Qwen vision tokens must form one contiguous sequence span")
                        before = valid_indices[valid_indices < vision_start]
                        after = valid_indices[valid_indices >= vision_end]
                        expanded_rows.append(torch.cat((
                            qwen[row, before], fused_tokens, qwen[row, after],
                        ), dim=0))
                        expanded_positions.append(torch.cat((
                            qwen_positions[row, before], fused_positions,
                            qwen_positions[row, after],
                        ), dim=0))
                        expanded_masks.append(torch.cat((
                            qwen_mask[row, before], fused_mask, qwen_mask[row, after],
                        ), dim=0))
                    max_tokens = max(row.shape[0] for row in expanded_rows)
                    tokens = qwen.new_zeros((batch, max_tokens, self.width))
                    mask = torch.zeros((batch, max_tokens), dtype=torch.bool, device=qwen.device)
                    positions = qwen_positions.new_zeros((batch, max_tokens, 3))
                    for row, (row_tokens, row_pos, row_mask) in enumerate(zip(
                        expanded_rows, expanded_positions, expanded_masks,
                    )):
                        row_length = row_tokens.shape[0]
                        tokens[row, :row_length] = row_tokens
                        mask[row, :row_length] = row_mask
                        positions[row, :row_length] = row_pos
            elif factor == 1:
                ref_tokens = reference_latent.flatten(2).transpose(1, 2)
                ref_tokens = self.reference_latent_in(
                    ref_tokens.to(dtype=self.reference_latent_in.weight.dtype),
                ) + self.reference_latent_type
                tokens = torch.cat((qwen, ref_tokens), dim=1)
                mask = torch.cat((qwen_mask, reference_mask), dim=1)
                positions = torch.cat((qwen_positions, reference_positions), dim=1)
            else:
                ref_mask_2d = reference_mask.reshape(batch, 1, ref_h, ref_w)
                masked_reference = reference_latent * ref_mask_2d.to(reference_latent.dtype)
                ref_features = self.reference_latent_in(
                    masked_reference.to(dtype=self.reference_latent_in.weight.dtype),
                )
                ref_h_down, ref_w_down = ref_features.shape[-2:]
                ref_tokens = ref_features.flatten(2).transpose(1, 2)
                reference_mask = F.max_pool2d(
                    ref_mask_2d.to(dtype=torch.float32),
                    kernel_size=factor, stride=factor,
                ).flatten(1).bool()
                reference_positions = reference_positions.to(torch.float32).reshape(
                    batch, ref_h // factor, factor, ref_w // factor, factor, 3,
                ).mean(dim=(2, 4)).reshape(batch, ref_h_down * ref_w_down, 3)
                ref_tokens = ref_tokens + self.reference_latent_type
                tokens = torch.cat((qwen, ref_tokens), dim=1)
                mask = torch.cat((qwen_mask, reference_mask), dim=1)
                positions = torch.cat((qwen_positions, reference_positions), dim=1)
        for block in self.blocks:
            tokens = block(tokens, mask)
        tokens = self.output_norm(tokens)
        return tokens.masked_fill(~mask[..., None], 0), mask, positions


METADATA_SCALE_MAPPINGS = (
    "linear",
    "one_plus_silu",
    "two_sigmoid",
    "silu1_normalized",
    "softplus1_normalized",
)
METADATA_FFN_GATE_MAPPINGS = ("linear", "silu")


def _apply_metadata_scale_mapping(raw_scale: torch.Tensor, mapping: str) -> torch.Tensor:
    """Map zero-initialized metadata projections to a unit-initialized scale."""
    if mapping == "linear":
        return 1.0 + raw_scale
    if mapping == "one_plus_silu":
        return 1.0 + F.silu(raw_scale)
    if mapping == "two_sigmoid":
        return 2.0 * torch.sigmoid(raw_scale)
    if mapping == "silu1_normalized":
        return F.silu(1.0 + raw_scale) / F.silu(raw_scale.new_ones(()))
    if mapping == "softplus1_normalized":
        return F.softplus(1.0 + raw_scale) / F.softplus(raw_scale.new_ones(()))
    raise ValueError(f"unknown metadata scale mapping: {mapping}")


def _apply_metadata_ffn_gate_mapping(raw_gate: torch.Tensor, mapping: str) -> torch.Tensor:
    """Map the zero-initialized metadata residual gate to its effective multiplier."""
    if mapping == "linear":
        return raw_gate
    if mapping == "silu":
        return F.silu(raw_gate)
    raise ValueError(f"unknown metadata FFN gate mapping: {mapping}")


class _TargetBlock(nn.Module):
    """Target-query-only DiT block; conditions are immutable cached K/V."""

    def __init__(
        self, width: int, heads: int, kv_heads: int, ff_mult: float,
        metadata_conditioning: str = "none",
        metadata_scale_mapping: str = "softplus1_normalized",
        metadata_ffn_gate_mapping: str = "silu",
        metadata_shift: bool = False,
        attention_head_gate: str = "input_silu",
        metadata_ffn_residual_gate: bool | None = None,
        fuse_same_input_projections: bool = True,
    ):
        super().__init__()
        if width % heads or heads % kv_heads or width // heads < 6 or (width // heads) % 2:
            raise ValueError("DiT requires valid GQA and an even head_dim >= 6")
        if metadata_conditioning not in (
            "none", "attn_concat_ffn_ada", "ada_attn_ffn",
        ):
            raise ValueError("unknown metadata conditioning variant")
        if metadata_scale_mapping not in METADATA_SCALE_MAPPINGS:
            raise ValueError("unknown metadata scale mapping")
        if metadata_ffn_gate_mapping not in METADATA_FFN_GATE_MAPPINGS:
            raise ValueError("unknown metadata FFN gate mapping")
        if metadata_shift and metadata_conditioning != "ada_attn_ffn":
            raise ValueError("metadata shift requires ada_attn_ffn conditioning")
        if attention_head_gate not in ("input_silu", "timestep_sigmoid"):
            raise ValueError("unknown attention head gate")
        if metadata_ffn_residual_gate is None:
            metadata_ffn_residual_gate = metadata_conditioning == "ada_attn_ffn"
        if metadata_ffn_residual_gate and metadata_conditioning != "ada_attn_ffn":
            raise ValueError("metadata FFN residual gate requires ada_attn_ffn conditioning")
        self.width = width
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = width // heads
        self.metadata_conditioning = metadata_conditioning
        self.metadata_scale_mapping = metadata_scale_mapping
        self.metadata_ffn_gate_mapping = metadata_ffn_gate_mapping
        self.metadata_shift = bool(metadata_shift)
        self.attention_head_gate = attention_head_gate
        self.metadata_ffn_residual_gate = bool(metadata_ffn_residual_gate)
        self.fuse_same_input_projections = bool(fuse_same_input_projections)
        self.norm1 = nn.RMSNorm(width)
        kv_width = kv_heads * self.head_dim
        if self.fuse_same_input_projections:
            fused_width = width + 2 * kv_width
            if attention_head_gate == "input_silu":
                fused_width += heads
                # Preserve the standalone gate's trainable bias without adding
                # bias terms to the bias-free Q/K/V projections.
                self.attn_head_gate_bias = nn.Parameter(torch.zeros(heads))
            self.target_qkv_proj = nn.Linear(width, fused_width, bias=False)
            if attention_head_gate == "input_silu":
                # The unfused gate's Linear is zero-initialized, so its initial
                # effective multiplier is exactly 1 + SiLU(0) = 1.
                gate_start = width + 2 * kv_width
                with torch.no_grad():
                    self.target_qkv_proj.weight[gate_start:].zero_()
        else:
            self.q_proj = nn.Linear(width, width, bias=False)
            self.k_proj = nn.Linear(width, kv_width, bias=False)
            self.v_proj = nn.Linear(width, kv_width, bias=False)
        if metadata_conditioning == "attn_concat_ffn_ada":
            # Separate W_m m terms implement W[x;m] without materializing
            # an expanded (B,T,D_meta) tensor.
            self.meta_q_proj = nn.Linear(width, width, bias=False)
            self.meta_k_proj = nn.Linear(width, kv_heads * self.head_dim, bias=False)
            self.meta_v_proj = nn.Linear(width, kv_heads * self.head_dim, bias=False)
            for layer in (self.meta_q_proj, self.meta_k_proj, self.meta_v_proj):
                nn.init.zeros_(layer.weight)
        self.condition_norm = nn.RMSNorm(width)
        if self.fuse_same_input_projections:
            self.condition_kv_proj = nn.Linear(width, 2 * kv_width, bias=False)
        else:
            self.condition_k_proj = nn.Linear(width, kv_width, bias=False)
            self.condition_v_proj = nn.Linear(width, kv_width, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.condition_k_norm = nn.RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(width, width, bias=False)
        if not self.fuse_same_input_projections or attention_head_gate == "timestep_sigmoid":
            self.attn_gate = nn.Linear(width, heads)
            nn.init.zeros_(self.attn_gate.weight)
            nn.init.zeros_(self.attn_gate.bias)
        if (
            self.metadata_ffn_residual_gate
            and not (self.fuse_same_input_projections and metadata_conditioning == "ada_attn_ffn")
        ):
            self.ffn_residual_gate = nn.Linear(width, width, bias=False)
            nn.init.zeros_(self.ffn_residual_gate.weight)
        self.norm2 = nn.RMSNorm(width)
        hidden = max(1, int(width * ff_mult))
        self.ffn_in = nn.Linear(width, 2 * hidden, bias=False)
        self.ffn_out = nn.Linear(hidden, width, bias=False)
        self.ffn_scale = nn.Linear(width, width)
        if metadata_conditioning == "attn_concat_ffn_ada" or (
            metadata_conditioning == "ada_attn_ffn" and not self.fuse_same_input_projections
        ):
            self.ffn_meta_scale = nn.Linear(width, width)
            nn.init.zeros_(self.ffn_meta_scale.weight)
            nn.init.zeros_(self.ffn_meta_scale.bias)
        if metadata_conditioning == "ada_attn_ffn":
            if self.fuse_same_input_projections:
                metadata_names = ["attn_scale", "ffn_scale"]
                if self.metadata_shift:
                    metadata_names.extend(("attn_shift", "ffn_shift"))
                if self.metadata_ffn_residual_gate:
                    metadata_names.append("ffn_gate")
                self.metadata_projection_names = tuple(metadata_names)
                self.metadata_projection = nn.Linear(
                    width, len(metadata_names) * width, bias=False,
                )
                nn.init.zeros_(self.metadata_projection.weight)
                # Existing scale/shift projections have trainable biases;
                # preserve the bias-free AdaLN-Zero residual gate exactly.
                biased_names = tuple(name for name in metadata_names if name != "ffn_gate")
                self.metadata_projection_bias_names = biased_names
                self.metadata_projection_bias = nn.Parameter(
                    torch.zeros(len(biased_names) * width)
                )
            else:
                self.attn_meta_scale = nn.Linear(width, width)
                nn.init.zeros_(self.attn_meta_scale.weight)
                nn.init.zeros_(self.attn_meta_scale.bias)
                if self.metadata_shift:
                    self.attn_meta_shift = nn.Linear(width, width)
                    self.ffn_meta_shift = nn.Linear(width, width)
                    for layer in (self.attn_meta_shift, self.ffn_meta_shift):
                        nn.init.zeros_(layer.weight)
                        nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.ffn_scale.weight)
        nn.init.zeros_(self.ffn_scale.bias)

    def metadata_projection_outputs(
        self, embedding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Project shared metadata embedding into branch-specific parameters."""
        if self.fuse_same_input_projections and self.metadata_conditioning == "ada_attn_ffn":
            values = self.metadata_projection(embedding).split(self.width, dim=-1)
            outputs = dict(zip(self.metadata_projection_names, values, strict=True))
            biases = self.metadata_projection_bias.split(self.width, dim=-1)
            for name, bias in zip(self.metadata_projection_bias_names, biases, strict=True):
                outputs[name] = outputs[name] + bias
            return outputs
        outputs = {
            "attn_scale": self.attn_meta_scale(embedding),
            "ffn_scale": self.ffn_meta_scale(embedding),
        }
        if self.metadata_shift:
            outputs["attn_shift"] = self.attn_meta_shift(embedding)
            outputs["ffn_shift"] = self.ffn_meta_shift(embedding)
        if self.metadata_ffn_residual_gate:
            outputs["ffn_gate"] = self.ffn_residual_gate(embedding)
        return outputs

    def prepare_condition(
        self,
        condition: torch.Tensor,
        mask: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, count, _ = condition.shape
        normalized = self.condition_norm(condition)
        if self.fuse_same_input_projections:
            key_value = self.condition_kv_proj(normalized)
            key, value = key_value.split(self.kv_heads * self.head_dim, dim=-1)
            key = key.reshape(batch, count, self.kv_heads, self.head_dim).transpose(1, 2)
            value = value.reshape(batch, count, self.kv_heads, self.head_dim).transpose(1, 2)
        else:
            key = self.condition_k_proj(normalized).reshape(
                batch, count, self.kv_heads, self.head_dim,
            ).transpose(1, 2)
            value = self.condition_v_proj(normalized).reshape(
                batch, count, self.kv_heads, self.head_dim,
            ).transpose(1, 2)
        key = apply_multimodal_rope(self.condition_k_norm(key), positions)
        # RMSNorm may promote K while autocast keeps the value projection in
        # BF16. SDPA requires Q/K/V to share a dtype, so store V in K's dtype.
        value = value.to(dtype=key.dtype)
        return key.masked_fill(~mask[:, None, :, None], 0), value.masked_fill(~mask[:, None, :, None], 0)

    def forward(
        self,
        target: torch.Tensor,
        timestep_embedding: torch.Tensor,
        target_positions: torch.Tensor,
        condition_key: torch.Tensor,
        condition_value: torch.Tensor,
        condition_mask: torch.Tensor,
        metadata_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, target_count, _ = target.shape
        expected_cache_shape = (
            batch, self.kv_heads, condition_mask.shape[1], self.head_dim,
        )
        if condition_mask.dtype != torch.bool or condition_mask.ndim != 2:
            raise TypeError("condition_mask must be bool (B, S)")
        if condition_mask.shape[0] != batch:
            raise ValueError("condition_mask batch dimension must match target")
        if condition_key.shape != expected_cache_shape or condition_value.shape != expected_cache_shape:
            raise ValueError("cached condition K/V shapes do not match the DiT block")
        if any(tensor.device != target.device for tensor in (
            condition_key, condition_value, condition_mask, target_positions,
        )):
            raise ValueError("target, positions, condition cache, and mask must share a device")
        normalized = self.norm1(target)
        if self.metadata_conditioning != "none":
            if metadata_embedding is None or metadata_embedding.shape != (batch, self.width):
                raise ValueError("metadata-conditioned blocks require metadata embedding (B,D)")
        elif metadata_embedding is not None:
            raise ValueError("metadata was provided to an unconditioned block")

        metadata_outputs = (
            self.metadata_projection_outputs(metadata_embedding)
            if self.metadata_conditioning == "ada_attn_ffn" else None
        )
        if self.metadata_conditioning == "ada_attn_ffn":
            assert metadata_outputs is not None
        # Also supplies the gate input on the legacy attention-concat path.
        attn_input = normalized
        if self.metadata_conditioning == "ada_attn_ffn":
            attn_scale = _apply_metadata_scale_mapping(
                metadata_outputs["attn_scale"], self.metadata_scale_mapping,
            )
            attn_input = normalized * attn_scale[:, None, :]
            if self.metadata_shift:
                attn_input = attn_input + metadata_outputs["attn_shift"][:, None, :]
        fused_gate_logits = None
        if self.fuse_same_input_projections:
            fused = self.target_qkv_proj(attn_input)
            kv_width = self.kv_heads * self.head_dim
            split_sizes = [self.width, kv_width, kv_width]
            if self.attention_head_gate == "input_silu":
                split_sizes.append(self.heads)
            projected = fused.split(split_sizes, dim=-1)
            query_input, key_input, value_input = projected[:3]
            if len(projected) == 4:
                fused_gate_logits = projected[3] + self.attn_head_gate_bias
            if self.metadata_conditioning == "attn_concat_ffn_ada":
                query_input = query_input + self.meta_q_proj(metadata_embedding)[:, None, :]
                key_input = key_input + self.meta_k_proj(metadata_embedding)[:, None, :]
                value_input = value_input + self.meta_v_proj(metadata_embedding)[:, None, :]
        elif self.metadata_conditioning == "attn_concat_ffn_ada":
            query_input = self.q_proj(normalized) + self.meta_q_proj(metadata_embedding)[:, None, :]
            key_input = self.k_proj(normalized) + self.meta_k_proj(metadata_embedding)[:, None, :]
            value_input = self.v_proj(normalized) + self.meta_v_proj(metadata_embedding)[:, None, :]
        else:
            query_input = self.q_proj(attn_input)
            key_input = self.k_proj(attn_input)
            value_input = self.v_proj(attn_input)
        query = query_input.reshape(
            batch, target_count, self.heads, self.head_dim,
        ).transpose(1, 2)
        target_key = key_input.reshape(
            batch, target_count, self.kv_heads, self.head_dim,
        ).transpose(1, 2)
        target_value = value_input.reshape(
            batch, target_count, self.kv_heads, self.head_dim,
        ).transpose(1, 2)
        query = apply_multimodal_rope(self.q_norm(query), target_positions)
        target_key = apply_multimodal_rope(self.k_norm(target_key), target_positions)
        if condition_key.dtype != query.dtype or condition_value.dtype != query.dtype:
            raise TypeError("condition cache dtype must match target Q/K/V dtype")
        keys = torch.cat((condition_key, target_key), dim=2)
        values = torch.cat((condition_value, target_value), dim=2)
        key_mask = torch.cat((condition_mask, torch.ones(
            (batch, target_count), device=target.device, dtype=torch.bool,
        )), dim=1)
        attended = F.scaled_dot_product_attention(
            query,
            keys,
            values,
            attn_mask=key_mask[:, None, None, :],
            dropout_p=0.0,
            enable_gqa=self.heads != self.kv_heads,
        )
        if self.attention_head_gate == "input_silu":
            # Per-token/per-head gate from the same (possibly Ada-modulated)
            # features that feed QKV. 1 + SiLU starts at identity.
            attn_gate_logits = (
                fused_gate_logits
                if fused_gate_logits is not None
                else self.attn_gate(attn_input)
            )
            gate = 1.0 + F.silu(attn_gate_logits)
            attended = attended * gate.transpose(1, 2).unsqueeze(-1).to(attended.dtype)
        else:
            # Legacy checkpoint path: one 2*sigmoid gate per sample/head,
            # conditioned only on timestep embedding.
            gate = 2.0 * torch.sigmoid(self.attn_gate(timestep_embedding))
            attended = attended * gate[:, :, None, None].to(attended.dtype)
        attended = attended.transpose(1, 2).reshape(batch, target_count, self.width)
        target = target + self.out_proj(attended)
        normalized = self.norm2(target)
        scale = 1.0 + self.ffn_scale(timestep_embedding)
        if self.metadata_conditioning == "attn_concat_ffn_ada":
            # Preserve the historical attention-concat + additive FFN-scale path.
            scale = scale + self.ffn_meta_scale(metadata_embedding)
        elif self.metadata_conditioning == "ada_attn_ffn":
            if metadata_embedding is None:
                raise ValueError("Ada-conditioned blocks require metadata embedding")
            raw_scale = metadata_outputs["ffn_scale"]
            metadata_scale = _apply_metadata_scale_mapping(
                raw_scale, self.metadata_scale_mapping,
            )
            # Add only the mapping's delta from identity to preserve timestep scale.
            scale = scale + (metadata_scale - 1.0)
        ffn_input = normalized * scale[:, None, :]
        if self.metadata_conditioning == "ada_attn_ffn" and self.metadata_shift:
            ffn_input = ffn_input + metadata_outputs["ffn_shift"][:, None, :]
        gate, value = self.ffn_in(ffn_input).chunk(2, dim=-1)
        ffn_output = self.ffn_out(F.silu(gate) * value)
        if self.metadata_ffn_residual_gate:
            # AdaLN-Zero-style residual gate: zero initialization closes the
            # FFN residual path initially; one channel gate is broadcast over T.
            residual_gate = _apply_metadata_ffn_gate_mapping(
                metadata_outputs["ffn_gate"], self.metadata_ffn_gate_mapping,
            )
            ffn_output = ffn_output * residual_gate[:, None, :].to(ffn_output.dtype)
        return target + ffn_output


class _OutputRefinementBlock(nn.Module):
    """Full-resolution self-attention block with target Ada and branch gates."""

    def __init__(
        self,
        width: int,
        heads: int,
        kv_heads: int,
        ff_mult: float = 2.0,
        use_ada: bool = False,
        metadata_scale_mapping: str = "softplus1_normalized",
        metadata_ffn_gate_mapping: str = "silu",
        metadata_shift: bool = False,
        metadata_ffn_residual_gate: bool = False,
        use_condition_attention: bool = False,
    ):
        super().__init__()
        if min(width, heads, kv_heads) <= 0 or width % heads or heads % kv_heads:
            raise ValueError("output-refinement heads must divide width and query heads")
        if metadata_scale_mapping not in METADATA_SCALE_MAPPINGS:
            raise ValueError("unknown metadata scale mapping")
        if metadata_ffn_gate_mapping not in METADATA_FFN_GATE_MAPPINGS:
            raise ValueError("unknown metadata FFN gate mapping")
        self.width = width
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = width // heads
        self.use_ada = bool(use_ada)
        self.metadata_scale_mapping = metadata_scale_mapping
        self.metadata_ffn_gate_mapping = metadata_ffn_gate_mapping
        self.metadata_shift = bool(metadata_shift)
        self.metadata_ffn_residual_gate = bool(metadata_ffn_residual_gate)
        self.use_condition_attention = bool(use_condition_attention)
        self.norm1 = nn.RMSNorm(width)
        kv_width = kv_heads * self.head_dim
        # QKV and the token/head attention gate share the same Ada-modulated input.
        self.qkv_gate_proj = nn.Linear(width, width + 2 * kv_width + heads, bias=False)
        self.attn_gate_bias = nn.Parameter(torch.zeros(heads))
        gate_start = width + 2 * kv_width
        nn.init.zeros_(self.qkv_gate_proj.weight[gate_start:])
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        if self.use_condition_attention:
            self.condition_norm = nn.RMSNorm(width)
            self.condition_kv_proj = nn.Linear(width, 2 * kv_heads * self.head_dim, bias=False)
            self.condition_k_norm = nn.RMSNorm(self.head_dim)
            self.condition_q_norm = nn.RMSNorm(width)
            self.condition_q_proj = nn.Linear(width, width, bias=False)
            self.condition_out_proj = nn.Linear(width, width, bias=False)
            nn.init.zeros_(self.condition_out_proj.weight)
        self.out_proj = nn.Linear(width, width, bias=False)
        self.norm2 = nn.RMSNorm(width)
        hidden = max(1, int(width * ff_mult))
        self.ffn_in = nn.Linear(width, 2 * hidden, bias=False)
        self.ffn_out = nn.Linear(hidden, width, bias=False)
        self.ffn_scale = nn.Linear(width, width)
        nn.init.zeros_(self.ffn_scale.weight)
        nn.init.zeros_(self.ffn_scale.bias)

        if self.use_ada:
            metadata_names = ["attn_scale", "ffn_scale"]
            if self.metadata_shift:
                metadata_names.extend(("attn_shift", "ffn_shift"))
            if self.metadata_ffn_residual_gate:
                metadata_names.append("ffn_gate")
            self.metadata_projection_names = tuple(metadata_names)
            self.metadata_projection = nn.Linear(
                width, len(metadata_names) * width, bias=False,
            )
            nn.init.zeros_(self.metadata_projection.weight)
            biased_names = tuple(name for name in metadata_names if name != "ffn_gate")
            self.metadata_projection_bias_names = biased_names
            self.metadata_projection_bias = nn.Parameter(
                torch.zeros(len(biased_names) * width),
            )
        else:
            self.metadata_projection_names = ()
            self.metadata_projection = None
            self.metadata_projection_bias_names = ()
            self.register_parameter("metadata_projection_bias", None)

    def _metadata_outputs(self, embedding: torch.Tensor | None) -> dict[str, torch.Tensor]:
        if not self.use_ada:
            return {}
        if embedding is None or embedding.shape[-1] != self.width:
            raise ValueError("Ada-enabled output refinement requires metadata embedding (B,D)")
        assert self.metadata_projection is not None
        assert self.metadata_projection_bias is not None
        values = self.metadata_projection(embedding).split(self.width, dim=-1)
        outputs = dict(zip(self.metadata_projection_names, values, strict=True))
        biases = self.metadata_projection_bias.split(self.width, dim=-1)
        for name, bias in zip(self.metadata_projection_bias_names, biases, strict=True):
            outputs[name] = outputs[name] + bias
        return outputs

    def prepare_condition(
        self,
        condition: torch.Tensor,
        mask: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_condition_attention:
            raise RuntimeError("condition attention is disabled for this output block")
        batch, count, _ = condition.shape
        normalized = self.condition_norm(condition)
        key, value = self.condition_kv_proj(normalized).split(
            self.kv_heads * self.head_dim, dim=-1,
        )
        key = key.reshape(batch, count, self.kv_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch, count, self.kv_heads, self.head_dim).transpose(1, 2)
        key = apply_multimodal_rope(self.condition_k_norm(key), positions)
        value = value.to(dtype=key.dtype)
        return (
            key.masked_fill(~mask[:, None, :, None], 0),
            value.masked_fill(~mask[:, None, :, None], 0),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        timestep_embedding: torch.Tensor,
        positions: torch.Tensor,
        metadata_embedding: torch.Tensor | None,
        condition_key: torch.Tensor | None = None,
        condition_value: torch.Tensor | None = None,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, count, _ = tokens.shape
        if timestep_embedding.shape != (batch, self.width):
            raise ValueError("output-refinement timestep embedding must have shape (B,D)")
        if self.use_ada and (
            metadata_embedding is None or metadata_embedding.shape != (batch, self.width)
        ):
            raise ValueError("Ada-enabled output refinement requires metadata embedding (B,D)")
        if positions.shape != (batch, count, 3):
            raise ValueError("output-refinement positions must have shape (B,T,3)")

        metadata = self._metadata_outputs(metadata_embedding)
        attn_input = self.norm1(tokens)
        if self.use_ada:
            attn_input = attn_input * _apply_metadata_scale_mapping(
                metadata["attn_scale"], self.metadata_scale_mapping,
            )[:, None, :]
        if self.use_ada and self.metadata_shift:
            attn_input = attn_input + metadata["attn_shift"][:, None, :]
        projected = self.qkv_gate_proj(attn_input)
        kv_width = self.kv_heads * self.head_dim
        query, key, value, gate_logits = projected.split(
            (self.width, kv_width, kv_width, self.heads), dim=-1,
        )
        query = query.reshape(batch, count, self.heads, self.head_dim).transpose(1, 2)
        key = key.reshape(batch, count, self.kv_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch, count, self.kv_heads, self.head_dim).transpose(1, 2)
        query = apply_multimodal_rope(self.q_norm(query), positions)
        key = apply_multimodal_rope(self.k_norm(key), positions)
        if self.use_condition_attention:
            if condition_key is None or condition_value is None or condition_mask is None:
                raise ValueError("condition-attention output blocks require cached condition K/V")
            if (
                condition_key.ndim != 4 or condition_key.shape != condition_value.shape
                or condition_key.shape[0] != batch
                or condition_key.shape[1] != self.kv_heads
                or condition_key.shape[3] != self.head_dim
                or condition_mask.shape != (batch, condition_key.shape[2])
                or condition_mask.dtype != torch.bool
            ):
                raise ValueError(
                    "invalid output condition cache; expected K/V shape "
                    "(B, kv_heads, condition_tokens, head_dim) and boolean mask (B, condition_tokens)"
                )
            if any(t.device != tokens.device for t in (
                condition_key, condition_value, condition_mask,
            )):
                raise ValueError("output condition cache must share the target device")
            if condition_key.dtype != query.dtype or condition_value.dtype != query.dtype:
                raise TypeError("output condition K/V dtype must match target Q/K/V dtype")
        elif any(item is not None for item in (condition_key, condition_value, condition_mask)):
            raise ValueError("condition cache was supplied to an output block with conditioning off")
        if value.dtype != query.dtype:
            value = value.to(query.dtype)
        attended = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0,
            enable_gqa=self.heads != self.kv_heads,
        )
        head_gate = 1.0 + F.silu(gate_logits + self.attn_gate_bias)
        attended = attended * head_gate.transpose(1, 2).unsqueeze(-1).to(attended.dtype)
        attended = attended.transpose(1, 2).reshape(batch, count, self.width)
        tokens = tokens + self.out_proj(attended)

        if self.use_condition_attention:
            assert condition_key is not None
            assert condition_value is not None
            assert condition_mask is not None
            condition_query = self.condition_q_proj(self.condition_q_norm(tokens))
            condition_query = condition_query.reshape(
                batch, count, self.heads, self.head_dim,
            ).transpose(1, 2)
            condition_query = apply_multimodal_rope(
                self.q_norm(condition_query), positions,
            )
            condition_attended = F.scaled_dot_product_attention(
                condition_query,
                condition_key,
                condition_value,
                attn_mask=condition_mask[:, None, None, :],
                dropout_p=0.0,
                enable_gqa=self.heads != self.kv_heads,
            )
            condition_attended = condition_attended.transpose(1, 2).reshape(
                batch, count, self.width,
            )
            tokens = tokens + self.condition_out_proj(condition_attended)

        ffn_scale = 1.0 + self.ffn_scale(timestep_embedding)
        if self.use_ada:
            metadata_ffn_scale = _apply_metadata_scale_mapping(
                metadata["ffn_scale"], self.metadata_scale_mapping,
            )
            ffn_scale = ffn_scale + (metadata_ffn_scale - 1.0)
        ffn_input = self.norm2(tokens) * ffn_scale[:, None, :]
        if self.use_ada and self.metadata_shift:
            ffn_input = ffn_input + metadata["ffn_shift"][:, None, :]
        gate, value = self.ffn_in(ffn_input).chunk(2, dim=-1)
        ffn_output = self.ffn_out(F.silu(gate) * value)
        if self.metadata_ffn_residual_gate:
            residual_gate = _apply_metadata_ffn_gate_mapping(
                metadata["ffn_gate"], self.metadata_ffn_gate_mapping,
            )
            ffn_output = ffn_output * residual_gate[:, None, :].to(ffn_output.dtype)
        return tokens + ffn_output


class NoVFCBDiT(nn.Module):
    """Flow-matching DiT with configurable target patches and a VLM prefix.

    Target patch size is tracked independently from reference-latent handling.
    Reference latents can be fused into Qwen vision tokens before the
    bidirectional condition adapter.
    """

    def __init__(
        self,
        *,
        qwen_dim: int,
        latent_channels: int = 16,
        width: int = 768,
        depth: int = 12,
        heads: int = 12,
        kv_heads: int = 4,
        adapter_depth: int = 2,
        ff_mult: float = 3.0,
        gradient_checkpointing: bool = False,
        latent_downsample_factor: int = 1,
        target_latent_downsample_factor: int | None = None,
        output_refinement_depth: int = 0,
        output_refinement_conditioning: str = "none",
        output_skip_fusion_mode: str = "add",
        output_head_ada_scale: bool = False,
        fuse_reference_latent_to_vision: bool = False,
        reference_latent_fusion_mode: str = "legacy_latent_to_qwen",
        metadata_conditioning: str = "none",
        metadata_scale_mapping: str = "softplus1_normalized",
        metadata_ffn_gate_mapping: str = "silu",
        metadata_shift: bool = False,
        attention_head_gate: str = "input_silu",
        metadata_ffn_residual_gate: bool | None = None,
        fuse_same_input_projections: bool = True,
    ):
        super().__init__()
        if width % heads or width // heads < 6 or (width // heads) % 2:
            raise ValueError("model head_dim must be even and at least 6")
        if latent_downsample_factor not in (1, 2, 4):
            raise ValueError("latent_downsample_factor must be one of 1, 2, or 4")
        if target_latent_downsample_factor is None:
            # Backward-compatible constructor behavior for older checkpoints
            # and direct callers that used one factor for both paths.
            target_latent_downsample_factor = latent_downsample_factor
        if target_latent_downsample_factor not in (1, 2, 4):
            raise ValueError("target_latent_downsample_factor must be one of 1, 2, or 4")
        if output_refinement_depth < 0:
            raise ValueError("output_refinement_depth must be non-negative")
        if output_refinement_conditioning not in ("none", "cross_attention"):
            raise ValueError("output_refinement_conditioning must be none or cross_attention")
        if output_refinement_conditioning != "none" and output_refinement_depth == 0:
            raise ValueError("output-refinement conditioning requires at least one refinement block")
        if output_skip_fusion_mode not in ("add", "concat_linear"):
            raise ValueError("output_skip_fusion_mode must be add or concat_linear")
        if output_head_ada_scale and output_refinement_depth == 0:
            raise ValueError("output-head Ada scale requires output refinement blocks")
        self.latent_channels = latent_channels
        self.width = width
        self.latent_downsample_factor = latent_downsample_factor
        self.target_latent_downsample_factor = target_latent_downsample_factor
        self.output_refinement_depth = int(output_refinement_depth)
        self.output_refinement_conditioning = output_refinement_conditioning
        self.output_skip_fusion_mode = output_skip_fusion_mode
        self.output_head_ada_scale = bool(output_head_ada_scale)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if metadata_conditioning not in (
            "none", "attn_concat_ffn_ada", "ada_attn_ffn",
        ):
            raise ValueError("unknown metadata conditioning variant")
        if metadata_scale_mapping not in METADATA_SCALE_MAPPINGS:
            raise ValueError("unknown metadata scale mapping")
        if metadata_ffn_gate_mapping not in METADATA_FFN_GATE_MAPPINGS:
            raise ValueError("unknown metadata FFN gate mapping")
        if metadata_shift and metadata_conditioning != "ada_attn_ffn":
            raise ValueError("metadata shift requires ada_attn_ffn conditioning")
        if attention_head_gate not in ("input_silu", "timestep_sigmoid"):
            raise ValueError("unknown attention head gate")
        if metadata_ffn_residual_gate is None:
            metadata_ffn_residual_gate = metadata_conditioning == "ada_attn_ffn"
        if metadata_ffn_residual_gate and metadata_conditioning != "ada_attn_ffn":
            raise ValueError("metadata FFN residual gate requires ada_attn_ffn conditioning")
        self.metadata_conditioning = metadata_conditioning
        self.metadata_scale_mapping = metadata_scale_mapping
        self.metadata_ffn_gate_mapping = metadata_ffn_gate_mapping
        self.metadata_shift = bool(metadata_shift)
        self.attention_head_gate = attention_head_gate
        self.metadata_ffn_residual_gate = bool(metadata_ffn_residual_gate)
        self.fuse_same_input_projections = bool(fuse_same_input_projections)
        self.meta_embedding = (
            nn.Sequential(nn.Linear(2, width), nn.SiLU(), nn.Linear(width, width))
            if metadata_conditioning != "none" else None
        )
        self.fuse_reference_latent_to_vision = bool(fuse_reference_latent_to_vision)
        if reference_latent_fusion_mode not in (
            "legacy_latent_to_qwen", "qwen_to_full_latent", "qwen_to_half_latent",
            "qwen_to_half_latent_add",
        ):
            raise ValueError("unknown reference latent fusion mode")
        self.reference_latent_fusion_mode = reference_latent_fusion_mode
        self.adapter = VLMAdapter(
            qwen_dim, latent_channels, width,
            depth=adapter_depth, ff_mult=2.0,
            latent_downsample_factor=latent_downsample_factor,
            fuse_reference_latent_to_vision=fuse_reference_latent_to_vision,
            reference_latent_fusion_mode=reference_latent_fusion_mode,
        )
        self.target_in = nn.Conv2d(
            latent_channels, width,
            kernel_size=target_latent_downsample_factor,
            stride=target_latent_downsample_factor,
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width),
        )
        self.blocks = nn.ModuleList(
            _TargetBlock(
                width, heads, kv_heads, ff_mult, metadata_conditioning,
                metadata_scale_mapping, metadata_ffn_gate_mapping, metadata_shift, attention_head_gate,
                self.metadata_ffn_residual_gate, self.fuse_same_input_projections,
            ) for _ in range(depth)
        )
        self.output_norm = nn.RMSNorm(width)
        if self.output_refinement_depth:
            if target_latent_downsample_factor == 1:
                self.output_upsample = nn.Identity()
            else:
                self.output_upsample = nn.ConvTranspose2d(
                    width, width,
                    kernel_size=target_latent_downsample_factor,
                    stride=target_latent_downsample_factor,
                )
            self.target_latent_skip = nn.Linear(latent_channels, width, bias=False)
            if output_skip_fusion_mode == "concat_linear":
                self.output_fusion = nn.Linear(2 * width, width)
                # Keep legacy checkpoint behavior when explicitly selected.
                with torch.no_grad():
                    self.output_fusion.weight.zero_()
                    self.output_fusion.weight[:, :width].copy_(torch.eye(width))
                    if self.output_fusion.bias is not None:
                        self.output_fusion.bias.zero_()
            else:
                self.output_fusion = None
                # Begin with the upsampled main branch unchanged, then learn
                # an additive target-latent skip from zero contribution.
                nn.init.zeros_(self.target_latent_skip.weight)
            self.output_refinement = nn.ModuleList(
                _OutputRefinementBlock(
                    width, heads, kv_heads, ff_mult=2.0,
                    use_ada=metadata_conditioning == "ada_attn_ffn",
                    metadata_scale_mapping=metadata_scale_mapping,
                    metadata_ffn_gate_mapping=metadata_ffn_gate_mapping,
                    metadata_shift=metadata_shift,
                    metadata_ffn_residual_gate=self.metadata_ffn_residual_gate,
                    use_condition_attention=output_refinement_conditioning == "cross_attention",
                )
                for _ in range(self.output_refinement_depth)
            )
            self.output = nn.Linear(width, latent_channels)
            if self.output_head_ada_scale:
                if metadata_conditioning != "ada_attn_ffn":
                    raise ValueError(
                        "output-head Ada scale requires ada_attn_ffn metadata conditioning"
                    )
                # The shared metadata embedding predicts a channel-wise scale
                # for the nonlinear correction branch only. Zero initialization
                # makes the mapped scale exactly one at startup.
                self.output_head_ada_proj = nn.Linear(width, width)
                nn.init.zeros_(self.output_head_ada_proj.weight)
                nn.init.zeros_(self.output_head_ada_proj.bias)
                self.output_head_u = nn.Linear(width, width, bias=False)
                self.output_head_g = nn.Linear(width, width, bias=False)
                self.output_head_d = nn.Linear(width, latent_channels, bias=False)
                nn.init.zeros_(self.output_head_d.weight)
            else:
                self.output_head_ada_proj = None
                self.output_head_u = None
                self.output_head_g = None
                self.output_head_d = None
        else:
            # Preserve the old checkpoint layout for direct callers and legacy
            # resume configs that explicitly select zero refinement blocks.
            self.output_upsample = nn.Identity()
            self.target_latent_skip = None
            self.output_fusion = None
            self.output_refinement = nn.ModuleList()
            self.output_head_ada_proj = None
            self.output_head_u = None
            self.output_head_g = None
            self.output_head_d = None
            if target_latent_downsample_factor == 1:
                self.output = nn.Conv2d(width, latent_channels, kernel_size=1)
            else:
                self.output = nn.ConvTranspose2d(
                    width, latent_channels,
                    kernel_size=target_latent_downsample_factor,
                    stride=target_latent_downsample_factor,
                )

    def load_checkpoint_state_dict(
        self, state_dict: dict[str, torch.Tensor], *, adapter_type: str | None,
    ) -> int:
        """Load current weights, upgrading older FFN checkpoints with unused attention tensors."""
        if adapter_type != "ffn":
            raise ValueError(
                "Only GatedFFN VLM adapter checkpoints are supported for inference/resume; "
                f"checkpoint adapter_type={adapter_type!r} uses a retired adapter architecture."
            )
        legacy_attention_parts = (
            ".norm1.", ".qkv_proj.", ".q_proj.", ".k_proj.", ".v_proj.",
            ".q_norm.", ".k_norm.", ".out_proj.",
        )
        compatible_weights = {
            name: value for name, value in state_dict.items()
            if not (
                name.startswith("adapter.blocks.")
                and any(part in name for part in legacy_attention_parts)
            )
        }
        dropped = len(state_dict) - len(compatible_weights)
        self.load_state_dict(compatible_weights, strict=True)
        if dropped:
            print(
                f"Dropped {dropped} unused frozen adapter-attention tensors from an older "
                "GatedFFN checkpoint."
            )
        return dropped

    def load_initialization_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        source_config: dict | None = None,
    ) -> tuple[int, int]:
        """Load shape-compatible weights and fold legacy concat fusions when possible."""
        source = dict(state_dict)
        source_config = source_config or {}
        source_reference_mode = source_config.get("reference_latent_fusion_mode")
        if (
            self.adapter.reference_latent_fusion_mode == "qwen_to_half_latent_add"
            and source_reference_mode == "qwen_to_half_latent"
        ):
            fusion_weight = source.get("adapter.reference_fusion.weight")
            latent_weight = source.get("adapter.reference_latent_in.weight")
            latent_module = self.adapter.reference_latent_in
            latent_module_bias = getattr(latent_module, "bias", None)
            if (
                fusion_weight is not None
                and latent_weight is not None
                and fusion_weight.shape[1] == self.adapter.qwen_dim + self.latent_channels
                and fusion_weight.shape[0] == self.width
                and latent_weight.shape[0] == self.latent_channels
                and tuple(latent_weight.shape[1:])
                == (self.latent_channels, 2, 2)
            ):
                source["adapter.reference_qwen_in.weight"] = fusion_weight[
                    :, :self.adapter.qwen_dim,
                ]
                old_bias = source.get("adapter.reference_fusion.bias")
                if old_bias is not None:
                    source["adapter.reference_qwen_in.bias"] = old_bias
                source["adapter.reference_latent_in.weight"] = torch.einsum(
                    "oj,jihw->oihw",
                    fusion_weight[:, self.adapter.qwen_dim:],
                    latent_weight,
                )
                latent_bias = source.get("adapter.reference_latent_in.bias")
                if latent_bias is not None and latent_module_bias is not None:
                    source["adapter.reference_latent_in.bias"] = torch.mv(
                        fusion_weight[:, self.adapter.qwen_dim:], latent_bias,
                    )
                elif latent_module_bias is not None:
                    source["adapter.reference_latent_in.bias"] = torch.zeros_like(
                        latent_module_bias, device="cpu",
                    )

        source_output_mode = source_config.get("output_skip_fusion_mode")
        if source_output_mode is None:
            # Some recorded runs serialized the parser's unresolved None even
            # though the constructed model used the legacy concat projection.
            # Infer that architecture from its distinctive fusion weight.
            source_output_mode = (
                "concat_linear" if "output_fusion.weight" in source else "add"
            )
        if self.output_refinement_depth and self.output_skip_fusion_mode == "add":
            fusion_weight = source.get("output_fusion.weight")
            skip_weight = source.get("target_latent_skip.weight")
            upsample_weight = source.get("output_upsample.weight")
            if (
                source_output_mode == "concat_linear"
                and fusion_weight is not None
                and skip_weight is not None
                and upsample_weight is not None
                and isinstance(self.output_upsample, nn.ConvTranspose2d)
            ):
                width = self.width
                main_weight = fusion_weight[:, :width]
                skip_fusion_weight = fusion_weight[:, width:]
                source["output_upsample.weight"] = torch.einsum(
                    "ab,ibxy->iaxy", main_weight, upsample_weight,
                )
                upsample_bias = source.get("output_upsample.bias")
                fusion_bias = source.get("output_fusion.bias")
                if upsample_bias is not None and fusion_bias is not None:
                    source["output_upsample.bias"] = main_weight @ upsample_bias + fusion_bias
                elif upsample_bias is not None:
                    source["output_upsample.bias"] = main_weight @ upsample_bias
                elif fusion_bias is not None:
                    source["output_upsample.bias"] = fusion_bias
                source["target_latent_skip.weight"] = skip_fusion_weight @ skip_weight

        destination = self.state_dict()
        compatible = {
            key: value for key, value in source.items()
            if key in destination and tuple(value.shape) == tuple(destination[key].shape)
        }
        if not compatible:
            raise ValueError("initialization checkpoint has no matching model tensors")
        missing, _unexpected = self.load_state_dict(compatible, strict=False)
        return len(compatible), len(missing)

    def prepare_condition(
        self,
        qwen_hidden: torch.Tensor,
        qwen_mask: torch.Tensor,
        qwen_positions: torch.Tensor,
        reference_latent: torch.Tensor | None = None,
        reference_mask: torch.Tensor | None = None,
        reference_positions: torch.Tensor | None = None,
        qwen_vision_mask: torch.Tensor | None = None,
    ) -> ConditionKVCache:
        condition, mask, positions = self.adapter(
            qwen_hidden, qwen_mask, qwen_positions,
            reference_latent, reference_mask, reference_positions,
            qwen_vision_mask,
        )
        caches = [
            block.prepare_condition(condition, mask, positions)
            for block in self.blocks
        ]
        output_caches = [
            block.prepare_condition(condition, mask, positions)
            for block in self.output_refinement
            if block.use_condition_attention
        ]
        return ConditionKVCache(
            keys=tuple(item[0] for item in caches),
            values=tuple(item[1] for item in caches),
            mask=mask,
            output_keys=tuple(item[0] for item in output_caches),
            output_values=tuple(item[1] for item in output_caches),
        )

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        condition_cache: ConditionKVCache,
        metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noisy_latent.ndim != 4 or noisy_latent.shape[1] != self.latent_channels:
            raise ValueError("noisy_latent must have shape (B, latent_channels, H, W)")
        batch, _channels, height, width = noisy_latent.shape
        factor = self.target_latent_downsample_factor
        if height % factor or width % factor:
            raise ValueError(
                "noisy_latent height and width must be divisible by "
                "latent_downsample_factor"
            )
        if len(condition_cache.keys) != len(self.blocks):
            raise ValueError("condition cache depth does not match DiT depth")
        expected_output_cache_depth = (
            self.output_refinement_depth
            if self.output_refinement_conditioning == "cross_attention" else 0
        )
        if len(condition_cache.output_keys) != expected_output_cache_depth:
            raise ValueError("condition cache depth does not match output-refinement conditioning")
        if condition_cache.mask.shape[0] != batch or condition_cache.mask.device != noisy_latent.device:
            raise ValueError("condition cache batch/device does not match target latent")
        dtype = self.target_in.weight.dtype
        target_features = self.target_in(noisy_latent.to(dtype=dtype))
        target_height, target_width = target_features.shape[-2:]
        target = target_features.flatten(2).transpose(1, 2)
        target_positions = grid_positions(
            batch, target_height, target_width,
            device=target.device,
            reference_id=0,
            spatial_stride=factor,
            center_offset=(factor - 1) / 2,
        )
        timestep = timestep.to(device=target.device, dtype=target.dtype).reshape(batch, 1)
        time = self.time_mlp(timestep)
        metadata_embedding = None
        if self.meta_embedding is not None:
            if metadata is None or metadata.shape != (batch, 2):
                raise ValueError("metadata-conditioned DiT requires metadata with shape (B,2)")
            metadata = metadata.to(
                device=target.device, dtype=next(self.meta_embedding.parameters()).dtype,
            )
            metadata_embedding = self.meta_embedding(metadata)
        elif metadata is not None:
            raise ValueError("metadata was provided to a DiT without metadata conditioning")
        for index, block in enumerate(self.blocks):
            block_inputs = (
                target, time, target_positions,
                condition_cache.keys[index], condition_cache.values[index],
                condition_cache.mask, metadata_embedding,
            )
            if self.training and self.gradient_checkpointing:
                target = checkpoint(block, *block_inputs, use_reentrant=False)
            else:
                target = block(*block_inputs)
        output_features = target.transpose(1, 2).reshape(
            batch, self.width, target_height, target_width,
        )
        if self.output_refinement_depth:
            output_features = self.output_upsample(output_features)
            if output_features.shape[-2:] != (height, width):
                raise RuntimeError(
                    "target feature upsampling did not restore the latent grid: "
                    f"expected {(height, width)}, got {tuple(output_features.shape[-2:])}"
                )
            if self.target_latent_skip is None:
                raise RuntimeError("output fusion modules are missing")
            upsampled_tokens = output_features.flatten(2).transpose(1, 2)
            skip_tokens = noisy_latent.permute(0, 2, 3, 1).to(
                dtype=self.target_latent_skip.weight.dtype,
            )
            # Keep the same row-major H,W order as output_features.flatten(2).
            skip_tokens = self.target_latent_skip(skip_tokens).flatten(1, 2)
            expected_tokens = (batch, height * width, self.width)
            if upsampled_tokens.shape != expected_tokens or skip_tokens.shape != expected_tokens:
                raise RuntimeError(
                    "output fusion token shapes do not match the latent grid: "
                    f"expected {expected_tokens}, got upsampled={tuple(upsampled_tokens.shape)} "
                    f"skip={tuple(skip_tokens.shape)}"
                )
            if self.output_skip_fusion_mode == "add":
                output_tokens = upsampled_tokens + skip_tokens
            else:
                if self.output_fusion is None:
                    raise RuntimeError("concatenation fusion layer is missing")
                output_tokens = self.output_fusion(
                    torch.cat((upsampled_tokens, skip_tokens), dim=-1),
                )
            output_positions = grid_positions(
                batch, height, width, device=output_tokens.device, reference_id=0,
            )
            output_cache_index = 0
            for block in self.output_refinement:
                if block.use_condition_attention:
                    condition_key = condition_cache.output_keys[output_cache_index]
                    condition_value = condition_cache.output_values[output_cache_index]
                    output_cache_index += 1
                    block_inputs = (
                        output_tokens, time, output_positions, metadata_embedding,
                        condition_key, condition_value, condition_cache.mask,
                    )
                else:
                    block_inputs = (
                        output_tokens, time, output_positions, metadata_embedding,
                    )
                if self.training and self.gradient_checkpointing:
                    output_tokens = checkpoint(
                        block, *block_inputs, use_reentrant=False,
                    )
                else:
                    output_tokens = block(*block_inputs)
            output_tokens = self.output_norm(output_tokens)
            with torch.autocast(device_type=output_tokens.device.type, enabled=False):
                head_input = output_tokens.float()
                output = F.linear(
                    head_input,
                    self.output.weight.float(),
                    self.output.bias.float() if self.output.bias is not None else None,
                )
                if self.output_head_ada_scale:
                    if metadata_embedding is None:
                        raise ValueError("output-head Ada scale requires metadata embedding")
                    assert self.output_head_ada_proj is not None
                    assert self.output_head_u is not None
                    assert self.output_head_g is not None
                    assert self.output_head_d is not None
                    raw_scale = F.linear(
                        metadata_embedding.float(),
                        self.output_head_ada_proj.weight.float(),
                        self.output_head_ada_proj.bias.float(),
                    )
                    correction_input = head_input * _apply_metadata_scale_mapping(
                        raw_scale, self.metadata_scale_mapping,
                    )[:, None, :]
                    value = F.linear(
                        correction_input, self.output_head_u.weight.float(),
                    )
                    gate = F.linear(
                        correction_input, self.output_head_g.weight.float(),
                    )
                    correction = F.linear(
                        value * F.silu(gate), self.output_head_d.weight.float(),
                    )
                    output = output + correction
            return output.transpose(1, 2).reshape(batch, self.latent_channels, height, width)

        # RMSNorm's normalized axis is the final channel dimension; the
        # convolutional feature map is stored as (B, C, H, W).
        output_features = self.output_norm(
            output_features.permute(0, 2, 3, 1),
        ).permute(0, 3, 1, 2).contiguous()
        # Keep the final prediction head and its backward pass in FP32. For a
        # 1x1 projection, use the algebraically equivalent per-pixel Linear
        # form to isolate Conv2d-specific backward failures without changing
        # the checkpoint parameter layout.
        with torch.autocast(device_type=output_features.device.type, enabled=False):
            output_features = output_features.float()
            if self.target_latent_downsample_factor == 1:
                output = F.linear(
                    output_features.permute(0, 2, 3, 1),
                    self.output.weight.flatten(1),
                    self.output.bias,
                ).permute(0, 3, 1, 2).contiguous()
            else:
                output = self.output(output_features)
        if output.shape != noisy_latent.shape:
            raise RuntimeError(
                "DiT output must restore the input latent shape: "
                f"expected {tuple(noisy_latent.shape)}, got {tuple(output.shape)}"
            )
        return output

    @torch.no_grad()
    def metadata_diagnostics(self, metadata: torch.Tensor) -> dict[str, torch.Tensor]:
        """Summarize Ada multipliers and shifts across blocks for logging.

        Reported scales are the metadata-derived multipliers alone; FFN
        timestep modulation is intentionally excluded. This method is opt-in
        at the training-callsite because it recomputes the small Ada projections.
        """
        if self.metadata_conditioning != "ada_attn_ffn" or self.meta_embedding is None:
            return {}
        if metadata.ndim != 2 or metadata.shape[-1] != 2:
            raise ValueError("metadata diagnostics expect (B,2) target-canvas metadata")
        parameter = next(self.meta_embedding.parameters())
        embedding = self.meta_embedding(metadata.to(device=parameter.device, dtype=parameter.dtype))
        scales: dict[str, list[torch.Tensor]] = {"attn": [], "ffn": []}
        shifts: dict[str, list[torch.Tensor]] = {"attn": [], "ffn": []}
        for block in self.blocks:
            outputs = block.metadata_projection_outputs(embedding)
            scales["attn"].append(_apply_metadata_scale_mapping(
                outputs["attn_scale"], self.metadata_scale_mapping,
            ))
            scales["ffn"].append(_apply_metadata_scale_mapping(
                outputs["ffn_scale"], self.metadata_scale_mapping,
            ))
            if self.metadata_shift:
                shifts["attn"].append(outputs["attn_shift"])
                shifts["ffn"].append(outputs["ffn_shift"])

        diagnostics: dict[str, torch.Tensor] = {}
        for branch, values in scales.items():
            combined = torch.cat([value.float().reshape(-1) for value in values])
            prefix = f"metadata_{branch}_scale"
            diagnostics[f"{prefix}_mean"] = combined.mean()
            diagnostics[f"{prefix}_min"] = combined.min()
            diagnostics[f"{prefix}_max"] = combined.max()
            diagnostics[f"{prefix}_near_zero_fraction"] = (combined.abs() < 0.1).float().mean()
            diagnostics[f"{prefix}_negative_fraction"] = (combined < 0).float().mean()
        for branch, values in shifts.items():
            if values:
                combined = torch.cat([value.float().reshape(-1) for value in values])
                diagnostics[f"metadata_{branch}_shift_rms"] = combined.square().mean().sqrt()
                diagnostics[f"metadata_{branch}_shift_absmax"] = combined.abs().max()
        return diagnostics


__all__ = [
    "ConditionKVCache",
    "METADATA_SCALE_MAPPINGS",
    "NoVFCBDiT",
    "VLMAdapter",
    "apply_multimodal_rope",
    "build_qwen_condition_positions",
    "grid_positions",
]
