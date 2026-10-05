"""Qwen3.5 semantic and raw visual feature extraction for Stage 0."""

from __future__ import annotations

import argparse
from typing import Any

import torch
from PIL import Image
from torch import nn


DEFAULT_QWEN35_MODEL = "Qwen/Qwen3.5-0.8B"


class _StopAfterConditionTap(Exception):
    """Internal control flow used to stop Qwen after capturing one condition tap."""


def parse_condition_layer(value: str) -> int | str:
    """Parse a one-based decoder block or the post-final-norm ``final`` tap."""
    if value.lower() == "final":
        return "final"
    try:
        layer = int(value)
    except ValueError as error:
        message = "condition layer must be a positive integer or 'final'"
        raise argparse.ArgumentTypeError(message) from error
    if layer <= 0:
        raise argparse.ArgumentTypeError(
            "integer condition layers are one-based and must be positive"
        )
    return layer


def validate_condition_layer(condition_layer: int | str, layer_count: int) -> None:
    """Validate an intermediate decoder-block tap or final normalized output."""
    if condition_layer == "final":
        return
    if not isinstance(condition_layer, int) or not 1 <= condition_layer <= layer_count:
        raise ValueError(
            f"condition_layer must be in [1, {layer_count}] or 'final'; "
            f"got {condition_layer!r}"
        )


def load_qwen35_encoder(
    *,
    model_id: str = DEFAULT_QWEN35_MODEL,
    device: torch.device | str,
    dtype: torch.dtype,
):
    """Load a frozen Qwen3.5 multimodal model and its processor."""
    try:
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as error:
        raise RuntimeError(
            "Qwen3.5 loading requires a Transformers version that exports "
            "AutoModelForMultimodalLM"
        ) from error

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=dtype)
    model = model.to(device=device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    text_config = getattr(model.config, "text_config", model.config)
    layer_count = int(getattr(text_config, "num_hidden_layers", 0))
    hidden_width = int(getattr(text_config, "hidden_size", 0))
    if layer_count <= 0 or hidden_width <= 0:
        raise ValueError("Could not read Qwen3.5 text layer count/hidden width")
    vision = _vision_model(model)
    vision_width = int(getattr(vision.config, "hidden_size", 0))
    if vision_width <= 0:
        raise ValueError("Could not read Qwen3.5 vision hidden width")
    return Qwen35FeatureEncoder(
        model=model,
        processor=processor,
        device=torch.device(device),
        layer_count=layer_count,
        condition_dim=hidden_width,
        teacher_dim=vision_width,
    )


def _vision_model(model: nn.Module) -> nn.Module:
    base = getattr(model, "model", model)
    vision = getattr(base, "visual", None)
    if vision is None:
        raise TypeError("Loaded Qwen3.5 model does not expose model.visual")
    return vision


class Qwen35FeatureEncoder:
    """Frozen Qwen3.5 feature provider with an explicit language-layer tap."""

    def __init__(
        self,
        *,
        model: nn.Module,
        processor: Any,
        device: torch.device,
        layer_count: int,
        condition_dim: int,
        teacher_dim: int,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.layer_count = layer_count
        self.condition_dim = condition_dim
        self.teacher_dim = teacher_dim

    def encode_condition(
        self,
        prompt: str,
        *,
        source_image: Image.Image | None = None,
        condition_layer: int | str,
        include_positions: bool = False,
        include_vision_mask: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Return a decoder-block or final-normalized hidden state.

        Integer taps are one-based decoder-block outputs before the final norm.
        The string ``final`` selects the output after the text model's final norm.
        """
        validate_condition_layer(condition_layer, self.layer_count)
        if not prompt.strip():
            raise ValueError("prompt must contain at least one non-whitespace character")
        content = []
        if source_image is not None:
            if not isinstance(source_image, Image.Image):
                raise TypeError("source_image must be a PIL.Image.Image")
            content.extend((
                {"type": "text", "text": "Source image:"},
                {"type": "image", "image": source_image.convert("RGB")},
            ))
        content.append({"type": "text", "text": "Instruction: " + prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Validate the processor mask before transfer; checking a CUDA scalar
        # after the Qwen forward would force a device-to-host synchronization.
        processor_attention_mask = inputs.get("attention_mask")
        input_ids = inputs.get("input_ids")
        if processor_attention_mask is None and input_ids is not None:
            processor_attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        qwen_positions = None
        vision_mask = None
        if include_vision_mask and not include_positions:
            raise ValueError("include_vision_mask requires include_positions")
        if include_positions:
            if processor_attention_mask is None or input_ids is None:
                raise ValueError("Qwen position metadata requires input_ids and attention_mask")
            modality_types = inputs.get("mm_token_type_ids")
            if modality_types is None:
                image_token_id = getattr(self.model.config, "image_token_id", None)
                if image_token_id is None:
                    raise ValueError("Qwen processor omitted mm_token_type_ids and image_token_id")
                modality_types = (input_ids == int(image_token_id)).to(dtype=torch.int32)
            vision_config = getattr(self.model.config, "vision_config", None)
            merge_size = int(getattr(vision_config, "spatial_merge_size", 0))
            from vfp_dit.model import build_qwen_condition_positions
            valid_tokens = processor_attention_mask.to(dtype=torch.bool)
            qwen_positions = build_qwen_condition_positions(
                modality_types,
                valid_tokens,
                inputs.get("image_grid_thw"),
                spatial_merge_size=merge_size,
                reference_id=1,
            ).to(device=self.device)
            if include_vision_mask:
                vision_mask = ((modality_types == 1) & valid_tokens).to(device=self.device)
        if processor_attention_mask is not None:
            if (
                processor_attention_mask.ndim != 2
                or processor_attention_mask.shape[0] != 1
                or processor_attention_mask.shape[1] == 0
            ):
                raise ValueError(
                    "Qwen processor attention mask must have non-empty shape (1,L), got "
                    + str(tuple(processor_attention_mask.shape))
                )
            if not processor_attention_mask.to(dtype=torch.bool).any(dim=-1).all():
                raise ValueError("Qwen processor returned an all-masked condition")
        inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        base = getattr(self.model, "model", self.model)
        text_model = getattr(base, "language_model", None)
        layers = getattr(text_model, "layers", None)
        if layers is None or len(layers) != self.layer_count:
            raise RuntimeError("Could not locate Qwen3.5 language-model blocks")
        captured: list[torch.Tensor] = []

        def capture_layer_output(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError("Qwen3.5 condition tap must return a tensor")
            captured.append(output)
            # Integer taps do not need deeper text blocks; the final-norm tap
            # does not need the LM head. Preserve the captured tensor exactly.
            raise _StopAfterConditionTap

        if condition_layer == "final":
            final_norm = getattr(text_model, "norm", None)
            if not isinstance(final_norm, nn.Module):
                raise RuntimeError("Could not locate Qwen3.5 final text normalization")
            tap_module = final_norm
        else:
            if not isinstance(condition_layer, int):
                raise TypeError("condition_layer must be an integer or 'final'")
            tap_module = layers[condition_layer - 1]
        handle = tap_module.register_forward_hook(capture_layer_output)
        try:
            with torch.inference_mode():
                try:
                    self.model(
                        **inputs,
                        output_hidden_states=False,
                        use_cache=False,
                        return_dict=True,
                    )
                except _StopAfterConditionTap:
                    pass
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(
                "Expected the selected Qwen3.5 block hook once, got "
                + str(len(captured))
            )
        hidden = captured[0]
        if (
            hidden.ndim != 3
            or hidden.shape[0] != 1
            or hidden.shape[-1] != self.condition_dim
        ):
            raise ValueError(
                "Qwen3.5 condition hidden must be (1,L,"
                + str(self.condition_dim) + "), got " + str(tuple(hidden.shape))
            )
        if processor_attention_mask is None:
            mask = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.bool)
        else:
            mask = processor_attention_mask.to(device=hidden.device, dtype=torch.bool)
        if mask.shape != hidden.shape[:2]:
            raise ValueError("Qwen3.5 attention mask does not match hidden sequence")
        if qwen_positions is not None:
            if qwen_positions.shape != (1, hidden.shape[1], 3):
                raise ValueError("Qwen condition positions do not match hidden sequence")
            if include_vision_mask:
                if vision_mask is None or vision_mask.shape != hidden.shape[:2]:
                    raise ValueError("Qwen vision-token mask does not match hidden sequence")
                return hidden[0].float(), mask[0], qwen_positions[0], vision_mask[0]
            return hidden[0].float(), mask[0], qwen_positions[0]
        return hidden[0].float(), mask[0]

    @torch.inference_mode()
    def encode_visual_teacher(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        """Return the raw pre-merger spatial map as (C,Hv,Wv), in float32.

        This is the final Qwen3.5 Vision block output before its irreversible
        spatial merger. Channel projection and fixed normalization are separate
        Stage-0 calibration steps and are intentionally not hidden here.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")
        image = image.convert("RGB")
        image_inputs = self.processor.image_processor(
            images=[image],
            return_tensors="pt",
        )
        pixel_values = image_inputs.get("pixel_values")
        grid_thw = image_inputs.get("image_grid_thw")
        if pixel_values is None or grid_thw is None:
            raise RuntimeError(
                "Qwen3.5 image processor must return pixel_values and image_grid_thw"
            )
        if grid_thw.shape != (1, 3):
            raise ValueError(
                "Expected one still-image image_grid_thw row, got "
                + str(tuple(grid_thw.shape))
            )
        # Read processor metadata while it is still on CPU. Reading .tolist()
        # after the device transfer would synchronize the GPU every image.
        frames, grid_height, grid_width = (
            int(value) for value in grid_thw[0].tolist()
        )
        if frames != 1 or grid_height <= 0 or grid_width <= 0:
            raise ValueError(
                "Expected a positive still-image grid (1,Hv,Wv), got "
                + str((frames, grid_height, grid_width))
            )
        expected_tokens = frames * grid_height * grid_width
        vision = _vision_model(self.model)
        parameter = next(vision.parameters(), None)
        if parameter is None:
            raise ValueError("Qwen3.5 visual tower has no parameters")
        pixel_values = pixel_values.to(
            device=self.device,
            dtype=parameter.dtype,
        )
        grid_thw = grid_thw.to(device=self.device)
        output = vision(pixel_values, grid_thw)
        tokens = getattr(output, "last_hidden_state", None)
        if tokens is None:
            raise RuntimeError("Qwen3.5 visual tower returned no last_hidden_state")
        if tokens.ndim != 2 or tokens.shape[0] != expected_tokens:
            raise ValueError(
                "Raw Qwen3.5 vision token/grid mismatch: grid="
                + str((frames, grid_height, grid_width))
                + ", tokens=" + str(tuple(tokens.shape))
            )
        if tokens.shape[1] != self.teacher_dim:
            raise ValueError(
                "Qwen3.5 vision width changed: expected "
                + str(self.teacher_dim) + ", got " + str(tokens.shape[1])
            )
        return (
            tokens.reshape(grid_height, grid_width, self.teacher_dim)
            .permute(2, 0, 1)
            .contiguous()
            .float()
        )
