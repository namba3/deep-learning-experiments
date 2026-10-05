"""Backward-compatible import path for adaptive RMS normalization."""

from .layers.adaptive_norm import AdaRMSScaleProjection, ScaleOnlyAdaRMSNorm

__all__ = ["AdaRMSScaleProjection", "ScaleOnlyAdaRMSNorm"]
