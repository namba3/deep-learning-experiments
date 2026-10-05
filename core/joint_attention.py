"""Backward-compatible import path for reusable joint attention."""

from .layers.joint_attention import JointKVSDPAAttention

__all__ = ["JointKVSDPAAttention"]
