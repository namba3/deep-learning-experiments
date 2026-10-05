import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class PatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        downsample_steps=1,
        hidden_channels=None,
    ):
        super().__init__()
        if downsample_steps < 0:
            raise ValueError("downsample_steps must be non-negative")
        if hidden_channels is None:
            hidden_channels = out_channels
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        layers = [
            GatedConv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(math.gcd(32, hidden_channels), hidden_channels),
        ]
        for step in range(downsample_steps):
            input_channels = hidden_channels if step == 0 else out_channels
            layers.extend((
                GatedConv2d(
                    input_channels, out_channels,
                    kernel_size=3, stride=2, padding=1,
                ),
                nn.GroupNorm(math.gcd(32, out_channels), out_channels),
            ))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class RMSNorm2d(nn.Module):
    """Apply channel-wise RMSNorm to an NCHW tensor.

    The previous implementation inherited from ``nn.LayerNorm`` while using
    only the nested ``nn.RMSNorm`` module.  That created unused ``weight`` and
    ``bias`` parameters in every instance.  Keep the nested module name so
    existing checkpoints remain compatible.
    """
    def __init__(self, num_channels):
        super().__init__()
        self.rmsnorm = nn.RMSNorm(num_channels)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Older checkpoints contain unused LayerNorm keys.  They are safe to
        # discard because the old forward path never read them.
        state_dict.pop(prefix + "weight", None)
        state_dict.pop(prefix + "bias", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.rmsnorm(x)
        x = x.permute(0, 3, 1, 2)
        return x

class GatedConv2d(nn.Module):
    """2倍のチャネルをSiLUゲートで分割・乗算する畳み込み。"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels,
            out_channels * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        value, gate = self.projection(x).chunk(2, dim=1)
        return F.silu(value) * gate

class GatedConvTranspose2d(nn.Module):
    """2倍のチャネルをSiLUゲートで分割・乗算する転置畳み込み。"""
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.projection = nn.ConvTranspose2d(
            in_channels,
            out_channels * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        value, gate = self.projection(x).chunk(2, dim=1)
        return F.silu(value) * gate

def _default_group_norm(channels):
    if channels <= 0:
        raise ValueError("GroupNorm channels must be positive")
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)

class PreNormConvFFNResidual2d(nn.Module):
    """Pre-norm convolutional FFN residual block for NCHW activations.

    ``state_layout`` selects the historic child names used by callers that
    must retain existing checkpoint keys: ``"named"`` stores ``norm``,
    ``conv1``, ``activation``, and ``conv2``; ``"sequential"`` stores the
    equivalent modules under ``block`` in that order.
    """
    def __init__(
        self,
        channels: int,
        hidden_channels: int | None = None,
        *,
        zero_last: bool = False,
        state_layout: str = "named",
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if hidden_channels is None:
            hidden_channels = channels * 2
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if state_layout not in {"named", "sequential"}:
            raise ValueError("state_layout must be 'named' or 'sequential'")
        norm = RMSNorm2d(channels)
        conv1 = nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1)
        activation = nn.SiLU()
        conv2 = nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1)
        if zero_last:
            nn.init.zeros_(conv2.weight)
            nn.init.zeros_(conv2.bias)
        self.state_layout = state_layout
        if state_layout == "named":
            self.norm = norm
            self.conv1 = conv1
            self.activation = activation
            self.conv2 = conv2
        else:
            self.block = nn.Sequential(norm, conv1, activation, conv2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Full-object checkpoints created before this shared class had no
        # ``state_layout`` attribute. Infer their registered child layout.
        state_layout = getattr(
            self, "state_layout", "sequential" if hasattr(self, "block") else "named",
        )
        if state_layout == "named":
            branch = self.conv2(self.activation(self.conv1(self.norm(x))))
        else:
            branch = self.block(x)
        return x + branch


class PreNormGatedConvFFNResidual2d(nn.Module):
    """Pre-norm gated-convolution FFN residual block for NCHW activations."""
    def __init__(
        self,
        channels: int,
        hidden_channels: int | None = None,
        *,
        state_layout: str = "named",
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if hidden_channels is None:
            hidden_channels = channels * 2
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if state_layout not in {"named", "sequential"}:
            raise ValueError("state_layout must be 'named' or 'sequential'")
        norm = RMSNorm2d(channels)
        conv1 = GatedConv2d(channels, hidden_channels, kernel_size=3, padding=1)
        conv2 = GatedConv2d(hidden_channels, channels, kernel_size=3, padding=1)
        self.state_layout = state_layout
        if state_layout == "named":
            self.norm = norm
            self.conv1 = conv1
            self.conv2 = conv2
        else:
            self.block = nn.Sequential(norm, conv1, conv2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state_layout = getattr(
            self, "state_layout", "sequential" if hasattr(self, "block") else "named",
        )
        if state_layout == "named":
            branch = self.conv2(self.conv1(self.norm(x)))
        else:
            branch = self.block(x)
        return x + branch


class GatedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_factory=None):
        super().__init__()
        if norm_factory is None:
            norm_factory = _default_group_norm
        self.block = nn.Sequential(
            GatedConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            norm_factory(out_channels),
            GatedConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            norm_factory(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else GatedConv2d(in_channels, out_channels, kernel_size=1, padding=0)
        )

    def forward(self, x):
        return self.block(x) + self.skip(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_factory=None):
        super().__init__()
        if norm_factory is None:
            norm_factory = _default_group_norm
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            norm_factory(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            norm_factory(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(self.block(x) + self.skip(x))

class ConvNeXtBlock(nn.Module):
    def __init__(self, channels, mlp_ratio=4, layer_scale_init_value=1e-6,kernel_size=7, padding=3):
        super().__init__()
        self.dwconv = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding, groups=channels,
        )
        self.norm = nn.LayerNorm(channels)
        self.pwconv1 = nn.Linear(channels, mlp_ratio * channels)
        self.activation = nn.GELU()
        self.pwconv2 = nn.Linear(mlp_ratio * channels, channels)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(channels)
        )

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.activation(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + x

class EfficientDownsample(nn.Module):
    """空間情報をチャネルへ再配置してから学習的に圧縮する。"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=1),
            nn.GroupNorm(min(32, out_channels), out_channels),
        )
        self.skip = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(self.main(x) + self.skip(x))

class EfficientUpsample(nn.Module):
    """チャネルを空間へ再配置して解像度を復元する。"""
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1),
            nn.PixelShuffle(2),
        ]
        if normalize:
            layers.append(nn.GroupNorm(min(32, out_channels), out_channels))
        self.main = nn.Sequential(
            *layers,
        )
        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(self.main(x) + self.skip(x))
