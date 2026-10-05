import torch
import torch.nn as nn
import torch.nn.functional as F
from ..kernels import gated_ffn, gated_silu



class GatedLinear(nn.Module):
    """
    Gate付き全結合層
    - GateにはSiLUを使用
    """
    def __init__(self, dim_in, dim_out, bias=False, backend="torch"):
        super().__init__()
        if backend not in {"auto", "torch", "naive", "triton"}:
            raise ValueError(f"unknown gated FFN backend: {backend}")
        self.backend = backend
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def forward(self, x):
        x1, x2 = self.proj(x).chunk(2, dim=-1)
        return gated_silu(x1, x2, backend=self.backend)

class GatedFFN(nn.Module):
    """A reusable gated FFN with an optional fused activation backend."""
    def __init__(self, dim_in, dim_hidden, dim_out=None, bias=False, backend="torch"):
        super().__init__()
        if dim_out is None:
            dim_out = dim_in
        if backend not in {"auto", "torch", "naive", "triton"}:
            raise ValueError(f"unknown gated FFN backend: {backend}")
        self.backend = backend
        self.gated = GatedLinear(
            dim_in, dim_hidden, bias=bias, backend="torch",
        )
        self.output = nn.Linear(dim_hidden, dim_out, bias=bias)

    def forward(self, x):
        if self.backend in {"auto", "triton"}:
            return gated_ffn(
                x,
                self.gated.proj.weight,
                self.gated.proj.bias,
                self.output.weight,
                self.output.bias,
                backend=self.backend,
            )
        return self.output(self.gated(x))

class BF16Linear(nn.Linear):
    """Linearの積だけBF16で計算し、出力をFP32に戻す。"""
    _save_parameters_as_bf16 = True

    def forward(self, x):
        weight = self.weight.to(dtype=torch.bfloat16)
        bias = self.bias.to(dtype=torch.bfloat16) if self.bias is not None else None
        return F.linear(x.to(dtype=torch.bfloat16), weight, bias).float()

class InterpolatedSuperLinear(nn.Module):
    """
    深さに応じて実効重みを生成する Linear。

    ``weight_bank`` は (num_steps, out_features, in_features) の
    スーパー行列です。depth_step=0.0～1.0 に応じて隣接する2枚の
    行列を線形補間し、forward 時に実際の Linear の重みとして使います。
    """
    def __init__(self, in_features, out_features, num_steps=4, bias=False):
        super().__init__()
        if num_steps < 2:
            raise ValueError("num_steps must be >= 2")
        self.in_features = in_features
        self.out_features = out_features
        self.num_steps = num_steps
        self.weight_bank = nn.Parameter(
            torch.empty(num_steps, out_features, in_features)
        )
        self.bias_bank = (
            nn.Parameter(torch.empty(num_steps, out_features))
            if bias else None
        )
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weight_bank:
            nn.init.xavier_uniform_(weight)
        if self.bias_bank is not None:
            nn.init.zeros_(self.bias_bank)

    def effective_parameters(self, depth_step):
        """depth_step から補間済みの (weight, bias) を返す。"""
        if not torch.is_tensor(depth_step):
            depth_step = self.weight_bank.new_tensor(depth_step)
        depth_step = depth_step.to(
            device=self.weight_bank.device, dtype=self.weight_bank.dtype
        ).clamp(0.0, 1.0)

        position = depth_step * (self.num_steps - 1)
        left = position.floor().long().clamp(max=self.num_steps - 1)
        right = (left + 1).clamp(max=self.num_steps - 1)
        fraction = position - left.to(position.dtype)

        weight = torch.lerp(
            self.weight_bank[left], self.weight_bank[right], fraction
        )
        if self.bias_bank is None:
            bias = None
        else:
            bias = torch.lerp(
                self.bias_bank[left], self.bias_bank[right], fraction
            )
        return weight, bias

    def forward(self, x, depth_step):
        weight, bias = self.effective_parameters(depth_step)
        return F.linear(x, weight, bias)

class InterpolatedGatedLinear(nn.Module):
    """DepthSuperLinear を使った GatedLinear。"""
    def __init__(self, dim_in, dim_out, num_steps=4, bias=False):
        super().__init__()
        self.proj = InterpolatedSuperLinear(
            dim_in, dim_out * 2, num_steps=num_steps, bias=bias
        )

    def forward(self, x, depth_step):
        x1, x2 = self.proj(x, depth_step).chunk(2, dim=-1)
        return F.silu(x1) * x2
