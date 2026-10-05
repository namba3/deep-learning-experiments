import heapq
import math
import re
import torch
import torch.nn as nn

from .layers import BF16Linear, DtypeAwareRMSNorm


def convert_linear_to_bf16(module, skip_modules=()):
    """指定module内のLinearをBF16計算・FP32出力のLinearへ置換する。"""
    if any(module is skipped for skipped in skip_modules):
        return module
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, BF16Linear):
            replacement = BF16Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            replacement.load_state_dict(child.state_dict())
            setattr(module, name, replacement)
        else:
            convert_linear_to_bf16(child, skip_modules)
    return module


def convert_rmsnorm_to_dtype_aware(module):
    """Replace RMSNorm modules with explicit mixed-precision dtype handling."""
    for name, child in list(module.named_children()):
        if isinstance(child, DtypeAwareRMSNorm):
            continue
        if isinstance(child, nn.RMSNorm):
            replacement = DtypeAwareRMSNorm(
                child.normalized_shape,
                eps=child.eps,
                elementwise_affine=child.elementwise_affine,
                device=(child.weight.device if child.weight is not None else None),
                dtype=(child.weight.dtype if child.weight is not None else None),
            )
            replacement.load_state_dict(child.state_dict())
            setattr(module, name, replacement)
        else:
            convert_rmsnorm_to_dtype_aware(child)
    return module


def format_bytes(num_bytes):
    """バイト数を2進単位で表示する。"""
    value = float(num_bytes)
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    for unit in units:
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} {units[-1]}"


def unique_storage_bytes(tensors):
    """共有を重複計上せず、実際のstorage容量を求める。"""
    total_bytes = 0
    seen = set()
    for tensor in tensors:
        if not torch.is_tensor(tensor):
            continue
        storage = tensor.untyped_storage()
        key = (
            tensor.device.type,
            tensor.device.index,
            storage.data_ptr(),
            storage.nbytes(),
        )
        if key in seen:
            continue
        seen.add(key)
        total_bytes += storage.nbytes()
    return total_bytes


def unique_storage_bytes_by_dtype(tensors):
    """dtypeごとに重複を除いたstorage容量を返す。"""
    totals = {}
    seen = set()
    for tensor in tensors:
        if not torch.is_tensor(tensor):
            continue
        storage = tensor.untyped_storage()
        key = (
            tensor.device.type,
            tensor.device.index,
            storage.data_ptr(),
            storage.nbytes(),
        )
        if key in seen:
            continue
        seen.add(key)
        dtype_name = str(tensor.dtype).replace("torch.", "")
        totals[dtype_name] = totals.get(dtype_name, 0) + storage.nbytes()
    return totals


def _storage_key(tensor):
    storage = tensor.untyped_storage()
    return (
        tensor.device.type,
        tensor.device.index,
        storage.data_ptr(),
        storage.nbytes(),
    )


def bf16_parameter_storage_keys(model):
    """保存時にBF16へ変換するParameterのstorageキーを集める。"""
    keys = set()
    for module in model.modules():
        if not getattr(module, "_save_parameters_as_bf16", False):
            continue
        for parameter in module.parameters(recurse=False):
            keys.add(_storage_key(parameter))
    return keys


def compact_state_dict(model, cast_bf16=False):
    """共有パラメータを重複保存せず、必要ならBF16で保存する。"""
    compact = {}
    seen = set()
    bf16_keys = bf16_parameter_storage_keys(model) if cast_bf16 else set()
    for name, tensor in model.state_dict().items():
        if not torch.is_tensor(tensor):
            continue
        storage = tensor.untyped_storage()
        key = (
            tensor.device.type,
            tensor.device.index,
            storage.data_ptr(),
            tensor.storage_offset(),
            tensor.numel(),
            tuple(tensor.stride()),
            str(tensor.dtype),
        )
        if key in seen:
            continue
        seen.add(key)
        saved_tensor = tensor.detach().cpu().contiguous()
        if _storage_key(tensor) in bf16_keys:
            saved_tensor = saved_tensor.to(dtype=torch.bfloat16)
        compact[name] = saved_tensor
    return compact


def print_model_info(model, cast_bf16=False):
    """学習開始前にパラメータ数と保存サイズの概算を表示する。"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    unique_param_bytes_by_dtype = unique_storage_bytes_by_dtype(
        model.parameters()
    )
    unique_param_bytes = sum(unique_param_bytes_by_dtype.values())
    compact_state = compact_state_dict(model, cast_bf16=cast_bf16)
    compact_bytes_by_dtype = {}
    for tensor in compact_state.values():
        if not torch.is_tensor(tensor):
            continue
        dtype_name = str(tensor.dtype).replace("torch.", "")
        compact_bytes_by_dtype[dtype_name] = (
            compact_bytes_by_dtype.get(dtype_name, 0)
            + tensor.numel() * tensor.element_size()
        )
    compact_bytes = sum(compact_bytes_by_dtype.values())

    print("========== Model Information ==========")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Unique parameter storage: {format_bytes(unique_param_bytes)}")
    print(
        "Unique storage by dtype: "
        + ", ".join(
            f"{dtype}={format_bytes(size)}"
            for dtype, size in sorted(unique_param_bytes_by_dtype.items())
        )
    )
    print(
        "Estimated safetensors payload: "
        f"{format_bytes(compact_bytes)}"
    )
    print(
        "Estimated payload by dtype: "
        + ", ".join(
            f"{dtype}={format_bytes(size)}"
            for dtype, size in sorted(compact_bytes_by_dtype.items())
        )
    )
    print("=======================================")


def compile_regexes(patterns, default):
    """正規表現文字列のリストをコンパイルする。"""
    raw_patterns = default if patterns is None else patterns
    try:
        return tuple(re.compile(pattern, re.IGNORECASE) for pattern in raw_patterns)
    except re.error as exc:
        raise ValueError(f"Invalid parameter regex: {exc}") from exc


def collect_parameter_metadata(model):
    """Parameter idから所属Module・完全名・Module型名を取得できるmetadataを作る。"""
    metadata = {}
    for module_name, module in model.named_modules():
        module_type = type(module).__name__.lower()
        for param_name, parameter in module.named_parameters(recurse=False):
            full_name = (
                f"{module_name}.{param_name}" if module_name else param_name
            ).lower()
            metadata[id(parameter)] = (module, full_name, module_type)
    return metadata


def parameter_matches_regexes(parameter_id, metadata, regexes):
    _, parameter_name, module_type = metadata.get(
        parameter_id, (None, "", "")
    )
    return any(
        regex.search(parameter_name) or regex.search(module_type)
        for regex in regexes
    )


def build_parameter_groups(model, target_param_regexes, weight_decay):
    """optimizer生成前にdecay対象と対象外のgroupを作る。"""
    regexes = compile_regexes(target_param_regexes, [r"linear", r"conv2d"])
    metadata = collect_parameter_metadata(model)
    decay_params = []
    no_decay_params = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter_matches_regexes(id(parameter), metadata, regexes):
            decay_params.append(parameter)
        else:
            no_decay_params.append(parameter)
    groups = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups


class DepthDistributionScheduler:
    """学習stepに応じて可変depthのサンプリング分布を変化させる。"""
    def __init__(
        self,
        min_depth,
        max_depth,
        total_steps,
        initial_bias=1.0,
        final_bias=16.0,
        schedule="cosine",
    ):
        if not 1 <= min_depth <= max_depth:
            raise ValueError("min_depth must be in [1, max_depth]")
        if initial_bias <= 0.0 or final_bias <= 0.0:
            raise ValueError("depth bias must be > 0")
        if schedule not in ("linear", "cosine"):
            raise ValueError("schedule must be 'linear' or 'cosine'")

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.total_steps = max(1, total_steps)
        self.initial_bias = initial_bias
        self.final_bias = final_bias
        self.schedule = schedule
        self.step_count = 0
        self.depth_choices = torch.arange(
            min_depth, max_depth + 1, dtype=torch.long
        )

    @property
    def progress(self):
        return min(self.step_count / self.total_steps, 1.0)

    @property
    def bias(self):
        if self.schedule == "linear":
            interpolation = self.progress
        else:
            interpolation = 0.5 * (
                1.0 - math.cos(math.pi * self.progress)
            )
        return self.initial_bias + (
            self.final_bias - self.initial_bias
        ) * interpolation

    def probabilities(self, bias=None):
        # depth数に依存して分布が過度に急峻にならないよう、
        # depth位置を0～1へ正規化する。final_bias=4.0なら、
        # max depthの重みはmin depthの4倍になる。
        depth_position = (
            self.depth_choices - self.min_depth
        ).float() / max(1, self.max_depth - self.min_depth)
        if bias is None:
            bias = self.bias
        weights = bias ** depth_position
        return weights / weights.sum()

    def sample(self):
        index = torch.multinomial(self.probabilities(), num_samples=1)
        return int(self.depth_choices[index].item())

    def expected_depth(self, bias=None):
        """指定したbias、または現在のbiasにおける期待depthを返す。"""
        return float(
            (
                self.depth_choices.float()
                * self.probabilities(bias=bias)
            ).sum().item()
        )

    def step(self, count=1):
        self.step_count += count

class GradSignFlipNoiseInjector:
    def __init__(
        self,
        optimizer,
        initial_flip_prob=0.3,
        final_flip_prob=0.0,
        total_steps=1000,
        schedule="cosine",
        weight_func="uniform",   # "uniform", "exp", "inv2" or "inv"
        tau=1e-3,
        clamp_probs=True,
        skip_threshold=1e-3,
        target_param_regexes=None,   # 適用対象の正規表現文字列リスト
        model=None,
    ):
        self.optimizer = optimizer
        self.initial_flip_prob = initial_flip_prob
        self.final_flip_prob = final_flip_prob
        self.total_steps = max(1, total_steps)
        self.schedule = schedule
        self.weight_func = weight_func
        self.tau = tau
        self.clamp_probs = clamp_probs
        self.skip_threshold = skip_threshold
        self.target_param_regexes = compile_regexes(
            target_param_regexes, [r"linear"]
        )
        self._param_metadata = (
            collect_parameter_metadata(model) if model is not None else {}
        )
        self._step_count = 0

    def _current_flip_prob(self):
        # total_steps以降は完全無効化
        if self._step_count >= self.total_steps:
            return 0.0

        t = self._step_count / self.total_steps
        if self.schedule == "linear":
            return self.initial_flip_prob + (self.final_flip_prob - self.initial_flip_prob) * t
        elif self.schedule == "cosine":
            return self.final_flip_prob + (self.initial_flip_prob - self.final_flip_prob) * (
                0.5 * (1 + math.cos(math.pi * t))
            )
        elif self.schedule == "exp":
            return self.final_flip_prob + (self.initial_flip_prob - self.final_flip_prob) * math.exp(-5 * t)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def _compute_weights(self, grad):
        if self.weight_func == "uniform":
            return torch.ones_like(grad)
        g_abs = grad.abs()
        tau = max(self.tau, 1e-12)
        t = g_abs / tau
        if self.weight_func == "exp":
            w = torch.exp(-t)
        elif self.weight_func == "inv2":
            w = 1.0 / (1.0 + t * t)
        elif self.weight_func == "inv":
            w = 1.0 / (1.0 + t)
        else:
            raise ValueError(f"Unknown weight_func: {self.weight_func}")
        return w

    def inject(self):
        p_avg = self._current_flip_prob()
        if p_avg <= self.skip_threshold:
            self._step_count += 1
            return

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if not parameter_matches_regexes(
                    id(p), self._param_metadata, self.target_param_regexes
                ):
                    continue

                # 勾配の大きさに基づく重み付け
                w = self._compute_weights(p.grad)
                w_mean = w.mean().clamp_min(1e-12)

                lam = p_avg / w_mean
                p_flip = lam * w

                if self.clamp_probs:
                    p_flip = p_flip.clamp(0.0, 1.0)

                mask = (torch.rand_like(p.grad) < p_flip).float()
                p.grad.mul_(1 - 2 * mask)

        self._step_count += 1

class WeightDecayScheduler:
    def __init__(
        self,
        optimizer,
        initial_weight_decay=1e-1,
        final_weight_decay=1e-4,
        total_steps=1000,
        schedule="linear",
        target_param_regexes=None,
        model=None,
    ):
        """
        optimizer: 元のtorch.optim.Optimizerインスタンス
        total_steps: 減衰スケジュールの全ステップ数
        schedule: "linear", "cosine", "exp" から選択
        """
        self.optimizer = optimizer
        self.initial_weight_decay = initial_weight_decay
        self.final_weight_decay = final_weight_decay
        self.total_steps = max(1, total_steps)
        self.schedule = schedule
        self._step_count = 0
        self.target_param_regexes = compile_regexes(
            target_param_regexes, [r"linear", r"conv2d"]
        )
        self._param_metadata = (
            collect_parameter_metadata(model) if model is not None else {}
        )
        # parameter groupはoptimizer生成前に分割済みであることを前提にする。
        self._decay_groups = [
            group for group in optimizer.param_groups
            if any(
                parameter_matches_regexes(
                    id(parameter), self._param_metadata,
                    self.target_param_regexes,
                )
                for parameter in group["params"]
            )
        ]

        self.step()

    def _current_weight_decay(self):
        t = min(self._step_count / self.total_steps, 1.0)
        if self.schedule == "linear":
            return self.initial_weight_decay + (self.final_weight_decay - self.initial_weight_decay) * t
        elif self.schedule == "cosine":
            return self.final_weight_decay + (self.initial_weight_decay - self.final_weight_decay) * (0.5 * (1 + math.cos(math.pi * t)))
        elif self.schedule == "exp":
            return self.final_weight_decay + (self.initial_weight_decay - self.final_weight_decay) * math.exp(-5 * t)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def step(self):
        weight_decay = self._current_weight_decay()
        for group in self._decay_groups:
            group["weight_decay"] = weight_decay
        self._step_count += 1

class ModelSnapshot:
    """
    モデルのスナップショットを保存し、最良モデルを追跡するためのユーティリティクラス
    内部でモデルの重みのコピーとその評価値、エポック数を保持する
    最大n個のスナップショットを最小ヒープで管理し、それらの平均重みを計算する機能も提供する
    評価値は高いほど良いとする
    """
    def __init__(self, max_snapshots=5):
        self.max_snapshots = max_snapshots
        self.snapshots = []  # (評価値, epoch, state_dict)のタプルのリスト

    def add_snapshot(self, model, score, epoch):
        state_dict_copy = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        snapshot = (score, epoch, state_dict_copy)
        if len(self.snapshots) < self.max_snapshots:
            heapq.heappush(self.snapshots, snapshot)
        else:
            heapq.heappushpop(self.snapshots, snapshot)

    def get_best_model(self):
        if not self.snapshots:
            return None
        best_snapshot = max(self.snapshots, key=lambda x: x[0])
        return best_snapshot[2]

    def get_average_model(self):
        if not self.snapshots:
            return None
        avg_state_dict = {}
        for key in self.snapshots[0][2].keys():
            avg_state_dict[key] = torch.mean(torch.stack([snap[2][key] for snap in self.snapshots]), dim=0)
        return avg_state_dict
