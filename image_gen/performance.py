"""Image-generation memory estimates and opt-in performance telemetry."""

from collections import Counter
from contextlib import contextmanager, nullcontext
import json
import time
import warnings

import torch

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import pynvml
except Exception:  # NVML is optional; CUDA memory metrics still work.
    pynvml = None

def tensor_storage_bytes(value):
    if not torch.is_tensor(value):
        return 0
    return value.numel() * value.element_size()

def module_storage_bytes(module):
    parameter_bytes = sum(
        tensor_storage_bytes(parameter)
        for parameter in module.parameters()
    )
    buffer_bytes = sum(
        tensor_storage_bytes(buffer)
        for buffer in module.buffers()
    )
    return parameter_bytes + buffer_bytes

def module_parameter_dtype_summary(*modules):
    counts = Counter()
    for module in modules:
        for parameter in module.parameters():
            counts[str(parameter.dtype).replace("torch.", "")] += parameter.numel()
    return " ".join(
        f"{dtype}={count:,}"
        for dtype, count in sorted(counts.items())
    ) or "none"

def cast_trainable_modules_dtype(modules, dtype):
    """Store all trainable model parameters and buffers in one dtype."""
    for module in modules:
        module.to(dtype=dtype)

class OptimizerBundle:
    """Coordinate multiple optimizers over disjoint parameter sets."""

    def __init__(self, optimizers):
        self.optimizers = dict(optimizers)
        if not self.optimizers:
            raise ValueError("OptimizerBundle requires at least one optimizer")

    @property
    def param_groups(self):
        return [
            group
            for optimizer in self.optimizers.values()
            for group in optimizer.param_groups
        ]

    @property
    def state(self):
        merged = {}
        for optimizer in self.optimizers.values():
            merged.update(optimizer.state)
        return merged

    def train(self):
        for optimizer in self.optimizers.values():
            if hasattr(optimizer, "train"):
                optimizer.train()

    def eval(self):
        for optimizer in self.optimizers.values():
            if hasattr(optimizer, "eval"):
                optimizer.eval()

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, timing=None):
        for name, optimizer in self.optimizers.items():
            if timing is None:
                optimizer.step()
            else:
                with timing.measure(f"optimizer[{name}]"):
                    if hasattr(optimizer, "step_with_performance"):
                        optimizer.step_with_performance(timing)
                    else:
                        optimizer.step()

    def state_dict(self):
        return {
            "optimizers": {
                name: optimizer.state_dict()
                for name, optimizer in self.optimizers.items()
            }
        }

    def load_state_dict(self, state):
        saved = state.get("optimizers", state)
        if set(saved) != set(self.optimizers):
            raise ValueError(
                "Optimizer bundle mismatch: "
                f"current={sorted(self.optimizers)}, saved={sorted(saved)}"
            )
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(saved[name])

def iter_optimizers(optimizer):
    if isinstance(optimizer, OptimizerBundle):
        return optimizer.optimizers.values()
    return (optimizer,)

def optimizer_step_with_timing(optimizer, timing):
    """Run optimizer step and expose per-optimizer timing when enabled."""
    with timing.measure("optimizer_step"):
        if isinstance(optimizer, OptimizerBundle):
            optimizer.step(timing=timing)
        else:
            with timing.measure(f"optimizer[{optimizer.__class__.__name__}]"):
                if hasattr(optimizer, "step_with_performance"):
                    optimizer.step_with_performance(timing)
                else:
                    optimizer.step()

def unique_optimizer_parameters(optimizer):
    seen = set()
    for optimizer_instance in iter_optimizers(optimizer):
        for group in optimizer_instance.param_groups:
            for parameter in group["params"]:
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    yield parameter

def estimate_optimizer_state_bytes(optimizer):
    """Estimate fully materialized optimizer-state storage.

    CAME keeps a full first moment plus factored row/column statistics for
    matrices and full second moments for vector/scalar parameters.
    """
    total = 0
    seen = set()
    for optimizer_instance in iter_optimizers(optimizer):
        optimizer_name = type(optimizer_instance).__name__
        for parameter in unique_optimizer_parameters(optimizer_instance):
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            numel = parameter.numel()
            parameter_estimator = getattr(
                optimizer_instance,
                "estimate_parameter_state_bytes",
                None,
            )
            if parameter_estimator is not None:
                total += int(parameter_estimator(parameter))
                continue
            if optimizer_name in {"CAME", "CAMEAutoSchedule"}:
                # CAME casts fp16/bf16 gradients to fp32 before allocating state.
                state_bytes = numel * 4
                if parameter.ndim >= 2:
                    row_numel = numel // parameter.shape[-1]
                    col_numel = numel // parameter.shape[-2]
                    state_bytes += 2 * (row_numel + col_numel) * 4
                else:
                    state_bytes += numel * 4
                total += state_bytes
            elif optimizer_name in {
                "RAdamScheduleFree", "AdamWScheduleFree", "AdamW",
                "AdamWFP32State", "AdamWAutoSchedule",
            }:
                # Schedule-Free: FP32 z + exp_avg_sq. AdamW: FP32 exp_avg
                # + exp_avg_sq. The parameter itself may be stored in BF16.
                total += 2 * numel * 4
            elif optimizer_name in {"Muon", "SingleDeviceMuon"}:
                # Muon keeps one FP32 momentum matrix per parameter.
                total += numel * 4
            elif optimizer_name in {"NorMuon", "AdaMuon"}:
                # Matrix momentum plus row-wise or element-wise FP32 statistics.
                if parameter.ndim >= 2:
                    total += numel * (2 if optimizer_name == "AdaMuon" else 1) * 4
                    if optimizer_name == "NorMuon":
                        total += parameter.shape[0] * 4
                else:
                    total += 2 * numel * 4
            elif optimizer_name == "SOAP":
                # Adam moments plus two Shampoo covariance matrices.
                if parameter.ndim >= 2:
                    rows = parameter.shape[0]
                    cols = parameter.numel() // rows
                    # exp_avg, exp_avg_sq, covariance matrices, and bases.
                    total += (2 * parameter.numel() + 2 * (rows * rows + cols * cols)) * 4
                else:
                    total += 2 * numel * 4
            elif optimizer_name == "Lion":
                total += numel * 4
            elif optimizer_name in {
                "APOLLO", "APOLLOMini", "APOLLOCAME",
                "APOLLOADAMW", "APOLLOAutoSchedule",
                "APOLLOADAMWAutoSchedule", "APOLLOLion", "RotAPOLLO",
                "DualRotAPOLLO",
                "APOLLOCAMEAutoSchedule",
            }:
                # APOLLO stores low-rank fp32 moments plus the random
                # projection.  APOLLO-CAME adds factored second/confidence
                # statistics in the same low-rank space.
                if parameter.ndim >= 2:
                    rank = 1
                    for group in optimizer_instance.param_groups:
                        if any(parameter is candidate for candidate in group["params"]):
                            rank = min(int(group.get("rank", 1)), min(parameter.shape))
                            break
                    rows = parameter.shape[0]
                    cols = parameter.numel() // rows
                    low_rank_elements = rank * max(rows, cols)
                    projection_elements = rank * min(rows, cols)
                    if optimizer_name in {"APOLLOCAME", "APOLLOCAMEAutoSchedule"}:
                        total += low_rank_elements * 4
                        total += 2 * (rank + max(rows, cols)) * 4
                    elif optimizer_name == "APOLLOLion":
                        total += low_rank_elements * 4
                    else:
                        total += low_rank_elements * 2 * 4
                    total += projection_elements * parameter.element_size()
                else:
                    # Account for the configured 1D fallback. AdamW-SF stores
                    # two parameter-dtype tensors; the legacy CAME-like path
                    # stores two FP32 tensors; SGD stores no optimizer state.
                    fallback = "adamw-sf"
                    for group in optimizer_instance.param_groups:
                        if any(parameter is candidate for candidate in group["params"]):
                            fallback = group.get("fallback", "adamw-sf")
                            break
                    if fallback == "came":
                        total += 2 * numel * 4
                    elif fallback == "adamw-sf":
                        total += 2 * numel * parameter.element_size()
    return total

def materialized_optimizer_state_bytes(optimizer):
    return sum(
        tensor_storage_bytes(value)
        for optimizer_instance in iter_optimizers(optimizer)
        for state in optimizer_instance.state.values()
        for value in state.values()
    )

def materialized_optimizer_state_dtype_summary(optimizer):
    counts = Counter()
    for optimizer_instance in iter_optimizers(optimizer):
        for state in optimizer_instance.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    counts[str(value.dtype).replace("torch.", "")] += value.numel()
    return " ".join(
        f"{dtype}={count:,}"
        for dtype, count in sorted(counts.items())
    ) or "none"

def format_memory_bytes(value):
    if value < 1024 ** 2:
        return f"{value / 1024:.1f} KiB"
    return f"{value / 1024 ** 2:.2f} MiB"

def report_runtime_cuda_memory(device, optimizer=None, label="runtime"):
    if device.type != "cuda":
        print(f"VRAM report ({label}): CUDA unavailable")
        return
    torch.cuda.synchronize(device)
    total = torch.cuda.get_device_properties(device).total_memory
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    print(f"VRAM {label}:")
    print(
        f"  allocated={format_memory_bytes(allocated)} ({allocated / total:.1%}) "
        f"reserved={format_memory_bytes(reserved)} ({reserved / total:.1%})"
    )
    print(
        f"  peak_allocated={format_memory_bytes(peak_allocated)} ({peak_allocated / total:.1%}) "
        f"peak_reserved={format_memory_bytes(peak_reserved)} ({peak_reserved / total:.1%}) "
        f"total={format_memory_bytes(total)}"
    )
    if optimizer is not None:
        print(
            f"  materialized_optimizer_state="
            f"{format_memory_bytes(materialized_optimizer_state_bytes(optimizer))}"
        )
        print(
            f"  optimizer_state_dtypes="
            f"{materialized_optimizer_state_dtype_summary(optimizer)}"
        )

def report_memory_estimate(device, dit, text_adapter, vae, text_encoder, optimizer):
    """Print persistent VRAM estimates and their composition ratios."""
    components = [
        ("DiT + adapter weights", module_storage_bytes(dit) + module_storage_bytes(text_adapter)),
        ("VAE weights/buffers", module_storage_bytes(vae)),
        ("Text encoder weights/buffers", module_storage_bytes(text_encoder)),
        ("Trainable gradients", sum(tensor_storage_bytes(parameter) for parameter in unique_optimizer_parameters(optimizer))),
        ("Optimizer state", estimate_optimizer_state_bytes(optimizer)),
    ]
    total = sum(value for _, value in components)
    print("VRAM persistent estimate (activations/workspaces excluded):")
    for name, value in components:
        ratio = value / total if total else 0.0
        print(f"  {name:<30} {format_memory_bytes(value):>12} ({ratio:>6.1%})")
    print(f"  {'estimated persistent total':<30} {format_memory_bytes(total):>12}")
    print(
        "  AMP persistent master weights: 0 B "
        "(autocast does not create a separate parameter copy)"
    )
    report_runtime_cuda_memory(device, optimizer, label="startup")

def query_gpu_telemetry(device):
    """Return best-effort NVML telemetry for one CUDA device."""
    if device.type != "cuda" or pynvml is None:
        return None
    try:
        if not getattr(query_gpu_telemetry, "_initialized", False):
            pynvml.nvmlInit()
            query_gpu_telemetry._initialized = True
        index = torch.cuda.current_device() if device.index is None else device.index
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        telemetry = {
            "gpu_utilization_percent": float(utilization.gpu),
            "memory_utilization_percent": float(utilization.memory),
            "temperature_c": float(
                pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU,
                )
            ),
            "sm_clock_mhz": float(
                pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            ),
            "memory_clock_mhz": float(
                pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            ),
            "power_w": float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0,
            "power_limit_w": float(
                pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            ) / 1000.0,
        }
        return telemetry
    except Exception:
        return None

def summarize_gpu_telemetry(samples):
    if not samples:
        return None
    summary = {"samples": len(samples)}
    for name in samples[0]:
        values = [sample[name] for sample in samples if name in sample]
        if not values:
            continue
        summary[name] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    return summary

class PerformanceAccumulator:
    """Accumulate timing and CUDA-memory metrics per optimizer interval."""
    def __init__(self, device, enabled=False, optimizer_breakdown=False):
        self.device = device
        self.enabled = enabled
        self.optimizer_breakdown = bool(optimizer_breakdown)
        self.use_cuda_events = enabled and device.type == "cuda"
        self.use_cuda_memory = self.use_cuda_events
        self.use_gpu_telemetry = self.use_cuda_memory and pynvml is not None
        self.gpu_telemetry_interval = 10
        self.gpu_telemetry_samples = []
        self.pending = {}
        self.cpu_totals = {}
        self.metric_totals = {}
        self.steps = 0
        self.reset_memory_stats()

    def reset_memory_stats(self):
        """Start a new CUDA-memory measurement interval."""
        if self.use_cuda_memory:
            torch.cuda.reset_peak_memory_stats(self.device)

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        if self.use_cuda_events:
            cpu_started = time.perf_counter()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self.pending.setdefault(name, []).append((start, end))
                self.cpu_totals[name] = (
                    self.cpu_totals.get(name, 0.0)
                    + time.perf_counter() - cpu_started
                )
        else:
            started = time.perf_counter()
            yield
            self.cpu_totals[name] = self.cpu_totals.get(name, 0.0) + time.perf_counter() - started

    @contextmanager
    def measure_host(self, name):
        """Measure a CPU-side span without creating CUDA events.

        This is intended for DataLoader waits and enqueue-side operations. On
        CUDA runs these values are reported under
        ``host_seconds_per_optimizer_step`` because the corresponding GPU
        work may execute asynchronously.
        """
        if not self.enabled:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            self.cpu_totals[name] = (
                self.cpu_totals.get(name, 0.0)
                + time.perf_counter() - started
            )

    def measure_optimizer(self, name):
        """Measure an optimizer sub-stage only when explicitly requested."""
        if not self.enabled or not self.optimizer_breakdown:
            return nullcontext()
        return self.measure(name)

    def begin_backward_measure(self, name):
        """Begin a backward hook span without affecting disabled training."""
        if not self.enabled:
            return None
        if self.use_cuda_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            return start, end, time.perf_counter()
        return time.perf_counter()

    def end_backward_measure(self, name, token):
        """Finish a backward hook span and add it to the current interval."""
        if token is None:
            return
        if self.use_cuda_events:
            start, end, host_started = token
            end.record()
            self.pending.setdefault(name, []).append((start, end))
            self.cpu_totals[name] = (
                self.cpu_totals.get(name, 0.0)
                + time.perf_counter() - host_started
            )
        else:
            self.cpu_totals[name] = (
                self.cpu_totals.get(name, 0.0)
                + time.perf_counter() - token
            )

    def step_completed(self):
        if self.enabled:
            self.steps += 1
            if (
                self.use_gpu_telemetry
                and (
                    self.steps == 1
                    or self.steps % self.gpu_telemetry_interval == 0
                )
            ):
                sample = query_gpu_telemetry(self.device)
                if sample is not None:
                    self.gpu_telemetry_samples.append(sample)

    def add_metric(self, name, value=1):
        """Accumulate a non-time performance metric for the interval."""
        if self.enabled:
            self.metric_totals[name] = self.metric_totals.get(name, 0) + value

    def report_and_reset(self):
        if not self.enabled or self.steps <= 0:
            return {}
        if self.use_cuda_events:
            torch.cuda.synchronize()
        if self.use_cuda_events:
            totals = {}
            for name, events in self.pending.items():
                totals[name] = sum(
                    start.elapsed_time(end) / 1000.0
                    for start, end in events
                )
        else:
            totals = dict(self.cpu_totals)
        averages = {name: value / self.steps for name, value in totals.items()}
        if self.use_cuda_events:
            averages["host_seconds_per_optimizer_step"] = {
                name: value / self.steps
                for name, value in self.cpu_totals.items()
            }
        averages["optimizer_steps"] = self.steps
        if self.metric_totals:
            averages["metrics"] = {
                name: value / self.steps
                for name, value in self.metric_totals.items()
            }
        if self.use_cuda_memory:
            properties = torch.cuda.get_device_properties(self.device)
            total_memory = int(properties.total_memory)
            peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
            peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
            averages["cuda_memory"] = {
                "current_allocated_bytes": int(
                    torch.cuda.memory_allocated(self.device)
                ),
                "current_reserved_bytes": int(
                    torch.cuda.memory_reserved(self.device)
                ),
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "total_bytes": total_memory,
                "peak_allocated_ratio": peak_allocated / max(total_memory, 1),
                "peak_reserved_ratio": peak_reserved / max(total_memory, 1),
            }
            sample = query_gpu_telemetry(self.device)
            if sample is not None:
                self.gpu_telemetry_samples.append(sample)
            gpu_telemetry = summarize_gpu_telemetry(
                self.gpu_telemetry_samples,
            )
            if gpu_telemetry is not None:
                averages["gpu_telemetry"] = gpu_telemetry
            # Do not include the next observation interval in this report.
            self.reset_memory_stats()
        self.pending.clear()
        self.cpu_totals.clear()
        self.metric_totals.clear()
        self.gpu_telemetry_samples.clear()
        self.steps = 0
        return averages

def install_backward_timing_hooks(model, performance):
    """Install optional backward spans for the main MMDiT block components.

    Full backward hooks add overhead and can interact with compiled graphs, so
    this is only called for the explicit, eager ``--perf-backward-breakdown``
    path.  The reported
    component spans overlap: a block contains its attention and FFN spans.
    """
    if not performance.enabled:
        return []
    handles = []
    targets = []
    for block in getattr(model, "blocks", ()):
        targets.append(("dit_block_backward", block))
        attention = block.joint_attn
        targets.append(("dit_attention_backward", attention))
        for name in ("latent_qkv", "image_qkv", "text_qkv"):
            projection = getattr(attention, name, None)
            if projection is not None:
                targets.append(("dit_attention_qkv_backward", projection))
        for name in (
            "latent_q_norm", "latent_k_norm",
            "image_q_norm", "image_k_norm",
            "text_q_norm", "text_k_norm",
        ):
            norm = getattr(attention, name, None)
            if norm is not None:
                targets.append(("dit_attention_norm_backward", norm))
        for name in (
            "latent_head_gate", "image_head_gate", "text_head_gate",
        ):
            gate = getattr(attention, name, None)
            if gate is not None:
                targets.append(("dit_attention_gate_backward", gate))
        for name in ("latent_out", "image_out", "text_out"):
            output = getattr(attention, name, None)
            if output is not None:
                targets.append(("dit_attention_output_backward", output))
        for ffn in (block.latent_ffn, block.image_ffn, block.text_ffn):
            targets.append(("dit_ffn_backward", ffn))

    for name, module in targets:
        tokens = []

        def pre_hook(_module, _grad_output, *, _name=name, _tokens=tokens):
            _tokens.append(performance.begin_backward_measure(_name))

        def post_hook(
            _module, _grad_input, _grad_output,
            *, _name=name, _tokens=tokens,
        ):
            if _tokens:
                performance.end_backward_measure(_name, _tokens.pop())

        handles.append(module.register_full_backward_pre_hook(pre_hook))
        handles.append(module.register_full_backward_hook(post_hook))
    return handles

PERFORMANCE_SUMMARY_KEYS = (
    "backward",
    "dit_forward_and_loss",
    "optimizer_step",
    "text_adapter",
    "text_encoder",
    "vae_encode",
)

def format_performance_summary(report):
    """Format only coarse timing fields for the standard output log."""
    return " ".join(
        f"{name}={report[name]:.4f}"
        for name in PERFORMANCE_SUMMARY_KEYS
        if name in report
    )

TIMING_SUMMARY_KEYS = PERFORMANCE_SUMMARY_KEYS

format_timing_summary = format_performance_summary

def write_performance_jsonl(path, report, *, epoch, global_step):
    """Append one detailed performance report to the run's JSONL artifact."""
    if not report:
        return
    record = {
        "epoch": int(epoch) + 1,
        "global_step": int(global_step),
        "optimizer_steps": int(report.get("optimizer_steps", 0)),
        "seconds_per_optimizer_step": {
            name: value
            for name, value in report.items()
            if name not in {
                "optimizer_steps", "cuda_memory", "gpu_telemetry", "metrics",
                "host_seconds_per_optimizer_step",
            }
        },
    }
    if "cuda_memory" in report:
        record["cuda_memory"] = report["cuda_memory"]
    if "gpu_telemetry" in report:
        record["gpu_telemetry"] = report["gpu_telemetry"]
    if "host_seconds_per_optimizer_step" in report:
        record["host_seconds_per_optimizer_step"] = report[
            "host_seconds_per_optimizer_step"
        ]
    if "metrics" in report:
        record["metrics"] = report["metrics"]
    with open(path, "a", encoding="utf-8") as file:
        json.dump(record, file, sort_keys=True)
        file.write("\n")

write_timing_jsonl = write_performance_jsonl
