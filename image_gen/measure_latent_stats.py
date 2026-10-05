"""Measure Qwen Image VAE latent statistics used by image_gen.

Statistics are accumulated per latent channel over image samples.  The default
sample count is 10,000 images, and CUDA can automatically search for a VAE
batch size while respecting a VRAM target.
"""

import argparse
import gc
import json
import math
import os
import sys
from collections import Counter
from contextlib import nullcontext
from statistics import median
from time import perf_counter

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runtime.progress import RichProgress  # noqa: E402

if __package__:
    from .train import (  # noqa: E402
        AspectRatioBatchSampler,
        HFDataset,
        RecordsDataset,
        collate,
    )
    from .cli import DEFAULT_DATASET_NAME
    from .inference import encode_images, resolve_vae_dtype, resolve_vae_latent_scale
else:
    from train import (  # noqa: E402
        AspectRatioBatchSampler,
        HFDataset,
        RecordsDataset,
        collate,
    )
    from cli import DEFAULT_DATASET_NAME
    from inference import encode_images, resolve_vae_dtype, resolve_vae_latent_scale


def collect_memory():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--records", default=None, help="Local JSONL/CSV records file")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--vae-model", required=True)
    parser.add_argument(
        "--vae-dtype", choices=["bf16", "fp32"], default="bf16",
        help="VAE parameter dtype. Default: bf16; use fp32 for compatibility.",
    )
    parser.add_argument("--latent-scale", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--bucket-step", type=int, default=32)
    parser.add_argument("--vae-batch-size", type=int, default=16,
                        help="VAE batch size when automatic search is disabled. Default: 16.")
    parser.add_argument("--batch-size", dest="vae_batch_size", type=int,
                        default=argparse.SUPPRESS,
                        help="Alias for --vae-batch-size.")
    parser.add_argument(
        "--auto-batch-size", action=argparse.BooleanOptionalAction, default=True,
        help="Search for the largest VAE batch size under the VRAM target. Default: enabled.",
    )
    parser.add_argument("--max-vae-batch-size", type=int, default=0,
                        help="Upper bound for automatic search; 0 means dataset limit.")
    parser.add_argument("--batch-benchmark-warmup", type=int, default=1,
                        help="Warmup batches per candidate. Default: 1.")
    parser.add_argument("--batch-benchmark-batches", type=int, default=3,
                        help="Timed batches per candidate. Default: 3.")
    parser.add_argument("--batch-memory-limit", type=float, default=0.90,
                        help="VRAM target for automatic search. Default: 0.90.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=10000,
                        help="Maximum images to measure. Default: 10000.")
    parser.add_argument("--output", default="output/image_gen/latent_stats.json")
    return parser.parse_args()


class RunningStats:
    def __init__(self):
        self.count = 0
        self.sum = None
        self.sum_sq = None
        self.minimum = None
        self.maximum = None

    def update(self, values):
        values = values.detach().float().permute(0, 2, 3, 1).reshape(-1, values.shape[1])
        values = values.cpu().double()
        if self.sum is None:
            channels = values.shape[1]
            self.sum = torch.zeros(channels, dtype=torch.float64)
            self.sum_sq = torch.zeros(channels, dtype=torch.float64)
            self.minimum = torch.full((channels,), float("inf"), dtype=torch.float64)
            self.maximum = torch.full((channels,), float("-inf"), dtype=torch.float64)
        self.count += values.shape[0]
        self.sum += values.sum(dim=0)
        self.sum_sq += values.square().sum(dim=0)
        self.minimum = torch.minimum(self.minimum, values.amin(dim=0))
        self.maximum = torch.maximum(self.maximum, values.amax(dim=0))

    def as_dict(self):
        if not self.count:
            raise ValueError("No latent values were measured")
        mean = self.sum / self.count
        variance = (self.sum_sq / self.count - mean.square()).clamp_min(0)
        std = variance.sqrt()
        rms = (self.sum_sq.sum() / (self.count * self.sum.numel())).sqrt()
        return {
            "count": self.count,
            "channels": int(self.sum.numel()),
            "global_mean": float(mean.mean()),
            "global_std": float(variance.mean().sqrt()),
            "global_rms": float(rms),
            "channel_mean": mean.float().tolist(),
            "channel_std": std.float().tolist(),
            "channel_min": self.minimum.float().tolist(),
            "channel_max": self.maximum.float().tolist(),
        }


def make_loader(dataset, batch_size, num_workers):
    return DataLoader(
        dataset,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
        batch_sampler=AspectRatioBatchSampler(dataset.bucket_ids, batch_size),
    )


def available_images(dataset, batch_size):
    counts = Counter(dataset.bucket_ids)
    return sum((count // batch_size) * batch_size for count in counts.values())


def build_dataset(args):
    if args.records:
        return RecordsDataset(args.records, args.image_size, args.bucket_step)
    if args.dataset_name.lower() == "nlphuji/flickr30k":
        if __package__:
            from .train import Flickr30KDataset
        else:
            from train import Flickr30KDataset
        return Flickr30KDataset(
            args.dataset_split, args.image_size, args.bucket_step, args.cache_dir,
        )
    return HFDataset(
        args.dataset_name, args.dataset_split, args.image_size,
        args.bucket_step, args.cache_dir,
    )


def is_cuda_oom(error):
    message = str(error).lower()
    return (
        isinstance(error, torch.cuda.OutOfMemoryError)
        or "out of memory" in message
        or "unable to find an engine to execute this computation" in message
        or "cudnn_status_alloc_failed" in message
        or "memoryallocation" in message
    )


def fit_peak_memory_line(observations):
    if len(observations) < 2:
        return None
    x_mean = sum(batch for batch, _ in observations) / len(observations)
    y_mean = sum(memory for _, memory in observations) / len(observations)
    denominator = sum((batch - x_mean) ** 2 for batch, _ in observations)
    if denominator <= 0:
        return None
    slope = sum(
        (batch - x_mean) * (memory - y_mean)
        for batch, memory in observations
    ) / denominator
    if slope <= 0:
        return None
    return y_mean - slope * x_mean, slope


def predicted_memory(observations, batch_size):
    line = fit_peak_memory_line(observations)
    if line is None:
        return None
    intercept, slope = line
    return intercept + slope * batch_size


def target_batch_size(observations, target_memory):
    line = fit_peak_memory_line(observations)
    if line is None:
        return None
    intercept, slope = line
    return math.floor((target_memory - intercept) / slope)


@torch.inference_mode()
def benchmark_batch_size(dataset, batch_size, vae, device, amp_context,
                         num_workers, warmup_batches, measured_batches):
    loader = make_loader(dataset, batch_size, num_workers)
    iterator = iter(loader)
    timings = []
    try:
        torch.cuda.reset_peak_memory_stats(device)
        for index in range(warmup_batches + measured_batches):
            images, _ = next(iterator)
            image_count = images.shape[0]
            images = images.to(device, non_blocking=True)
            torch.cuda.synchronize(device)
            started = perf_counter()
            with amp_context:
                latent = encode_images(vae, images)
            torch.cuda.synchronize(device)
            if index >= warmup_batches:
                timings.append((perf_counter() - started) / image_count)
            del latent, images
            collect_memory()
        if not timings:
            raise ValueError("No batches available for VAE benchmark")
        peak_reserved = torch.cuda.max_memory_reserved(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory
        return median(timings), peak_reserved, peak_reserved / total_memory
    except StopIteration:
        if not timings:
            raise ValueError("No batches available for VAE benchmark")
        peak_reserved = torch.cuda.max_memory_reserved(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory
        return median(timings), peak_reserved, peak_reserved / total_memory
    except RuntimeError as error:
        if not is_cuda_oom(error):
            raise
        return None
    finally:
        del iterator, loader
        collect_memory()


def find_batch_size(dataset, requested_limit, vae, device, amp_context,
                    num_workers, warmup_batches, measured_batches, memory_limit):
    bucket_limit = max(Counter(dataset.bucket_ids).values())
    upper_limit = min(
        requested_limit if requested_limit > 0 else bucket_limit,
        bucket_limit,
    )
    if upper_limit < 1:
        raise ValueError("The dataset must contain at least one sample in a bucket")

    candidates = []
    candidate = 1
    while candidate < upper_limit:
        candidates.append(candidate)
        candidate *= 2
    candidates.append(upper_limit)
    candidates = list(dict.fromkeys(candidates))
    print(f"benchmarking VAE batch sizes: {candidates}")

    total_memory = torch.cuda.get_device_properties(device).total_memory
    observations = []
    selected = None
    last_candidate = 0
    candidate_index = 0
    while candidate_index < len(candidates):
        nominal = candidates[candidate_index]
        candidate_index += 1
        predicted = predicted_memory(observations, nominal)
        if predicted is not None and predicted >= total_memory:
            target = target_batch_size(observations, total_memory * memory_limit)
            if target is None or target <= last_candidate:
                print(
                    f"  vae batch={nominal}: predicted VRAM reaches 100%; stopping"
                )
                break
            candidate = min(target, nominal, upper_limit)
            if candidate <= last_candidate:
                break
            print(
                f"  vae batch={nominal}: predicted VRAM reaches 100%; "
                f"trying batch={candidate} for ~{memory_limit:.1%}"
            )
        else:
            candidate = nominal

        result = benchmark_batch_size(
            dataset, candidate, vae, device, amp_context,
            num_workers, warmup_batches, measured_batches,
        )
        if result is None:
            print(f"  vae batch={candidate}: OOM")
            break
        per_image, peak_reserved, peak_ratio = result
        print(
            f"  vae batch={candidate}: {per_image * 1000:.3f} ms/image "
            f"peak_reserved={peak_reserved / 2**30:.2f}/"
            f"{total_memory / 2**30:.2f} GiB ({peak_ratio:.1%})"
        )
        if peak_ratio >= 1.0:
            print(f"  vae batch={candidate}: VRAM exceeded; stopping")
            break
        observations.append((candidate, peak_reserved))
        selected = candidate
        last_candidate = candidate
    if selected is None:
        raise RuntimeError("No VAE batch size fits on the available CUDA memory")
    return selected


def main():
    args = parse_args()
    if args.max_images <= 0:
        raise ValueError("--max-images must be positive")
    if args.batch_memory_limit <= 0 or args.batch_memory_limit >= 1:
        raise ValueError("--batch-memory-limit must be between 0 and 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    vae_dtype = resolve_vae_dtype(device, args.vae_dtype)
    amp_context = (
        torch.autocast("cuda", dtype=amp_dtype)
        if device.type == "cuda" else nullcontext()
    )
    dataset = build_dataset(args)

    from diffusers import AutoencoderKL
    if "qwen" in args.vae_model.lower():
        from diffusers import AutoencoderKLQwenImage
        vae = AutoencoderKLQwenImage.from_pretrained(
            args.vae_model, subfolder="vae", torch_dtype=vae_dtype,
        ).to(device).eval()
    else:
        vae = AutoencoderKL.from_pretrained(
            args.vae_model, subfolder="vae", torch_dtype=vae_dtype,
        ).to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    print(f"vae dtype={vae_dtype}")
    args.latent_scale = resolve_vae_latent_scale(vae, args.latent_scale)

    if args.auto_batch_size:
        if device.type != "cuda":
            raise ValueError("--auto-batch-size requires CUDA; specify --no-auto-batch-size on CPU")
        args.vae_batch_size = find_batch_size(
            dataset, args.max_vae_batch_size, vae, device, amp_context,
            args.num_workers, args.batch_benchmark_warmup,
            args.batch_benchmark_batches, args.batch_memory_limit,
        )
        print(f"selected VAE batch size: {args.vae_batch_size}")

    loader = make_loader(dataset, args.vae_batch_size, args.num_workers)
    total_images = min(
        available_images(dataset, args.vae_batch_size), args.max_images,
    )
    if total_images <= 0:
        raise ValueError("No complete VAE batches are available")

    with torch.inference_mode():
        probe, _ = next(iter(loader))
        probe = probe[:1].to(device, non_blocking=True)
        with amp_context:
            probe_latent = encode_images(vae, probe, args.latent_scale)
        probe_shape = list(probe_latent.shape)
        del probe, probe_latent
        collect_memory()

        stats = RunningStats()
        measured_images = 0
        progress = RichProgress(
            total=total_images, description="measuring VAE latent stats", unit="image"
        )
        progress.__enter__()
        try:
            for images, _ in loader:
                remaining = total_images - measured_images
                if remaining <= 0:
                    break
                images = images[:remaining].to(device, non_blocking=True)
                with amp_context:
                    latent = encode_images(vae, images, args.latent_scale)
                stats.update(latent)
                measured_images += images.shape[0]
                progress.update(images.shape[0])
                progress.set_postfix(images=measured_images)
                del latent, images
                collect_memory()
        finally:
            progress.close()

    result = stats.as_dict()
    output = {
        "config": {
            "dataset_name": args.dataset_name if not args.records else args.records,
            "vae_model": args.vae_model,
            "vae_dtype": str(vae_dtype).replace("torch.", ""),
            "image_size": args.image_size,
            "bucket_step": args.bucket_step,
            "vae_batch_size": args.vae_batch_size,
            "max_images": args.max_images,
            "measured_images": measured_images,
            "latent_scale": args.latent_scale,
            "latent_shape_probe": probe_shape,
        },
        "image_latent": result,
    }
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"\nsaved latent statistics: {args.output}")
    print(f"image RMS={result['global_rms']:.6f}")


if __name__ == "__main__":
    main()
