import argparse
import json
import math
import os
import re
import sys
from datetime import datetime
from time import perf_counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import save_image
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not __package__ and PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from core.layers import (  # noqa: E402
	EfficientDownsample,
	EfficientUpsample,
	GatedConv2d,
	GatedConvTranspose2d,
	GatedResBlock,
	PreNormConvFFNResidual2d,
	PreNormGatedConvFFNResidual2d,
	ResBlock,
	RMSNorm2d,
	RotaryEmbedding2D,
)
from runtime.memory import maybe_collect_memory  # noqa: E402
from runtime.data import build_dataloader_options  # noqa: E402
from runtime.metrics import write_standard_training_metrics  # noqa: E402
from runtime.preflight import build_training_preflight  # noqa: E402
from runtime.validation import ValidationTimer, build_validation_report  # noqa: E402
from runtime.progress import RichProgress  # noqa: E402
from runtime.run import RunRecorder  # noqa: E402
from runtime.signal import GracefulStop  # noqa: E402
from runtime.sampler import (  # noqa: E402
	ResumableAspectRatioBatchSampler,
	ResumableRandomSampler,
)
from runtime.checkpoint import (  # noqa: E402
	load_training_state,
	make_training_state,
	restore_rng_state,
	save_training_state,
)
from runtime.config import (  # noqa: E402
	CONFIG_SCHEMA_VERSION,
	apply_saved_config,
	cli_option_provided,
	validate_config_schema,
)
from runtime.device import add_device_argument, resolve_device  # noqa: E402
from core.utils import ModelSnapshot  # noqa: E402
from image_ae.adapter_training import (  # noqa: E402
	DEFAULT_LORA_TARGETS,  # noqa: F401 - backwards-compatible module export
	add_adapter_arguments,
	enable_adapter,
	optimizer_parameters as adapter_optimizer_parameters,
	resolve_adapter_config,
)
from optimizers.factory import (  # noqa: E402
	add_optimizer_argument,
	build_optimizer,
	is_schedule_free_optimizer,
)
from optimizers.lr_scheduler import (  # noqa: E402
	add_lr_scheduler_arguments,
	build_lr_scheduler,
)

# Increment when checkpoint architecture metadata becomes incompatible.
NETWORK_CONFIG_VERSION = "7"
NETWORK_CONFIG_METADATA_KEY = "image_ae.network_config"
NETWORK_CONFIG_VERSION_METADATA_KEY = "image_ae.network_config_version"
TRAINING_CONFIG_METADATA_KEY = "image_ae.training_config"
LEGACY_NETWORK_CONFIG_METADATA_KEY = "cifar10_ae.network_config"
LEGACY_NETWORK_CONFIG_VERSION_METADATA_KEY = "cifar10_ae.network_config_version"
LEGACY_TRAINING_CONFIG_METADATA_KEY = "cifar10_ae.training_config"
DEFAULT_DATASET = "flickr30k"
DATASET_IDS = {
	"flickr30k": "lmms-lab-encoder/flickr30k",
	"imagenet1k": "ILSVRC/imagenet-1k",
	"mini-imagenet": "timm/mini-imagenet",
}
SAMPLE_IMAGE_INTERVAL = 100
INIT_SHARED_PREFIXES = (
	"input_adapter.",
	"reconstruction_head.",
	"latent_mu.",
	"latent_to_bottleneck.",
)
def checkpoint_network_config(args):
	return {
		"encoder": args.encoder,
		"decoder": args.decoder,
		"latent_channels": args.latent_channels,
		"bottleneck_channels": args.bottleneck_channels,
		"hidden_channels": args.hidden_channels,
		"encoder_blocks": args.encoder_blocks,
		"decoder_blocks": args.decoder_blocks,
		"encoder_layers": args.encoder_layers,
		"encoder_window_size": args.encoder_window_size,
		"decoder_layers": args.decoder_layers,
		"downsample_stages": args.downsample_stages,
		"dataset": args.dataset,
		"image_size": args.image_size,
		"bucket_step": args.bucket_step,
		"vae": args.vae,
	}

def checkpoint_training_config(args):
	config = {
		"epochs": args.epochs,
		"batch_size": args.batch_size,
		"lr": args.lr,
		"loss_fn": args.loss_fn,
		"huber_beta": args.huber_beta,
		"optimizer": args.optimizer,
		"auto_schedule": args.auto_schedule,
		"lr_scheduler": args.lr_scheduler,
		"warmup_steps": args.warmup_steps,
		"warmup_ratio": args.warmup_ratio,
		"min_lr_ratio": args.min_lr_ratio,
		"lr_step_size": args.lr_step_size,
		"lr_gamma": args.lr_gamma,
		"lr_milestones": args.lr_milestones,
		"lr_num_cycles": args.lr_num_cycles,
		"lr_power": args.lr_power,
		"force_scheduler": args.force_scheduler,
		"apollo_rank": args.apollo_rank,
		"apollo_scale": args.apollo_scale,
		"apollo_update_proj_gap": args.apollo_update_proj_gap,
		"apollo_projection_refresh_mode": args.apollo_projection_refresh_mode,
		"apollo_projection_refresh_window": args.apollo_projection_refresh_window,
		"apollo_projection_refresh_mix": args.apollo_projection_refresh_mix,
		"apollo_projection_refresh_state": args.apollo_projection_refresh_state,
		"apollo_orthogonal_refresh_rate": args.apollo_orthogonal_refresh_rate,
		"apollo_orthogonal_refresh_direction": args.apollo_orthogonal_refresh_direction,
		"apollo_scale_front": args.apollo_scale_front,
		"apollo_disable_norm_growth_limiter": args.apollo_disable_norm_growth_limiter,
		"apollo_norm_growth_rate": args.apollo_norm_growth_rate,
		"apollo_fallback": args.apollo_fallback,
		"apollo_matrix_fallback": args.apollo_matrix_fallback,
		"apollo_came_backend": args.apollo_came_backend,
		"came_lrsf_rank": args.came_lrsf_rank,
		"came_lrsf_beta1": args.came_lrsf_beta1,
		"came_lrsf_warmup_steps": args.came_lrsf_warmup_steps,
		"came_lrsf_r": args.came_lrsf_r,
		"came_lrsf_weight_lr_power": args.came_lrsf_weight_lr_power,
		"came_lrsf_seed": args.came_lrsf_seed,
		"came_lrsf_refresh_mode": args.came_lrsf_refresh_mode,
		"came_lrsf_refresh_interval": args.came_lrsf_refresh_interval,
		"came_lrsf_refresh_window": args.came_lrsf_refresh_window,
		"came_lrsf_refresh_mix": args.came_lrsf_refresh_mix,
		"came_lrsf_orthogonal_refresh_rate": args.came_lrsf_orthogonal_refresh_rate,
		"came_lrsf_orthogonal_refresh_direction": args.came_lrsf_orthogonal_refresh_direction,
		"came_lrsf_orthogonal_refresh_signal": args.came_lrsf_orthogonal_refresh_signal,
		"num_workers": args.num_workers,
		"seed": args.seed,
		"vae": args.vae,
		"kl_weight": args.kl_weight,
		"latent_cycle_consistency": args.latent_cycle_consistency,
		"latent_cycle_weight": args.latent_cycle_weight,
		"latent_cycle_z1_weight": args.latent_cycle_z1_weight,
		"latent_cycle_z2_weight": args.latent_cycle_z2_weight,
		"image_cycle_consistency": args.image_cycle_consistency,
		"image_cycle_weight": args.image_cycle_weight,
		"wavelet_loss": args.wavelet_loss,
		"wavelet_loss_weight": args.wavelet_loss_weight,
		"wavelet_levels": args.wavelet_levels,
		"latent_variance_loss": args.latent_variance_loss,
		"latent_variance_weight": args.latent_variance_weight,
		"channel_decorrelation_loss": args.channel_decorrelation_loss,
		"channel_decorrelation_weight": args.channel_decorrelation_weight,
		"input_noise_std": args.input_noise_std,
		"input_blur_sigma": args.input_blur_sigma,
		"input_bit_depth": args.input_bit_depth,
		"imagenet_train_samples": args.imagenet_train_samples,
		"imagenet_val_samples": args.imagenet_val_samples,
		"cifar10_train_samples": getattr(args, "cifar10_train_samples", 0),
		"cifar10_val_samples": getattr(args, "cifar10_val_samples", 0),
		"dataset_split": args.dataset_split,
		"validation_split": args.validation_split,
		"lora_rank": args.lora_rank,
		"adapter": args.adapter,
		"lora_alpha": args.lora_alpha,
		"lora_dropout": args.lora_dropout,
		"lora_target": args.lora_target,
		"adapter_init": args.adapter_init,
	}
	for key in ("merged_adapter", "merged_from"):
		if hasattr(args, key):
			config[key] = getattr(args, key)
	return config

def checkpoint_metadata(args, epoch=None):
	network_config = checkpoint_network_config(args)
	network_config["config_schema_version"] = CONFIG_SCHEMA_VERSION
	training_config = checkpoint_training_config(args)
	training_config["config_schema_version"] = CONFIG_SCHEMA_VERSION
	metadata = {
		NETWORK_CONFIG_VERSION_METADATA_KEY: NETWORK_CONFIG_VERSION,
		NETWORK_CONFIG_METADATA_KEY: json.dumps(
			network_config, sort_keys=True,
		),
		TRAINING_CONFIG_METADATA_KEY: json.dumps(
			training_config, sort_keys=True,
		),
	}
	if epoch is not None:
		metadata["image_ae.epoch"] = str(epoch)
	return metadata

def save_model_checkpoint(state_dict, path, args, epoch=None):
	save_file(state_dict, path, metadata=checkpoint_metadata(args, epoch))


def save_training_checkpoint(
	state_dict, path, args, *, optimizer, scheduler, epoch, global_step, extra=None,
):
	"""Save model weights and the state required for a full training resume."""
	save_model_checkpoint(state_dict, path, args, epoch=epoch)
	save_training_state(
		path,
		make_training_state(
			optimizer=optimizer,
			scheduler=scheduler,
			epoch=epoch,
			global_step=global_step,
			extra=extra,
		),
	)

def checkpoint_epoch(path):
	with safe_open(path, framework="pt", device="cpu") as checkpoint:
		metadata = checkpoint.metadata() or {}
	try:
		return int(metadata.get("image_ae.epoch", metadata.get("cifar10_ae.epoch", "0")))
	except ValueError:
		return 0

def checkpoint_is_vae(path):
	with safe_open(path, framework="pt", device="cpu") as checkpoint:
		keys = set(checkpoint.keys())
	return "latent_logvar.weight" in keys or "conv_logvar.weight" in keys

def load_resume_state_dict(path, vae):
	state_dict = load_file(path, device="cpu")
	if not vae:
		state_dict = {
			key: value for key, value in state_dict.items()
			if not key.startswith("latent_logvar.")
			and not key.startswith("conv_logvar.")
		}
	return state_dict

def initialize_shared_parameters(model, path):
	source_state = load_file(path, device="cpu")
	target_state = model.state_dict()
	transferred = []
	with torch.no_grad():
		for key, target in target_state.items():
			if not key.startswith(INIT_SHARED_PREFIXES):
				continue
			source = source_state.get(key)
			if source is None or source.shape != target.shape:
				continue
			target.copy_(source.to(device=target.device, dtype=target.dtype))
			transferred.append(key)
	return transferred

def network_config_from_checkpoint_name(path):
	name = os.path.basename(path)
	match = re.search(
		r"(?:(?P<dataset>cifar10|flickr30k|imagenet1k|mini-imagenet)_)?"
		r"enc-(?P<encoder>.+?)_dec-(?P<decoder>.+?)"
		r"(?:_size(?P<image_size>\d+))?"
		r"_latent-ch(?P<latent_channels>\d+)"
		r"_bottleneck-ch(?P<bottleneck_channels>\d+)"
		r"_ds(?P<downsample_stages>\d+)"
		r"_eb(?P<encoder_blocks>\d+)_db(?P<decoder_blocks>\d+)"
		r"(?:_el(?P<encoder_layers>\d+)_ew(?P<encoder_window_size>\d+))?"
		r"(?:_dl(?P<decoder_layers>\d+))?"
		r"(?P<vae>_vae)?"
		r"(?:_hidden-ch(?P<hidden_channels>\d+))?",
		name,
	)
	if not match:
		return None
	values = match.groupdict()
	config = {
		"encoder": values["encoder"],
		"decoder": values["decoder"],
		"latent_channels": int(values["latent_channels"]),
		"bottleneck_channels": int(values["bottleneck_channels"]),
		"hidden_channels": (
			int(values["hidden_channels"])
			if values["hidden_channels"] is not None else None
		),
		"encoder_blocks": int(values["encoder_blocks"]),
		"decoder_blocks": int(values["decoder_blocks"]),
		"downsample_stages": int(values["downsample_stages"]),
		"encoder_layers": (
			int(values["encoder_layers"])
			if values["encoder_layers"] is not None else 4
		),
		"encoder_window_size": (
			int(values["encoder_window_size"])
			if values["encoder_window_size"] is not None else 8
		),
		"decoder_layers": (
			int(values["decoder_layers"])
			if values["decoder_layers"] is not None else 2
		),
		"vae": values["vae"] is not None,
	}
	if values["dataset"] is not None:
		config["dataset"] = values["dataset"]
	if values["image_size"] is not None:
		config["image_size"] = int(values["image_size"])
	return config

def resume_network_config(path):
	with safe_open(path, framework="pt", device="cpu") as checkpoint:
		metadata = checkpoint.metadata() or {}
	version = metadata.get(
		NETWORK_CONFIG_VERSION_METADATA_KEY,
		metadata.get(LEGACY_NETWORK_CONFIG_VERSION_METADATA_KEY),
	)
	config_key = (
		NETWORK_CONFIG_METADATA_KEY
		if metadata.get(NETWORK_CONFIG_METADATA_KEY)
		else LEGACY_NETWORK_CONFIG_METADATA_KEY
	)
	if version == NETWORK_CONFIG_VERSION and metadata.get(config_key):
		config = json.loads(metadata[config_key])
		validate_config_schema(config, key=config_key, path=path)
		return config
	if version is None and not metadata.get(config_key):
		return network_config_from_checkpoint_name(path)
	if version != NETWORK_CONFIG_VERSION:
		raise ValueError(
			f"resume checkpoint network config version {version!r} is incompatible "
			f"with current version {NETWORK_CONFIG_VERSION!r}"
		)
	return None

def resume_training_config(path):
	with safe_open(path, framework="pt", device="cpu") as checkpoint:
		metadata = checkpoint.metadata() or {}
	config_key = (
		TRAINING_CONFIG_METADATA_KEY
		if metadata.get(TRAINING_CONFIG_METADATA_KEY)
		else LEGACY_TRAINING_CONFIG_METADATA_KEY
	)
	if metadata.get(config_key):
		config = json.loads(metadata[config_key])
		validate_config_schema(config, key=config_key, path=path)
		return config
	return None

def group_norm(num_channels):
	if num_channels <= 0:
		raise ValueError("GroupNorm channels must be positive")
	groups = min(32, num_channels)
	while num_channels % groups:
		groups -= 1
	return nn.GroupNorm(groups, num_channels)

class Random90Rotation:
	def __call__(self, img):
		k = torch.randint(0, 4, (1,)).item()
		return img.rotate(90 * k, resample=transforms.InterpolationMode.NEAREST)

def make_bucket_shapes(image_size, bucket_step=32):
	"""Create near-equal-area aspect-ratio buckets aligned to the AE stride."""
	if image_size <= 0 or bucket_step <= 0:
		raise ValueError("image_size and bucket_step must be positive")
	ratios = (0.667, 0.8, 1.0, 1.25, 1.5)
	shapes = []
	for ratio in ratios:
		height = int(round(math.sqrt(image_size * image_size / ratio) / bucket_step)) * bucket_step
		width = int(round(height * ratio / bucket_step)) * bucket_step
		shapes.append((max(bucket_step, height), max(bucket_step, width)))
	return tuple(dict.fromkeys(shapes))


def assign_bucket(width, height, bucket_shapes):
	"""Map an image to the closest logarithmic aspect-ratio bucket."""
	aspect = width / max(height, 1)
	return min(
		range(len(bucket_shapes)),
		key=lambda index: abs(
			math.log(aspect)
			- math.log(bucket_shapes[index][1] / bucket_shapes[index][0])
		),
	)


def bucket_transform(shape, train):
	transforms_list = [
		transforms.Resize(max(shape), interpolation=transforms.InterpolationMode.BICUBIC),
		transforms.CenterCrop(shape),
	]
	if train:
		transforms_list.extend((
			transforms.RandomHorizontalFlip(),
			transforms.ColorJitter(
				brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01,
			),
		))
	transforms_list.append(transforms.ToTensor())
	return transforms.Compose(transforms_list)


class AspectRatioBatchSampler(ResumableAspectRatioBatchSampler):
	"""Yield checkpointable batches whose images share one resolution bucket."""


class HuggingFaceImageDataset(torch.utils.data.Dataset):
	"""Apply fixed or aspect-ratio-bucketed transforms to HF image rows."""
	def __init__(self, dataset, transform=None, image_size=None, bucket_step=32, train=False):
		self.dataset = dataset
		self.train = train
		self.bucket_shapes = (
			make_bucket_shapes(image_size, bucket_step) if image_size is not None else None
		)
		self.transform = transform
		if self.bucket_shapes is not None:
			self.bucket_transforms = tuple(
				bucket_transform(shape, train) for shape in self.bucket_shapes
			)
			self.bucket_ids = []
			for index in range(len(self.dataset)):
				image = self.dataset[index]["image"]
				self.bucket_ids.append(assign_bucket(*image.size, self.bucket_shapes))
		else:
			self.bucket_ids = None

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		row = self.dataset[index]
		image = row["image"].convert("RGB")
		if self.bucket_shapes is not None:
			transform = self.bucket_transforms[self.bucket_ids[index]]
		else:
			transform = self.transform
		return transform(image), row.get("label", 0)

class BasicCNNEncoder(nn.Module):
	def __init__(self, in_channels, feature_channels=256):
		super().__init__()
		stage_channels = feature_channels // 4
		mid_channels = feature_channels // 2
		self.layers = nn.Sequential(
			nn.Conv2d(in_channels, stage_channels, kernel_size=4, stride=2, padding=1),
			group_norm(stage_channels),
			nn.SiLU(),
			ResBlock(stage_channels, stage_channels),
			nn.Conv2d(stage_channels, mid_channels, kernel_size=4, stride=2, padding=1),
			group_norm(mid_channels),
			nn.SiLU(),
			ResBlock(mid_channels, mid_channels),
			nn.Conv2d(mid_channels, feature_channels, kernel_size=4, stride=2, padding=1),
			group_norm(feature_channels),
			nn.SiLU(),
			ResBlock(feature_channels, feature_channels),
		)
		self.output_norm = RMSNorm2d(feature_channels)

	def forward(self, x):
		return self.output_norm(self.layers(x))

class GatedCNNEncoder(nn.Module):
	def __init__(self, in_channels, feature_channels=256):
		super().__init__()
		stage_channels = feature_channels // 4
		mid_channels = feature_channels // 2
		self.layers = nn.Sequential(
			GatedConv2d(in_channels, stage_channels, kernel_size=4, stride=2, padding=1),
			group_norm(stage_channels),
			GatedResBlock(stage_channels, stage_channels),
			GatedConv2d(stage_channels, mid_channels, kernel_size=4, stride=2, padding=1),
			group_norm(mid_channels),
			GatedResBlock(mid_channels, mid_channels),
			GatedConv2d(mid_channels, feature_channels, kernel_size=4, stride=2, padding=1),
			group_norm(feature_channels),
			GatedResBlock(feature_channels, feature_channels),
		)
		self.output_norm = RMSNorm2d(feature_channels)

	def forward(self, x):
		return self.output_norm(self.layers(x))

class DCAEEncoder(nn.Module):
	def __init__(self, in_channels, feature_channels=256):
		super().__init__()
		stage_channels = feature_channels // 4
		mid_channels = feature_channels // 2
		self.layers = nn.Sequential(
			nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
			group_norm(16),
			nn.SiLU(),
			EfficientDownsample(16, stage_channels),
			ResBlock(stage_channels, stage_channels),
			EfficientDownsample(stage_channels, mid_channels),
			ResBlock(mid_channels, mid_channels),
			EfficientDownsample(mid_channels, feature_channels),
			ResBlock(feature_channels, feature_channels),
		)
		self.output_norm = RMSNorm2d(feature_channels)

	def forward(self, x):
		return self.output_norm(self.layers(x))

class ResidualConvFFNBlock(PreNormConvFFNResidual2d):
	"""Checkpoint-compatible wrapper for the shared pre-norm Conv FFN."""
	def __init__(self, channels, hidden_channels=None):
		super().__init__(channels, hidden_channels, state_layout="sequential")

class ResidualConvFFNEncoder(nn.Module):
	def __init__(self, in_channels=16, feature_channels=256,
			 num_blocks=1, hidden_channels=None, downsample_stages=3):
		super().__init__()
		if num_blocks <= 0:
			raise ValueError("num_blocks must be positive")
		if downsample_stages <= 0:
			raise ValueError("downsample_stages must be positive")
		stage_channels = [
			feature_channels // (2 ** (downsample_stages - index - 1))
			for index in range(downsample_stages)
		]
		self.stem_downsample = nn.Sequential(
			nn.Conv2d(in_channels, stage_channels[0], kernel_size=4, stride=2, padding=1),
			group_norm(stage_channels[0]),
			*(ResidualConvFFNBlock(stage_channels[0], hidden_channels=hidden_channels)
			  for _ in range(num_blocks)),
		)
		self.downsample_stages = nn.ModuleList(
			nn.Sequential(
				nn.Conv2d(stage_channels[index - 1], stage_channels[index], kernel_size=4, stride=2, padding=1),
				group_norm(stage_channels[index]),
				*(ResidualConvFFNBlock(stage_channels[index], hidden_channels=hidden_channels)
				  for _ in range(num_blocks)),
			)
			for index in range(1, downsample_stages)
		)
		self.output_norm = RMSNorm2d(feature_channels)

	def forward(self, x):
		x = self.stem_downsample(x)
		for stage in self.downsample_stages:
			x = stage(x)
		return self.output_norm(x)

class GatedResidualConvFFNBlock(PreNormGatedConvFFNResidual2d):
	"""Checkpoint-compatible wrapper for the shared gated Conv FFN."""
	def __init__(self, channels, hidden_channels=None):
		super().__init__(channels, hidden_channels, state_layout="sequential")

class GatedResidualConvFFNEncoder(nn.Module):
	def __init__(self, in_channels=16, feature_channels=256,
			 num_blocks=1, hidden_channels=None, downsample_stages=3):
		super().__init__()
		if num_blocks <= 0:
			raise ValueError("num_blocks must be positive")
		if downsample_stages <= 0:
			raise ValueError("downsample_stages must be positive")
		stage_channels = [
			feature_channels // (2 ** (downsample_stages - index - 1))
			for index in range(downsample_stages)
		]
		self.stem_downsample = nn.Sequential(
			GatedConv2d(in_channels, stage_channels[0], kernel_size=4, stride=2, padding=1),
			group_norm(stage_channels[0]),
			*(GatedResidualConvFFNBlock(stage_channels[0], hidden_channels=hidden_channels)
			  for _ in range(num_blocks)),
		)
		self.downsample_stages = nn.ModuleList(
			nn.Sequential(
				GatedConv2d(stage_channels[index - 1], stage_channels[index], kernel_size=4, stride=2, padding=1),
				group_norm(stage_channels[index]),
				*(GatedResidualConvFFNBlock(stage_channels[index], hidden_channels=hidden_channels)
				  for _ in range(num_blocks)),
			)
			for index in range(1, downsample_stages)
		)
		self.output_norm = RMSNorm2d(feature_channels)

	def forward(self, x):
		x = self.stem_downsample(x)
		for stage in self.downsample_stages:
			x = stage(x)
		return self.output_norm(x)

def window_partition(x, window_size):
	"""Partition an NHWC tensor into non-overlapping local windows."""
	batch, height, width, channels = x.shape
	if height % window_size or width % window_size:
		raise ValueError("window_partition requires dimensions divisible by window_size")
	return x.view(
		batch, height // window_size, window_size,
		width // window_size, window_size, channels,
	).permute(0, 1, 3, 2, 4, 5).reshape(
		batch * (height // window_size) * (width // window_size),
		window_size * window_size,
		channels,
	)


def window_reverse(windows, window_size, height, width, batch):
	"""Reverse :func:`window_partition` for a padded spatial grid."""
	return windows.view(
		batch, height // window_size, width // window_size,
		window_size, window_size, -1,
	).permute(0, 1, 3, 2, 4, 5).reshape(batch, height, width, -1)


class WindowSelfAttention(nn.Module):
	"""Local SDPA with shifted windows, 2D RoPE, and head-wise gates."""
	def __init__(self, embed_dim, num_heads, window_size=8, shift_size=0,
				 use_head_gate=True):
		super().__init__()
		if embed_dim % num_heads != 0:
			raise ValueError("embed_dim must be divisible by num_heads")
		head_dim = embed_dim // num_heads
		if head_dim % 4 != 0:
			raise ValueError("head_dim must be divisible by 4 for 2D RoPE")
		if not 0 <= shift_size < window_size:
			raise ValueError("shift_size must be in [0, window_size)")
		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.head_dim = head_dim
		self.window_size = window_size
		self.shift_size = shift_size
		self.qkv = nn.Linear(embed_dim, embed_dim * 3)
		self.output = nn.Linear(embed_dim, embed_dim)
		self.head_gate = nn.Linear(embed_dim, num_heads) if use_head_gate else None
		if self.head_gate is not None:
			nn.init.zeros_(self.head_gate.weight)
			nn.init.zeros_(self.head_gate.bias)
		self._mask_cache = {}
		self.rotary_embedding = RotaryEmbedding2D(
			head_dim, height=window_size, width=window_size,
		)

	def _attention_mask(self, height, width, device, dtype):
		cache_key = (height, width, device.type, device.index, dtype)
		cached = self._mask_cache.get(cache_key)
		if cached is not None:
			return cached
		window_size = self.window_size
		shift_size = self.shift_size
		valid = torch.ones((1, height, width, 1), device=device, dtype=torch.bool)
		valid = F.pad(
			valid,
			(0, 0, 0, (-width) % window_size, 0, (-height) % window_size),
		)
		padded_height, padded_width = valid.shape[1:3]
		if shift_size:
			valid = torch.roll(valid, shifts=(-shift_size, -shift_size), dims=(1, 2))
			region = torch.zeros(
				(1, padded_height, padded_width, 1), device=device, dtype=torch.int64,
			)
			count = 0
			for height_slice in (
				slice(0, -window_size), slice(-window_size, -shift_size),
				slice(-shift_size, None),
			):
				for width_slice in (
					slice(0, -window_size), slice(-window_size, -shift_size),
					slice(-shift_size, None),
				):
					region[:, height_slice, width_slice, :] = count
					count += 1
			region_windows = window_partition(region, window_size).squeeze(-1)
			region_delta = region_windows.unsqueeze(1) - region_windows.unsqueeze(2)
			mask = torch.zeros_like(region_delta, dtype=dtype)
			mask.masked_fill_(region_delta.ne(0), float("-inf"))
		else:
			mask = torch.zeros(
				(
					padded_height // window_size * padded_width // window_size,
					window_size * window_size,
					window_size * window_size,
				),
				device=device,
				dtype=dtype,
			)
		valid_windows = window_partition(valid, window_size).squeeze(-1)
		invalid_keys = ~valid_windows
		mask = mask.masked_fill(invalid_keys[:, None, :], float("-inf"))
		self._mask_cache[cache_key] = mask
		return mask

	def forward(self, x):
		batch, height, width, channels = x.shape
		window_size = self.window_size
		pad_height = (-height) % window_size
		pad_width = (-width) % window_size
		padded = F.pad(x, (0, 0, 0, pad_width, 0, pad_height))
		padded_height, padded_width = padded.shape[1:3]
		if self.shift_size:
			padded = torch.roll(
				padded,
				shifts=(-self.shift_size, -self.shift_size), dims=(1, 2),
			)
		windows = window_partition(padded, window_size)
		window_batch = windows.shape[0]
		qkv = self.qkv(windows).reshape(
			window_batch, window_size * window_size, 3,
			self.num_heads, self.head_dim,
		).permute(2, 0, 3, 1, 4)
		query, key = self.rotary_embedding(
			qkv[0], qkv[1], grid_shape=(window_size, window_size),
		)
		mask = self._attention_mask(height, width, x.device, query.dtype)
		mask = mask.unsqueeze(0).expand(batch, -1, -1, -1).reshape(
			window_batch, 1, window_size * window_size, window_size * window_size,
		)
		attention = F.scaled_dot_product_attention(
			query, key, qkv[2], attn_mask=mask, dropout_p=0.0,
		)
		if self.head_gate is not None:
			# Gate each SDPA head independently for every query token.  The
			# factor 2 makes the zero-initialized gate an exact 1x identity,
			# while still allowing the model to suppress or amplify a head.
			gate = 2.0 * torch.sigmoid(self.head_gate(windows))
			attention = attention * gate.transpose(1, 2).unsqueeze(-1)
		attention = attention.transpose(1, 2).reshape(window_batch, -1, channels)
		output = self.output(attention)
		output = window_reverse(
			output, window_size, padded_height, padded_width, batch,
		)
		if self.shift_size:
			output = torch.roll(
				output, shifts=(self.shift_size, self.shift_size), dims=(1, 2),
			)
		return output[:, :height, :width, :]


class WindowTransformerBlock(nn.Module):
	def __init__(self, embed_dim, num_heads, window_size=8, shift_size=0,
				 dropout=0.1, use_head_gate=True):
		super().__init__()
		self.attention = WindowSelfAttention(
			embed_dim, num_heads, window_size=window_size, shift_size=shift_size,
			use_head_gate=use_head_gate,
		)
		self.norm1 = nn.RMSNorm(embed_dim)
		self.norm2 = nn.RMSNorm(embed_dim)
		self.ffn = nn.Sequential(
			nn.Linear(embed_dim, embed_dim * 4),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(embed_dim * 4, embed_dim),
			nn.Dropout(dropout),
		)

	def forward(self, x):
		residual = x
		x_nhwc = x.permute(0, 2, 3, 1)
		x = residual + self.attention(self.norm1(x_nhwc)).permute(0, 3, 1, 2)
		x_nhwc = x.permute(0, 2, 3, 1)
		return x + self.ffn(self.norm2(x_nhwc)).permute(0, 3, 1, 2)


def hierarchical_stage_channels(feature_channels):
	"""Return the low-to-high widths used by the three spatial stages."""
	if feature_channels <= 0 or feature_channels % 32 != 0:
		raise ValueError("feature_channels must be a positive multiple of 32")
	return (
		feature_channels // 4,
		feature_channels // 2,
		feature_channels,
	)


def hierarchical_stage_heads():
	"""Keep the per-head dimension constant across the three stages."""
	return (2, 4, 8)


def make_window_stage(channels, heads, num_layers, window_size,
					  use_head_gate=True):
	return nn.Sequential(*(
		WindowTransformerBlock(
			channels,
			heads,
			window_size=window_size,
			shift_size=0 if index % 2 == 0 else window_size // 2,
			use_head_gate=use_head_gate,
		)
		for index in range(num_layers)
	))


class HierarchicalWindowAttentionEncoder(nn.Module):
	"""Three-level shifted-window encoder with constant per-head width.

	The input is reduced by one factor of two before each stage.  This keeps
	attention local at high resolutions while allowing the later stages to
	model a larger receptive field at a lower token count.
	"""
	def __init__(self, in_channels=16, feature_channels=256, num_layers=2,
				 window_size=8, use_head_gate=True):
		super().__init__()
		if num_layers <= 0:
			raise ValueError("num_layers must be positive")
		if window_size <= 0:
			raise ValueError("window_size must be positive")
		stage_channels = hierarchical_stage_channels(feature_channels)
		stage_heads = hierarchical_stage_heads()
		self.stem_downsample = nn.Sequential(
			GatedConv2d(in_channels, stage_channels[0], kernel_size=4, stride=2, padding=1),
			group_norm(stage_channels[0]),
			nn.SiLU(),
		)
		self.stages = nn.ModuleList([
			make_window_stage(
				channels, heads, num_layers, window_size, use_head_gate,
			)
			for channels, heads in zip(stage_channels, stage_heads)
		])
		self.downsample_stages = nn.ModuleList([
			nn.Sequential(
				GatedConv2d(
					stage_channels[index], stage_channels[index + 1],
					kernel_size=4, stride=2, padding=1,
				),
				group_norm(stage_channels[index + 1]),
				nn.SiLU(),
			)
			for index in range(len(stage_channels) - 1)
		])
		self.output_norm = RMSNorm2d(feature_channels)

	def forward(self, x):
		x = self.stem_downsample(x)
		for index, stage in enumerate(self.stages):
			x = stage(x)
			if index < len(self.downsample_stages):
				x = self.downsample_stages[index](x)
		return self.output_norm(x)


class HierarchicalWindowAttentionDecoder(nn.Module):
	"""Mirror of :class:`HierarchicalWindowAttentionEncoder`.

	The decoder processes the bottleneck at the coarsest level, then alternates
	window blocks and transposed-convolution upsampling until it returns the
	16-channel feature map consumed by ``reconstruction_head``.  Decoder
	upsampling intentionally uses ordinary convolutions rather than gated
	convolutions, matching the more stable reconstruction-side behavior.
	"""
	def __init__(self, feature_channels=256, num_layers=2, window_size=8,
				 use_head_gate=True):
		super().__init__()
		if num_layers <= 0:
			raise ValueError("num_layers must be positive")
		if window_size <= 0:
			raise ValueError("window_size must be positive")
		stage_channels = hierarchical_stage_channels(feature_channels)
		stage_heads = hierarchical_stage_heads()
		self.stages = nn.ModuleList([
			make_window_stage(
				channels, heads, num_layers, window_size, use_head_gate,
			)
			for channels, heads in zip(stage_channels[::-1], stage_heads[::-1])
		])
		upsample_channels = (stage_channels[1], stage_channels[0], 16)
		input_channels = (stage_channels[2], stage_channels[1], stage_channels[0])
		self.upsample_stages = nn.ModuleList([
			nn.Sequential(
				RMSNorm2d(in_channels),
				nn.ConvTranspose2d(
					in_channels, out_channels, kernel_size=4, stride=2, padding=1,
				),
				group_norm(out_channels),
				nn.SiLU(),
			)
			for in_channels, out_channels in zip(input_channels, upsample_channels)
		])

	def forward(self, x):
		for index, stage in enumerate(self.stages):
			x = stage(x)
			x = self.upsample_stages[index](x)
		return x

class BasicCNNDecoder(nn.Module):
	def __init__(self, feature_channels=256):
		super().__init__()
		upsample_stage_channels = feature_channels // 4
		upsample_mid_channels = feature_channels // 2
		decoder_feature_channels = 16
		self.layers = nn.Sequential(
			RMSNorm2d(feature_channels),
			ResBlock(feature_channels, feature_channels, norm_factory=RMSNorm2d),
			nn.ConvTranspose2d(feature_channels, upsample_mid_channels, kernel_size=4, stride=2, padding=1),
			RMSNorm2d(upsample_mid_channels),
			ResBlock(upsample_mid_channels, upsample_mid_channels, norm_factory=RMSNorm2d),
			nn.ConvTranspose2d(upsample_mid_channels, upsample_stage_channels, kernel_size=4, stride=2, padding=1),
			RMSNorm2d(upsample_stage_channels),
			ResBlock(upsample_stage_channels, upsample_stage_channels, norm_factory=RMSNorm2d),
			nn.ConvTranspose2d(upsample_stage_channels, decoder_feature_channels, kernel_size=4, stride=2, padding=1),
		)

	def forward(self, x):
		return self.layers(x)

class ResidualConvFFNDecoder(nn.Module):
	def __init__(self, feature_channels=256, hidden_channels=None, num_blocks=1,
			 downsample_stages=3):
		super().__init__()
		if num_blocks <= 0:
			raise ValueError("num_blocks must be positive")
		if downsample_stages <= 0:
			raise ValueError("downsample_stages must be positive")
		upsample_channels = [
			feature_channels // (2 ** index)
			for index in range(1, downsample_stages)
		]
		self.upsample_stages = nn.ModuleList(
			nn.Sequential(
				RMSNorm2d(feature_channels // (2 ** (index - 1))),
				nn.ConvTranspose2d(
					feature_channels // (2 ** (index - 1)), upsample_channels[index - 1],
					kernel_size=4, stride=2, padding=1,
				),
				*(ResidualConvFFNBlock(upsample_channels[index - 1], hidden_channels=hidden_channels)
				  for _ in range(num_blocks)),
			)
			for index in range(1, downsample_stages)
		)
		last_channels = upsample_channels[-1] if upsample_channels else feature_channels
		self.stem_upsample = nn.Sequential(
			RMSNorm2d(last_channels),
			nn.ConvTranspose2d(last_channels, 16, kernel_size=4, stride=2, padding=1),
			*(ResidualConvFFNBlock(16, hidden_channels=hidden_channels)
			  for _ in range(num_blocks)),
		)

	def forward(self, x):
		for stage in self.upsample_stages:
			x = stage(x)
		return self.stem_upsample(x)

class GatedResidualConvFFNDecoder(nn.Module):
	def __init__(self, feature_channels=256, hidden_channels=None, num_blocks=1,
			 downsample_stages=3):
		super().__init__()
		if num_blocks <= 0:
			raise ValueError("num_blocks must be positive")
		if downsample_stages <= 0:
			raise ValueError("downsample_stages must be positive")
		upsample_channels = [
			feature_channels // (2 ** index)
			for index in range(1, downsample_stages)
		]
		self.upsample_stages = nn.ModuleList(
			nn.Sequential(
				RMSNorm2d(feature_channels // (2 ** (index - 1))),
				GatedConvTranspose2d(
					feature_channels // (2 ** (index - 1)), upsample_channels[index - 1],
				),
				*(GatedResidualConvFFNBlock(upsample_channels[index - 1], hidden_channels=hidden_channels)
				  for _ in range(num_blocks)),
			)
			for index in range(1, downsample_stages)
		)
		last_channels = upsample_channels[-1] if upsample_channels else feature_channels
		self.stem_upsample = nn.Sequential(
			RMSNorm2d(last_channels),
			GatedConvTranspose2d(last_channels, 16),
			*(GatedResidualConvFFNBlock(16, hidden_channels=hidden_channels)
			  for _ in range(num_blocks)),
		)

	def forward(self, x):
		for stage in self.upsample_stages:
			x = stage(x)
		return self.stem_upsample(x)

class DCAEDecoder(nn.Module):
	def __init__(self, feature_channels=256):
		super().__init__()
		upsample_stage_channels = feature_channels // 4
		upsample_mid_channels = feature_channels // 2
		decoder_feature_channels = 16
		self.layers = nn.Sequential(
			RMSNorm2d(feature_channels),
			ResBlock(feature_channels, feature_channels, norm_factory=RMSNorm2d),
			EfficientUpsample(feature_channels, upsample_mid_channels),
			ResBlock(upsample_mid_channels, upsample_mid_channels, norm_factory=RMSNorm2d),
			EfficientUpsample(upsample_mid_channels, upsample_stage_channels),
			ResBlock(upsample_stage_channels, upsample_stage_channels, norm_factory=RMSNorm2d),
			EfficientUpsample(upsample_stage_channels, decoder_feature_channels, normalize=False),
		)

	def forward(self, x):
		return self.layers(x)

def canonical_encoder_type(encoder_type):
	return {
		"transformer": "window_transformer",
		"sliding_window": "window_transformer",
		"basic_cnn": "cnn",
		"gated_residual_cnn": "gated_cnn",
		"efficient_residual": "dc_ae",
	}.get(encoder_type, encoder_type)

def canonical_decoder_type(decoder_type):
	return {
		"basic_cnn": "cnn",
		"efficient_residual": "dc_ae",
	}.get(decoder_type, decoder_type)

def build_encoder(encoder_type, bottleneck_channels, encoder_blocks, hidden_channels,
				  downsample_stages, encoder_layers=2, encoder_window_size=8):
	encoder_type = canonical_encoder_type(encoder_type)
	encoder_factories = {
		"window_transformer": lambda: HierarchicalWindowAttentionEncoder(
				in_channels=16, feature_channels=bottleneck_channels,
				num_layers=encoder_layers, window_size=encoder_window_size,
		),
		"residual_conv_ffn": lambda: ResidualConvFFNEncoder(
			in_channels=16,
				feature_channels=bottleneck_channels,
				num_blocks=encoder_blocks,
				hidden_channels=hidden_channels,
				downsample_stages=downsample_stages,
		),
		"gated_residual_conv_ffn": lambda: GatedResidualConvFFNEncoder(
			in_channels=16,
				feature_channels=bottleneck_channels,
				num_blocks=encoder_blocks,
				hidden_channels=hidden_channels,
				downsample_stages=downsample_stages,
		),
		"cnn": lambda: BasicCNNEncoder(
			in_channels=16, feature_channels=bottleneck_channels,
		),
		"gated_cnn": lambda: GatedCNNEncoder(
			in_channels=16, feature_channels=bottleneck_channels,
		),
		"dc_ae": lambda: DCAEEncoder(
			in_channels=16, feature_channels=bottleneck_channels,
		),
	}
	try:
		return encoder_factories[encoder_type]()
	except KeyError as error:
		raise ValueError(f"Unknown encoder type: {encoder_type}") from error

def build_decoder(decoder_type, bottleneck_channels, decoder_blocks, hidden_channels,
				  downsample_stages, decoder_layers=2, decoder_window_size=8):
	decoder_type = canonical_decoder_type(decoder_type)
	decoder_factories = {
		"window_transformer": lambda: HierarchicalWindowAttentionDecoder(
				feature_channels=bottleneck_channels,
				num_layers=decoder_layers, window_size=decoder_window_size,
		),
		"cnn": lambda: BasicCNNDecoder(feature_channels=bottleneck_channels),
		"residual_conv_ffn": lambda: ResidualConvFFNDecoder(
			feature_channels=bottleneck_channels,
				hidden_channels=hidden_channels,
				num_blocks=decoder_blocks,
				downsample_stages=downsample_stages,
		),
		"gated_residual_conv_ffn": lambda: GatedResidualConvFFNDecoder(
			feature_channels=bottleneck_channels,
				hidden_channels=hidden_channels,
				num_blocks=decoder_blocks,
				downsample_stages=downsample_stages,
		),
		"dc_ae": lambda: DCAEDecoder(feature_channels=bottleneck_channels),
	}
	try:
		return decoder_factories[decoder_type]()
	except KeyError as error:
		raise ValueError(f"Unknown decoder type: {decoder_type}") from error

class ImageAE(nn.Module):
	def __init__(self, latent_channels=48, encoder_type="window_transformer",
			 decoder_type="window_transformer", bottleneck_channels=256,
			 encoder_blocks=1, decoder_blocks=1, vae=False, hidden_channels=None,
			 downsample_stages=3, encoder_layers=2, encoder_window_size=8,
			 decoder_layers=2):
		super().__init__()
		encoder_type = canonical_encoder_type(encoder_type)
		decoder_type = canonical_decoder_type(decoder_type)
		if latent_channels <= 0:
			raise ValueError("latent_channels must be positive")
		if bottleneck_channels <= 0 or bottleneck_channels % 4 != 0:
			raise ValueError("bottleneck_channels must be a positive multiple of 4")
		if bottleneck_channels % 16 != 0:
				raise ValueError(
				"bottleneck_channels must be divisible by 16 for decoder output stages"
			)
		if (encoder_type == "window_transformer" or decoder_type == "window_transformer") \
				and bottleneck_channels % 32 != 0:
			raise ValueError(
				"bottleneck_channels must be divisible by 32 for hierarchical window stages"
			)
		if downsample_stages <= 0 or downsample_stages > 5:
			raise ValueError("downsample_stages must be between 1 and 5")
		if bottleneck_channels % (2 ** (downsample_stages - 1)) != 0:
			raise ValueError("bottleneck_channels must support the selected downsample_stages")
		if downsample_stages != 3 and encoder_type not in (
			"residual_conv_ffn", "gated_residual_conv_ffn",
		):
			raise ValueError(
				"downsample_stages != 3 is supported only by residual Conv FFN encoders"
			)
		if downsample_stages != 3 and decoder_type not in (
			"residual_conv_ffn", "gated_residual_conv_ffn",
		):
			raise ValueError(
				"downsample_stages != 3 is supported only by residual Conv FFN decoders"
			)
		if encoder_blocks <= 0 or decoder_blocks <= 0:
			raise ValueError("encoder_blocks and decoder_blocks must be positive")
		if encoder_layers <= 0:
			raise ValueError("encoder_layers must be positive")
		if decoder_layers <= 0:
			raise ValueError("decoder_layers must be positive")
		if encoder_window_size <= 0:
			raise ValueError("encoder_window_size must be positive")
		self.latent_channels = latent_channels
		self.downsample_stages = downsample_stages
		self.vae = vae
		self.input_adapter = GatedConv2d(3, 16, kernel_size=3, padding=1)
		self.encoder = build_encoder(
			encoder_type, bottleneck_channels, encoder_blocks, hidden_channels,
			downsample_stages, encoder_layers, encoder_window_size,
		)
		self.latent_mu = nn.Conv2d(bottleneck_channels, latent_channels, kernel_size=1)
		if self.vae:
			self.latent_logvar = nn.Conv2d(
				bottleneck_channels, latent_channels, kernel_size=1,
			)
		self.latent_to_bottleneck = nn.Conv2d(
			latent_channels, bottleneck_channels, kernel_size=1,
		)
		self.decoder = build_decoder(
			decoder_type, bottleneck_channels, decoder_blocks, hidden_channels,
			downsample_stages, decoder_layers, encoder_window_size,
		)
		self.reconstruction_head = nn.Sequential(
			# Final decoder feature map -> RGB reconstruction.
			RMSNorm2d(16),
			nn.Conv2d(16, 3, kernel_size=3, padding=1),
			nn.Sigmoid(),
		)

	def _validate_input(self, x):
		if x.ndim != 4 or x.shape[1] != 3:
			raise ValueError(
				"ImageAE expects RGB NCHW input with shape (B, 3, H, W), "
				f"got {tuple(x.shape)}"
			)
		divisor = 2 ** self.downsample_stages
		height, width = x.shape[-2:]
		if height < divisor or width < divisor or height % divisor or width % divisor:
			raise ValueError(
				"ImageAE input height and width must be divisible by "
				f"2**downsample_stages ({divisor}), got {height}x{width}"
			)

	def encode(self, x, return_stats=False):
		self._validate_input(x)
		hidden = self.encoder(self.input_adapter(x))
		mu = self.latent_mu(hidden)
		if not self.vae:
			return mu
		logvar = self.latent_logvar(hidden)
		if self.training:
			latent = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
		else:
			latent = mu
		if return_stats:
			return latent, mu, logvar
		return latent

	def decode(self, z):
		hidden = self.latent_to_bottleneck(z)
		hidden = self.decoder(hidden)
		return self.reconstruction_head(hidden)

	def encode_mean(self, x):
		"""Encode without VAE sampling for deterministic consistency targets."""
		self._validate_input(x)
		hidden = self.encoder(self.input_adapter(x))
		return self.latent_mu(hidden)

	def forward(self, x):
		if self.vae:
			latent, mu, logvar = self.encode(x, return_stats=True)
			return self.decode(latent), latent, mu, logvar
		latent = self.encode(x)
		return self.decode(latent), latent


def compute_kl_loss(mu, logvar):
	"""Return KL divergence averaged per latent element and per sample."""
	return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

def compute_reconstruction_loss(reconstruction, target, loss_name, huber_beta):
	return compute_reconstruction_losses(reconstruction, target, huber_beta)[loss_name]

def compute_reconstruction_losses(reconstruction, target, huber_beta):
	"""Return reconstruction errors averaged over batch, channels, and pixels."""
	return {
		"MSE": F.mse_loss(reconstruction, target, reduction="mean"),
		"L1": F.l1_loss(reconstruction, target, reduction="mean"),
		"Huber": F.smooth_l1_loss(
			reconstruction, target, beta=huber_beta, reduction="mean",
		),
	}

def format_metric_value(value):
	"""Format small metrics without hiding them through fixed-point rounding."""
	value = float(value)
	if value == 0.0 or abs(value) >= 1e-4:
		return f"{value:.6f}"
	return f"{value:.3e}"

def format_loss_with_contribution(value, total, weight=1.0):
	if not weight:
		return f"{format_metric_value(value)} (-)"
	weighted_value = value * weight
	percentage = 100.0 * weighted_value / max(abs(total), 1e-12)
	return f"{format_metric_value(value)} ({percentage:5.1f}%)"

def format_loss_breakdown(total, components):
	return " ".join(
		f"{name}={format_loss_with_contribution(value, total, weight)}"
		for name, (value, weight) in components.items()
	)

def format_loss_comparison(name, train_value, train_total, test_value, test_total,
								 weight=1.0):
	train_text = format_loss_with_contribution(train_value, train_total, weight)
	test_text = format_loss_with_contribution(test_value, test_total, weight)
	return f"  {name:<16} {train_text:>16}  {test_text:>16}"

def format_metric_comparison(name, train_value, test_value):
	return (
		f"  {name:<16} {format_metric_value(train_value):>16}  "
		f"{format_metric_value(test_value):>16}"
	)

def compute_wavelet_loss(reconstruction, target, levels=1):
	"""Compare multi-scale 2D Haar wavelet coefficients."""
	if levels <= 0 or int(levels) != levels:
		raise ValueError("wavelet levels must be a positive integer")
	if reconstruction.ndim != 4 or target.ndim != 4:
		raise ValueError("wavelet inputs must have shape (batch, channels, height, width)")
	if reconstruction.shape != target.shape:
		raise ValueError(
		"wavelet inputs must have identical shapes, "
		f"got {tuple(reconstruction.shape)} and {tuple(target.shape)}"
	)
	min_spatial = min(int(target.shape[-2]), int(target.shape[-1]))
	if min_spatial < 2:
		raise ValueError("wavelet inputs require height and width >= 2")
	max_levels = min_spatial.bit_length() - 1
	if levels > max_levels:
		raise ValueError(
		f"wavelet levels={levels} exceed the spatial limit {max_levels} "
		f"for input size {tuple(target.shape[-2:])}"
	)
	channels = target.size(1)
	sqrt_two_inv = 2 ** -0.5
	haar_filters = torch.tensor([
		[1, 1, 1, 1],
		[1, -1, 1, -1],
		[1, 1, -1, -1],
		[1, -1, -1, 1],
	], device=target.device, dtype=target.dtype) * (sqrt_two_inv ** 2)
	filters = haar_filters.reshape(4, 1, 2, 2).repeat(channels, 1, 1, 1)
	current_reconstruction = reconstruction
	current_target = target
	loss = torch.zeros((), device=target.device)
	for _ in range(levels):
		coefficients_reconstruction = F.conv2d(
			current_reconstruction, filters, stride=2, groups=channels,
		)
		coefficients_target = F.conv2d(
			current_target, filters, stride=2, groups=channels,
		)
		loss = loss + F.l1_loss(
			coefficients_reconstruction, coefficients_target, reduction="mean",
		)
		current_reconstruction = coefficients_reconstruction[:, 0::4]
		current_target = coefficients_target[:, 0::4]
	return loss / levels

def compute_latent_variance_loss(z):
	"""Keep each latent channel's batch/spatial standard deviation near one."""
	values = z.permute(1, 0, 2, 3).reshape(z.size(1), -1)
	standard_deviation = torch.sqrt(values.var(dim=1, unbiased=False) + 1e-4)
	return F.relu(1.0 - standard_deviation).pow(2).mean()

def compute_channel_decorrelation_loss(z):
	"""Penalize correlations between different latent channels."""
	if z.size(1) < 2:
		# There are no off-diagonal channel pairs to decorrelate. Keep a
		# differentiable zero so this term remains safe in a composed loss.
		return z.sum() * 0.0
	values = z.permute(1, 0, 2, 3).reshape(z.size(1), -1)
	values = values - values.mean(dim=1, keepdim=True)
	standard_deviation = torch.sqrt(values.pow(2).mean(dim=1) + 1e-4)
	correlation = (values @ values.t()) / values.size(1)
	correlation = correlation / (standard_deviation[:, None] * standard_deviation[None, :])
	off_diagonal = ~torch.eye(z.size(1), dtype=torch.bool, device=z.device)
	return correlation[off_diagonal].pow(2).mean()

CORRUPTION_PROBABILITY = 0.5

def corrupt_encoder_input(images, noise_std=0.0, blur_sigma=0.0, bit_depth=0):
	"""Apply optional corruption while keeping images as clean reconstruction targets."""
	# 6-bit quantization is a recommended starting point for bit-depth corruption.
	if blur_sigma > 0:
		radius = max(1, int(math.ceil(3.0 * blur_sigma)))
		blurred = transforms.functional.gaussian_blur(
			images, kernel_size=2 * radius + 1, sigma=blur_sigma,
		)
		mask = torch.rand(images.size(0), 1, 1, 1, device=images.device)
		images = torch.where(mask < CORRUPTION_PROBABILITY, blurred, images)
	if noise_std > 0:
		noisy = (images + noise_std * torch.randn_like(images)).clamp(0.0, 1.0)
		mask = torch.rand(images.size(0), 1, 1, 1, device=images.device)
		images = torch.where(mask < CORRUPTION_PROBABILITY, noisy, images)
	if bit_depth > 0:
		levels = 2 ** bit_depth
		quantized = torch.round(images * (levels - 1)) / (levels - 1)
		mask = torch.rand(images.size(0), 1, 1, 1, device=images.device)
		images = torch.where(mask < CORRUPTION_PROBABILITY, quantized, images)
	return images

def compute_cycle_consistency_losses(model, reconstruction, target, z1,
									 loss_name, huber_beta, z1_weight, z2_weight,
									 compute_latent, compute_image):
	z2 = model.encode_mean(reconstruction)
	latent_cycle_loss = torch.zeros((), device=target.device)
	image_cycle_loss = torch.zeros((), device=target.device)
	if compute_latent:
		z1_normalized = F.normalize(z1.flatten(1), dim=1).reshape_as(z1)
		z2_normalized = F.normalize(z2.flatten(1), dim=1).reshape_as(z2)
		loss_z1 = F.smooth_l1_loss(z2_normalized, z1_normalized.detach())
		loss_z2 = F.smooth_l1_loss(z2_normalized.detach(), z1_normalized)
		weight_sum = z1_weight + z2_weight
		latent_cycle_loss = (z1_weight * loss_z1 + z2_weight * loss_z2) / weight_sum
	if compute_image:
		second_reconstruction = model.decode(z2)
		image_cycle_loss = compute_reconstruction_loss(
			second_reconstruction, target, loss_name, huber_beta,
		)
	return latent_cycle_loss, image_cycle_loss

def evaluate_saved_model(model, loader, device, loss_name, huber_beta, kl_weight,
					 latent_cycle_consistency=False, latent_cycle_weight=0.0,
					 latent_cycle_z1_weight=0.8, latent_cycle_z2_weight=0.2,
					 image_cycle_consistency=False, image_cycle_weight=0.01,
					 wavelet_loss=False, wavelet_loss_weight=0.05, wavelet_levels=1,
					 latent_variance_loss=False, latent_variance_weight=0.01,
					 channel_decorrelation_loss=False, channel_decorrelation_weight=0.001):
	model.eval()
	total_loss = 0.0
	total_reconstruction = 0.0
	total_kl = 0.0
	total_latent_cycle = 0.0
	total_image_cycle = 0.0
	total_wavelet = 0.0
	total_latent_variance = 0.0
	total_channel_decorrelation = 0.0
	total_images = 0
	with torch.no_grad():
		for images, _ in loader:
			images = images.to(device, non_blocking=True)
			outputs = model(images)
			if model.vae:
				reconstructions, _, mu, logvar = outputs
				z1 = mu
				kl_loss = compute_kl_loss(mu, logvar)
			else:
				reconstructions, z1 = outputs
				kl_loss = torch.zeros((), device=images.device)
			reconstruction_losses = compute_reconstruction_losses(
				reconstructions, images, huber_beta
			)
			reconstruction_loss = reconstruction_losses[loss_name]
			if latent_cycle_consistency or image_cycle_consistency:
				latent_cycle_loss, image_cycle_loss = compute_cycle_consistency_losses(
					model, reconstructions, images, z1, loss_name, huber_beta,
					latent_cycle_z1_weight, latent_cycle_z2_weight,
					latent_cycle_consistency, image_cycle_consistency,
				)
				if not latent_cycle_consistency:
					latent_cycle_loss = torch.zeros((), device=images.device)
				if not image_cycle_consistency:
					image_cycle_loss = torch.zeros((), device=images.device)
			else:
				latent_cycle_loss = torch.zeros((), device=images.device)
				image_cycle_loss = torch.zeros((), device=images.device)
			wavelet_metric = compute_wavelet_loss(
				reconstructions, images, levels=wavelet_levels,
			)
			wavelet_loss_value = wavelet_metric if wavelet_loss else torch.zeros(
				(), device=images.device,
			)
			if latent_variance_loss:
				latent_variance_loss_value = compute_latent_variance_loss(z1)
			else:
				latent_variance_loss_value = torch.zeros((), device=images.device)
			if channel_decorrelation_loss:
				channel_decorrelation_loss_value = compute_channel_decorrelation_loss(z1)
			else:
				channel_decorrelation_loss_value = torch.zeros((), device=images.device)
			loss = (
				reconstruction_loss + kl_weight * kl_loss
				+ latent_cycle_weight * latent_cycle_loss
				+ image_cycle_weight * image_cycle_loss
				+ wavelet_loss_weight * wavelet_loss_value
				+ latent_variance_weight * latent_variance_loss_value
				+ channel_decorrelation_weight * channel_decorrelation_loss_value
			)
			total_loss += loss.item() * images.size(0)
			total_reconstruction += reconstruction_loss.item() * images.size(0)
			total_kl += kl_loss.item() * images.size(0)
			total_latent_cycle += latent_cycle_loss.item() * images.size(0)
			total_image_cycle += image_cycle_loss.item() * images.size(0)
			total_wavelet += wavelet_metric.item() * images.size(0)
			total_latent_variance += latent_variance_loss_value.item() * images.size(0)
			total_channel_decorrelation += channel_decorrelation_loss_value.item() * images.size(0)
			total_images += images.size(0)
	return (
		total_loss / total_images,
		total_reconstruction / total_images,
		total_kl / total_images,
		total_latent_cycle / total_images,
		total_image_cycle / total_images,
		total_wavelet / total_images,
		total_latent_variance / total_images,
		total_channel_decorrelation / total_images,
	)


def select_hf_split(dataset, requested, fallbacks, role):
	"""Select a split while handling mirrors with only one available split."""
	if not hasattr(dataset, "keys"):
		return dataset, requested
	available = list(dataset.keys())
	for split in (requested, *fallbacks):
		if split and split in dataset:
			return dataset[split], split
	if len(available) == 1:
		split = available[0]
		print(f"{role} split={requested!r} unavailable; using {split!r}")
		return dataset[split], split
	raise ValueError(
		f"Unable to select {role} split={requested!r}; available={available}"
	)


def load_hf_image_datasets(args):
	try:
		from datasets import load_dataset
	except ImportError as error:
		raise RuntimeError(
			"Hugging Face Datasets is required for image datasets. "
			"Install it with: pip install datasets"
		) from error
	dataset_name = DATASET_IDS[args.dataset]
	loaded = load_dataset(dataset_name, cache_dir=args.hf_cache_dir)
	train_data, train_split = select_hf_split(
		loaded, args.dataset_split, ("train", "test"), "train",
	)
	validation_request = args.validation_split or "validation"
	validation_data, validation_split = select_hf_split(
		loaded, validation_request, ("test", args.dataset_split), "validation",
	)
	if (train_split == validation_split or train_data is validation_data) and len(train_data) > 1:
		# Some Flickr30k mirrors expose all 30k images under one split only.
		# Keep validation disjoint and deterministic in that case.
		split_index = max(1, min(len(train_data) - 1, int(len(train_data) * 0.95)))
		full_data = train_data
		train_data = full_data.select(range(split_index))
		validation_data = full_data.select(range(split_index, len(full_data)))
		train_split = f"{train_split}[0:{split_index}]"
		validation_split = f"{validation_split}[{split_index}:]"
	if args.dataset == "imagenet1k":
		if args.imagenet_train_samples > 0:
			train_data = train_data.select(
				range(min(args.imagenet_train_samples, len(train_data)))
			)
		if args.imagenet_val_samples > 0:
			validation_data = validation_data.select(
				range(min(args.imagenet_val_samples, len(validation_data)))
			)
	train_dataset = HuggingFaceImageDataset(
		train_data, image_size=args.image_size, bucket_step=args.bucket_step, train=True,
	)
	validation_dataset = HuggingFaceImageDataset(
		validation_data, image_size=args.image_size, bucket_step=args.bucket_step, train=False,
	)
	return train_dataset, validation_dataset, train_split, validation_split


def make_image_loader(
	dataset, batch_size, num_workers, device, *, shuffle, drop_last, seed=None,
	stream=0,
):
	loader_kwargs = build_dataloader_options(
		num_workers=num_workers,
		pin_memory=device.type == "cuda",
		seed=seed,
		stream=stream,
	)
	if getattr(dataset, "bucket_ids", None) is None:
		sampler = ResumableRandomSampler(dataset, seed=seed) if shuffle else None
		return DataLoader(
			dataset,
			batch_size=batch_size,
			shuffle=False if sampler is not None else shuffle,
			sampler=sampler,
			**loader_kwargs,
		)
	return DataLoader(
		dataset,
		batch_sampler=AspectRatioBatchSampler(
			dataset.bucket_ids, batch_size, drop_last=drop_last, shuffle=shuffle,
			seed=seed,
		),
		**loader_kwargs,
	)

def parse_args(argv=None, *, adapter_only=False):
	parser = argparse.ArgumentParser(description="Train an image autoencoder or VAE")
	parser.add_argument(
		"--dataset", choices=["flickr30k", "cifar10", "imagenet1k", "mini-imagenet"], default=None,
		help="Dataset to use. Default: flickr30k.",
	)
	parser.add_argument("--data-dir", default="../cifar10/data")
	parser.add_argument(
		"--image-size", type=int, default=None,
		help="Reference bucket area size. Default: 256 for image datasets, 32 for CIFAR-10.",
	)
	parser.add_argument(
		"--hf-cache-dir", default=None,
		help="Hugging Face Datasets cache directory.",
	)
	parser.add_argument("--dataset-split", default="train", help="Training split for Hugging Face datasets.")
	parser.add_argument(
		"--validation-split", default=None,
		help="Validation split. Defaults to validation, test, or the only available split.",
	)
	parser.add_argument(
		"--imagenet-train-samples", type=int, default=0,
		help="Number of ImageNet training samples to use. 0 uses all samples.",
	)
	parser.add_argument(
		"--imagenet-val-samples", type=int, default=0,
		help="Number of ImageNet validation samples to use. 0 uses all samples.",
	)
	parser.add_argument(
		"--cifar10-train-samples", type=int, default=0,
		help="Number of CIFAR-10 training samples to use. 0 uses all samples.",
	)
	parser.add_argument(
		"--cifar10-val-samples", type=int, default=0,
		help="Number of CIFAR-10 validation samples to use. 0 uses all samples.",
	)
	parser.add_argument("--output-dir", default="output")
	parser.add_argument(
		"--run-name", default=None,
		help="Optional human-readable name included in the run directory.",
	)
	add_device_argument(parser)
	parser.add_argument(
		"--dry-run", action="store_true",
		help="共通設定を検証し、データやモデルを読み込まずに終了",
	)
	parser.add_argument(
		"--validate-only", action="store_true",
		help="データとモデルの構成を検証し、学習せずに終了",
	)
	parser.add_argument(
		"--log-dir", default=None,
		help="Optional legacy TensorBoard root. Default: the current run's tensorboard directory.",
	)
	parser.add_argument("--epochs", type=int, default=100)
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument(
		"--input-noise-std", type=float, default=0.0,
		help="Gaussian noise std applied before encoding. 0 disables it.",
	)
	parser.add_argument(
		"--input-blur-sigma", type=float, default=0.0,
		help="Gaussian blur sigma applied before encoding. 0 disables it.",
	)
	parser.add_argument(
		"--input-bit-depth", type=int, default=0,
		help="Bit depth applied before encoding. Default: disabled; 6 is recommended; 0 disables it.",
	)
	parser.add_argument(
		"--downsample-stages", type=int, default=None,
		help="Number of spatial downsampling stages. Default: 3.",
	)
	parser.add_argument(
		"--bucket-step", type=int, default=32,
		help="Height/width alignment for aspect-ratio buckets. Default: 32.",
	)
	parser.add_argument(
		"--latent-channels", "--latent-dim", dest="latent_channels", type=int, default=None,
		help="Number of channels in the spatial latent. Spatial size depends on input size and downsample stages.",
	)
	parser.add_argument(
		"--bottleneck-channels", "--feature-channels", dest="bottleneck_channels",
		type=int, default=None,
		help="Number of channels immediately before and after the latent projection. Default: 256.",
	)
	parser.add_argument(
		"--hidden-channels", type=int, default=None,
		help="Hidden channels in Conv FFN blocks. Default: channels * 2.",
	)
	parser.add_argument("--encoder", choices=["window_transformer", "residual_conv_ffn", "gated_residual_conv_ffn", "basic_cnn", "gated_residual_cnn", "efficient_residual", "cnn", "gated_cnn", "dc_ae"], default=None,
			help="Encoder architecture. Default: window_transformer.")
	parser.add_argument(
		"--encoder-layers", type=int, default=None,
		help="Number of shifted-window blocks per encoder stage. Default: 2.",
	)
	parser.add_argument(
		"--encoder-window-size", type=int, default=None,
		help="Local attention window size in encoder tokens. Default: 8.",
	)
	parser.add_argument(
		"--encoder-blocks-per-stage", "--encoder-blocks", "--residual-layers",
		dest="encoder_blocks", type=int, default=None,
		help="Number of Conv FFN residual blocks per encoder stage. Default: 1.",
	)
	parser.add_argument(
		"--decoder-blocks-per-stage", "--decoder-blocks", dest="decoder_blocks", type=int, default=None,
		help="Number of Conv FFN residual blocks per decoder stage. Default: 1.",
	)
	parser.add_argument(
		"--decoder-layers", type=int, default=None,
		help="Number of shifted-window blocks per decoder stage. Default: 2.",
	)
	parser.add_argument("--decoder", choices=["window_transformer", "basic_cnn", "residual_conv_ffn", "gated_residual_conv_ffn", "efficient_residual", "cnn", "dc_ae"], default=None,
			help="Decoder architecture. Default: window_transformer.")
	parser.add_argument("--lr", type=float, default=2e-4)
	parser.add_argument("--loss-fn", "--reconstruction-loss", dest="loss_fn",
				choices=["MSE", "Huber", "L1"], default="MSE",
				help="Reconstruction loss function. Default: MSE.")
	parser.add_argument("--huber-beta", type=float, default=0.05,
				help="Huber quadratic-region threshold when using Huber loss.")
	add_optimizer_argument(
		parser, default="AdamW", include_apollo_mini=True,
		include_came_lrsf=True, include_came_sf=True,
		include_apollo_came_lrsf=True,
	)
	parser.add_argument(
		"--came-lrsf-rank", type=int, default=4,
		help="Low-rank Schedule-Free delta rank for CAME-LRSF. Default: 4.",
	)
	parser.add_argument(
		"--came-lrsf-beta1", type=float, default=0.9,
		help="Schedule-Free interpolation beta for CAME-LRSF. Default: 0.9.",
	)
	parser.add_argument(
		"--came-lrsf-warmup-steps", type=int, default=0,
		help="Linear learning-rate warmup steps for CAME-LRSF. Default: 0.",
	)
	parser.add_argument(
		"--came-lrsf-r", type=float, default=0.0,
		help="Schedule-Free weighting exponent for CAME-LRSF. Default: 0.",
	)
	parser.add_argument(
		"--came-lrsf-weight-lr-power", type=float, default=2.0,
		help="Learning-rate weighting power for CAME-LRSF. Default: 2.",
	)
	parser.add_argument(
		"--came-lrsf-seed", type=int, default=0,
		help="Projection seed for CAME-LRSF. Default: 0.",
	)
	parser.add_argument(
		"--came-lrsf-refresh-mode", choices=("none", "hard", "smooth"),
		default="none",
		help="Schedule-Free delta projection refresh mode. Default: none.",
	)
	parser.add_argument(
		"--came-lrsf-refresh-interval", type=int, default=200,
		help="Steps between LRSF delta projection refreshes. Default: 200.",
	)
	parser.add_argument(
		"--came-lrsf-refresh-window", type=int, default=200,
		help="Steps used for smooth LRSF delta projection transition. Default: 200.",
	)
	parser.add_argument(
		"--came-lrsf-refresh-mix",
		choices=("linear", "smoothstep", "stochastic", "ema"),
		default="smoothstep",
		help="Smooth refresh mixing curve. Default: smoothstep.",
	)
	parser.add_argument(
		"--came-lrsf-orthogonal-refresh-rate", type=float, default=0.0,
		help=(
			"Per-step tangent-space rotation rate for the LRSF projection. "
			"Zero disables it. Default: 0."
		),
	)
	parser.add_argument(
		"--came-lrsf-orthogonal-refresh-direction",
		choices=("random", "loss_directed", "loss_lowering"), default="random",
		help="Direction for LRSF orthogonal refresh. Default: random.",
	)
	parser.add_argument(
		"--came-lrsf-orthogonal-refresh-signal",
		choices=("gradient", "effective_update"), default="gradient",
		help=(
			"Signal for LRSF loss-directed refresh: gradient or effective_update. "
			"Default: gradient."
		),
	)
	parser.add_argument(
		"--apollo-rank", type=int, default=8,
		help="Auxiliary rank for APOLLO matrix parameters. Default: 8.",
	)
	parser.add_argument(
		"--apollo-scale", type=float, default=1.0,
		help="Gradient scale for APOLLO. Default: 1.0.",
	)
	parser.add_argument(
		"--apollo-update-proj-gap", type=int, default=200,
		help="Refresh interval for APOLLO projection bases. Default: 200.",
	)
	parser.add_argument(
		"--apollo-projection-refresh-mode", choices=("none", "hard", "smooth"),
		default="hard",
		help="APOLLO projection refresh mode: none, hard, or smooth. Default: hard.",
	)
	parser.add_argument(
		"--apollo-projection-refresh-window", type=int, default=200,
		help="APOLLO smooth refresh window in optimizer steps. Default: 200.",
	)
	parser.add_argument(
		"--apollo-projection-refresh-mix",
		choices=("linear", "smoothstep", "stochastic", "ema"),
		default="smoothstep",
		help="APOLLO smooth refresh mixing curve. Default: smoothstep.",
	)
	parser.add_argument(
		"--apollo-projection-refresh-state", choices=("reset", "transport"),
		default="reset",
		help="APOLLO moment handling at refresh: reset or overlap transport. Default: reset.",
	)
	parser.add_argument(
		"--apollo-orthogonal-refresh-rate", type=float, default=0.0,
		help=(
			"Per-step tangent-space rotation rate for the APOLLO projection. "
			"Zero disables it. Default: 0."
		),
	)
	parser.add_argument(
		"--apollo-orthogonal-refresh-direction",
		choices=("random", "loss_directed"), default="random",
		help="Direction for APOLLO orthogonal refresh. Default: random.",
	)
	parser.add_argument(
		"--apollo-scale-front", action=argparse.BooleanOptionalAction,
		default=False,
		help="Apply APOLLO scale before the norm-growth limiter. Default: disabled.",
	)
	parser.add_argument(
		"--apollo-disable-norm-growth-limiter",
		action=argparse.BooleanOptionalAction, default=True,
		help=(
			"Disable APOLLO's norm-growth limiter. This is the default; pass "
			"--no-apollo-disable-norm-growth-limiter to enable the limiter."
		),
	)
	parser.add_argument(
		"--apollo-norm-growth-rate", type=float, default=1.01,
		help="Maximum consecutive APOLLO update-norm growth. Default: 1.01.",
	)
	parser.add_argument(
		"--apollo-fallback", choices=("came", "sgd", "adamw-sf"), default="adamw-sf",
		help="Fallback optimizer for APOLLO 1D parameters. Default: adamw-sf.",
	)
	parser.add_argument(
		"--apollo-matrix-fallback", choices=("apollo", "came", "auto", "adamw-sf", "auto-sf"), default="auto-sf",
		help="Matrix fallback: auto-sf selects by AdamW-SF state size; default: auto-sf.",
	)
	parser.add_argument(
		"--apollo-came-backend", choices=("auto", "torch", "triton"), default="torch",
		help="Backend for APOLLO-CAME updates. Default: torch.",
	)
	add_lr_scheduler_arguments(
		parser, default="constant", include_force_scheduler=True,
	)
	parser.add_argument("--num-workers", type=int, default=4,
					help="DataLoaderのワーカープロセス数。デフォルト: 4")
	parser.add_argument("--gc-interval", type=int, default=100,
					help="NバッチごとにPython GCを実行。0で無効")
	parser.add_argument("--empty-cache-interval", type=int, default=0,
					help="NバッチごとにCUDAキャッシュを解放。0で無効")
	parser.add_argument(
		"--resume", default=None,
		help=(
			"Load model weights. If a sibling .resume.pt file exists, also restore "
			"optimizer, scheduler, epoch, and RNG state; otherwise use weights-only resume."
		),
	)
	parser.add_argument(
		"--init-checkpoint", default=None,
		help="Initialize shared input_adapter/latent_mu/latent_to_bottleneck/reconstruction_head weights from a checkpoint for a new run.",
	)
	if adapter_only:
		add_adapter_arguments(parser)
	parser.add_argument(
		"--init-freeze-epochs", type=int, default=1,
		help="Freeze transferred shared parameters for this many initial epochs. Default: 1.",
	)
	parser.add_argument(
		"--resume-epoch", type=int, default=0,
		help="Completed epochs in a checkpoint without epoch metadata. Default: 0.",
	)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument(
		"--vae", action=argparse.BooleanOptionalAction, default=False,
		help="Enable variational latent sampling and KL-divergence regularization.",
	)
	parser.add_argument(
		"--kl-weight", type=float, default=1e-4,
		help="Weight of the VAE KL-divergence loss. Default: 1e-4.",
	)
	parser.add_argument(
		"--latent-cycle-consistency", action=argparse.BooleanOptionalAction, default=False,
		help="Add latent cycle consistency between the original and re-encoded latents.",
	)
	parser.add_argument(
		"--latent-cycle-weight", type=float, default=0.05,
		help="Overall latent cycle consistency loss weight. Default: 0.05.",
	)
	parser.add_argument(
		"--latent-cycle-z1-weight", type=float, default=0.8,
		help="Gradient weight for matching re-encoded z2 to original z1. Default: 0.8.",
	)
	parser.add_argument(
		"--latent-cycle-z2-weight", type=float, default=0.2,
		help="Gradient weight for matching original z1 to re-encoded z2. Default: 0.2.",
	)
	parser.add_argument(
		"--image-cycle-consistency", action=argparse.BooleanOptionalAction, default=False,
		help="Add image cycle consistency between the original and twice-decoded images.",
	)
	parser.add_argument(
		"--image-cycle-weight", type=float, default=0.01,
		help="Image cycle consistency loss weight. Default: 0.01.",
	)
	parser.add_argument(
		"--wavelet-loss", action=argparse.BooleanOptionalAction, default=False,
		help="Add multi-scale Haar wavelet loss to the reconstruction loss.",
	)
	parser.add_argument(
		"--wavelet-loss-weight", type=float, default=0.05,
		help="Wavelet loss weight. Default: 0.05.",
	)
	parser.add_argument(
		"--wavelet-levels", type=int, default=1,
		help="Number of Haar wavelet decomposition levels. Default: 1.",
	)
	parser.add_argument(
		"--latent-variance-loss", action=argparse.BooleanOptionalAction, default=False,
		help="Add latent variance regularization.",
	)
	parser.add_argument(
		"--latent-variance-weight", type=float, default=0.01,
		help="Latent variance loss weight. Default: 0.01.",
	)
	parser.add_argument(
		"--channel-decorrelation-loss", action=argparse.BooleanOptionalAction, default=False,
		help="Add latent channel decorrelation regularization.",
	)
	parser.add_argument(
		"--channel-decorrelation-weight", type=float, default=0.001,
		help="Channel decorrelation loss weight. Default: 0.001.",
	)
	args = parser.parse_args(argv)
	if not adapter_only:
		# Keep checkpoint metadata fields stable without exposing adapter options
		# from the ordinary full-model training entrypoint.
		args.lora_base_checkpoint = None
		args.lora_rank = 0
		args.adapter = "none"
		args.lora_alpha = None
		args.lora_dropout = 0.0
		args.lora_target = None
		args.adapter_init = "identity"
	return args

def main(argv=None, *, adapter_only=False):
	args = parse_args(argv, adapter_only=adapter_only)
	cli_argv = sys.argv[1:] if argv is None else argv
	script_name = "image_ae.train_adapter" if adapter_only else "image_ae.train"
	if args.dry_run and args.validate_only:
		raise ValueError("--dry-run and --validate-only cannot be used together")
	if args.resume and args.init_checkpoint:
		raise ValueError("--resume and --init-checkpoint cannot be used together")
	if args.init_freeze_epochs < 0:
		raise ValueError("--init-freeze-epochs must be >= 0")
	resume_path = None
	init_path = None
	if args.init_checkpoint:
		init_path = args.init_checkpoint
		if not os.path.isfile(init_path):
			candidate = os.path.join(args.output_dir, init_path)
			if os.path.isfile(candidate):
				init_path = candidate
		if not os.path.isfile(init_path):
			raise FileNotFoundError(f"Initialization checkpoint not found: {args.init_checkpoint}")
	lora_base_path = None
	if args.lora_base_checkpoint:
		lora_base_path = args.lora_base_checkpoint
		if not os.path.isfile(lora_base_path):
			candidate = os.path.join(args.output_dir, lora_base_path)
			if os.path.isfile(candidate):
				lora_base_path = candidate
		if not os.path.isfile(lora_base_path):
			raise FileNotFoundError(
				f"LoRA base checkpoint not found: {args.lora_base_checkpoint}"
			)
	if args.resume:
		resume_path = args.resume
		if not os.path.isfile(resume_path):
			candidate = os.path.join(args.output_dir, resume_path)
			if os.path.isfile(candidate):
				resume_path = candidate
		if not os.path.isfile(resume_path):
			raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
		resume_config = resume_network_config(resume_path)
		if resume_config:
			network_overridden_keys = []
			apply_saved_config(
				args,
				resume_config,
				argv=cli_argv,
				keys=(
					"encoder", "decoder", "latent_channels", "bottleneck_channels",
					"hidden_channels", "encoder_blocks", "decoder_blocks",
					"encoder_layers", "encoder_window_size", "decoder_layers",
					"downsample_stages",
					"dataset", "image_size", "bucket_step",
				),
				overridden_keys=network_overridden_keys,
			)
			if resume_config.get("vae") and not cli_option_provided("--vae", "--no-vae"):
				args.vae = True
		if not cli_option_provided("--vae", "--no-vae"):
			args.vae = checkpoint_is_vae(resume_path)
		if resume_config:
			print(f"Restored network config from: {resume_path}")
			if network_overridden_keys:
				print(
					"CLI overrides checkpoint network settings: "
					+ ", ".join(network_overridden_keys)
				)
		resume_training = resume_training_config(resume_path)
		if (
			not adapter_only
			and resume_training
			and resume_training.get("adapter", "none") != "none"
		):
			raise ValueError(
			"adapter checkpoint must be resumed with image_ae.train_adapter"
			)
		if args.lora_rank > 0 and int((resume_training or {}).get("lora_rank", 0)) <= 0:
			raise ValueError(
				"--lora-rank with --resume requires a LoRA checkpoint; "
				"use --lora-base-checkpoint for a full base checkpoint"
			)
		if resume_training:
			overridden_training_keys = []
			training_options = {
				"epochs": ("--epochs",),
				"batch_size": ("--batch-size",),
				"lr": ("--lr",),
				"loss_fn": ("--loss-fn", "--reconstruction-loss"),
				"huber_beta": ("--huber-beta",),
				"optimizer": ("--optimizer",),
				"auto_schedule": ("--auto-schedule", "--no-auto-schedule"),
				"lr_scheduler": ("--lr-scheduler",),
				"warmup_steps": ("--warmup-steps",),
				"warmup_ratio": ("--warmup-ratio",),
				"min_lr_ratio": ("--min-lr-ratio",),
				"lr_step_size": ("--lr-step-size",),
				"lr_gamma": ("--lr-gamma",),
				"lr_milestones": ("--lr-milestones",),
				"lr_num_cycles": ("--lr-num-cycles",),
				"lr_power": ("--lr-power",),
				"force_scheduler": ("--force-scheduler",),
				"apollo_rank": ("--apollo-rank",),
				"apollo_scale": ("--apollo-scale",),
				"apollo_update_proj_gap": ("--apollo-update-proj-gap",),
				"apollo_projection_refresh_mode": ("--apollo-projection-refresh-mode",),
				"apollo_projection_refresh_window": ("--apollo-projection-refresh-window",),
				"apollo_projection_refresh_mix": ("--apollo-projection-refresh-mix",),
				"apollo_projection_refresh_state": ("--apollo-projection-refresh-state",),
				"apollo_orthogonal_refresh_rate": (
					"--apollo-orthogonal-refresh-rate",
				),
				"apollo_orthogonal_refresh_direction": (
					"--apollo-orthogonal-refresh-direction",
				),
				"apollo_scale_front": ("--apollo-scale-front", "--no-apollo-scale-front"),
				"apollo_disable_norm_growth_limiter": (
					"--apollo-disable-norm-growth-limiter",
					"--no-apollo-disable-norm-growth-limiter",
				),
				"apollo_norm_growth_rate": ("--apollo-norm-growth-rate",),
				"apollo_fallback": ("--apollo-fallback",),
				"apollo_matrix_fallback": ("--apollo-matrix-fallback",),
				"apollo_came_backend": ("--apollo-came-backend",),
				"came_lrsf_rank": ("--came-lrsf-rank",),
				"came_lrsf_beta1": ("--came-lrsf-beta1",),
				"came_lrsf_warmup_steps": ("--came-lrsf-warmup-steps",),
				"came_lrsf_r": ("--came-lrsf-r",),
				"came_lrsf_weight_lr_power": ("--came-lrsf-weight-lr-power",),
				"came_lrsf_seed": ("--came-lrsf-seed",),
				"came_lrsf_refresh_mode": ("--came-lrsf-refresh-mode",),
				"came_lrsf_refresh_interval": ("--came-lrsf-refresh-interval",),
				"came_lrsf_refresh_window": ("--came-lrsf-refresh-window",),
				"came_lrsf_refresh_mix": ("--came-lrsf-refresh-mix",),
				"came_lrsf_orthogonal_refresh_rate": (
					"--came-lrsf-orthogonal-refresh-rate",
				),
				"came_lrsf_orthogonal_refresh_direction": (
					"--came-lrsf-orthogonal-refresh-direction",
				),
				"came_lrsf_orthogonal_refresh_signal": (
					"--came-lrsf-orthogonal-refresh-signal",
				),
				"num_workers": ("--num-workers",),
				"seed": ("--seed",),
				"vae": ("--vae", "--no-vae"),
				"kl_weight": ("--kl-weight",),
				"latent_cycle_consistency": ("--latent-cycle-consistency", "--no-latent-cycle-consistency"),
				"latent_cycle_weight": ("--latent-cycle-weight",),
				"latent_cycle_z1_weight": ("--latent-cycle-z1-weight",),
				"latent_cycle_z2_weight": ("--latent-cycle-z2-weight",),
				"image_cycle_consistency": ("--image-cycle-consistency", "--no-image-cycle-consistency"),
				"image_cycle_weight": ("--image-cycle-weight",),
				"wavelet_loss": ("--wavelet-loss", "--no-wavelet-loss"),
				"wavelet_loss_weight": ("--wavelet-loss-weight",),
				"wavelet_levels": ("--wavelet-levels",),
				"latent_variance_loss": ("--latent-variance-loss", "--no-latent-variance-loss"),
				"latent_variance_weight": ("--latent-variance-weight",),
				"channel_decorrelation_loss": ("--channel-decorrelation-loss", "--no-channel-decorrelation-loss"),
				"channel_decorrelation_weight": ("--channel-decorrelation-weight",),
				"input_noise_std": ("--input-noise-std",),
				"input_blur_sigma": ("--input-blur-sigma",),
				"input_bit_depth": ("--input-bit-depth",),
				"imagenet_train_samples": ("--imagenet-train-samples",),
				"imagenet_val_samples": ("--imagenet-val-samples",),
				"cifar10_train_samples": ("--cifar10-train-samples",),
				"cifar10_val_samples": ("--cifar10-val-samples",),
				"dataset_split": ("--dataset-split",),
				"validation_split": ("--validation-split",),
				"lora_rank": ("--lora-rank",),
				"adapter": ("--adapter",),
				"lora_alpha": ("--lora-alpha",),
				"lora_dropout": ("--lora-dropout",),
				"lora_target": ("--lora-target",),
				"adapter_init": ("--adapter-init",),
			}
			restored_training_keys = apply_saved_config(
				args,
				resume_training,
				training_options,
				argv=cli_argv,
				keys=tuple(training_options),
				overridden_keys=overridden_training_keys,
			)
			if restored_training_keys:
				print(
					"Restored training settings from checkpoint: "
					+ ", ".join(restored_training_keys)
				)
			if overridden_training_keys:
				print(
					"CLI overrides checkpoint training settings: "
					+ ", ".join(overridden_training_keys)
				)
	adapter_type = (
		resolve_adapter_config(
			args, resume_path=resume_path, base_path=lora_base_path,
		)
		if adapter_only
		else "none"
	)
	args.dataset = args.dataset or DEFAULT_DATASET
	args.encoder = args.encoder or "window_transformer"
	args.decoder = args.decoder or "window_transformer"
	if args.latent_channels is None:
		args.latent_channels = 16
	args.bottleneck_channels = args.bottleneck_channels or 256
	args.encoder_blocks = args.encoder_blocks or 1
	args.decoder_blocks = args.decoder_blocks or 1
	args.encoder_layers = args.encoder_layers or 2
	args.encoder_window_size = args.encoder_window_size or 8
	args.decoder_layers = args.decoder_layers or 2
	args.downsample_stages = args.downsample_stages or 3
	args.encoder = canonical_encoder_type(args.encoder)
	args.decoder = canonical_decoder_type(args.decoder)
	if args.resume_epoch < 0:
		raise ValueError("--resume-epoch must be >= 0")
	default_image_size = 32 if args.dataset == "cifar10" else 256
	args.image_size = default_image_size if args.image_size is None else args.image_size
	if args.image_size <= 0:
		raise ValueError("--image-size must be positive")
	if args.latent_channels <= 0:
		raise ValueError("--latent-channels must be positive")
	if args.apollo_rank <= 0:
		raise ValueError("--apollo-rank must be positive")
	if args.apollo_scale <= 0.0:
		raise ValueError("--apollo-scale must be positive")
	if args.apollo_update_proj_gap <= 0:
		raise ValueError("--apollo-update-proj-gap must be positive")
	if args.apollo_projection_refresh_window < 0:
		raise ValueError("--apollo-projection-refresh-window must be non-negative")
	if args.apollo_projection_refresh_mode == "smooth" and args.apollo_projection_refresh_window <= 0:
		raise ValueError(
			"--apollo-projection-refresh-window must be positive for smooth mode"
		)
	if args.apollo_orthogonal_refresh_rate < 0.0:
		raise ValueError(
			"--apollo-orthogonal-refresh-rate must be non-negative"
		)
	if args.apollo_norm_growth_rate <= 1.0:
		raise ValueError("--apollo-norm-growth-rate must be greater than 1")
	if args.came_lrsf_rank <= 0:
		raise ValueError("--came-lrsf-rank must be positive")
	if not 0.0 < args.came_lrsf_beta1 < 1.0:
		raise ValueError("--came-lrsf-beta1 must be between 0 and 1")
	if args.came_lrsf_warmup_steps < 0:
		raise ValueError("--came-lrsf-warmup-steps must be non-negative")
	if args.came_lrsf_r < 0.0 or args.came_lrsf_weight_lr_power < 0.0:
		raise ValueError("CAME-LRSF weighting values must be non-negative")
	if args.came_lrsf_refresh_interval < 0:
		raise ValueError("--came-lrsf-refresh-interval must be non-negative")
	if args.came_lrsf_refresh_window < 0:
		raise ValueError("--came-lrsf-refresh-window must be non-negative")
	if args.came_lrsf_refresh_mode != "none":
		if args.came_lrsf_refresh_interval <= 0:
			raise ValueError("refresh interval must be positive when refresh is enabled")
		if args.came_lrsf_refresh_mode == "smooth" and args.came_lrsf_refresh_window <= 0:
			raise ValueError("smooth refresh window must be positive")
	if args.came_lrsf_orthogonal_refresh_rate < 0.0:
		raise ValueError(
			"--came-lrsf-orthogonal-refresh-rate must be non-negative"
		)
	if args.dataset == "cifar10" and args.image_size != 32:
		raise ValueError("CIFAR-10 currently requires --image-size 32")
	if args.bucket_step <= 0:
		raise ValueError("--bucket-step must be positive")
	if args.bucket_step % (2 ** args.downsample_stages) != 0:
		raise ValueError(
			"--bucket-step must be divisible by 2**downsample_stages: "
			f"got bucket_step={args.bucket_step}, stages={args.downsample_stages}"
		)
	if args.image_size % (2 ** args.downsample_stages) != 0:
		raise ValueError(
			"--image-size must be divisible by 2**downsample_stages: "
			f"got image_size={args.image_size}, stages={args.downsample_stages}"
		)
	if args.huber_beta <= 0:
		raise ValueError("--huber-beta must be > 0")
	if args.input_noise_std < 0 or args.input_blur_sigma < 0:
		raise ValueError("input noise std and blur sigma must be >= 0")
	if args.encoder_layers <= 0 or args.decoder_layers <= 0:
		raise ValueError("encoder and decoder layer counts must be positive")
	if args.encoder_window_size <= 0:
		raise ValueError("--encoder-window-size must be positive")
	if args.input_bit_depth < 0 or args.input_bit_depth > 8:
		raise ValueError("input bit depth must be between 0 and 8")
	if args.kl_weight < 0:
		raise ValueError("--kl-weight must be >= 0")
	if args.latent_cycle_weight < 0:
		raise ValueError("--latent-cycle-weight must be >= 0")
	if args.latent_cycle_z1_weight < 0 or args.latent_cycle_z2_weight < 0:
		raise ValueError("latent cycle z1/z2 weights must be >= 0")
	if args.latent_cycle_z1_weight + args.latent_cycle_z2_weight <= 0:
		raise ValueError("latent cycle z1/z2 weights must not both be zero")
	if args.image_cycle_weight < 0:
		raise ValueError("--image-cycle-weight must be >= 0")
	if args.wavelet_loss_weight < 0:
		raise ValueError("--wavelet-loss-weight must be >= 0")
	if args.wavelet_levels <= 0:
		raise ValueError("--wavelet-levels must be positive")
	if args.latent_variance_weight < 0:
		raise ValueError("--latent-variance-weight must be >= 0")
	if args.channel_decorrelation_weight < 0:
		raise ValueError("--channel-decorrelation-weight must be >= 0")
	if args.imagenet_train_samples < 0 or args.imagenet_val_samples < 0:
		raise ValueError("ImageNet sample counts must be >= 0")
	if args.cifar10_train_samples < 0 or args.cifar10_val_samples < 0:
		raise ValueError("CIFAR-10 sample counts must be >= 0")
	torch.manual_seed(args.seed)
	device = resolve_device(args.device)
	os.makedirs(args.output_dir, exist_ok=True)
	run_recorder = RunRecorder(
		args.output_dir,
		script=script_name,
		config=vars(args),
		run_name=args.run_name,
	)
	run_recorder.install_exception_hook()
	preflight = build_training_preflight(
		script=script_name,
		device=device,
		dtype=torch.float32,
		output_dir=args.output_dir,
		epochs=args.epochs,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		seed=args.seed,
		resume=args.resume,
		extra={"optimizer": args.optimizer},
	)
	run_recorder.record("preflight", **preflight)
	if args.dry_run:
		run_recorder.finish(status="dry_run")
		print("Dry run completed; no dataset or model was loaded.")
		return
	validation_timer = ValidationTimer(device) if args.validate_only else None
	checkpoint_dir = str(run_recorder.checkpoints_dir)
	artifact_dir = str(run_recorder.artifacts_dir)

	if args.dataset == "cifar10":
		train_transform = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			Random90Rotation(),
			transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
			transforms.RandomAffine(
				degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), fill=0,
			),
			transforms.ToTensor(),
		])
		test_transform = transforms.ToTensor()
		train_dataset = datasets.CIFAR10(
			root=args.data_dir, train=True, download=True, transform=train_transform,
		)
		test_dataset = datasets.CIFAR10(
			root=args.data_dir, train=False, download=True, transform=test_transform,
		)
		if args.cifar10_train_samples > 0:
			train_dataset = Subset(
				train_dataset,
				range(min(args.cifar10_train_samples, len(train_dataset))),
			)
		if args.cifar10_val_samples > 0:
			test_dataset = Subset(
				test_dataset,
				range(min(args.cifar10_val_samples, len(test_dataset))),
			)
		train_split, validation_split = "train", "test"
	else:
		train_dataset, test_dataset, train_split, validation_split = load_hf_image_datasets(args)
	train_loader = make_image_loader(
		train_dataset, args.batch_size, args.num_workers, device,
		shuffle=True, drop_last=True, seed=args.seed,
	)
	test_loader = make_image_loader(
		test_dataset, args.batch_size, args.num_workers, device,
		shuffle=False, drop_last=False, seed=args.seed, stream=1,
	)
	train_order = (
		train_loader.batch_sampler
		if isinstance(train_loader.batch_sampler, ResumableAspectRatioBatchSampler)
		else train_loader.sampler
	)
	loader_steps_per_epoch = len(train_loader)

	model = ImageAE(
		args.latent_channels,
		bottleneck_channels=args.bottleneck_channels,
		encoder_blocks=args.encoder_blocks,
		decoder_blocks=args.decoder_blocks,
			encoder_type=args.encoder,
			encoder_layers=args.encoder_layers,
			encoder_window_size=args.encoder_window_size,
			decoder_layers=args.decoder_layers,
		decoder_type=args.decoder,
		vae=args.vae,
		downsample_stages=args.downsample_stages,
		).to(device)
	if lora_base_path:
		print(f"Loading LoRA base weights: {lora_base_path}")
		base_state = load_file(lora_base_path, device="cpu")
		model.load_state_dict(base_state, strict=True)
		del base_state
	if adapter_type != "none":
		matched_lora_targets, lora_trainable_count = enable_adapter(model, args)
		print(
			f"Enabled {adapter_type}: rank={args.lora_rank} "
			f"alpha={args.lora_alpha or args.lora_rank:g} "
			f"targets={len(matched_lora_targets)} "
			f"trainable_parameters={lora_trainable_count:,}"
		)
	init_frozen_parameter_names = set()
	if init_path:
		transferred_keys = initialize_shared_parameters(model, init_path)
		if args.init_freeze_epochs > 0:
			init_frozen_parameter_names = {
				name for name, _ in model.named_parameters()
				if name in transferred_keys
			}
			for name, parameter in model.named_parameters():
				if name in init_frozen_parameter_names:
					parameter.requires_grad_(False)
		print(
			f"initialized shared parameters: {len(transferred_keys)} "
			f"from {init_path}"
		)
		if init_frozen_parameter_names:
			print(
			f"frozen initialized parameters for {args.init_freeze_epochs} epoch(s)"
			)
	if args.validate_only:
		assert validation_timer is not None
		model.eval()
		with torch.inference_mode():
			sample_images, _ = next(iter(test_loader))
			sample_images = sample_images.to(device)
			sample_output = model(sample_images)
			sample_reconstruction = sample_output[0]
			sample_latent = sample_output[1]
		if sample_reconstruction.shape != sample_images.shape:
			raise ValueError(
				"Image AE validation produced an unexpected reconstruction shape: "
				f"{tuple(sample_reconstruction.shape)}"
			)
		if not torch.isfinite(sample_reconstruction).all() or not torch.isfinite(sample_latent).all():
			raise ValueError("Image AE validation produced non-finite outputs")
		validation = build_validation_report(
			script=script_name,
			device=device,
			dtype=torch.float32,
			train_examples=len(train_dataset),
			eval_examples=len(test_dataset),
			model_parameters=sum(parameter.numel() for parameter in model.parameters()),
			trainable_parameters=sum(
				parameter.numel()
				for parameter in model.parameters()
				if parameter.requires_grad
			),
			steps_per_epoch=loader_steps_per_epoch,
			measurements=validation_timer.finish(),
			extra={
				"dataset": args.dataset,
				"input_shape": list(sample_images.shape),
				"latent_shape": list(sample_latent.shape),
				"reconstruction_shape": list(sample_reconstruction.shape),
				"bucket_shapes": getattr(train_dataset, "bucket_shapes", None),
				"downsample_stages": args.downsample_stages,
			},
		)
		run_recorder.record("validation", **validation)
		run_recorder.finish(status="validate_only")
		print("Validation completed; training was not started.")
		return
	optimizer_parameters = (
		adapter_optimizer_parameters(model)
		if adapter_type != "none"
		else model.parameters()
	)
	optimizer = build_optimizer(
		args.optimizer,
		optimizer_parameters,
		lr=args.lr,
		weight_decay=1e-4,
		args=args,
	)
	is_schedule_free = is_schedule_free_optimizer(args.optimizer)
	scheduler = None
	if not is_schedule_free or args.force_scheduler:
		scheduler = build_lr_scheduler(
			optimizer, args, max(1, args.epochs * loader_steps_per_epoch),
		)
	print(
		f"lr_scheduler={scheduler.name if scheduler is not None else 'disabled'} "
		f"warmup_steps={scheduler.warmup_steps if scheduler is not None else 0}"
	)
	start_epoch = 0
	resume_state = None
	resume_global_step = None
	if args.resume:
		print(f"Resuming training from: {resume_path}")
		model.load_state_dict(load_resume_state_dict(resume_path, args.vae))
		resume_state = load_training_state(resume_path)
		if resume_state is not None:
			stored_epoch = int(resume_state["epoch"])
			resume_global_step = int(resume_state["global_step"])
			start_epoch = args.resume_epoch if args.resume_epoch > 0 else stored_epoch
			saved_sampler = (resume_state.get("extra") or {}).get("sampler")
			if saved_sampler is not None:
				train_order.load_state_dict(saved_sampler)
				if args.resume_epoch > 0:
					train_order.set_epoch(start_epoch)
			print(
				f"Restored full training state: epoch={start_epoch} "
				f"global_step={resume_global_step}"
			)
		else:
			stored_epoch = checkpoint_epoch(resume_path)
			start_epoch = args.resume_epoch if args.resume_epoch > 0 else stored_epoch
			print(
				f"Resuming weights-only; completed epochs restored: {start_epoch}"
			)
	if start_epoch >= args.epochs:
		raise ValueError(
			f"resume starts at epoch {start_epoch}, but --epochs is {args.epochs}; "
			"set --epochs to a larger total epoch count"
		)

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_output_dir = str(run_recorder.run_dir)
	vae_tag = "_vae" if args.vae else ""
	artifact_tag = (
		f"{args.dataset}_enc-{args.encoder}_dec-{args.decoder}"
		f"_size{args.image_size}"
		f"_latent-ch{args.latent_channels}"
		f"_bottleneck-ch{args.bottleneck_channels}"
		f"_ds{args.downsample_stages}_eb{args.encoder_blocks}"
		f"_db{args.decoder_blocks}_el{args.encoder_layers}"
		f"_ew{args.encoder_window_size}_dl{args.decoder_layers}{vae_tag}"
	)
	if args.hidden_channels is not None:
		artifact_tag += f"_hidden-ch{args.hidden_channels}"
	if args.latent_cycle_consistency:
		artifact_tag += (
			f"_cycle-w{args.latent_cycle_weight:g}"
			f"-z1{args.latent_cycle_z1_weight:g}"
			f"-z2{args.latent_cycle_z2_weight:g}"
		)
	if args.image_cycle_consistency:
		artifact_tag += f"_image-cycle-w{args.image_cycle_weight:g}"
	if args.wavelet_loss:
		artifact_tag += (
			f"_wavelet-w{args.wavelet_loss_weight:g}"
			f"-l{args.wavelet_levels}"
		)
	if args.latent_variance_loss:
		artifact_tag += f"_latent-var-w{args.latent_variance_weight:g}"
	if args.channel_decorrelation_loss:
		artifact_tag += f"_decor-w{args.channel_decorrelation_weight:g}"
	if args.input_noise_std > 0:
		artifact_tag += f"_noise-s{args.input_noise_std:g}"
	if args.input_blur_sigma > 0:
		artifact_tag += f"_blur-s{args.input_blur_sigma:g}"
	if args.input_bit_depth > 0:
		artifact_tag += f"_bits{args.input_bit_depth}"
	if args.dataset != "cifar10":
		if args.imagenet_train_samples > 0:
			artifact_tag += f"_train{args.imagenet_train_samples}"
		if args.imagenet_val_samples > 0:
			artifact_tag += f"_val{args.imagenet_val_samples}"
	writer = SummaryWriter(
		os.path.join(args.log_dir, f"image_ae_{timestamp}_{artifact_tag}")
		if args.log_dir is not None else run_recorder.tensorboard_dir
	)
	fixed_images, _ = next(iter(test_loader))
	fixed_images = fixed_images[:16].to(device)
	w0, h0, c0 = fixed_images.shape[3], fixed_images.shape[2], fixed_images.shape[1]
	if w0 % (2 ** args.downsample_stages) != 0 or h0 % (2 ** args.downsample_stages) != 0:
		raise ValueError(
			"input dimensions must be divisible by 2**downsample_stages: "
			f"got ({w0}, {h0}) and {args.downsample_stages} stages"
		)
	w1 = w0 // (2 ** args.downsample_stages)
	h1 = h0 // (2 ** args.downsample_stages)
	c1 = model.latent_channels
	width_scale = w1 / w0
	height_scale = h1 / h0
	channel_scale = c1 / c0
	compression_rate = (w0 * h0 * c0) / (w1 * h1 * c1)
	print(f"device={device}, parameters={sum(p.numel() for p in model.parameters()):,}")
	print(f"optimizer={args.optimizer}, lr={args.lr}, loss_fn={args.loss_fn}")
	print(
		f"dataset={args.dataset}, train_split={train_split}, "
		f"validation_split={validation_split}, image_size={args.image_size}, "
		f"bucket_step={args.bucket_step}"
	)
	print(f"run output directory: {run_output_dir}")
	if args.dataset != "cifar10":
		print(
			f"samples: train={'all' if args.imagenet_train_samples == 0 else args.imagenet_train_samples}, "
			f"validation={'all' if args.imagenet_val_samples == 0 else args.imagenet_val_samples}"
		)
	else:
		print(
			f"samples: train={'all' if args.cifar10_train_samples == 0 else args.cifar10_train_samples}, "
			f"validation={'all' if args.cifar10_val_samples == 0 else args.cifar10_val_samples}"
		)
	if getattr(train_dataset, "bucket_shapes", None) is not None:
		print(f"resolution buckets: {train_dataset.bucket_shapes}")
	print("loss/metric reduction: reconstruction/wavelet=mean per pixel-channel, kl=mean per latent element")
	print(f"input size (w0, h0, c0): ({w0}, {h0}, {c0})")
	print(f"latent size (w1, h1, c1): ({w1}, {h1}, {c1})")
	print(f"bottleneck channels (before/after latent projection): {args.bottleneck_channels}")
	print(
		f"latent cycle consistency: {'on' if args.latent_cycle_consistency else 'off'}"
		f" (weight={args.latent_cycle_weight:g}, "
		f"z1:z2={args.latent_cycle_z1_weight:g}:{args.latent_cycle_z2_weight:g})"
	)
	print(
		f"image cycle consistency: {'on' if args.image_cycle_consistency else 'off'}"
		f" (weight={args.image_cycle_weight:g})"
	)
	print(
		f"wavelet loss: {'on' if args.wavelet_loss else 'off'}"
		f" (weight={args.wavelet_loss_weight:g}, levels={args.wavelet_levels}, "
		"metric=always)"
	)
	print(
		f"latent variance loss: {'on' if args.latent_variance_loss else 'off'}"
		f" (weight={args.latent_variance_weight:g})"
	)
	print(
		f"channel decorrelation loss: {'on' if args.channel_decorrelation_loss else 'off'}"
		f" (weight={args.channel_decorrelation_weight:g})"
	)
	print(
		f"encoder input corruption: noise_std={args.input_noise_std:g}, "
		f"blur_sigma={args.input_blur_sigma:g}, bit_depth={args.input_bit_depth}"
	)
	print(
		f"scale (w1/w0, h1/h0, c1/c0): "
		f"({width_scale:.6f}, {height_scale:.6f}, {channel_scale:.6f})"
	)
	print(f"total compression rate: {compression_rate:.4f}x")
	model_snapshot = ModelSnapshot(max_snapshots=5)
	stop_controller = GracefulStop(
		"Ctrl-C received; finishing the current batch and saving a checkpoint..."
	)
	stop_controller.install()
	latest_path = os.path.join(
		checkpoint_dir, f"checkpoint_{artifact_tag}_latest.safetensors"
	)
	global_step = (
		resume_global_step
		if resume_global_step is not None
		else start_epoch * loader_steps_per_epoch
	)
	if resume_state is not None:
		optimizer.load_state_dict(resume_state["optimizer"])
		saved_scheduler = resume_state.get("scheduler")
		if scheduler is not None and saved_scheduler is not None:
			if "total_steps" in saved_scheduler:
				scheduler.total_steps = int(saved_scheduler["total_steps"])
			scheduler.load_state_dict(saved_scheduler)
		elif scheduler is None and saved_scheduler is not None:
			raise ValueError(
				"resume checkpoint contains an LR scheduler, but the current run "
				"disabled it"
			)
		elif scheduler is not None:
			scheduler.step(global_step)
		restore_rng_state(resume_state["rng"])
	for epoch in range(start_epoch, args.epochs):
		epoch_started_at = perf_counter()
		if train_order.epoch != epoch:
			train_order.set_epoch(epoch)
		steps_in_epoch = len(train_loader)
		if init_frozen_parameter_names and epoch >= args.init_freeze_epochs:
			for name, parameter in model.named_parameters():
				if name in init_frozen_parameter_names:
					parameter.requires_grad_(True)
			init_frozen_parameter_names.clear()
			print(f"initialized parameters unfrozen at epoch {epoch + 1}")
		model.train()
		if hasattr(optimizer, "train"):
			optimizer.train()
		progress = RichProgress(
			train_loader,
			description=f"epoch {epoch + 1}/{args.epochs}",
		)
		progress.set_status(
			loss="waiting for first batch",
			reconstruction="rmse=-",
			latent=f"{'mu' if args.vae else 'z'}_std=- [-,-] mean=-",
			sample="in - steps",
		)
		running_loss = 0.0
		running_reconstruction = 0.0
		running_reconstruction_metrics = {name: 0.0 for name in ("MSE", "L1", "Huber")}
		running_kl = 0.0
		running_latent_cycle = 0.0
		running_image_cycle = 0.0
		running_wavelet = 0.0
		running_latent_variance = 0.0
		running_channel_decorrelation = 0.0
		num_images = 0
		for step_index, (images, _) in enumerate(progress, start=1):
			step_started_at = perf_counter()
			images = images.to(device, non_blocking=True)
			optimizer.zero_grad(set_to_none=True)
			encoder_images = corrupt_encoder_input(
				images, args.input_noise_std, args.input_blur_sigma, args.input_bit_depth,
			)
			outputs = model(encoder_images)
			if model.vae:
				reconstructions, _, mu, logvar = outputs
				z1 = mu
				kl_loss = compute_kl_loss(mu, logvar)
			else:
				reconstructions, z1 = outputs
				kl_loss = torch.zeros((), device=images.device)
			reconstruction_losses = compute_reconstruction_losses(
				reconstructions, images, args.huber_beta
			)
			reconstruction_loss = reconstruction_losses[args.loss_fn]
			if args.latent_cycle_consistency or args.image_cycle_consistency:
				latent_cycle_loss, image_cycle_loss = compute_cycle_consistency_losses(
					model, reconstructions, images, z1, args.loss_fn, args.huber_beta,
					args.latent_cycle_z1_weight, args.latent_cycle_z2_weight,
					args.latent_cycle_consistency, args.image_cycle_consistency,
				)
				if not args.latent_cycle_consistency:
					latent_cycle_loss = torch.zeros((), device=images.device)
				if not args.image_cycle_consistency:
					image_cycle_loss = torch.zeros((), device=images.device)
			else:
				latent_cycle_loss = torch.zeros((), device=images.device)
				image_cycle_loss = torch.zeros((), device=images.device)
			if args.wavelet_loss:
				wavelet_loss_value = compute_wavelet_loss(
					reconstructions, images, levels=args.wavelet_levels,
				)
				wavelet_metric = wavelet_loss_value.detach()
			else:
				with torch.no_grad():
					wavelet_metric = compute_wavelet_loss(
						reconstructions.detach(), images, levels=args.wavelet_levels,
					)
				wavelet_loss_value = torch.zeros((), device=images.device)
			if args.latent_variance_loss:
				latent_variance_loss_value = compute_latent_variance_loss(z1)
			else:
				latent_variance_loss_value = torch.zeros((), device=images.device)
			if args.channel_decorrelation_loss:
				channel_decorrelation_loss_value = compute_channel_decorrelation_loss(z1)
			else:
				channel_decorrelation_loss_value = torch.zeros((), device=images.device)
			loss = (
				reconstruction_loss + args.kl_weight * kl_loss
				+ args.latent_cycle_weight * latent_cycle_loss
				+ args.image_cycle_weight * image_cycle_loss
				+ args.wavelet_loss_weight * wavelet_loss_value
				+ args.latent_variance_weight * latent_variance_loss_value
				+ args.channel_decorrelation_weight * channel_decorrelation_loss_value
			)
			loss.backward()
			optimizer.step()
			global_step += 1
			if isinstance(train_order, ResumableAspectRatioBatchSampler):
				train_order.set_position(train_order.position + 1)
			else:
				train_order.set_position(train_order.position + images.size(0))
			if scheduler is not None:
				scheduler.step(global_step)
				step_loss_value = loss.item()
				step_reconstruction_value = reconstruction_loss.item()
				step_kl_value = kl_loss.item()
				step_latent_cycle_value = latent_cycle_loss.item()
				step_image_cycle_value = image_cycle_loss.item()
				step_latent_variance_value = latent_variance_loss_value.item()
				step_channel_decorrelation_value = channel_decorrelation_loss_value.item()
				running_loss += step_loss_value * images.size(0)
				running_reconstruction += step_reconstruction_value * images.size(0)
				running_kl += step_kl_value * images.size(0)
				running_latent_cycle += step_latent_cycle_value * images.size(0)
				running_image_cycle += step_image_cycle_value * images.size(0)
				running_wavelet += wavelet_metric.item() * images.size(0)
				running_latent_variance += step_latent_variance_value * images.size(0)
				running_channel_decorrelation += step_channel_decorrelation_value * images.size(0)
				for name, metric in reconstruction_losses.items():
						running_reconstruction_metrics[name] += metric.item() * images.size(0)
				num_images += images.size(0)
				step_seconds = perf_counter() - step_started_at
				step_loss_components = {
					"recon": (step_reconstruction_value, 1.0),
				}
				if args.vae and args.kl_weight > 0:
					step_loss_components["kl"] = (step_kl_value, args.kl_weight)
				if args.latent_cycle_consistency and args.latent_cycle_weight > 0:
					step_loss_components["lc"] = (
						step_latent_cycle_value, args.latent_cycle_weight,
					)
				if args.image_cycle_consistency and args.image_cycle_weight > 0:
					step_loss_components["ic"] = (
						step_image_cycle_value, args.image_cycle_weight,
					)
				if args.wavelet_loss and args.wavelet_loss_weight > 0:
					step_loss_components["w"] = (
						wavelet_loss_value.item(), args.wavelet_loss_weight,
					)
				if args.latent_variance_loss and args.latent_variance_weight > 0:
					step_loss_components["lv"] = (
						step_latent_variance_value, args.latent_variance_weight,
					)
				if args.channel_decorrelation_loss and args.channel_decorrelation_weight > 0:
					step_loss_components["cd"] = (
						step_channel_decorrelation_value,
						args.channel_decorrelation_weight,
					)
				latent_values = z1.detach().float().permute(1, 0, 2, 3).reshape(z1.size(1), -1)
				latent_channel_mean = latent_values.mean(dim=1)
				latent_channel_std = torch.sqrt(
					latent_values.var(dim=1, unbiased=False) + 1e-4
				)
				latent_std_mean = latent_channel_std.mean().item()
				latent_std_min = latent_channel_std.min().item()
				latent_std_max = latent_channel_std.max().item()
				latent_mean_abs = latent_channel_mean.abs().mean().item()
				reconstruction_rmse = math.sqrt(
					max(reconstruction_losses["MSE"].item(), 0.0)
				)
				next_sample = SAMPLE_IMAGE_INTERVAL - (global_step % SAMPLE_IMAGE_INTERVAL)
				progress.set_status(
					step=f"{step_index}/{steps_in_epoch} "
					f"global_step={global_step} time={step_seconds:.3f}s",
					loss=(
						f"total={format_metric_value(step_loss_value)} "
						f"components={format_loss_breakdown(step_loss_value, step_loss_components)}"
					),
					reconstruction=(
						f"rmse={format_metric_value(reconstruction_rmse)} "
						f"{args.loss_fn.lower()}={format_metric_value(step_reconstruction_value)}"
					),
					latent=(
						f"{'mu' if args.vae else 'z'}_std="
						f"{format_metric_value(latent_std_mean)} "
						f"[{format_metric_value(latent_std_min)},"
						f"{format_metric_value(latent_std_max)}] "
						f"mean={format_metric_value(latent_mean_abs)}"
					),
					sample=f"in {next_sample} steps",
				)
				maybe_collect_memory(
					global_step,
					gc_interval=args.gc_interval,
					empty_cache_interval=args.empty_cache_interval,
				)
				if global_step % SAMPLE_IMAGE_INTERVAL == 0:
					model.eval()
					with torch.no_grad():
						sample_latent = model.encode(fixed_images)
						sample_reconstructions = model.decode(sample_latent).clamp(0, 1)
					sample_comparison = torch.stack(
						(fixed_images, sample_reconstructions), dim=1,
					).flatten(0, 1).cpu()
					save_image(
						sample_comparison,
						os.path.join(
							artifact_dir,
							f"reconstruction_{artifact_tag}_step_latest.png",
						),
						nrow=2,
					)
					model.train()
			if stop_controller.requested:
				break
		if stop_controller.requested:
			save_training_checkpoint(
				model.state_dict(), latest_path, args,
				optimizer=optimizer, scheduler=scheduler,
				epoch=epoch, global_step=global_step,
				extra={"sampler": train_order.state_dict()},
			)
			stop_controller.restore()
			print(
				f"Interrupted after {epoch} completed epochs; "
				f"global_step={global_step}; saved={latest_path}"
			)
			run_recorder.finish(
			status="interrupted",
			checkpoints=[latest_path],
		)
			writer.close()
			return
		train_loss = running_loss / num_images
		train_recon = running_reconstruction / num_images
		train_reconstruction_metrics = {
			name: value / num_images
			for name, value in running_reconstruction_metrics.items()
		}
		train_kl = running_kl / num_images
		train_latent_cycle = running_latent_cycle / num_images
		train_image_cycle = running_image_cycle / num_images
		train_wavelet = running_wavelet / num_images
		train_latent_variance = running_latent_variance / num_images
		train_channel_decorrelation = running_channel_decorrelation / num_images

		model.eval()
		if hasattr(optimizer, "eval"):
			optimizer.eval()
		with torch.no_grad():
			val_loss = 0.0
			val_recon = 0.0
			val_reconstruction_metrics = {name: 0.0 for name in ("MSE", "L1", "Huber")}
			val_kl = 0.0
			val_latent_cycle = 0.0
			val_image_cycle = 0.0
			val_wavelet = 0.0
			val_latent_variance = 0.0
			val_channel_decorrelation = 0.0
			val_count = 0
			for images, _ in test_loader:
				images = images.to(device, non_blocking=True)
				encoder_images = corrupt_encoder_input(
					images, args.input_noise_std, args.input_blur_sigma, args.input_bit_depth,
				)
				outputs = model(encoder_images)
				if model.vae:
					reconstructions, _, mu, logvar = outputs
					z1 = mu
					kl_loss = compute_kl_loss(mu, logvar)
				else:
					reconstructions, z1 = outputs
					kl_loss = torch.zeros((), device=images.device)
				reconstruction_losses = compute_reconstruction_losses(
					reconstructions, images, args.huber_beta
				)
				reconstruction_loss = reconstruction_losses[args.loss_fn]
				if args.latent_cycle_consistency or args.image_cycle_consistency:
					latent_cycle_loss, image_cycle_loss = compute_cycle_consistency_losses(
						model, reconstructions, images, z1, args.loss_fn, args.huber_beta,
						args.latent_cycle_z1_weight, args.latent_cycle_z2_weight,
						args.latent_cycle_consistency, args.image_cycle_consistency,
					)
					if not args.latent_cycle_consistency:
						latent_cycle_loss = torch.zeros((), device=images.device)
					if not args.image_cycle_consistency:
						image_cycle_loss = torch.zeros((), device=images.device)
				else:
					latent_cycle_loss = torch.zeros((), device=images.device)
					image_cycle_loss = torch.zeros((), device=images.device)
				wavelet_metric = compute_wavelet_loss(
					reconstructions, images, levels=args.wavelet_levels,
				)
				wavelet_loss_value = wavelet_metric if args.wavelet_loss else torch.zeros(
					(), device=images.device,
				)
				if args.latent_variance_loss:
					latent_variance_loss_value = compute_latent_variance_loss(z1)
				else:
					latent_variance_loss_value = torch.zeros((), device=images.device)
				if args.channel_decorrelation_loss:
					channel_decorrelation_loss_value = compute_channel_decorrelation_loss(z1)
				else:
					channel_decorrelation_loss_value = torch.zeros((), device=images.device)
				loss = (
					reconstruction_loss + args.kl_weight * kl_loss
					+ args.latent_cycle_weight * latent_cycle_loss
					+ args.image_cycle_weight * image_cycle_loss
					+ args.wavelet_loss_weight * wavelet_loss_value
					+ args.latent_variance_weight * latent_variance_loss_value
					+ args.channel_decorrelation_weight * channel_decorrelation_loss_value
				)
				val_loss += loss.item() * images.size(0)
				val_recon += reconstruction_loss.item() * images.size(0)
				for name, metric in reconstruction_losses.items():
					val_reconstruction_metrics[name] += metric.item() * images.size(0)
				val_kl += kl_loss.item() * images.size(0)
				val_latent_cycle += latent_cycle_loss.item() * images.size(0)
				val_image_cycle += image_cycle_loss.item() * images.size(0)
				val_wavelet += wavelet_metric.item() * images.size(0)
				val_latent_variance += latent_variance_loss_value.item() * images.size(0)
				val_channel_decorrelation += channel_decorrelation_loss_value.item() * images.size(0)
				val_count += images.size(0)
			test_loss = val_loss / val_count
			test_recon = val_recon / val_count
			test_reconstruction_metrics = {
				name: value / val_count
				for name, value in val_reconstruction_metrics.items()
			}
			test_kl = val_kl / val_count
			test_latent_cycle = val_latent_cycle / val_count
			test_image_cycle = val_image_cycle / val_count
			test_wavelet = val_wavelet / val_count
			test_latent_variance = val_latent_variance / val_count
			test_channel_decorrelation = val_channel_decorrelation / val_count
			fixed_latent = model.encode(fixed_images)
			reconstructions = model.decode(fixed_latent)
		model_snapshot.add_snapshot(model, -test_loss, epoch + 1)

		writer.add_scalar("test/loss/total", test_loss, epoch + 1)
		writer.add_scalar("train/loss/reconstruction", train_recon, epoch + 1)
		writer.add_scalar("test/loss/reconstruction", test_recon, epoch + 1)
		train_metrics = {"reconstruction/MSE": train_reconstruction_metrics["MSE"],
			"reconstruction/RMSE": math.sqrt(max(train_reconstruction_metrics["MSE"], 0.0)),
			"reconstruction/L1": train_reconstruction_metrics["L1"],
			"reconstruction/Huber": train_reconstruction_metrics["Huber"],
			"wavelet": train_wavelet}
		test_metrics = {"reconstruction/MSE": test_reconstruction_metrics["MSE"],
			"reconstruction/RMSE": math.sqrt(max(test_reconstruction_metrics["MSE"], 0.0)),
			"reconstruction/L1": test_reconstruction_metrics["L1"],
			"reconstruction/Huber": test_reconstruction_metrics["Huber"],
			"wavelet": test_wavelet}
		if args.vae:
			train_metrics["kl"] = train_kl
			test_metrics["kl"] = test_kl
		if args.latent_cycle_consistency:
			train_metrics["latent_cycle"] = train_latent_cycle
			test_metrics["latent_cycle"] = test_latent_cycle
		if args.image_cycle_consistency:
			train_metrics["image_cycle"] = train_image_cycle
			test_metrics["image_cycle"] = test_image_cycle
		if args.latent_variance_loss:
			train_metrics["latent_variance"] = train_latent_variance
			test_metrics["latent_variance"] = test_latent_variance
		if args.channel_decorrelation_loss:
			train_metrics["channel_decorrelation"] = train_channel_decorrelation
			test_metrics["channel_decorrelation"] = test_channel_decorrelation
		for name, value in train_metrics.items():
			writer.add_scalar(f"train/metric/{name}", value, epoch + 1)
		for name, value in test_metrics.items():
			writer.add_scalar(f"test/metric/{name}", value, epoch + 1)
		write_standard_training_metrics(
			writer,
			step=epoch + 1,
			train_loss=train_loss,
			eval_loss=test_loss,
			learning_rate=float(optimizer.param_groups[0]["lr"]),
			scheduled_learning_rate=float(
				optimizer.param_groups[0].get(
					"scheduled_lr", optimizer.param_groups[0]["lr"]
				)
			),
			extra={
				"train/reconstruction": train_recon,
				"eval/reconstruction": test_recon,
			},
		)
		epoch_elapsed = perf_counter() - epoch_started_at
		effective_lr = float(optimizer.param_groups[0]["lr"])
		scheduled_lr = float(
			optimizer.param_groups[0].get("scheduled_lr", effective_lr)
		)
		run_recorder.record_training_step(
			global_step=global_step,
			epoch=epoch + 1,
			train_loss=train_loss,
			eval_loss=test_loss,
			effective_lr=effective_lr,
			scheduled_lr=scheduled_lr,
			step_time_sec=epoch_elapsed / max(steps_in_epoch, 1),
			steps_per_second=steps_in_epoch / max(epoch_elapsed, 1e-6),
			samples_per_second=num_images / max(epoch_elapsed, 1e-6),
			metrics={
				"train_reconstruction": train_recon,
				"eval_reconstruction": test_recon,
			},
		)
		loss_components = {
			"reconstruction": train_recon,
		}
		test_loss_components = {
			"reconstruction": test_recon,
		}
		if args.vae and args.kl_weight > 0:
			loss_components["kl"] = args.kl_weight * train_kl
			test_loss_components["kl"] = args.kl_weight * test_kl
		if args.latent_cycle_consistency and args.latent_cycle_weight > 0:
			loss_components["latent_cycle"] = args.latent_cycle_weight * train_latent_cycle
			test_loss_components["latent_cycle"] = args.latent_cycle_weight * test_latent_cycle
		if args.image_cycle_consistency and args.image_cycle_weight > 0:
			loss_components["image_cycle"] = args.image_cycle_weight * train_image_cycle
			test_loss_components["image_cycle"] = args.image_cycle_weight * test_image_cycle
		if args.wavelet_loss and args.wavelet_loss_weight > 0:
			loss_components["wavelet"] = args.wavelet_loss_weight * train_wavelet
			test_loss_components["wavelet"] = args.wavelet_loss_weight * test_wavelet
		if args.latent_variance_loss and args.latent_variance_weight > 0:
			loss_components["latent_variance"] = args.latent_variance_weight * train_latent_variance
			test_loss_components["latent_variance"] = args.latent_variance_weight * test_latent_variance
		if args.channel_decorrelation_loss and args.channel_decorrelation_weight > 0:
			loss_components["channel_decorrelation"] = args.channel_decorrelation_weight * train_channel_decorrelation
			test_loss_components["channel_decorrelation"] = args.channel_decorrelation_weight * test_channel_decorrelation
		for name in loss_components:
			writer.add_scalar(
				f"contribution/train/loss/{name}",
				100.0 * loss_components[name] / max(abs(train_loss), 1e-12), epoch + 1,
			)
			writer.add_scalar(
				f"contribution/test/loss/{name}",
				100.0 * test_loss_components[name] / max(abs(test_loss), 1e-12), epoch + 1,
			)
		comparison_images = torch.stack(
			(fixed_images, reconstructions), dim=1,
		).flatten(0, 1)
		writer.add_images(
			f"reconstruction/{args.loss_fn}",
			comparison_images,
			epoch + 1,
		)
		save_image(
			comparison_images,
			os.path.join(artifact_dir, f"reconstruction_{artifact_tag}_latest.png"),
			nrow=2,
		)
		train_order.set_epoch(epoch + 1)
		save_training_checkpoint(
			model.state_dict(), latest_path, args,
			optimizer=optimizer, scheduler=scheduler,
			epoch=epoch + 1, global_step=global_step,
			extra={"sampler": train_order.state_dict()},
		)
		loss_report = [
			format_loss_comparison(
				"recon(MSE)", train_reconstruction_metrics["MSE"], train_loss,
				test_reconstruction_metrics["MSE"], test_loss,
				args.loss_fn == "MSE",
			),
			format_loss_comparison(
				"recon(L1)", train_reconstruction_metrics["L1"], train_loss,
				test_reconstruction_metrics["L1"], test_loss,
				args.loss_fn == "L1",
			),
			format_loss_comparison(
				"recon(Huber)", train_reconstruction_metrics["Huber"], train_loss,
				test_reconstruction_metrics["Huber"], test_loss,
				args.loss_fn == "Huber",
			),
			format_loss_comparison(
				"wavelet", train_wavelet, train_loss, test_wavelet, test_loss,
				args.wavelet_loss_weight if args.wavelet_loss else 0.0,
			),
		]
		if args.vae and args.kl_weight > 0:
			loss_report.append(
				format_loss_comparison(
					"kl", train_kl, train_loss, test_kl, test_loss, args.kl_weight,
				)
			)
		if args.latent_cycle_consistency:
			loss_report.append(
				format_loss_comparison(
					"latent_cycle", train_latent_cycle, train_loss,
					test_latent_cycle, test_loss, args.latent_cycle_weight,
				)
			)
		if args.image_cycle_consistency:
			loss_report.append(
				format_loss_comparison(
					"image_cycle", train_image_cycle, train_loss,
					test_image_cycle, test_loss, args.image_cycle_weight,
				)
			)
		if args.latent_variance_loss:
			loss_report.append(
				format_loss_comparison(
					"latent_variance", train_latent_variance, train_loss,
					test_latent_variance, test_loss, args.latent_variance_weight,
				)
			)
		if args.channel_decorrelation_loss:
			loss_report.append(
				format_loss_comparison(
					"channel_decor", train_channel_decorrelation, train_loss,
					test_channel_decorrelation, test_loss,
					args.channel_decorrelation_weight,
				)
			)
		print(
			f"epoch {epoch + 1:03d}/{args.epochs}: "
			f"train_loss={format_metric_value(train_loss)} "
			f"test_loss={format_metric_value(test_loss)}\n"
			f"  {'loss component':<16} {'train':>16}  {'test':>16}\n"
			+ "\n".join(loss_report)
			+ "\n"
			+ f"  {'metric':<16} {'train':>16}  {'test':>16}\n"
			+ format_metric_comparison(
				"recon(RMSE)",
				math.sqrt(max(train_reconstruction_metrics["MSE"], 0.0)),
				math.sqrt(max(test_reconstruction_metrics["MSE"], 0.0)),
			)
		)
		if stop_controller.requested:
			stop_controller.restore()
			writer.close()
			print(f"Interrupted after epoch {epoch + 1}; saved={latest_path}")
			run_recorder.finish(
			status="interrupted",
			checkpoints=[latest_path],
		)
			return

	stop_controller.restore()
	best_state_dict = model_snapshot.get_best_model()
	model.load_state_dict(best_state_dict)
	best_loss, best_recon, best_kl, best_latent_cycle, best_image_cycle, best_wavelet, best_latent_variance, best_channel_decorrelation = evaluate_saved_model(
		model, test_loader, device, args.loss_fn, args.huber_beta, args.kl_weight,
		args.latent_cycle_consistency, args.latent_cycle_weight,
		args.latent_cycle_z1_weight, args.latent_cycle_z2_weight,
		args.image_cycle_consistency, args.image_cycle_weight,
		args.wavelet_loss, args.wavelet_loss_weight, args.wavelet_levels,
		args.latent_variance_loss, args.latent_variance_weight,
		args.channel_decorrelation_loss, args.channel_decorrelation_weight,
	)
	best_path = os.path.join(
		checkpoint_dir, f"checkpoint_{artifact_tag}_best.safetensors"
	)
	save_model_checkpoint(best_state_dict, best_path, args, epoch=args.epochs)
	print(
		f"best weights: test_loss={format_metric_value(best_loss)} "
		f"test_recon={format_metric_value(best_recon)} test_kl={format_metric_value(best_kl)} "
		f"test_cycle={format_metric_value(best_latent_cycle)} "
		f"test_image_cycle={format_metric_value(best_image_cycle)} "
		f"test_wavelet={format_metric_value(best_wavelet)} "
		f"test_latent_variance={format_metric_value(best_latent_variance)} "
		f"test_channel_decor={format_metric_value(best_channel_decorrelation)} "
		f"saved={best_path}"
	)

	avg_state_dict = model_snapshot.get_average_model()
	model.load_state_dict(avg_state_dict)
	avg_loss, avg_recon, avg_kl, avg_latent_cycle, avg_image_cycle, avg_wavelet, avg_latent_variance, avg_channel_decorrelation = evaluate_saved_model(
		model, test_loader, device, args.loss_fn, args.huber_beta, args.kl_weight,
		args.latent_cycle_consistency, args.latent_cycle_weight,
		args.latent_cycle_z1_weight, args.latent_cycle_z2_weight,
		args.image_cycle_consistency, args.image_cycle_weight,
		args.wavelet_loss, args.wavelet_loss_weight, args.wavelet_levels,
		args.latent_variance_loss, args.latent_variance_weight,
		args.channel_decorrelation_loss, args.channel_decorrelation_weight,
	)
	avg_path = os.path.join(
		checkpoint_dir, f"checkpoint_{artifact_tag}_avg.safetensors"
	)
	save_model_checkpoint(avg_state_dict, avg_path, args, epoch=args.epochs)
	print(
		f"average weights: test_loss={format_metric_value(avg_loss)} "
		f"test_recon={format_metric_value(avg_recon)} test_kl={format_metric_value(avg_kl)} "
		f"test_cycle={format_metric_value(avg_latent_cycle)} "
		f"test_image_cycle={format_metric_value(avg_image_cycle)} "
		f"test_wavelet={format_metric_value(avg_wavelet)} "
		f"test_latent_variance={format_metric_value(avg_latent_variance)} "
		f"test_channel_decor={format_metric_value(avg_channel_decorrelation)} "
		f"saved={avg_path}"
	)
	writer.close()
	run_recorder.finish(
		checkpoints=[latest_path, best_path, avg_path],
	)

if __name__ == "__main__":
	main()
