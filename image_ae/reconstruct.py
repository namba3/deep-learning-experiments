import argparse
import json
import os
import re
import sys
import tomllib
from collections import Counter

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torchvision import datasets, transforms
from torchvision.utils import save_image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not __package__ and PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)
try:
	from .train import (
		DATASET_IDS, ImageAE, HuggingFaceImageDataset, NETWORK_CONFIG_METADATA_KEY,
		NETWORK_CONFIG_VERSION, NETWORK_CONFIG_VERSION_METADATA_KEY,
		LEGACY_NETWORK_CONFIG_METADATA_KEY, LEGACY_NETWORK_CONFIG_VERSION_METADATA_KEY,
		canonical_decoder_type, canonical_encoder_type, corrupt_encoder_input,
	)
except ImportError:  # noqa: E402
	from image_ae.train import (
		DATASET_IDS, ImageAE, HuggingFaceImageDataset, NETWORK_CONFIG_METADATA_KEY,
		NETWORK_CONFIG_VERSION, NETWORK_CONFIG_VERSION_METADATA_KEY,
		LEGACY_NETWORK_CONFIG_METADATA_KEY, LEGACY_NETWORK_CONFIG_VERSION_METADATA_KEY,
		canonical_decoder_type, canonical_encoder_type, corrupt_encoder_input,
	)

NETWORK_CONFIG_KEYS = (
	"encoder", "decoder", "latent_channels", "bottleneck_channels",
	"hidden_channels", "encoder_blocks", "decoder_blocks", "downsample_stages",
	"encoder_layers", "encoder_window_size", "decoder_layers", "bucket_step",
)

def load_network_config_file(path):
	with open(path, "rb") as config_file:
		if path.lower().endswith(".toml"):
			config = tomllib.load(config_file)
		else:
			config = json.load(config_file)
	if "network" in config:
		config = config["network"]
	return config

def checkpoint_metadata(path):
	with safe_open(path, framework="pt", device="cpu") as checkpoint:
		return checkpoint.metadata() or {}

def network_config_from_checkpoint_name(path):
	"""Recover architecture settings from the standard training artifact name."""
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
		r"(?:_hidden-ch(?P<hidden_channels>\d+))?"
		r"",
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

def resolve_network_config(args, state_dict):
	metadata = checkpoint_metadata(args.weights)
	checkpoint_version = metadata.get(
		NETWORK_CONFIG_VERSION_METADATA_KEY,
		metadata.get(LEGACY_NETWORK_CONFIG_VERSION_METADATA_KEY),
	)
	checkpoint_config = None
	config_key = (
		NETWORK_CONFIG_METADATA_KEY
		if metadata.get(NETWORK_CONFIG_METADATA_KEY)
		else LEGACY_NETWORK_CONFIG_METADATA_KEY
	)
	if metadata.get(config_key):
		checkpoint_config = json.loads(metadata[config_key])

	config = {}
	if checkpoint_version == NETWORK_CONFIG_VERSION and checkpoint_config:
		config.update(checkpoint_config)
	elif checkpoint_version is None and checkpoint_config is None:
		filename_config = network_config_from_checkpoint_name(args.weights)
		if filename_config:
			config.update(filename_config)
			print("warning: restoring network config from checkpoint filename")
	else:
		message = (
			f"checkpoint network config version is incompatible: "
			f"found {checkpoint_version!r}, expected {NETWORK_CONFIG_VERSION!r}. "
			"Automatic architecture restoration is disabled. Provide all architecture "
			"options explicitly or use --network-config, then add "
			"--allow-incompatible-checkpoint to proceed."
		)
		if not args.allow_incompatible_checkpoint:
			raise ValueError(message)
		print(f"warning: {message}")

	if args.network_config:
		config.update(load_network_config_file(args.network_config))
	for key in NETWORK_CONFIG_KEYS + ("dataset", "image_size"):
		value = getattr(args, key, None)
		if value is not None:
			config[key] = value

	if "bucket_step" not in config:
		config["bucket_step"] = 32
	if "encoder_layers" not in config:
		config["encoder_layers"] = 2
	if "encoder_window_size" not in config:
		config["encoder_window_size"] = 8
	if "decoder_layers" not in config:
		config["decoder_layers"] = 2
	missing = [
		key for key in NETWORK_CONFIG_KEYS
		if key not in config and key not in {
			"bucket_step", "encoder_layers", "encoder_window_size", "decoder_layers",
		}
	]
	if missing:
		raise ValueError(
			"Missing network architecture settings: " + ", ".join(missing)
			+ ". Provide --network-config or explicit architecture options."
		)
	if "dataset" not in config:
		config["dataset"] = "flickr30k"
	if "image_size" not in config:
		config["image_size"] = 32 if config["dataset"] == "cifar10" else 256
	config["encoder"] = canonical_encoder_type(config["encoder"])
	config["decoder"] = canonical_decoder_type(config["decoder"])
	return config


def parse_args():
	parser = argparse.ArgumentParser(
		description="Encode and decode images with a trained autoencoder."
	)
	parser.add_argument("--weights", required=True, help="Path to a safetensors checkpoint.")
	parser.add_argument(
		"--dataset", choices=["flickr30k", "cifar10", "imagenet1k", "mini-imagenet"], default=None,
		help="Dataset to use. Restored from checkpoint when omitted.",
	)
	parser.add_argument("--data-dir", default="../cifar10/data")
	parser.add_argument(
		"--image-size", type=int, default=None,
		help="Evaluation crop size. Default: 32 for CIFAR-10, 256 for ImageNet datasets.",
	)
	parser.add_argument("--hf-cache-dir", default=None)
	parser.add_argument("--bucket-step", type=int, default=None)
	parser.add_argument(
		"--imagenet-val-samples", type=int, default=0,
		help="Number of ImageNet validation samples to use. 0 uses all samples.",
	)
	parser.add_argument(
		"--output",
		default="reconstruction_from_checkpoint.png",
		help="Output PNG path. Original images are in the first column, reconstructions in the second.",
	)
	parser.add_argument("--count", type=int, default=16, help="Number of test images.")
	parser.add_argument("--input-noise-std", type=float, default=0.0)
	parser.add_argument("--input-blur-sigma", type=float, default=0.0)
	parser.add_argument(
		"--input-bit-depth", type=int, default=0,
		help="Bit depth applied before encoding. Default: disabled; 6 is recommended; 0 disables it.",
	)
	parser.add_argument(
		"--downsample-stages", type=int, default=None,
		help="Number of spatial downsampling stages shared by encoder and decoder. Default: 3 (32x32 -> 4x4).",
	)
	parser.add_argument(
		"--encoder",
		choices=[
			"window_transformer", "residual_conv_ffn", "gated_residual_conv_ffn",
			"basic_cnn", "gated_residual_cnn", "efficient_residual",
			"cnn", "gated_cnn", "dc_ae",
		],
		default=None,
	)
	parser.add_argument(
		"--decoder",
		choices=["window_transformer", "basic_cnn", "residual_conv_ffn", "gated_residual_conv_ffn", "efficient_residual", "cnn", "dc_ae"],
		default=None,
	)
	parser.add_argument(
		"--latent-channels", "--latent-dim", dest="latent_channels", type=int, default=None,
		help="Number of channels in the spatial latent. Spatial size depends on input size and downsample stages.",
	)
	parser.add_argument("--bottleneck-channels", "--feature-channels", dest="bottleneck_channels", type=int, default=None)
	parser.add_argument("--hidden-channels", type=int, default=None)
	parser.add_argument(
		"--encoder-blocks-per-stage", "--encoder-blocks", dest="encoder_blocks", type=int, default=None,
	)
	parser.add_argument("--encoder-layers", type=int, default=None)
	parser.add_argument("--encoder-window-size", type=int, default=None)
	parser.add_argument("--decoder-layers", type=int, default=None)
	parser.add_argument(
		"--decoder-blocks-per-stage", "--decoder-blocks", dest="decoder_blocks", type=int, default=None,
	)
	parser.add_argument("--network-config", default=None, help="JSON or TOML network configuration file.")
	parser.add_argument(
		"--allow-incompatible-checkpoint", action="store_true",
		help="Allow execution with explicit architecture settings when checkpoint metadata is incompatible.",
	)
	parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
	return parser.parse_args()


def main():
	args = parse_args()
	state_dict = load_file(args.weights, device="cpu")
	config = resolve_network_config(args, state_dict)
	for key in NETWORK_CONFIG_KEYS + ("dataset", "image_size"):
		setattr(args, key, config[key])
	default_image_size = 32 if args.dataset == "cifar10" else 256
	args.image_size = default_image_size if args.image_size is None else args.image_size
	if args.image_size <= 0:
		raise ValueError("--image-size must be positive")
	if args.dataset == "cifar10" and args.image_size != 32:
		raise ValueError("CIFAR-10 currently requires --image-size 32")
	if args.bucket_step <= 0 or args.bucket_step % (2 ** args.downsample_stages) != 0:
		raise ValueError("--bucket-step must be positive and divisible by 2**downsample_stages")
	if args.image_size % (2 ** args.downsample_stages) != 0:
		raise ValueError(
			"--image-size must be divisible by 2**downsample_stages: "
			f"got image_size={args.image_size}, stages={args.downsample_stages}"
		)
	args.encoder = canonical_encoder_type(args.encoder)
	args.decoder = canonical_decoder_type(args.decoder)
	if args.count <= 0:
		raise ValueError("--count must be positive")
	if args.imagenet_val_samples < 0:
		raise ValueError("--imagenet-val-samples must be >= 0")
	if args.input_noise_std < 0 or args.input_blur_sigma < 0:
		raise ValueError("input noise std and blur sigma must be >= 0")
	if args.input_bit_depth < 0 or args.input_bit_depth > 8:
		raise ValueError("input bit depth must be between 0 and 8")
	if args.downsample_stages <= 0 or args.downsample_stages > 5:
		raise ValueError("--downsample-stages must be between 1 and 5")

	device = torch.device(
		"cuda" if args.device == "cuda" or (
			args.device == "auto" and torch.cuda.is_available()
		) else "cpu"
	)
	vae = "latent_logvar.weight" in state_dict or "conv_logvar.weight" in state_dict
	model = ImageAE(
		latent_channels=args.latent_channels,
		encoder_type=args.encoder,
		encoder_layers=args.encoder_layers,
		encoder_window_size=args.encoder_window_size,
		decoder_layers=args.decoder_layers,
		decoder_type=args.decoder,
		bottleneck_channels=args.bottleneck_channels,
		hidden_channels=args.hidden_channels,
		encoder_blocks=args.encoder_blocks,
		decoder_blocks=args.decoder_blocks,
		vae=vae,
		downsample_stages=args.downsample_stages,
	)
	model.load_state_dict(state_dict)
	model.to(device)
	model.eval()

	if args.dataset == "cifar10":
		dataset = datasets.CIFAR10(
			root=args.data_dir, train=False, download=True, transform=transforms.ToTensor(),
		)
	else:
		from datasets import load_dataset
		dataset_id = DATASET_IDS[args.dataset]
		hf_dataset = load_dataset(dataset_id, cache_dir=args.hf_cache_dir)
		if hasattr(hf_dataset, "keys"):
			validation_split = (
				"validation" if "validation" in hf_dataset
				else "test" if "test" in hf_dataset
				else next(iter(hf_dataset))
			)
			hf_dataset = hf_dataset[validation_split]
		if args.imagenet_val_samples > 0:
			hf_dataset = hf_dataset.select(
				range(min(args.imagenet_val_samples, len(hf_dataset)))
			)
		dataset = HuggingFaceImageDataset(
			hf_dataset,
			image_size=args.image_size,
			bucket_step=args.bucket_step,
			train=False,
		)
	if isinstance(dataset, torch.utils.data.IterableDataset):
		samples = []
		for sample in dataset:
			if len(samples) >= args.count:
				break
			samples.append(sample)
	else:
		count = min(args.count, len(dataset))
		if getattr(dataset, "bucket_ids", None) is not None:
			bucket = Counter(dataset.bucket_ids).most_common(1)[0][0]
			indices = [
				index for index, bucket_id in enumerate(dataset.bucket_ids)
				if bucket_id == bucket
			][:count]
			samples = [dataset[index] for index in indices]
		else:
			samples = [dataset[index] for index in range(count)]
	count = len(samples)
	images = torch.stack([sample[0] for sample in samples]).to(device)
	labels = [sample[1] for sample in samples]

	with torch.no_grad():
		encoder_images = corrupt_encoder_input(
			images, args.input_noise_std, args.input_blur_sigma, args.input_bit_depth,
		)
		latent = model.encode(encoder_images)
		reconstructions = model.decode(latent).clamp(0, 1)

	output_dir = os.path.dirname(args.output)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)
	comparison = torch.stack((images.cpu(), reconstructions.cpu()), dim=1).flatten(0, 1)
	save_image(comparison, args.output, nrow=2)
	print(f"device={device}, vae={vae}, images={count}")
	print(f"latent shape={tuple(latent.shape)}")
	print(f"saved reconstruction comparison to {args.output}")
	print(f"labels={labels}")


if __name__ == "__main__":
	main()
