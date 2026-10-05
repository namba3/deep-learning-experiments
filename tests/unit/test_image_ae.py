import pytest
import torch

from optimizers.lr_scheduler import LearningRateSchedule
from runtime.checkpoint import load_training_state
from core.low_rank import (
	inject_adapter,
	inject_lora,
	mark_only_adapter_trainable,
	mark_only_lora_trainable,
)
from image_ae.train import (
	AspectRatioBatchSampler,
	DEFAULT_LORA_TARGETS,
	ImageAE,
	assign_bucket,
	compute_channel_decorrelation_loss,
	compute_wavelet_loss,
	checkpoint_training_config,
	group_norm,
	make_bucket_shapes,
	parse_args,
	save_training_checkpoint,
	WindowSelfAttention,
)
from image_ae.train_adapter import main as train_adapter_main


def test_default_image_ae_adapter_targets_include_ffn_output_linear():
	assert r"\.ffn\.3$" in DEFAULT_LORA_TARGETS
	assert r"\.ffn\.2$" not in DEFAULT_LORA_TARGETS


def test_resolution_buckets_are_stride_aligned_and_aspect_aware():
	shapes = make_bucket_shapes(256, 32)
	assert len(shapes) == 5
	assert all(height % 32 == 0 and width % 32 == 0 for height, width in shapes)
	assert assign_bucket(640, 360, shapes) == 4
	assert assign_bucket(360, 640, shapes) == 0


def test_aspect_ratio_batch_sampler_keeps_bucket_members_together():
	sampler = AspectRatioBatchSampler([0, 1, 0, 1, 0, 1], batch_size=2, shuffle=False)
	batches = list(sampler)
	assert batches == [[0, 2], [1, 3]]
	for batch in batches:
		assert len({sampler.bucket_ids[index] for index in batch}) == 1


def test_image_ae_accepts_rectangular_resolution():
	model = ImageAE(
		latent_channels=8,
		bottleneck_channels=64,
		encoder_type="window_transformer",
		decoder_type="window_transformer",
		encoder_layers=2,
		decoder_layers=2,
		downsample_stages=3,
	)
	image = torch.rand(2, 3, 64, 48)
	reconstruction, latent = model(image)
	assert reconstruction.shape == image.shape
	assert latent.shape == (2, 8, 8, 6)
	loss = reconstruction.square().mean() + latent.square().mean()
	loss.backward()
	assert all(
		parameter.grad is None or torch.isfinite(parameter.grad).all()
		for parameter in model.parameters()
	)


def test_image_ae_lora_targets_window_transformer_linears():
	torch.manual_seed(0)
	model = ImageAE(
		latent_channels=16,
		bottleneck_channels=64,
		encoder_layers=1,
		decoder_layers=1,
		encoder_window_size=4,
		encoder_type="window_transformer",
		decoder_type="window_transformer",
	).eval()
	matched = inject_lora(
		model,
		[r"\.attention\.(qkv|output)$", r"\.ffn\.2$"],
		rank=2,
	)
	trainable = mark_only_lora_trainable(model)

	assert matched
	assert trainable == sum(
		parameter.numel()
		for name, parameter in model.named_parameters()
		if "lora_" in name
	)
	reconstruction, latent = model(torch.randn(1, 3, 32, 32))[:2]
	assert reconstruction.shape == (1, 3, 32, 32)
	assert latent.shape[1] == 16
	reconstruction.square().mean().backward()
	assert all(
		parameter.grad is None or torch.isfinite(parameter.grad).all()
		for parameter in model.parameters()
	)


def test_image_ae_rglu_lora_forward_backward():
	torch.manual_seed(1)
	model = ImageAE(
		latent_channels=8,
		bottleneck_channels=64,
		encoder_layers=1,
		decoder_layers=1,
		encoder_window_size=4,
		encoder_type="window_transformer",
		decoder_type="window_transformer",
	).eval()
	matched = inject_adapter(
		model,
		"rglu_lora",
		[r"\.attention\.(qkv|output)$", r"\.ffn\.2$"],
		rank=2,
	)
	trainable = mark_only_adapter_trainable(model)

	assert matched
	assert trainable > 0
	reconstruction, latent = model(torch.randn(1, 3, 32, 32))[:2]
	assert reconstruction.shape == (1, 3, 32, 32)
	assert latent.shape[1] == 8
	reconstruction.square().mean().backward()
	assert all(
		parameter.grad is None or torch.isfinite(parameter.grad).all()
		for parameter in model.parameters()
	)


def test_image_ae_minimum_resolution_matches_downsample_stages():
	model = ImageAE(
		latent_channels=4,
		bottleneck_channels=16,
		encoder_type="residual_conv_ffn",
		decoder_type="residual_conv_ffn",
		downsample_stages=3,
	)
	image = torch.rand(1, 3, 8, 16)
	reconstruction, latent = model(image)

	assert reconstruction.shape == image.shape
	assert latent.shape == (1, 4, 1, 2)
	assert torch.isfinite(reconstruction).all()
	assert torch.isfinite(latent).all()

	with pytest.raises(ValueError, match="divisible by 2\\*\\*downsample_stages"):
		model(torch.rand(1, 3, 7, 16))


def test_image_ae_rejects_non_rgb_or_non_nchw_input():
	model = ImageAE(
		latent_channels=4,
		bottleneck_channels=16,
		encoder_type="residual_conv_ffn",
		decoder_type="residual_conv_ffn",
		downsample_stages=3,
	)
	with pytest.raises(ValueError, match="RGB NCHW"):
		model(torch.rand(1, 1, 8, 8))


def test_channel_decorrelation_loss_is_zero_and_finite_for_one_channel():
	latent = torch.randn(2, 1, 4, 4, requires_grad=True)
	loss = compute_channel_decorrelation_loss(latent)

	assert loss.item() == 0.0
	assert torch.isfinite(loss)
	loss.backward()
	assert latent.grad is not None
	assert torch.equal(latent.grad, torch.zeros_like(latent.grad))


def test_group_norm_and_image_ae_reject_invalid_channels():
	norm = group_norm(48)
	assert norm.num_groups == 24
	with pytest.raises(ValueError, match="channels must be positive"):
		group_norm(0)
	for latent_channels in (0, -1):
		with pytest.raises(ValueError, match="latent_channels must be positive"):
			ImageAE(latent_channels=latent_channels)


def test_image_ae_checkpoint_writes_model_and_full_resume_sidecar(tmp_path, monkeypatch):
	monkeypatch.setattr("sys.argv", ["image_ae.train"])
	args = parse_args()
	parameter = torch.nn.Parameter(torch.ones(2))
	optimizer = torch.optim.AdamW([parameter], lr=0.1)
	scheduler = LearningRateSchedule(
		optimizer, "cosine", total_steps=10, warmup_steps=2,
	)
	parameter.grad = torch.ones_like(parameter)
	optimizer.step()
	scheduler.step(1)
	checkpoint = tmp_path / "checkpoint_latest.safetensors"

	save_training_checkpoint(
		{"weight": torch.ones(1)}, checkpoint, args,
		optimizer=optimizer, scheduler=scheduler,
		epoch=2, global_step=7,
	)

	assert checkpoint.is_file()
	state = load_training_state(checkpoint)
	assert state is not None
	assert state["epoch"] == 2
	assert state["global_step"] == 7
	assert state["scheduler"]["last_step"] == 1


def test_image_ae_parser_accepts_apollo_settings(monkeypatch):
	monkeypatch.setattr(
		"sys.argv",
		[
			"image_ae.train", "--optimizer", "APOLLO-Mini",
			"--apollo-rank", "4", "--apollo-scale", "0.5",
			"--apollo-update-proj-gap", "10",
			"--apollo-projection-refresh-mode", "smooth",
			"--apollo-projection-refresh-window", "4",
			"--apollo-projection-refresh-mix", "stochastic",
			"--apollo-projection-refresh-state", "transport",
			"--apollo-orthogonal-refresh-rate", "0.05",
			"--apollo-orthogonal-refresh-direction", "loss_directed",
			"--apollo-disable-norm-growth-limiter",
		],
	)
	args = parse_args()

	assert args.optimizer == "APOLLO-Mini"
	assert args.apollo_rank == 4
	assert args.apollo_scale == 0.5
	assert args.apollo_update_proj_gap == 10
	assert args.apollo_projection_refresh_mode == "smooth"
	assert args.apollo_projection_refresh_window == 4
	assert args.apollo_projection_refresh_mix == "stochastic"
	assert args.apollo_projection_refresh_state == "transport"
	assert args.apollo_orthogonal_refresh_rate == 0.05
	assert args.apollo_orthogonal_refresh_direction == "loss_directed"
	assert args.apollo_disable_norm_growth_limiter is True


def test_image_ae_parser_accepts_lora_settings(monkeypatch):
	monkeypatch.setattr(
		"sys.argv",
		[
			"image_ae.train_adapter",
			"--lora-base-checkpoint", "base.safetensors",
			"--lora-rank", "4",
			"--lora-alpha", "8",
			"--lora-dropout", "0.1",
			"--lora-target", r"\.attention\.qkv$",
			"--lora-target", r"\.ffn\.2$",
			"--adapter-init", "lora_warm",
		],
	)
	args = parse_args(adapter_only=True)

	assert args.lora_base_checkpoint == "base.safetensors"
	assert args.lora_rank == 4
	assert args.lora_alpha == 8
	assert args.lora_dropout == 0.1
	assert args.lora_target == [r"\.attention\.qkv$", r"\.ffn\.2$"]
	assert args.adapter_init == "lora_warm"
	assert checkpoint_training_config(args)["adapter_init"] == "lora_warm"


def test_image_ae_normal_parser_rejects_adapter_options(monkeypatch):
	monkeypatch.setattr(
		"sys.argv",
		["image_ae.train", "--adapter", "lora", "--lora-rank", "1"],
	)
	with pytest.raises(SystemExit):
		parse_args()


def test_image_ae_adapter_entrypoint_requires_adapter():
	with pytest.raises(ValueError, match="requires --adapter"):
		train_adapter_main(["--dry-run"])


def test_image_ae_parser_accepts_cifar10_sample_limits(monkeypatch):
	monkeypatch.setattr(
		"sys.argv",
		[
			"image_ae.train", "--dataset", "cifar10",
			"--cifar10-train-samples", "32",
			"--cifar10-val-samples", "16",
		],
	)
	args = parse_args()

	assert args.cifar10_train_samples == 32
	assert args.cifar10_val_samples == 16
	assert checkpoint_training_config(args)["cifar10_train_samples"] == 32


def test_image_ae_parser_accepts_came_lrsf_settings(monkeypatch):
	monkeypatch.setattr(
		"sys.argv",
		[
			"image_ae.train", "--optimizer", "CAME-LRSF",
			"--came-lrsf-rank", "4", "--came-lrsf-beta1", "0.85",
			"--came-lrsf-orthogonal-refresh-rate", "0.05",
			"--came-lrsf-orthogonal-refresh-direction", "loss_directed",
			"--came-lrsf-orthogonal-refresh-signal", "effective_update",
		],
	)
	args = parse_args()

	assert args.optimizer == "CAME-LRSF"
	assert args.came_lrsf_rank == 4
	assert args.came_lrsf_beta1 == 0.85
	assert args.came_lrsf_orthogonal_refresh_rate == 0.05
	assert args.came_lrsf_orthogonal_refresh_direction == "loss_directed"
	assert args.came_lrsf_orthogonal_refresh_signal == "effective_update"


def test_wavelet_loss_rejects_levels_beyond_spatial_resolution():
	target = torch.rand(1, 3, 4, 4)
	reconstruction = target.clone().requires_grad_()
	loss = compute_wavelet_loss(reconstruction, target, levels=2)
	assert torch.isfinite(loss)
	loss.backward()
	assert reconstruction.grad is not None
	with pytest.raises(ValueError, match="exceed the spatial limit"):
		compute_wavelet_loss(reconstruction.detach(), target, levels=3)
	with pytest.raises(ValueError, match="height and width >= 2"):
		compute_wavelet_loss(torch.rand(1, 3, 1, 4), torch.rand(1, 3, 1, 4))


def test_window_self_attention_handles_minimum_and_non_multiple_rectangles():
	layer = WindowSelfAttention(
		embed_dim=32,
		num_heads=4,
		window_size=4,
		shift_size=2,
	)
	inputs = torch.randn(2, 1, 3, 32, requires_grad=True)
	output = layer(inputs)

	assert output.shape == inputs.shape
	assert torch.isfinite(output).all()
	output.square().mean().backward()
	assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


def test_window_self_attention_rejects_invalid_head_and_shift_boundaries():
	with pytest.raises(ValueError, match="head_dim must be divisible by 4"):
		WindowSelfAttention(embed_dim=24, num_heads=4, window_size=2)
	with pytest.raises(ValueError, match="shift_size"):
		WindowSelfAttention(embed_dim=32, num_heads=4, window_size=2, shift_size=2)

	layer = WindowSelfAttention(
		embed_dim=16,
		num_heads=4,
		window_size=1,
	)
	output = layer(torch.randn(1, 1, 1, 16))
	assert output.shape == (1, 1, 1, 16)
	assert torch.isfinite(output).all()
