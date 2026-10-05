import pytest
import torch

from cifar10.train import CIFAR10ViT
from cifar10.train_adapter import main as train_adapter_main
from core.mhla import attention as triton_attention
from core.mhla import triton_available, triton_unavailable_reason
from core.layers import Grid2DMHLA
from core.low_rank import inject_lora, mark_only_lora_trainable


@pytest.mark.parametrize("attention_type", ["full", "window", "window_mhla"])
def test_cifar10_vit_uses_hierarchical_attention_and_variable_depth(attention_type):
	model = CIFAR10ViT(
		patch_size=2,
		embed_dim=128,
		num_layers=6,
		num_heads=8,
		attention_type=attention_type,
		window_size=4,
	)
	assert model.pooling.attn.kv_heads == 4
	assert model.pooling.attn.num_heads == 8
	assert model.pooling.attn.head_gate is not None
	with torch.no_grad():
		pool_gate = 2.0 * torch.sigmoid(
			model.pooling.attn.head_gate(torch.zeros(1, 1, 128))
		)
	assert torch.allclose(pool_gate, torch.ones_like(pool_gate))
	images = torch.rand(2, 3, 32, 32)

	for depth in (1, 6):
		logits = model(images, depth=depth)
		assert logits.shape == (2, 10)

	attention = model.encoder.stages[0][0].attention
	with torch.no_grad():
		initial_gate = 2.0 * torch.sigmoid(
			attention.head_gate(torch.zeros(1, 256, 32))
		)
	assert torch.allclose(initial_gate, torch.ones_like(initial_gate))
	if attention_type == "window_mhla":
		assert model.encoder.stages[0][1].mhla is not None

	logits.square().mean().backward()
	assert all(
		parameter.grad is None or torch.isfinite(parameter.grad).all()
		for parameter in model.parameters()
	)


def test_cifar10_vit_supports_tiny_imagenet_image_and_class_shapes():
	model = CIFAR10ViT(
		patch_size=2,
		embed_dim=32,
		num_layers=1,
		num_heads=4,
		attention_type="full",
		window_size=4,
		img_size=64,
		num_classes=200,
	).eval()
	logits = model(torch.rand(1, 3, 64, 64))

	assert logits.shape == (1, 200)
	assert torch.isfinite(logits).all()


def test_grid2d_mhla_vectorized_backend_handles_rectangular_grid():
	module = Grid2DMHLA(
		dim=64,
		heads=8,
		kv_heads=4,
		block_size=4,
		backend="vectorized",
	)
	tokens = torch.randn(2, 15, 64, requires_grad=True)
	output = module(tokens, height=3, width=5)
	assert output.shape == tokens.shape
	output.square().mean().backward()
	assert tokens.grad is not None
	assert torch.isfinite(tokens.grad).all()


def test_cifar10_vit_lora_targets_attention_and_freezes_base():
	torch.manual_seed(0)
	model = CIFAR10ViT(
		patch_size=2,
		embed_dim=128,
		num_layers=1,
		num_heads=8,
		attention_type="full",
		window_size=4,
	).eval()
	matched = inject_lora(
		model,
		[r"\.attention\.(qkv|output)$", r"^pooling\.attn\.(q_proj|kv_proj|out_proj)$"],
		rank=2,
	)
	trainable = mark_only_lora_trainable(model)

	assert len(matched) == 5
	assert trainable > 0
	assert all(
		parameter.requires_grad == ("lora_" in name)
		for name, parameter in model.named_parameters()
	)
	logits = model(torch.randn(2, 3, 32, 32))
	logits.square().mean().backward()
	assert all(
		parameter.grad is None or torch.isfinite(parameter.grad).all()
		for parameter in model.parameters()
	)


def test_train_adapter_requires_adapter_and_base_checkpoint():
	with pytest.raises(ValueError, match="requires --adapter"):
		train_adapter_main([
			"--dry-run", "--epochs", "1", "--batch-size", "2",
		])

	with pytest.raises(ValueError, match="requires --init-checkpoint or --resume"):
		train_adapter_main([
			"--dry-run", "--adapter", "lora", "--lora-rank", "2",
			"--epochs", "1", "--batch-size", "2",
		])


def test_train_adapter_supports_random_base_initialization(tmp_path):
	output_dir = str(tmp_path / "adapter-run")
	result = train_adapter_main([
		"--dry-run", "--base-init", "random", "--adapter", "lora",
		"--lora-rank", "2", "--epochs", "1", "--batch-size", "2",
		"--output-dir", output_dir,
	])

	assert result is None


def test_train_adapter_rejects_checkpoint_with_random_base_initialization(tmp_path):
	with pytest.raises(ValueError, match="cannot be combined"):
		train_adapter_main([
			"--dry-run", "--base-init", "random", "--init-checkpoint",
			str(tmp_path / "base.safetensors"), "--adapter", "lora",
			"--lora-rank", "2", "--epochs", "1", "--batch-size", "2",
		])


def test_train_adapter_restricts_lora_warm_initialization_to_gated_adapters():
	with pytest.raises(ValueError, match="only supported"):
		train_adapter_main([
			"--dry-run", "--base-init", "random", "--adapter", "lora",
			"--lora-rank", "2", "--adapter-init", "lora_warm",
		])


def test_grid2d_mhla_explicit_triton_reports_cpu_requirement():
	if torch.cuda.is_available():
		pytest.skip("This test checks the CPU-side explicit-backend error")
	module = Grid2DMHLA(dim=64, heads=8, kv_heads=4, backend="triton")
	with pytest.raises(RuntimeError, match="backend='triton'"):
		module(torch.randn(1, 16, 64), height=4, width=4)


def test_mhla_triton_contract_rejects_non_contiguous_inputs():
	query = torch.randn(1, 2, 4, 8)
	key = torch.randn(1, 2, 4, 8)
	value = torch.randn(1, 2, 4, 8).transpose(2, 3).contiguous().transpose(2, 3)

	assert value.shape == query.shape
	assert not value.is_contiguous()
	assert isinstance(triton_available(query, key, value), bool)
	assert not triton_available(query, key, value)
	assert "contiguous flags" in triton_unavailable_reason(query, key, value)
	with pytest.raises(RuntimeError, match="contiguous Q/K/V"):
		triton_attention(
			query, key, value,
			torch.zeros(1, 4, dtype=torch.long),
			torch.ones(1, 4, dtype=torch.bool),
			heads=2,
			kv_heads=2,
		)
