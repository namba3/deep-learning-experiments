import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchinfo import summary
from safetensors.torch import save_file, load_file
import argparse
import os
from datetime import datetime
from time import perf_counter
from torch.utils.tensorboard import SummaryWriter
from aptx_activation import APTx

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.progress import RichProgress
from runtime.data import build_dataloader_options
from runtime.metrics import build_standard_progress_rows, write_standard_training_metrics
from runtime.memory import maybe_collect_memory
from runtime.preflight import build_training_preflight
from runtime.validation import ValidationTimer, build_validation_report
from runtime.run import RunRecorder
from runtime.signal import GracefulStop
from runtime.sampler import ResumableRandomSampler
from runtime.checkpoint import (
    load_training_state,
    make_training_state,
    restore_rng_state,
    save_training_state,
)
from runtime.config import (
    apply_saved_config,
    checkpoint_config_metadata,
    read_checkpoint_config,
)
from runtime.device import add_device_argument, resolve_device
from optimizers.factory import (
    add_optimizer_argument,
    build_optimizer,
    is_schedule_free_optimizer,
)
from optimizers.lr_scheduler import add_lr_scheduler_arguments, build_lr_scheduler
from core.utils import (
    GradSignFlipNoiseInjector,
    WeightDecayScheduler,
    ModelSnapshot,
    build_parameter_groups,
    DepthDistributionScheduler,
    convert_linear_to_bf16,
    format_bytes,
    compact_state_dict,
    print_model_info,
)
from core.layers import (
	GatedLinear,
	GatedConv2d,
	RMSNorm2d,
	AttentionPoolingWithGroupedQueryAttention,
	RotaryEmbedding2D,
	Grid2DMHLA,
)
from cifar10.adapter_training import (
    add_adapter_arguments,
    build_adapter_parameter_groups,
    enable_adapter,
    resolve_adapter_config,
)

# ========= ハイパーパラメータ =========
NUM_EPOCHS = 100
BATCH_SIZE =  256
WEIGHT_DECAY = 1e-1
IMG_SIZE = 32
PATCH_SIZE = 2 # image_aeと同じstem stride。最初のattention gridは16x16。
EMBED_DIM = 256
NUM_LAYERS = 6 # Transformerブロックの数
NUM_HEADS = 8 # マルチヘッドアテンションのヘッド数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CIFAR10_CONFIG_METADATA_KEY = "cifar10.config"


def resolve_resume_path(path, output_dir):
    if os.path.isfile(path):
        return path
    candidate = os.path.join(output_dir, path)
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f"Resume checkpoint not found: {path}")

# デフォルトは全FP32。BF16は --bf16 指定時だけTransformer内部で使用する。

# ========= モデル定義 =========
class PatchEmbed(nn.Module):
    """image_aeと同じ入力stemを持つfeature-map embedder。

    まずRGBを16chへ写像し、その後に4x4/stride=2のGatedConvで
    ``embed_dim``（C/4）へdownsampleする。従来の単純なpatch分割よりも
    局所畳み込みを先に行うため、階層encoderの最初のstageへ自然に接続できる。
    """
    def __init__(self, img_size=32, patch_size=2, in_chans=3, embed_dim=128):
        super().__init__()
        if patch_size != 2:
            raise ValueError(
                "PatchEmbed now follows image_ae and requires patch_size=2"
            )
        if img_size % 2 != 0:
            raise ValueError("img_size must be divisible by 2")
        self.grid_size = img_size // 2
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Sequential(
            GatedConv2d(in_chans, 16, kernel_size=3, padding=1),
            GatedConv2d(16, embed_dim, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(min(32, embed_dim), embed_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        x = self.proj(x)                       # (B, C/4, 16, 16)
        return x.flatten(2).transpose(1, 2)    # (B, N, C/4)

class FullSelfAttention(nn.Module):
	"""Full-token SDPA with 2D RoPE and query-dependent head gates."""
	def __init__(self, embed_dim, num_heads, grid_size, dropout=0.1):
		super().__init__()
		if embed_dim % num_heads != 0:
			raise ValueError("embed_dim must be divisible by num_heads")
		head_dim = embed_dim // num_heads
		if head_dim % 4 != 0:
			raise ValueError("head_dim must be divisible by 4 for 2D RoPE")
		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.head_dim = head_dim
		self.dropout = dropout
		self.qkv = nn.Linear(embed_dim, embed_dim * 3)
		self.output = nn.Linear(embed_dim, embed_dim)
		self.head_gate = nn.Linear(embed_dim, num_heads)
		self.head_gate._preserve_init = True
		nn.init.zeros_(self.head_gate.weight)
		nn.init.zeros_(self.head_gate.bias)
		self.rotary_embedding = RotaryEmbedding2D(
			head_dim, height=grid_size, width=grid_size,
		)

	def forward(self, x):
		batch, tokens, channels = x.shape
		qkv = self.qkv(x).reshape(
			batch, tokens, 3, self.num_heads, self.head_dim,
		).permute(2, 0, 3, 1, 4)
		query, key = self.rotary_embedding(qkv[0], qkv[1])
		attention = F.scaled_dot_product_attention(
			query,
			key,
			qkv[2],
			dropout_p=self.dropout if self.training else 0.0,
		)
		# Keep the gate at exactly 1x at initialization while allowing each
		# head to be suppressed or amplified during training.
		gate = 2.0 * torch.sigmoid(self.head_gate(x))
		attention = attention * gate.transpose(1, 2).unsqueeze(-1)
		attention = attention.transpose(1, 2).reshape(batch, tokens, channels)
		return self.output(attention)


class FullAttentionTransformerBlock(nn.Module):
	"""Pre-norm full-attention block used at one spatial hierarchy level."""
	def __init__(self, embed_dim, num_heads, grid_size, drop_out=0.1):
		super().__init__()
		self.norm1 = nn.RMSNorm(embed_dim)
		self.attention = FullSelfAttention(
			embed_dim, num_heads, grid_size=grid_size, dropout=drop_out,
		)
		self.attention_dropout = nn.Dropout(drop_out)
		self.norm2 = nn.RMSNorm(embed_dim)
		self.ffn = nn.Sequential(
			GatedLinear(embed_dim, embed_dim * 3),
			nn.Dropout(drop_out),
			nn.Linear(embed_dim * 3, embed_dim, bias=False),
		)
		self.ffn_dropout = nn.Dropout(drop_out)

	def forward(self, x):
		batch, channels, height, width = x.shape
		tokens = x.flatten(2).transpose(1, 2)
		tokens = tokens + self.attention_dropout(
			self.attention(self.norm1(tokens))
		)
		tokens = tokens + self.ffn_dropout(
			self.ffn(self.norm2(tokens))
		)
		return tokens.transpose(1, 2).reshape(batch, channels, height, width)


def window_partition(x, window_size):
	"""Partition an NHWC tensor into non-overlapping local windows."""
	batch, height, width, channels = x.shape
	if height % window_size or width % window_size:
		raise ValueError("window_partition requires divisible dimensions")
	return x.view(
		batch, height // window_size, window_size,
		width // window_size, window_size, channels,
	).permute(0, 1, 3, 2, 4, 5).reshape(
		batch * (height // window_size) * (width // window_size),
		window_size * window_size, channels,
	)


def window_reverse(windows, window_size, height, width, batch):
	"""Reverse :func:`window_partition` for a padded spatial grid."""
	return windows.view(
		batch, height // window_size, width // window_size,
		window_size, window_size, -1,
	).permute(0, 1, 3, 2, 4, 5).reshape(batch, height, width, -1)


class WindowSelfAttention(nn.Module):
	"""Local SDPA with alternating shifted windows, 2D RoPE, and head gates."""
	def __init__(self, embed_dim, num_heads, window_size=4, shift_size=0,
				 dropout=0.1):
		super().__init__()
		if embed_dim % num_heads != 0:
			raise ValueError("embed_dim must be divisible by num_heads")
		head_dim = embed_dim // num_heads
		if head_dim % 4 != 0:
			raise ValueError("head_dim must be divisible by 4 for 2D RoPE")
		if window_size <= 0 or not 0 <= shift_size < window_size:
			raise ValueError("window_size must be positive and shift valid")
		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.head_dim = head_dim
		self.window_size = window_size
		self.shift_size = shift_size
		self.dropout = dropout
		self.qkv = nn.Linear(embed_dim, embed_dim * 3)
		self.output = nn.Linear(embed_dim, embed_dim)
		self.head_gate = nn.Linear(embed_dim, num_heads)
		self.head_gate._preserve_init = True
		nn.init.zeros_(self.head_gate.weight)
		nn.init.zeros_(self.head_gate.bias)
		self.rotary_embedding = RotaryEmbedding2D(
			head_dim, height=window_size, width=window_size,
		)
		self._mask_cache = {}

	def _attention_mask(self, height, width, device, dtype):
		cache_key = (height, width, device.type, device.index, dtype)
		cached = self._mask_cache.get(cache_key)
		if cached is not None:
			return cached
		window_size = self.window_size
		valid = torch.ones((1, height, width, 1), device=device, dtype=torch.bool)
		valid = F.pad(
			valid,
			(0, 0, 0, (-width) % window_size, 0, (-height) % window_size),
		)
		padded_height, padded_width = valid.shape[1:3]
		if self.shift_size:
			valid = torch.roll(
				valid,
				shifts=(-self.shift_size, -self.shift_size),
				dims=(1, 2),
			)
			region = torch.zeros(
				(1, padded_height, padded_width, 1),
				device=device,
				dtype=torch.int64,
			)
			count = 0
			for height_slice in (
				slice(0, -window_size),
				slice(-window_size, -self.shift_size),
				slice(-self.shift_size, None),
			):
				for width_slice in (
					slice(0, -window_size),
					slice(-window_size, -self.shift_size),
					slice(-self.shift_size, None),
				):
					region[:, height_slice, width_slice, :] = count
					count += 1
			region_windows = window_partition(region, window_size).squeeze(-1)
			region_delta = (
				region_windows.unsqueeze(1) - region_windows.unsqueeze(2)
			)
			mask = torch.zeros_like(region_delta, dtype=dtype)
			mask.masked_fill_(region_delta.ne(0), float("-inf"))
		else:
			window_count = padded_height // window_size * padded_width // window_size
			mask = torch.zeros(
				(window_count, window_size * window_size, window_size * window_size),
				device=device,
				dtype=dtype,
			)
		valid_windows = window_partition(valid, window_size).squeeze(-1)
		mask = mask.masked_fill(~valid_windows[:, None, :], float("-inf"))
		self._mask_cache[cache_key] = mask
		return mask

	def forward(self, x):
		batch, height, width, channels = x.shape
		window_size = self.window_size
		padded = F.pad(
			x,
			(0, 0, 0, (-width) % window_size, 0, (-height) % window_size),
		)
		padded_height, padded_width = padded.shape[1:3]
		if self.shift_size:
			padded = torch.roll(
				padded,
				shifts=(-self.shift_size, -self.shift_size),
				dims=(1, 2),
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
			query,
			key,
			qkv[2],
			attn_mask=mask,
			dropout_p=self.dropout if self.training else 0.0,
		)
		gate = 2.0 * torch.sigmoid(self.head_gate(windows))
		attention = attention * gate.transpose(1, 2).unsqueeze(-1)
		attention = attention.transpose(1, 2).reshape(
			window_batch, window_size * window_size, channels,
		)
		output = self.output(attention)
		output = window_reverse(
			output, window_size, padded_height, padded_width, batch,
		)
		if self.shift_size:
			output = torch.roll(
				output,
				shifts=(self.shift_size, self.shift_size),
				dims=(1, 2),
			)
		return output[:, :height, :width, :]


class WindowAttentionTransformerBlock(nn.Module):
	"""Pre-norm shifted-window block for one spatial hierarchy level."""
	def __init__(self, embed_dim, num_heads, window_size=4,
				 shift_size=0, drop_out=0.1):
		super().__init__()
		self.norm1 = nn.RMSNorm(embed_dim)
		self.attention = WindowSelfAttention(
			embed_dim,
			num_heads,
			window_size=window_size,
			shift_size=shift_size,
			dropout=drop_out,
		)
		self.attention_dropout = nn.Dropout(drop_out)
		self.norm2 = nn.RMSNorm(embed_dim)
		self.ffn = nn.Sequential(
			GatedLinear(embed_dim, embed_dim * 3),
			nn.Dropout(drop_out),
			nn.Linear(embed_dim * 3, embed_dim, bias=False),
		)
		self.ffn_dropout = nn.Dropout(drop_out)

	def forward(self, x):
		batch, channels, height, width = x.shape
		nhwc = x.permute(0, 2, 3, 1)
		nhwc = self.norm1(nhwc)
		x = x + self.attention_dropout(self.attention(nhwc).permute(0, 3, 1, 2))
		nhwc = x.permute(0, 2, 3, 1)
		x = x + self.ffn_dropout(self.ffn(self.norm2(nhwc)).permute(0, 3, 1, 2))
		return x.reshape(batch, channels, height, width)


class WindowMHLACompositeBlock(nn.Module):
	"""Window block with an optional 2D-grid MHLA residual path."""
	def __init__(self, embed_dim, num_heads, window_size=4, shift_size=0,
				 mhla_block_size=8, use_mhla=False, mhla_backend="auto",
				 drop_out=0.1):
		super().__init__()
		self.window_block = WindowAttentionTransformerBlock(
			embed_dim,
			num_heads,
			window_size=window_size,
			shift_size=shift_size,
			drop_out=drop_out,
		)
		# Preserve the same inspection path as the pure window block.
		self.attention = self.window_block.attention
		self.mhla_norm = nn.RMSNorm(embed_dim) if use_mhla else None
		self.mhla = (
			Grid2DMHLA(
				dim=embed_dim,
				heads=num_heads,
				kv_heads=max(num_heads // 2, 1),
				block_size=mhla_block_size,
				backend=mhla_backend,
			)
			if use_mhla else None
		)
		self.mhla_dropout = nn.Dropout(drop_out) if use_mhla else None

	def forward(self, x):
		x = self.window_block(x)
		if self.mhla is None:
			return x
		batch, channels, height, width = x.shape
		tokens = x.flatten(2).transpose(1, 2)
		mhla_output = self.mhla(
			self.mhla_norm(tokens), height, width,
		)
		x = tokens + self.mhla_dropout(mhla_output)
		return x.transpose(1, 2).reshape(batch, channels, height, width)


class HierarchicalFullAttentionEncoder(nn.Module):
	"""CIFAR encoder with full attention at each resolution stage.

	The total ``num_layers`` blocks are distributed across three stages.  The
	``depth`` argument keeps the existing variable-depth training behavior by
	activating only the first ``depth`` blocks while retaining all downsampling
	steps.
	"""
	def __init__(self, patch_size=2, feature_channels=256, num_layers=6,
				 num_heads=8, drop_out=0.1, compute_dtype=None,
				 attention_type="full", window_size=4, mhla_block_size=None,
				 mhla_backend="auto", img_size=IMG_SIZE):
		super().__init__()
		if attention_type not in ("full", "window", "window_mhla"):
			raise ValueError("attention_type must be 'full', 'window', or 'window_mhla'")
		if window_size <= 0:
			raise ValueError("window_size must be positive")
		if mhla_block_size is None:
			mhla_block_size = window_size * 2
		if mhla_block_size <= 0:
			raise ValueError("mhla_block_size must be positive")
		if mhla_backend not in ("auto", "vectorized", "triton"):
			raise ValueError("mhla_backend must be 'auto', 'vectorized', or 'triton'")
		if feature_channels <= 0 or feature_channels % 32 != 0:
			raise ValueError("feature_channels must be a positive multiple of 32")
		if num_layers < 1:
			raise ValueError("num_layers must be >= 1")
		if num_heads < 4 or num_heads % 4 != 0:
			raise ValueError("num_heads must be a positive multiple of 4")
		if feature_channels % num_heads != 0:
			raise ValueError("feature_channels must be divisible by num_heads")
		if img_size <= 0 or img_size % patch_size != 0:
			raise ValueError("img_size must be positive and divisible by patch_size")
		stage_channels = (
			feature_channels // 4,
			feature_channels // 2,
			feature_channels,
		)
		stage_heads = (num_heads // 4, num_heads // 2, num_heads)
		grid_size = img_size // patch_size
		self.num_layers = num_layers
		self.img_size = img_size
		self.compute_dtype = compute_dtype
		self.attention_type = attention_type
		self.window_size = window_size
		self.mhla_block_size = mhla_block_size
		self.mhla_backend = mhla_backend
		self.patch_embed = PatchEmbed(
			img_size=img_size,
			patch_size=patch_size,
			in_chans=3,
			embed_dim=stage_channels[0],
		)
		base_layers, remainder = divmod(num_layers, len(stage_channels))
		stage_layer_counts = tuple(
			base_layers + (index < remainder)
			for index in range(len(stage_channels))
		)
		self.stage_layer_counts = stage_layer_counts
		self.stages = nn.ModuleList([
			nn.ModuleList([
				(
					FullAttentionTransformerBlock(
						channels, heads,
						grid_size=grid_size // (2 ** index),
						drop_out=drop_out,
					)
					if attention_type == "full" else WindowMHLACompositeBlock(
						channels, heads,
						window_size=window_size,
						shift_size=0 if block_index % 2 == 0 else window_size // 2,
						mhla_block_size=mhla_block_size,
						mhla_backend=mhla_backend,
						use_mhla=(
							attention_type == "window_mhla"
							and block_index % 2 == 1
						),
						drop_out=drop_out,
					)
				)
				for block_index in range(layer_count)
			])
			for index, (channels, heads, layer_count) in enumerate(
				zip(stage_channels, stage_heads, stage_layer_counts)
			)
		])
		self.downsample_stages = nn.ModuleList([
			nn.Sequential(
				nn.Conv2d(
					stage_channels[index], stage_channels[index + 1],
					kernel_size=4, stride=2, padding=1, bias=False,
				),
				RMSNorm2d(stage_channels[index + 1]),
				APTx(trainable=True),
			)
			for index in range(len(stage_channels) - 1)
		])
		self.output_norm = RMSNorm2d(feature_channels)

	def forward(self, x, depth=None):
		if depth is None:
			depth = self.num_layers
		if not isinstance(depth, int) or not 1 <= depth <= self.num_layers:
			raise ValueError(f"depth must be an integer in [1, {self.num_layers}]")
		x = self.patch_embed(x)
		batch, tokens, channels = x.shape
		grid_size = int(tokens ** 0.5)
		if grid_size * grid_size != tokens:
				raise ValueError("image classifier requires a square patch grid")
		x = x.transpose(1, 2).reshape(batch, channels, grid_size, grid_size)
		remaining = depth
		for index, stage in enumerate(self.stages):
			active_blocks = min(remaining, len(stage))
			for block in stage[:active_blocks]:
				x = block(x)
			remaining -= active_blocks
			if index < len(self.downsample_stages):
				x = self.downsample_stages[index](x)
		return self.output_norm(x).flatten(2).transpose(1, 2)


class CIFAR10ViT(nn.Module):
	def __init__(self, patch_size=2, embed_dim=128, num_layers=3, num_heads=8,
				 compute_dtype=None, attention_type="full", window_size=4,
				 mhla_block_size=None, mhla_backend="auto", img_size=IMG_SIZE,
				 num_classes=10):
		super().__init__()
		self.encoder = HierarchicalFullAttentionEncoder(
			patch_size=patch_size,
			feature_channels=embed_dim,
			num_layers=num_layers,
			num_heads=num_heads,
			drop_out=0.1,
			compute_dtype=compute_dtype,
				 attention_type=attention_type,
				 window_size=window_size,
				 mhla_block_size=mhla_block_size,
				 mhla_backend=mhla_backend,
				 img_size=img_size,
		)
		self.pooling = AttentionPoolingWithGroupedQueryAttention(
			embed_dim, num_heads, kv_heads=max(num_heads // 2, 1), dropout=0.1,
		)
		self.head = nn.Sequential(
			 nn.RMSNorm(embed_dim),
			 GatedLinear(embed_dim, embed_dim),
			 nn.Dropout(0.1),
			 nn.Linear(embed_dim, num_classes, bias=False),
		)

	def forward(self, x, depth=None):
		x = self.encoder(x, depth=depth)
		x, _ = self.pooling(x)
		return self.head(x)

# ========= 初期化関数 =========
def init_weights(m):
    if isinstance(m, nn.Linear) and not getattr(m, "_preserve_init", False):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

# ========= 評価関数 =========
def evaluate(model, loader, device, criterion, depth=None):
    model.eval()
    correct, total = 0, 0
    loss_total = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logit = model(images, depth=depth)
            labels_vector = nn.functional.one_hot(labels, num_classes=10).float()
            loss = criterion(logit, labels, labels_vector)
            loss_total += loss.item()
            preds = logit.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    loss_avg = loss_total / len(loader)
    return 100.0 * correct / total, loss_avg

# ========= メイントレーニング =========
def main(argv=None, *, adapter_only=False):
    script_name = "cifar10.train_adapter" if adapter_only else "cifar10.train"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", type=str, default=None,
        help=(
            "Load model weights. If a sibling .resume.pt exists, also restore "
            "optimizer, scheduler, epoch, and RNG state; otherwise use "
            "weights-only resume."
        ),
    )
    parser.add_argument(
        "--init-checkpoint", type=str, default=None,
        help=(
            "Initialize model weights only from a checkpoint; optimizer, "
            "scheduler, epoch, and RNG state are not restored."
        ),
    )
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="output",
                        help="保存先ディレクトリ")
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
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Deterministic train sampler seed. Default: process seed.",
    )
    parser.add_argument(
        "--checkpoint-interval-steps", type=int, default=0,
        help="Save a resumable latest checkpoint every N optimizer steps; 0 disables it.",
    )
    add_optimizer_argument(parser, default="AdamW")
    add_lr_scheduler_arguments(
        parser, default="cosine", include_force_scheduler=True,
    )
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="学習率の設定")
    parser.add_argument("--loss-fn", type=str, default="CrossEntropy",
                        choices=["CrossEntropy", "KLDiv"],
                        help="損失関数の選択")
    parser.add_argument("--show-model", action="store_true",
                        help="モデルの概要を表示")
    parser.add_argument("--transform-degrees", type=float, default=10.0,
                        help="ランダムアフィン変換の回転角度の最大値")
    parser.add_argument("--transform-shear", type=float, default=10.0,
                        help="ランダムアフィン変換のせん断角度の最大値")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoaderのワーカープロセス数。デフォルト: 4")
    parser.add_argument("--variable-depth", action="store_true",
                        help="学習時に使用するTransformer block数をランダムに変更")
    parser.add_argument(
        "--attention-type", choices=["full", "window", "window_mhla"], default="full",
        help="Transformer attention type. Default: full.",
    )
    parser.add_argument(
        "--window-size", type=int, default=4,
        help="Window side length when --attention-type=window. Default: 4.",
    )
    parser.add_argument(
        "--mhla-block-size", type=int, default=None,
        help="2D MHLA block side length. Default: window-size * 2.",
    )
    parser.add_argument(
        "--mhla-backend", choices=["auto", "vectorized", "triton"], default="auto",
        help="Backend for 2D MHLA. auto uses Triton on compatible CUDA tensors.",
    )
    parser.add_argument("--min-depth", type=int, default=1,
                        help="variable-depth時の最小block数")
    parser.add_argument("--depth-max-bias", type=float, default=16.0,
                        help="終了時のmax depthへの偏重。開始時は一様分布")
    parser.add_argument("--depth-eval-interval", type=int, default=5,
                        help="Depth Test Accを評価するepoch間隔。最終epochは必ず評価")
    parser.add_argument("--bf16", action="store_true",
                        help="Transformer内部をBF16で計算・保存する（デフォルトは全FP32）")
    parser.add_argument("--grad-sign-flip-prob", type=float, default=0.0,
                        help="Linear系パラメータに適用する勾配符号反転の初期確率")
    parser.add_argument("--gc-interval", type=int, default=100,
                        help="NバッチごとにPython GCを実行。0で無効")
    parser.add_argument("--empty-cache-interval", type=int, default=0,
                        help="NバッチごとにCUDAキャッシュを解放。0で無効")
    if adapter_only:
        add_adapter_arguments(parser)
    args = parser.parse_args(argv)
    if not adapter_only:
        # Keep the old metadata keys stable while removing adapter options
        # from the normal-training CLI.
        args.lora_rank = 0
        args.adapter = "none"
        args.lora_alpha = None
        args.lora_dropout = 0.0
        args.lora_target = None
    if args.dry_run and args.validate_only:
        raise ValueError("--dry-run and --validate-only cannot be used together")
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint cannot be used together")
    resume_path = resolve_resume_path(args.resume, args.output_dir) if args.resume else None
    if adapter_only and args.base_init == "random" and args.init_checkpoint:
        raise ValueError(
            "--base-init random cannot be combined with --init-checkpoint"
        )
    init_path = resolve_resume_path(args.init_checkpoint, args.output_dir) if args.init_checkpoint else None
    if resume_path:
        resume_config = read_checkpoint_config(resume_path, CIFAR10_CONFIG_METADATA_KEY)
        if (
            not adapter_only
            and resume_config
            and resume_config.get("adapter", "none") != "none"
        ):
            raise ValueError(
                "adapter checkpoint must be resumed with "
                "cifar10.train_adapter"
            )
        overridden_config_keys = []
        restored_config_keys = apply_saved_config(
            args,
            resume_config,
            {"auto_schedule": ("--auto-schedule", "--no-auto-schedule")},
            argv=sys.argv[1:] if argv is None else argv,
            keys=tuple(
                key for key in vars(args)
                if key not in {
                    "resume", "init_checkpoint", "output_dir", "run_name", "device", "dry_run", "validate_only", "show_model",
                }
            ),
            overridden_keys=overridden_config_keys,
        )
        if restored_config_keys:
            print(
                "Restored settings from checkpoint: "
                + ", ".join(restored_config_keys)
            )
        if overridden_config_keys:
            print(
                "CLI overrides checkpoint settings: "
                + ", ".join(overridden_config_keys)
            )
    global DEVICE
    DEVICE = resolve_device(args.device, default=DEVICE)
    if args.warmup_steps is None and args.warmup_ratio is None:
        # Preserve the historical 10% warmup when no scheduler warmup flag
        # was supplied explicitly.
        args.warmup_ratio = 0.1
    elif args.warmup_steps is None:
        args.warmup_steps = 0
    elif args.warmup_ratio is None:
        args.warmup_ratio = 0.0
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.seed is not None and args.seed < 0:
        raise ValueError("--seed must be >= 0")
    if args.checkpoint_interval_steps < 0:
        raise ValueError("--checkpoint-interval-steps must be >= 0")
    if adapter_only:
        adapter_type = resolve_adapter_config(
            args, resume_path=resume_path, init_path=init_path,
        )
    else:
        adapter_type = "none"
    if not 0.0 <= args.grad_sign_flip_prob <= 1.0:
        raise ValueError("--grad-sign-flip-prob must be in [0.0, 1.0]")
    if args.depth_eval_interval < 1:
        raise ValueError("--depth-eval-interval must be >= 1")
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    if args.mhla_block_size is not None and args.mhla_block_size <= 0:
        raise ValueError("--mhla-block-size must be positive")

    if args.dry_run:
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
            device=DEVICE,
            dtype=torch.bfloat16 if args.bf16 else torch.float32,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            resume=args.resume,
            extra={"optimizer": args.optimizer, "dry_run": True},
        )
        run_recorder.record("preflight", **preflight)
        run_recorder.finish(status="dry_run")
        print("Dry run completed; no dataset or model was loaded.")
        return
    validation_timer = ValidationTimer(DEVICE) if args.validate_only else None

    # ========= データ準備 =========
    normalize_mean = (0.4914, 0.4822, 0.4965)
    normalize_std = (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.RandomAffine(degrees=args.transform_degrees, shear=args.transform_shear),
        transforms.RandomPerspective(distortion_scale=0.1),
        # RandomResizedCropをアフィン変換等の前ではなく後に適用することで、空白部分ができにくくする
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    train_sampler = ResumableRandomSampler(train_dataset, seed=args.seed)
    args.seed = train_sampler.seed
    train_loader_options = build_dataloader_options(
        num_workers=args.num_workers,
        pin_memory=DEVICE.type == "cuda",
        seed=args.seed,
        stream=0,
    )
    test_loader_options = build_dataloader_options(
        num_workers=args.num_workers,
        pin_memory=DEVICE.type == "cuda",
        seed=args.seed,
        stream=1,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        **train_loader_options,
    )
    test_loader  = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        **test_loader_options,
    )

    # 保存ディレクトリ作成
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
        device=DEVICE,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        resume=args.resume,
        extra={"optimizer": args.optimizer},
    )
    run_recorder.record("preflight", **preflight)
    checkpoint_dir = str(run_recorder.checkpoints_dir)
    tensorboard_dir = str(run_recorder.tensorboard_dir)

    model = CIFAR10ViT(patch_size=PATCH_SIZE,
                     embed_dim=EMBED_DIM,
                     num_layers=NUM_LAYERS,
                     num_heads=NUM_HEADS,
                     compute_dtype=torch.bfloat16 if args.bf16 else None,
                     attention_type=args.attention_type,
                     window_size=args.window_size,
                     mhla_block_size=args.mhla_block_size,
                     mhla_backend=args.mhla_backend).to(DEVICE)
    model.apply(init_weights)
    if args.bf16:
        # Transformer本体だけをBF16化する。
        # patch embedding、pooling、出力headはFP32に残す。
        convert_linear_to_bf16(
            model.encoder,
        )

    # ========== safetensorsから復元 ==========
    resume_state = None
    if resume_path or init_path:
        load_path = resume_path or init_path
        if resume_path:
            resume_state = load_training_state(resume_path)
            state_message = (
                "(full state sidecar found)"
                if resume_state is not None
                else "(weights-only resume)"
            )
            print(f"Loading model weights: {resume_path} {state_message}")
        else:
            print(f"Initializing model weights only: {init_path}")
        if resume_path and adapter_type != "none":
            matched_lora_targets, _ = enable_adapter(model, args)
            print(
                f"Enabled {adapter_type}: rank={args.lora_rank} "
                f"alpha={args.lora_alpha or args.lora_rank:g} "
                f"targets={len(matched_lora_targets)}"
            )
        state_dict = load_file(load_path, device="cpu")
        # compact保存では共有パラメータの別名キーを省略しているため、
        # 共有先の代表キーだけを読み込む。
        model.load_state_dict(state_dict, strict=False)
        print("Loaded pretrained weights.")

    if adapter_type != "none" and not (resume_path and adapter_type != "none"):
        matched_lora_targets, trainable_count = enable_adapter(model, args)
        print(
            f"Enabled {adapter_type}: rank={args.lora_rank} "
            f"alpha={args.lora_alpha or args.lora_rank:g} "
            f"targets={len(matched_lora_targets)} "
            f"trainable_parameters={trainable_count:,}"
        )

    print_model_info(model, cast_bf16=args.bf16)

    if args.show_model:
        summary(model, input_size=(args.batch_size, 3, IMG_SIZE, IMG_SIZE))
        for name, p in model.named_parameters():
            print(name, list(p.shape))

    if args.validate_only:
        assert validation_timer is not None
        model.eval()
        with torch.inference_mode():
            sample_images, _ = next(iter(test_loader))
            sample_images = sample_images.to(DEVICE)
            sample_logits = model(sample_images)
        if sample_logits.shape != (sample_images.size(0), 10):
            raise ValueError(
                "CIFAR-10 validation produced an unexpected output shape: "
                f"{tuple(sample_logits.shape)}"
            )
        if not torch.isfinite(sample_logits).all():
            raise ValueError("CIFAR-10 validation produced non-finite logits")
        validation = build_validation_report(
            script=script_name,
            device=DEVICE,
            dtype=torch.bfloat16 if args.bf16 else torch.float32,
            train_examples=len(train_dataset),
            eval_examples=len(test_dataset),
            model_parameters=sum(parameter.numel() for parameter in model.parameters()),
            trainable_parameters=sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            steps_per_epoch=len(train_loader),
            measurements=validation_timer.finish(),
            extra={
                "input_shape": list(sample_images.shape),
                "output_shape": list(sample_logits.shape),
                "num_classes": 10,
                "attention_type": args.attention_type,
            },
        )
        run_recorder.record("validation", **validation)
        run_recorder.finish(status="validate_only")
        print("Validation completed; training was not started.")
        return

    # 損失関数設定
    if args.loss_fn == "KLDiv":
        _kl = nn.KLDivLoss(reduction='batchmean')

        def criterion(outputs, labels, labels_vector):
            return _kl(nn.functional.log_softmax(outputs, dim=1), labels_vector)
    else:
        _ce = nn.CrossEntropyLoss()

        def criterion(outputs, labels, labels_vector):
            return _ce(outputs, labels)

    lr = args.lr
    optimizer_param_groups = (
        build_adapter_parameter_groups(model, WEIGHT_DECAY)
        if adapter_type != "none"
        else build_parameter_groups(
            model,
            target_param_regexes=[r"linear", r"conv2d", r"super_weight"],
            weight_decay=WEIGHT_DECAY,
        )
    )
    optimizer = build_optimizer(
        args.optimizer,
        optimizer_param_groups,
        lr=lr,
        weight_decay=WEIGHT_DECAY,
        args=args,
    )
    is_schedule_free = is_schedule_free_optimizer(args.optimizer)

    total_optimizer_steps = max(1, args.epochs * len(train_loader))

    noiseInjector = GradSignFlipNoiseInjector(
        optimizer,
        initial_flip_prob=args.grad_sign_flip_prob,
        final_flip_prob=0.0,
        total_steps=total_optimizer_steps,
        schedule="cosine",
        target_param_regexes=[r"linear", r"super_weight"],
        model=model,
    )
    weightDecayScheduler = WeightDecayScheduler(
        optimizer,
        initial_weight_decay=WEIGHT_DECAY,
        final_weight_decay=WEIGHT_DECAY*1e-2,
        total_steps=total_optimizer_steps,
        schedule="cosine",
        target_param_regexes=[r"linear", r"conv2d", r"super_weight"],
        model=model,
    )

    scheduler = None
    if not is_schedule_free or args.force_scheduler:
        scheduler = build_lr_scheduler(
            optimizer, args, total_optimizer_steps,
        )

    # 重み・勾配はFP32で保持するため、GradScalerは使用しない。
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    start_epoch = 0
    global_step = 0

    timestamp = datetime.now().strftime("%Y%m%d%H%M")

    print("Starting training...")
    print(f"    Using optimizer: {args.optimizer}")
    print(
        f"    LR scheduler: {scheduler.name if scheduler is not None else 'disabled'} "
        f"warmup_steps={scheduler.warmup_steps if scheduler is not None else 0}"
    )
    print(f"    Epochs: {args.epochs}, Batch size: {args.batch_size}, Learning rate: {lr:.0e}")
    print(f"    Steps per epoch: {len(train_loader)}, Total steps: {total_optimizer_steps}")
    compute_dtype_name = "torch.bfloat16" if args.bf16 else "torch.float32"
    print(f"    Transformer compute dtype: {compute_dtype_name}")
    print(
        f"    Attention type: {args.attention_type}"
        + (
            f" (window={args.window_size}"
            f", mhla_block={args.mhla_block_size or args.window_size * 2}"
            f", mhla_backend={args.mhla_backend})"
            if args.attention_type == "window_mhla"
            else f" (window={args.window_size})"
            if args.attention_type == "window" else ""
        )
    )
    print("    Embedding / Norm / Pooling / Head compute dtype: torch.float32")
    if args.variable_depth:
        if not 1 <= args.min_depth <= NUM_LAYERS:
            raise ValueError(f"--min-depth must be in [1, {NUM_LAYERS}]")
        if args.depth_max_bias <= 0.0:
            raise ValueError("--depth-max-bias must be > 0")
        depth_scheduler = DepthDistributionScheduler(
            min_depth=args.min_depth,
            max_depth=NUM_LAYERS,
            total_steps=total_optimizer_steps,
            initial_bias=1.0,
            final_bias=args.depth_max_bias,
            schedule="cosine",
        )
        print(f"    Variable depth: {args.min_depth}～{NUM_LAYERS} blocks")
        def format_depth_probabilities(probabilities):
            return ", ".join(
                f"{int(depth)}={prob:.3f}"
                for depth, prob in zip(
                    depth_scheduler.depth_choices, probabilities
                )
            )

        print(
            "    Initial depth probabilities: "
            + format_depth_probabilities(
                depth_scheduler.probabilities(
                    bias=depth_scheduler.initial_bias
                )
            )
        )
        print(
            "    Final depth probabilities: "
            + format_depth_probabilities(
                depth_scheduler.probabilities(
                    bias=depth_scheduler.final_bias
                )
            )
        )
    else:
        depth_scheduler = None

    if resume_state is not None:
        start_epoch = int(resume_state["epoch"])
        global_step = int(resume_state["global_step"])
        optimizer.load_state_dict(resume_state["optimizer"])
        saved_scheduler = resume_state.get("scheduler")
        if scheduler is not None and saved_scheduler is not None:
            if scheduler.total_steps != saved_scheduler.get("total_steps"):
                scheduler.total_steps = int(saved_scheduler["total_steps"])
            scheduler.load_state_dict(saved_scheduler)
        elif scheduler is None and saved_scheduler is not None:
            raise ValueError(
                "resume checkpoint contains an LR scheduler, but the current "
                "run disabled it"
            )
        elif scheduler is not None:
            scheduler.step(global_step)

        extra = resume_state.get("extra") or {}
        saved_sampler = extra.get("sampler")
        if saved_sampler is not None:
            train_sampler.load_state_dict(saved_sampler)
            start_epoch = train_sampler.epoch
        saved_scaler = extra.get("scaler")
        if saved_scaler:
            scaler.load_state_dict(saved_scaler)
        saved_noise = extra.get("noise_injector")
        if saved_noise:
            noiseInjector.total_steps = int(saved_noise["total_steps"])
            noiseInjector._step_count = int(saved_noise["step_count"])
        saved_decay = extra.get("weight_decay_scheduler")
        if saved_decay:
            weightDecayScheduler.total_steps = int(saved_decay["total_steps"])
            weightDecayScheduler._step_count = int(saved_decay["step_count"])
            current_decay = weightDecayScheduler._current_weight_decay()
            for group in weightDecayScheduler._decay_groups:
                group["weight_decay"] = current_decay
        saved_depth = extra.get("depth_scheduler")
        if saved_depth is not None and depth_scheduler is None:
            raise ValueError(
                "resume checkpoint contains variable-depth state, but the "
                "current run disabled --variable-depth"
            )
        if depth_scheduler is not None and saved_depth is not None:
            for key in ("min_depth", "max_depth", "schedule"):
                if saved_depth.get(key) != getattr(depth_scheduler, key):
                    raise ValueError(
                        "Depth scheduler configuration mismatch for "
                        f"{key}: current={getattr(depth_scheduler, key)!r}, "
                        f"saved={saved_depth.get(key)!r}"
                    )
            depth_scheduler.step_count = int(saved_depth["step_count"])
        restore_rng_state(resume_state["rng"])
        print(
            f"Restored full training state: epoch={start_epoch}, "
            f"global_step={global_step}"
        )
    elif resume_path:
        print("Starting a new optimizer schedule from weights-only resume")
    if start_epoch >= args.epochs:
        raise ValueError(
            f"resume starts at epoch {start_epoch}, but --epochs is {args.epochs}; "
            "set --epochs to a larger total epoch count"
        )

    def resume_extra_state():
        extra = {
            "sampler": train_sampler.state_dict(),
            "scaler": scaler.state_dict(),
            "noise_injector": {
                "total_steps": noiseInjector.total_steps,
                "step_count": noiseInjector._step_count,
            },
            "weight_decay_scheduler": {
                "total_steps": weightDecayScheduler.total_steps,
                "step_count": weightDecayScheduler._step_count,
            },
        }
        if depth_scheduler is not None:
            extra["depth_scheduler"] = {
                "min_depth": depth_scheduler.min_depth,
                "max_depth": depth_scheduler.max_depth,
                "schedule": depth_scheduler.schedule,
                "step_count": depth_scheduler.step_count,
            }
        return extra

    model_snapshot = ModelSnapshot(max_snapshots=5)

    latest_path = os.path.join(
        checkpoint_dir,
        f"cifar10_vit_{timestamp}_{args.optimizer}_latest.safetensors",
    )

    def save_training_checkpoint(path, epoch_number):
        save_file(
            compact_state_dict(model, cast_bf16=args.bf16),
            path,
            metadata=checkpoint_config_metadata(args, CIFAR10_CONFIG_METADATA_KEY),
        )
        save_training_state(
            path,
            make_training_state(
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch_number,
                global_step=global_step,
                extra=resume_extra_state(),
            ),
        )

    stop_controller = GracefulStop(
        "Ctrl-C received; finishing the current batch and saving a checkpoint..."
    )
    stop_controller.install()

    def finish_interrupted(epoch_number):
        save_training_checkpoint(latest_path, epoch_number)
        run_recorder.record(
            "checkpoint",
            kind="interrupted",
            epoch=epoch_number,
            global_step=global_step,
            sampler_position=train_sampler.position,
            path=latest_path,
        )
        if writer is not None:
            writer.close()
        stop_controller.restore()
        run_recorder.finish(status="interrupted", checkpoints=[latest_path])

    writer = None
    for epoch in range(start_epoch, args.epochs):
        epoch_started_at = perf_counter()
        if train_sampler.epoch != epoch:
            train_sampler.set_epoch(epoch)
        steps_in_epoch = len(train_loader)
        samples_seen = train_sampler.position
        model.train()
        if is_schedule_free:
            optimizer.train()
        if depth_scheduler is not None:
            print(
                f"[Epoch {epoch + 1}] Depth distribution "
                f"(bias={depth_scheduler.bias:.4f}, "
                f"expected_depth={depth_scheduler.expected_depth():.3f}): "
                + format_depth_probabilities(
                    depth_scheduler.probabilities()
                )
            )
        pbar = RichProgress(train_loader, description=f"Epoch {epoch+1}/{args.epochs}")

        prev_weights_flat = torch.cat([p.data.clone().detach().flatten() for p in model.parameters()])

        correct_train, num_images_train = 0, 0
        running_loss = 0.0
        for batch_index, (images, labels) in enumerate(pbar, start=1):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            labels_vector = nn.functional.one_hot(labels, num_classes=10).float()

            train_depth = None
            if args.variable_depth:
                train_depth = depth_scheduler.sample()

            optimizer.zero_grad()
            # モデル全体のautocastは使わず、Transformer内部で指定した射影・
            # FFN・AttentionだけがBF16経路を使う。
            logit = model(images, depth=train_depth)
            loss = criterion(logit, labels, labels_vector)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            noiseInjector.inject()

            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            samples_seen += labels.size(0)
            train_sampler.set_position(samples_seen)
            if scheduler is not None:
                scheduler.step(global_step)
            weightDecayScheduler.step()
            if depth_scheduler is not None:
                depth_scheduler.step()

            running_loss += loss.item()
            correct = (logit.argmax(dim=1) == labels).sum().item()
            current_acc = 100.0 * correct / labels.size(0)
            correct_train += correct
            num_images_train += labels.size(0)
            pbar.set_status(build_standard_progress_rows(
                step=batch_index,
                total_steps=steps_in_epoch,
                global_step=global_step,
                loss=f"{loss.item():.4f}",
                learning_rate=f"{optimizer.param_groups[0]['lr']:.4e}",
                extra={"accuracy": f"{current_acc:.2f}%"},
            ))
            maybe_collect_memory(
                global_step,
                gc_interval=args.gc_interval,
                empty_cache_interval=args.empty_cache_interval,
            )
            if (
                args.checkpoint_interval_steps > 0
                and global_step % args.checkpoint_interval_steps == 0
            ):
                save_training_checkpoint(latest_path, epoch)
                run_recorder.record(
                    "checkpoint",
                    kind="latest",
                    epoch=epoch,
                    global_step=global_step,
                    sampler_position=train_sampler.position,
                    path=latest_path,
                )
            if stop_controller.requested:
                finish_interrupted(epoch)
                return

        if is_schedule_free:
            optimizer.eval()

        acc_train = correct_train / num_images_train * 100.0
        loss_train = running_loss / max(1, steps_in_epoch)
        depth_metrics = {}
        should_eval_depths = (
            args.variable_depth
            and (
                (epoch + 1) % args.depth_eval_interval == 0
                or epoch + 1 == args.epochs
            )
        )
        if should_eval_depths:
            mid_depth = (args.min_depth + NUM_LAYERS) // 2
            eval_depths = sorted({
                args.min_depth,
                mid_depth,
                NUM_LAYERS,
            })
            for eval_depth in eval_depths:
                depth_acc, depth_loss = evaluate(
                    model,
                    test_loader,
                    DEVICE,
                    criterion,
                    depth=eval_depth,
                )
                depth_metrics[eval_depth] = {
                    "acc": depth_acc,
                    "loss": depth_loss,
                }
            # max depth の結果を通常の test 指標として再利用する。
            acc_test = depth_metrics[NUM_LAYERS]["acc"]
            loss_test = depth_metrics[NUM_LAYERS]["loss"]
        else:
            # Depth評価を省略するepochは、通常のmax depthだけ評価する。
            acc_test, loss_test = evaluate(
                model, test_loader, DEVICE, criterion
            )
        if stop_controller.requested:
            finish_interrupted(epoch)
            return
        scheduled_lr = optimizer.param_groups[0].get('scheduled_lr', optimizer.param_groups[0]['lr'])
        current_weights_flat = torch.cat([p.data.clone().detach().flatten() for p in model.parameters()])
        delta = current_weights_flat - prev_weights_flat
        base = (prev_weights_flat + current_weights_flat) / 2.0 + 1e-10
        rel_change = torch.norm(delta) / torch.norm(base)

        print(f"[Epoch {epoch+1}] Train Loss: {loss_train:.4f} | Test Loss: {loss_test:.4f} | Train Acc: {acc_train:.2f} | Test Acc: {acc_test:.2f}% | scheduled LR: {scheduled_lr:.4e} | Relative Weight Change: {rel_change:.4e}")
        run_recorder.record(
            "epoch",
            epoch=epoch + 1,
            train_loss=loss_train,
            test_loss=loss_test,
            train_accuracy=acc_train,
            test_accuracy=acc_test,
            learning_rate=scheduled_lr,
        )
        epoch_elapsed = perf_counter() - epoch_started_at
        run_recorder.record_training_step(
            global_step=global_step,
            epoch=epoch + 1,
            train_loss=loss_train,
            eval_loss=loss_test,
            effective_lr=float(optimizer.param_groups[0]["lr"]),
            scheduled_lr=float(scheduled_lr),
            step_time_sec=epoch_elapsed / max(steps_in_epoch, 1),
            steps_per_second=steps_in_epoch / max(epoch_elapsed, 1e-6),
            samples_per_second=num_images_train / max(epoch_elapsed, 1e-6),
            metrics={
                "train_accuracy": acc_train,
                "eval_accuracy": acc_test,
                "weight_change_relative": float(rel_change),
            },
        )
        if depth_metrics:
            depth_log = " | ".join(
                f"d={depth}: {metrics['acc']:.2f}%"
                for depth, metrics in depth_metrics.items()
            )
            print(f"[Depth Test Acc] {depth_log}")

        model_snapshot.add_snapshot(model, acc_test, epoch)

        if writer is None:
            # Initialize TensorBoard writer
            writer = SummaryWriter(tensorboard_dir)

        # Log training loss
        writer.add_scalar('Loss/train', loss_train, epoch)
        # Log test loss
        writer.add_scalar('Loss/test', loss_test, epoch)
        # Log training accuracy
        writer.add_scalar('Accuracy/train', acc_train, epoch)
        # Log test accuracy
        writer.add_scalar('Accuracy/test', acc_test, epoch)
        # Log effective learning rate
        writer.add_scalar('LearningRate/scheduled', scheduled_lr, epoch)
        # Log weight change norm
        writer.add_scalar('WeightChange/relative', rel_change, epoch)
        write_standard_training_metrics(
            writer,
            step=epoch + 1,
            train_loss=loss_train,
            eval_loss=loss_test,
            learning_rate=float(optimizer.param_groups[0]['lr']),
            scheduled_learning_rate=float(scheduled_lr),
            extra={
                "train/accuracy": acc_train,
                "eval/accuracy": acc_test,
            },
        )
        for depth, metrics in depth_metrics.items():
            writer.add_scalar(
                f'Accuracy/test_depth_{depth}', metrics['acc'], epoch
            )
            writer.add_scalar(
                f'Loss/test_depth_{depth}', metrics['loss'], epoch
            )

        train_sampler.set_epoch(epoch + 1)
        save_training_checkpoint(latest_path, epoch + 1)
        run_recorder.record(
            "checkpoint",
            kind="latest",
            epoch=epoch + 1,
            global_step=global_step,
            path=latest_path,
        )

    print("Training complete!")
    writer.close()
    stop_controller.restore()

    # ========= safetensors形式で保存 =========
    # safetensors保存用のファイル名
    save_path = os.path.join(checkpoint_dir, f"cifar10_vit_{timestamp}_{args.optimizer}_epoch{args.epochs}acc{acc_test:.2f}_loss{loss_test:.4f}.safetensors")
    # 共有パラメータを重複させずに保存
    state_dict = compact_state_dict(model, cast_bf16=args.bf16)
    save_file(
        state_dict,
        save_path,
        metadata=checkpoint_config_metadata(args, CIFAR10_CONFIG_METADATA_KEY),
    )
    save_training_state(
        save_path,
        make_training_state(
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=args.epochs,
            global_step=global_step,
            extra=resume_extra_state(),
        ),
    )
    print(
        f"Saved model weights and full resume state to {save_path} "
        f"({format_bytes(os.path.getsize(save_path))})"
    )
    run_recorder.record("checkpoint", kind="final", path=save_path)

    # ========= 最良モデルの重みを safetensors形式で保存 =========
    best_state_dict = model_snapshot.get_best_model()
    model.load_state_dict(best_state_dict)
    acc_best, loss_best = evaluate(model, test_loader, DEVICE, criterion)
    print(f"Best model from snapshots - Test Acc: {acc_best:.2f}%, Test Loss: {loss_best:.4f}")
    best_save_path = os.path.join(checkpoint_dir, f"cifar10_vit_{timestamp}_{args.optimizer}_best_acc{acc_best:.2f}_loss{loss_best:.4f}.safetensors")
    save_file(
        compact_state_dict(model, cast_bf16=args.bf16),
        best_save_path,
        metadata=checkpoint_config_metadata(args, CIFAR10_CONFIG_METADATA_KEY),
    )
    print(
        f"Saved best model weights to {best_save_path} "
        f"({format_bytes(os.path.getsize(best_save_path))})"
    )
    run_recorder.record("checkpoint", kind="best", path=best_save_path)

    # ========= スナップショットの平均モデルの重みを safetensors形式で保存 =========
    avg_state_dict = model_snapshot.get_average_model()
    model.load_state_dict(avg_state_dict)
    acc_avg, loss_avg = evaluate(model, test_loader, DEVICE, criterion)
    print(f"Average model from snapshots - Test Acc: {acc_avg:.2f}, Test Loss: {loss_avg:.4f}")
    avg_save_path = os.path.join(checkpoint_dir, f"cifar10_vit_{timestamp}_{args.optimizer}_avg_acc{acc_avg:.2f}_loss{loss_avg:.4f}.safetensors")
    save_file(
        compact_state_dict(model, cast_bf16=args.bf16),
        avg_save_path,
        metadata=checkpoint_config_metadata(args, CIFAR10_CONFIG_METADATA_KEY),
    )
    print(
        f"Saved average model weights to {avg_save_path} "
        f"({format_bytes(os.path.getsize(avg_save_path))})"
    )
    run_recorder.record("checkpoint", kind="average", path=avg_save_path)
    run_recorder.finish(
        checkpoints=[latest_path, save_path, best_save_path, avg_save_path],
    )

if __name__ == "__main__":
    main()
