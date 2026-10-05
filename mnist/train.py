import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchinfo import summary
from torch.utils.data import DataLoader
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
)
from core.layers import GatedLinear, PositionalEncoding2D_SineCosine, RMSNorm2d, KVTransformerEncoder, AttentionPoolingWithKVSelfAttention

# ========= ハイパーパラメータ =========
NUM_EPOCHS = 50
BATCH_SIZE = 256
WEIGHT_DECAY = 1e-1
IMG_SIZE = 28
PATCH_SIZE = 4 # 28x28画像を4x4パッチに分割 -- > 49パッチ
EMBED_DIM = 64
NUM_LAYERS = 2 # transformerブロックの数
NUM_HEADS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MNIST_CONFIG_METADATA_KEY = "mnist.config"


def resolve_resume_path(path, output_dir):
    if os.path.isfile(path):
        return path
    candidate = os.path.join(output_dir, path)
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f"Resume checkpoint not found: {path}")

# ========= モデル定義 =========
class PatchEmbed(nn.Module):
    """
    画像をパッチに分割し、線形埋め込みに変換する
    """
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj1 = nn.Sequential(
            # 事前に畳み込みで特徴抽出を行う
            nn.Conv2d(in_chans, embed_dim, kernel_size=5, padding=2, bias=False),
            RMSNorm2d(embed_dim),
            APTx(trainable=True),
            # 空間ダウンサンプリング
            nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False),
            RMSNorm2d(embed_dim),
            APTx(trainable=True),
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            RMSNorm2d(embed_dim),
            APTx(trainable=True),
        )
        self.proj2_dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.proj1(x)                       # (B, D, H', W')
        x = x + self.proj2_dropout(self.proj2(x))
        x = x.flatten(2).transpose(1, 2)       # (B, N, D)
        return x

class MNISTViT(nn.Module):
    def __init__(self, embed_dim=128, num_layers=3, num_heads=8, drop_out=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(embed_dim=embed_dim, in_chans=1, img_size=IMG_SIZE, patch_size=PATCH_SIZE)
        self.pos_encoding = PositionalEncoding2D_SineCosine(embed_dim=embed_dim, grid_size=IMG_SIZE // PATCH_SIZE)
        # self.encoder = TransformerEncoder(num_layers, embed_dim, num_heads, drop_out)
        # self.pooling = AttentionPoolingWithGMHA(embed_dim, num_heads, dropout=drop_out)
        self.encoder = KVTransformerEncoder(num_layers, embed_dim, num_heads, drop_out)
        self.pooling = AttentionPoolingWithKVSelfAttention(embed_dim, num_heads, dropout=drop_out)
        self.head = nn.Sequential(
            nn.RMSNorm(embed_dim),
            GatedLinear(embed_dim, embed_dim),
            nn.Dropout(drop_out),
            nn.Linear(embed_dim, 10, bias=False),
        )

    def forward(self, x):
        x = self.patch_embed(x)   # (B, N, D)
        x = self.pos_encoding(x) # (B, N, D)
        x = self.encoder(x) # (B, N, D)
        # x = x.mean(dim=1)  # (B, D)  # 平均プーリング
        x, _ = self.pooling(x)  # (B, D)  # Attentionプーリング
        return self.head(x)

# ========= 初期化関数 =========
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

# ========= 評価関数 =========
def evaluate(model, loader, device, criterion):
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            logit  = model(images)
            preds = logit.argmax(dim=1)
            loss = criterion(logit, labels, nn.functional.one_hot(labels, num_classes=10).float())

            total += labels.size(0)
            correct += (preds == labels).sum().item()
            total_loss += loss.item()
    return 100.0 * correct / total, total_loss / len(loader)

# ========= メイントレーニング =========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", type=str, default=None,
        help=(
            "Load model weights. If a sibling .resume.pt exists, also restore "
            "optimizer, scheduler, scaler, epoch, and RNG state; otherwise "
            "use weights-only resume."
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
        parser, default="auto", include_force_scheduler=True,
    )
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="学習率の設定")
    parser.add_argument("--loss-fn", type=str, default="CrossEntropy",
                        choices=["CrossEntropy", "KLDiv"],
                        help="損失関数の選択")
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--show-model", action="store_true",
                        help="モデルの概要を表示")
    parser.add_argument("--transform-degrees", type=float, default=10.0,
                        help="ランダムアフィン変換の回転角度の最大値")
    parser.add_argument("--transform-shear", type=float, default=10.0,
                        help="ランダムアフィン変換のせん断角度の最大値")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoaderのワーカープロセス数。デフォルト: 4")
    parser.add_argument("--gc-interval", type=int, default=100,
                        help="NバッチごとにPython GCを実行。0で無効")
    parser.add_argument("--empty-cache-interval", type=int, default=0,
                        help="NバッチごとにCUDAキャッシュを解放。0で無効")
    args = parser.parse_args()
    if args.dry_run and args.validate_only:
        raise ValueError("--dry-run and --validate-only cannot be used together")
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint cannot be used together")
    resume_path = resolve_resume_path(args.resume, args.output_dir) if args.resume else None
    init_path = resolve_resume_path(args.init_checkpoint, args.output_dir) if args.init_checkpoint else None
    if resume_path:
        resume_config = read_checkpoint_config(resume_path, MNIST_CONFIG_METADATA_KEY)
        overridden_config_keys = []
        restored_config_keys = apply_saved_config(
            args,
            resume_config,
            {"auto_schedule": ("--auto-schedule", "--no-auto-schedule")},
            argv=sys.argv[1:],
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

    if args.dry_run:
        os.makedirs(args.output_dir, exist_ok=True)
        run_recorder = RunRecorder(
            args.output_dir,
            script="mnist.train",
            config=vars(args),
            run_name=args.run_name,
        )
        run_recorder.install_exception_hook()
        preflight = build_training_preflight(
            script="mnist.train",
            device=DEVICE,
            dtype=torch.float32,
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
    normalize_mean = (0.1307,)
    normalize_std = (0.3081,)
    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(), # MNISTでは左右反転が文字認識に悪影響の可能性がある
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

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

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
        script="mnist.train",
        config=vars(args),
        run_name=args.run_name,
    )
    run_recorder.install_exception_hook()
    preflight = build_training_preflight(
        script="mnist.train",
        device=DEVICE,
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
    checkpoint_dir = str(run_recorder.checkpoints_dir)
    tensorboard_dir = str(run_recorder.tensorboard_dir)

    model = MNISTViT(embed_dim=EMBED_DIM,
                     num_layers=NUM_LAYERS,
                     num_heads=NUM_HEADS,
                     drop_out=args.dropout).to(DEVICE)
    model.apply(init_weights)

    if args.show_model:
        summary(model, input_size=(args.batch_size, 1, IMG_SIZE, IMG_SIZE))
        for name, p in model.named_parameters():
            print(name, list(p.shape))

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
        state_dict = load_file(load_path, device="cpu")
        model.load_state_dict(state_dict)
        print("Loaded pretrained weights.")

    if args.validate_only:
        assert validation_timer is not None
        model.eval()
        with torch.inference_mode():
            sample_images, _ = next(iter(test_loader))
            sample_images = sample_images.to(DEVICE)
            sample_logits = model(sample_images)
        if sample_logits.shape != (sample_images.size(0), 10):
            raise ValueError(
                "MNIST validation produced an unexpected output shape: "
                f"{tuple(sample_logits.shape)}"
            )
        if not torch.isfinite(sample_logits).all():
            raise ValueError("MNIST validation produced non-finite logits")
        validation = build_validation_report(
            script="mnist.train",
            device=DEVICE,
            dtype=torch.float32,
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
    optimizer_param_groups = build_parameter_groups(
        model,
        target_param_regexes=[r"linear", r"conv2d"],
        weight_decay=WEIGHT_DECAY,
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
        initial_flip_prob=0.0,
        final_flip_prob=0.0,
        total_steps=total_optimizer_steps,
        schedule="cosine",
        target_param_regexes=[r"linear"],
        model=model,
    )
    weightDecayScheduler = WeightDecayScheduler(
        optimizer,
        initial_weight_decay=WEIGHT_DECAY,
        final_weight_decay=WEIGHT_DECAY*1e-2,
        total_steps=total_optimizer_steps,
        schedule="cosine",
        target_param_regexes=[r"linear", r"conv2d"],
        model=model,
    )

    scheduler = None
    if not is_schedule_free or args.force_scheduler:
        scheduler = build_lr_scheduler(
            optimizer, args, total_optimizer_steps,
        )

    # Mixed Precision Training用のGradScaler
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 0
    global_step = 0
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
        return {
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

    timestamp = datetime.now().strftime("%Y%m%d%H%M")

    writer = None

    print("Starting training...")
    print(f"    Using optimizer: {args.optimizer}")
    print(
        f"    LR scheduler: {scheduler.name if scheduler is not None else 'disabled'} "
        f"warmup_steps={scheduler.warmup_steps if scheduler is not None else 0}"
    )
    print(f"    Epochs: {args.epochs}, Batch size: {args.batch_size}, Learning rate: {lr:.0e}")
    print(f"    Steps per epoch: {len(train_loader)}, Total steps: {total_optimizer_steps}")

    model_snapshot = ModelSnapshot(max_snapshots=5)

    latest_path = os.path.join(
        checkpoint_dir,
        f"mnist_vit_{timestamp}_{args.optimizer}_latest.safetensors",
    )

    def save_training_checkpoint(path, epoch_number):
        save_file(
            {key: value.detach().cpu() for key, value in model.state_dict().items()},
            path,
            metadata=checkpoint_config_metadata(args, MNIST_CONFIG_METADATA_KEY),
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

    for epoch in range(start_epoch, args.epochs):
        epoch_started_at = perf_counter()
        if train_sampler.epoch != epoch:
            train_sampler.set_epoch(epoch)
        steps_in_epoch = len(train_loader)
        samples_seen = train_sampler.position
        model.train()
        if is_schedule_free:
            optimizer.train()
        pbar = RichProgress(train_loader, description=f"Epoch {epoch+1}/{args.epochs}")

        prev_weights_flat = torch.cat([p.data.clone().detach().flatten() for p in model.parameters()])

        correct, total = 0, 0
        loss_train_total = 0.0
        for batch_index, (images, labels) in enumerate(pbar, start=1):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            labels_vector = nn.functional.one_hot(labels, num_classes=10).float()

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logit  = model(images)
                loss = criterion(logit, labels, labels_vector)
                preds = logit.argmax(dim=1)
                acc = 100.0 * (preds == labels).sum().item() / labels.size(0)
                loss_train_total += loss.item()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

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

            pbar.set_status(build_standard_progress_rows(
                step=batch_index,
                total_steps=steps_in_epoch,
                global_step=global_step,
                loss=f"{loss.item():.4f}",
                learning_rate=f"{optimizer.param_groups[0]['lr']:.4e}",
                extra={"accuracy": f"{acc:.2f}%"},
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

        acc_train = 100.0 * correct / total
        loss_train = loss_train_total / max(1, steps_in_epoch)
        acc_test, loss_test = evaluate(model, test_loader, DEVICE, criterion)
        if stop_controller.requested:
            finish_interrupted(epoch)
            return
        scheduled_lr = optimizer.param_groups[0].get('scheduled_lr', optimizer.param_groups[0]['lr'])
        current_weights_flat = torch.cat([p.data.clone().detach().flatten() for p in model.parameters()])
        delta = current_weights_flat - prev_weights_flat
        base = (prev_weights_flat + current_weights_flat) / 2.0 + 1e-10
        rel_change = torch.norm(delta) / torch.norm(base)

        print(f"[Epoch {epoch+1}] Train Loss: {loss_train:.4f} | Test Loss: {loss_test:.4f} | Train Acc: {acc_train:.2f}% | Test Acc: {acc_test:.2f}% | scheduled LR: {scheduled_lr:.4e} | Relative Weight Change: {rel_change:.4e}")
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
            samples_per_second=total / max(epoch_elapsed, 1e-6),
            metrics={
                "train_accuracy": acc_train,
                "eval_accuracy": acc_test,
                "weight_change_relative": float(rel_change),
            },
        )

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

    # ========= 最終epochの重みを safetensors形式で保存 =========
    # safetensors保存用のファイル名
    save_path = os.path.join(checkpoint_dir, f"mnist_vit_{timestamp}_{args.optimizer}_epoch{args.epochs}_acc{acc_test:.2f}_loss{loss_test:.4f}.safetensors")
    # CPUに移動してstate_dictを取得
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    # safetensors形式で保存
    save_file(
        state_dict,
        save_path,
        metadata=checkpoint_config_metadata(args, MNIST_CONFIG_METADATA_KEY),
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
    print(f"Saved model weights and full resume state to {save_path}")
    run_recorder.record("checkpoint", kind="final", path=save_path)

    # ========= 最良モデルの重みを safetensors形式で保存 =========
    best_state_dict = model_snapshot.get_best_model()
    model.load_state_dict(best_state_dict)
    acc_best, loss_best = evaluate(model, test_loader, DEVICE, criterion)
    print(f"Best model from snapshots - Test Acc: {acc_best:.2f}%, Test Loss: {loss_best:.4f}")
    save_path_best = os.path.join(checkpoint_dir, f"mnist_vit_{timestamp}_{args.optimizer}_best_acc{acc_best:.2f}_loss{loss_best:.4f}.safetensors")
    save_file(
        best_state_dict,
        save_path_best,
        metadata=checkpoint_config_metadata(args, MNIST_CONFIG_METADATA_KEY),
    )
    print(f"Saved best model weights to {save_path_best}")
    run_recorder.record("checkpoint", kind="best", path=save_path_best)

    # ========= スナップショットの平均モデルの重みを safetensors形式で保存 =========
    avg_state_dict = model_snapshot.get_average_model()
    model.load_state_dict(avg_state_dict)
    acc_avg, loss_avg = evaluate(model, test_loader, DEVICE, criterion)
    print(f"Average model from snapshots - Test Acc: {acc_avg:.2f}%, Test Loss: {loss_avg:.4f}")
    save_path_avg = os.path.join(checkpoint_dir, f"mnist_vit_{timestamp}_{args.optimizer}_avg_acc{acc_avg:.2f}_loss{loss_avg:.4f}.safetensors")
    save_file(
        avg_state_dict,
        save_path_avg,
        metadata=checkpoint_config_metadata(args, MNIST_CONFIG_METADATA_KEY),
    )
    print(f"Saved average model weights to {save_path_avg}")
    run_recorder.record("checkpoint", kind="average", path=save_path_avg)
    run_recorder.finish(
        checkpoints=[save_path, save_path_best, save_path_avg],
    )

if __name__ == "__main__":
    main()
