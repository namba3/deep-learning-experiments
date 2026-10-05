"""Checkpoint restoration helpers for image generation training."""

from contextlib import contextmanager
from dataclasses import dataclass

from image_gen.checkpoint import (
    checkpoint_epoch,
    checkpoint_info,
    load_resume_weights,
    save_checkpoint,
)
from runtime.checkpoint import (
    load_training_state,
    make_training_state,
    restore_rng_state,
)


@contextmanager
def optimizer_evaluation_mode(optimizer):
    """Use Schedule-Free evaluation parameters while sampling or saving."""
    if hasattr(optimizer, "eval"):
        optimizer.eval()
    try:
        yield
    finally:
        if hasattr(optimizer, "train"):
            optimizer.train()


@dataclass
class ResumeTrainingState:
    """Resolved epoch and sidecar state for a resumed training run."""

    start_epoch: int
    global_step: int | None
    state: dict | None


def restore_training_checkpoint(
    args,
    resume_path,
    resume_config,
    dit,
    text_adapter,
    batch_sampler,
):
    """Load checkpoint weights and recover epoch, step, and sampler position."""
    start_epoch = 0
    global_step = None
    state = None
    if resume_path:
        config = resume_config or checkpoint_info(resume_path)
        load_resume_weights(resume_path, dit, text_adapter)
        state = load_training_state(resume_path)
        if state is not None:
            start_epoch = (
                args.resume_epoch
                if args.resume_epoch > 0
                else int(state["epoch"])
            )
            global_step = int(state["global_step"])
            saved_sampler = (state.get("extra") or {}).get("sampler")
            if saved_sampler is not None:
                batch_sampler.load_state_dict(saved_sampler)
                if args.resume_epoch > 0:
                    batch_sampler.set_epoch(start_epoch)
            print(
                f"resuming full training state from epoch {start_epoch}, "
                f"global_step={global_step}: {resume_path}"
            )
        else:
            stored_epoch = checkpoint_epoch(resume_path)
            start_epoch = args.resume_epoch if args.resume_epoch > 0 else stored_epoch
            global_step = config.get("global_step")
            print(f"resuming weights-only from epoch {start_epoch}: {resume_path}")

    if start_epoch >= args.epochs:
        raise ValueError(
            f"resume starts at epoch {start_epoch}, but --epochs is {args.epochs}; "
            "set --epochs to a larger total epoch count"
        )
    return ResumeTrainingState(
        start_epoch=start_epoch,
        global_step=global_step,
        state=state,
    )


def restore_resume_optimizer_state(state, optimizer, lr_scheduler, global_step):
    """Restore optimizer, scheduler, and random state after construction."""
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        saved_scheduler = state.get("scheduler")
        if saved_scheduler is not None:
            lr_scheduler.load_state_dict(saved_scheduler)
        else:
            lr_scheduler.step(global_step)
        restore_rng_state(state["rng"])
    else:
        lr_scheduler.step(global_step)


def save_training_checkpoint(
    path,
    epoch,
    step,
    *,
    optimizer,
    scheduler,
    batch_sampler,
    dit,
    text_adapter,
    args,
):
    """Save weights with optimizer, scheduler, and sampler sidecar state."""
    state = make_training_state(
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        global_step=step,
        extra={"sampler": batch_sampler.state_dict()},
    )
    with optimizer_evaluation_mode(optimizer):
        save_checkpoint(
            dit, text_adapter, path, args, epoch, step,
            resume_state=state,
        )
