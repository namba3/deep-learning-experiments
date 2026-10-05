"""Runtime helpers for releasing temporary Python and CUDA allocations."""

import gc

import torch


def collect_memory(*, python_gc=True, empty_cache=True):
    """Collect Python garbage and optionally release PyTorch's CUDA cache."""
    if python_gc:
        gc.collect()
    if empty_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()


def maybe_collect_memory(step, *, gc_interval=0, empty_cache_interval=0):
    """Run cleanup on configured step intervals.

    ``empty_cache`` is deliberately independent because it synchronizes with
    the CUDA allocator and can noticeably reduce throughput when called often.
    """
    if gc_interval < 0 or empty_cache_interval < 0:
        raise ValueError("memory cleanup intervals must be >= 0")
    do_gc = gc_interval > 0 and step % gc_interval == 0
    do_empty_cache = empty_cache_interval > 0 and step % empty_cache_interval == 0
    if do_gc or do_empty_cache:
        collect_memory(python_gc=do_gc, empty_cache=do_empty_cache)
