import torch

from image_gen.train import JointMHLALayoutCache


def test_mhla_layout_cache_does_not_retain_inference_tensors():
    cache = JointMHLALayoutCache()
    key = ("cpu", None, 4, 4, 2, 2, 8, 2, 1, 1)

    with torch.inference_mode():
        value = (
            [torch.tensor([0, 1]), torch.tensor([2, 3])],
            torch.tensor([0, 1]),
            (torch.tensor([[0, 1], [2, 3]]), torch.ones(2, 2, dtype=torch.bool)),
        )
        cache.put(key, value)

    cached_blocks, cached_modalities, cached_layout = cache.get(key)
    tensors = [*cached_blocks, cached_modalities, *cached_layout]
    assert all(not torch.is_inference(tensor) for tensor in tensors)
