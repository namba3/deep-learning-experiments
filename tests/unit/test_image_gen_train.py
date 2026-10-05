import torch
import pytest
from PIL import Image as PILImage

from image_gen import train as image_train
from image_gen.train import RecordsDataset, apply_reconstruction_loss_cap


def test_image_gen_parser_accepts_apollo_refresh_mode(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "image_gen.train",
            "--vae-model", "test-vae",
            "--apollo-projection-refresh-mode", "smooth",
            "--apollo-projection-refresh-window", "4",
            "--apollo-projection-refresh-mix", "stochastic",
            "--apollo-projection-refresh-state", "transport",
            "--apollo-orthogonal-refresh-rate", "0.05",
        ],
    )

    args = image_train.parse_args()

    assert args.apollo_projection_refresh_mode == "smooth"
    assert args.apollo_projection_refresh_window == 4
    assert args.apollo_projection_refresh_mix == "stochastic"
    assert args.apollo_projection_refresh_state == "transport"
    assert args.apollo_orthogonal_refresh_rate == 0.05


def test_reconstruction_loss_cap_limits_post_cap_share():
    diffusion = torch.tensor(4.0, requires_grad=True)
    weighted_reconstruction = torch.tensor(8.0, requires_grad=True)

    capped, scale = apply_reconstruction_loss_cap(
        diffusion, weighted_reconstruction, max_contribution=0.25,
    )

    assert scale.item() == torch.tensor(1 / 6).item()
    assert capped.item() == torch.tensor(4 / 3).item()
    assert capped.item() / (diffusion.item() + capped.item()) == pytest.approx(0.25)
    capped.backward()
    assert weighted_reconstruction.grad.item() == scale.item()


def test_reconstruction_loss_cap_keeps_small_loss_unchanged():
    diffusion = torch.tensor(4.0)
    weighted_reconstruction = torch.tensor(0.5)

    capped, scale = apply_reconstruction_loss_cap(
        diffusion, weighted_reconstruction, max_contribution=0.25,
    )

    assert scale.item() == 1.0
    assert capped.item() == weighted_reconstruction.item()


def test_records_dataset_closes_images_after_bucket_scan_and_getitem(
    tmp_path, monkeypatch,
):
    image_path = tmp_path / "sample.png"
    PILImage.new("RGB", (16, 16), color=(128, 64, 32)).save(image_path)

    opened = []
    real_open = image_train.Image.open

    class TrackedImage:
        def __init__(self, image):
            self._image = image
            self.closed = False

        @property
        def size(self):
            return self._image.size

        def convert(self, mode):
            return self._image.convert(mode)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.closed = True
            self._image.close()

    def tracked_open(*args, **kwargs):
        tracked = TrackedImage(real_open(*args, **kwargs))
        opened.append(tracked)
        return tracked

    monkeypatch.setattr(image_train.Image, "open", tracked_open)
    dataset = RecordsDataset(
        records_path="",
        image_size=16,
        bucket_step=8,
        records=[{"image": str(image_path), "caption": "sample"}],
    )

    image, caption = dataset[0]

    assert image.shape == (3, 16, 16)
    assert caption == "sample"
    assert len(opened) == 2
    assert all(item.closed for item in opened)
