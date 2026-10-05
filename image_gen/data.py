"""Image dataset loading, transforms, and aspect-ratio batching."""

import csv
import json
import math
import os
import zipfile

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from runtime.sampler import ResumableAspectRatioBatchSampler


def make_bucket_shapes(image_size, step=32):
    """Return near-equal-area H/W buckets, aligned for common VAE strides."""
    ratios = (0.667, 0.8, 1.0, 1.25, 1.5)
    shapes = []
    for ratio in ratios:
        height = int(round(math.sqrt(image_size * image_size / ratio) / step) * step)
        width = int(round(height * ratio / step) * step)
        shapes.append((max(step, height), max(step, width)))
    return tuple(dict.fromkeys(shapes))

def assign_bucket(width, height, bucket_shapes):
    aspect = width / max(height, 1)
    return min(
        range(len(bucket_shapes)),
        key=lambda index: abs(math.log(aspect) - math.log(bucket_shapes[index][1] / bucket_shapes[index][0])),
    )

class AspectRatioBatchSampler(ResumableAspectRatioBatchSampler):
    """Yield checkpointable batches whose images share one resolution bucket."""


class AddGaussianNoise:
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, image):
        return (image + torch.randn_like(image) * self.std).clamp(0.0, 1.0)

def image_transform(shape):
    return transforms.Compose([
        transforms.Resize(max(shape), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(shape),
        # Keep augmentations weak so caption/object placement remains aligned.
        transforms.ColorJitter(
            brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01,
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        ], p=0.15),
        transforms.RandomRotation(
            degrees=2,
            interpolation=transforms.InterpolationMode.BICUBIC,
            fill=(0, 0, 0),
        ),
        transforms.ToTensor(),
        AddGaussianNoise(std=0.01),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

class RecordsDataset(Dataset):
    def __init__(self, records_path, image_size, bucket_step, records=None):
        self.root = os.path.dirname(os.path.abspath(records_path)) if records_path else ""
        self.records = records
        if self.records is None:
            self.records = []
            if records_path.endswith(".csv"):
                with open(records_path, newline="", encoding="utf-8") as f:
                    self.records = list(csv.DictReader(f))
            else:
                with open(records_path, encoding="utf-8") as f:
                    self.records = [json.loads(line) for line in f if line.strip()]
        self.bucket_shapes = make_bucket_shapes(image_size, bucket_step)
        self.bucket_ids = []
        for record in self.records:
            from PIL import Image
            path = record.get("image", record.get("path"))
            if not os.path.isabs(path):
                path = os.path.join(self.root, path)
            with Image.open(path) as image:
                self.bucket_ids.append(assign_bucket(*image.size, self.bucket_shapes))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from PIL import Image
        record = self.records[index]
        path = record.get("image", record.get("path"))
        caption = row_caption(record)
        if not os.path.isabs(path):
            path = os.path.join(self.root, path)
        with Image.open(path) as source_image:
            image = source_image.convert("RGB")
        return image_transform(self.bucket_shapes[self.bucket_ids[index]])(image), str(caption)

class Flickr30KDataset(RecordsDataset):
    """Load the legacy nlphuji/flickr30k repository without its dataset script."""
    def __init__(self, split, image_size, bucket_step, cache_dir=None):
        from huggingface_hub import hf_hub_download

        cache_root = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        csv_path = hf_hub_download(
            "nlphuji/flickr30k", "flickr_annotations_30k.csv",
            repo_type="dataset", cache_dir=cache_dir,
        )
        zip_path = hf_hub_download(
            "nlphuji/flickr30k", "flickr30k-images.zip",
            repo_type="dataset", cache_dir=cache_dir,
        )
        image_root = os.path.join(cache_root, "flickr30k-images")
        os.makedirs(image_root, exist_ok=True)
        if not os.path.isdir(os.path.join(image_root, "flickr30k-images")):
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(image_root)

        records = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if split not in ("all", "*", row.get("split", "")):
                    continue
                captions = json.loads(row["raw"])
                records.append({
                    "image": os.path.join(image_root, "flickr30k-images", row["filename"]),
                    "caption": captions,
                })
        if not records:
            raise ValueError(f"No Flickr30K records found for split={split!r}")
        super().__init__(None, image_size, bucket_step, records=records)

class HFDataset(Dataset):
    def __init__(self, dataset_name, split, image_size, bucket_step, cache_dir=None):
        from datasets import load_dataset
        self.selected_split = split
        loaded = load_dataset(dataset_name, cache_dir=cache_dir)
        if hasattr(loaded, "keys"):
            available_splits = list(loaded.keys())
            if split in loaded:
                self.dataset = loaded[split]
            elif len(available_splits) == 1:
                selected_split = available_splits[0]
                print(
                    f"split={split!r} is unavailable for {dataset_name}; "
                    f"using the only available split={selected_split!r}"
                )
                self.selected_split = selected_split
                self.dataset = loaded[selected_split]
            else:
                raise ValueError(
                    f"Unknown split {split!r} for {dataset_name}. "
                    f"Available splits: {available_splits}"
                )
        else:
            self.dataset = loaded
        self.bucket_shapes = make_bucket_shapes(image_size, bucket_step)
        self.bucket_ids = []
        for index in range(len(self.dataset)):
            image = self.dataset[index]["image"]
            self.bucket_ids.append(assign_bucket(*image.size, self.bucket_shapes))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        image = row["image"].convert("RGB")
        return image_transform(self.bucket_shapes[self.bucket_ids[index]])(image), row_caption(row)

def collate(batch):
    images, captions = zip(*batch)
    return torch.stack(images), list(captions)

def row_caption(row):
    """Support common caption column names used by Flickr30K mirrors."""
    for key in ("caption", "text", "captions", "sentence", "sentences"):
        if key in row and row[key] is not None:
            caption = row[key]
            if isinstance(caption, dict):
                caption = caption.get("text", caption.get("raw", ""))
            if isinstance(caption, list):
                caption = caption[torch.randint(len(caption), ()).item()]
            return str(caption)
    raise KeyError(
        "Dataset row has no supported caption column: "
        "caption/text/captions/sentence/sentences"
    )
