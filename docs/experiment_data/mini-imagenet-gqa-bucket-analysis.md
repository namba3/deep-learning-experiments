# Mini-ImageNet resolution and aspect buckets

## Dataset scan

`mini_imagenet_gqa/analyze_image_buckets.py` decoded the cached `timm/mini-imagenet` image records offline on 2026-09-25. It measured the original PIL image dimensions across all splits: train 50,000, validation 10,000, and test 5,000. This scan does not copy the Arrow data or contact the Hub. Reproduce it with:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python3 -m mini_imagenet_gqa.analyze_image_buckets \
  --cache-dir ./path/to/huggingface/datasets --offline
```

The train split contains 10,093 distinct width/height pairs. Its most common shapes are 500×375 (23.09%), 500×333 (8.09%), 375×500 (6.61%), and 333×500 (2.94%). Validation and test have the same dominant shapes, so the source aspect distribution is consistent across splits.

| Split | Aspect ratio p05 / p50 / p95 | Pixel area p05 / p50 / p95 | Short side p05 / p50 / p95 |
| --- | ---: | ---: | ---: |
| Train | 0.666 / 1.333 / 1.531 | 38,400 / 187,500 / 405,008 | 162 / 375 / 538 |
| Validation | 0.666 / 1.333 / 1.527 | 37,411 / 187,500 / 413,267 | 161 / 375 / 550 |
| Test | 0.668 / 1.333 / 1.502 | 136,900 / 187,500 / 307,200 | 300 / 375 / 500 |

The source resolution has a long tail: train pixel area ranges from 1,650 to 28,512,640, while p01–p99 is 11,625–1,616,724. Native source dimensions should therefore describe the selected model input bucket, not select unbounded target resolutions. For the first comparison, keep approximately the same token budget and vary aspect ratio; a separate resolution-tier experiment would change compute and should be measured independently.

## Proposed first bucket set

Choose the nearest bucket by absolute log aspect-ratio distance, then resize while preserving aspect ratio and randomly crop to that bucket. All dimensions are divisible by 8, matching the classifier's three stride-2 stages. The token count column is the final spatial token count after those stages.

| Bucket (H×W) | W/H | Final tokens | Input pixels vs 64×64 | Train assignment |
| --- | ---: | ---: | ---: | ---: |
| 80×56 | 0.700 | 70 | +9.4% | 11.31% |
| 72×56 | 0.778 | 63 | −1.6% | 13.36% |
| 64×64 | 1.000 | 64 | 0% | 8.53% |
| 56×72 | 1.286 | 63 | −1.6% | 39.41% |
| 56×80 | 1.429 | 70 | +9.4% | 27.40% |

The plan follows the dataset's dominant portrait, square, 4:3 landscape, and 3:2 landscape modes while keeping per-image final token count between 63 and 70. Across the train assignment, the weighted mean is 65.79 final tokens per image (+2.8% versus 64). The assignment distribution is similar on validation (10.87%, 13.55%, 8.92%, 40.04%, 26.62%) and test (7.60%, 16.90%, 10.84%, 48.54%, 16.12%).

For a geometry-only resize-to-cover plus crop estimate, the assigned bucket retains at least 90% of source content for 92.39% of train, 93.02% of validation, and 95.10% of test images. It retains at least 80% for 97.44%, 97.71%, and 99.08%, respectively. The long-tail extreme aspect ratios account for most of the large crops; a first pass can route them to the nearest endpoint bucket rather than adding sparse buckets.

## Implementation implications

The trainer now assigns each image from its source aspect ratio, produces the selected `(H,W)` shape, and batches examples with identical bucket shapes. It passes `[log(sqrt(H*W)), log(W/H)]` from that selected post-transform bucket to the model. The model accepts positive H/W dimensions divisible by the number of stride-2 stages; every proposed bucket satisfies this constraint.

This is a proposed first aspect-bucket screen, not a claim that these are final optimal sizes. It keeps compute close to the current 64×64 baseline and avoids using source pixel count as a proxy for a desired generation/classification resolution.
