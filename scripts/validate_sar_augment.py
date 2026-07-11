#!/usr/bin/env python
"""Validate SAR-TPT stage-three guided augmentation on a single image.

This acceptance script uses a simple synthetic or file-provided mask and does
not require CLIP. It verifies the output contract and prints crop statistics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as transforms

from data.sar_augment import SARAugMixAugmenter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SAR guided augmentation with a provided/synthetic mask")
    parser.add_argument("--image", type=Path, required=True, help="Path to a single RGB image")
    parser.add_argument("--mask", type=Path, default=None, help="Optional torch .pt mask [H,W]")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--n-views", type=int, default=4)
    parser.add_argument("--tau-cov", type=float, default=0.6)
    parser.add_argument("--max-crop-trials", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = Image.open(args.image).convert("RGB")

    if args.mask is not None:
        mask = torch.load(args.mask, map_location="cpu")
    else:
        mask = torch.zeros(args.resolution, args.resolution)
        margin = args.resolution // 4
        mask[margin:-margin, margin:-margin] = 1

    augmenter = SARAugMixAugmenter(
        base_transform=transforms.Compose([
            transforms.Resize(args.resolution),
            transforms.CenterCrop(args.resolution),
        ]),
        preprocess=transforms.ToTensor(),
        n_views=args.n_views,
        augmix=False,
        mask_provider=lambda _image: mask,
        tau_cov=args.tau_cov,
        max_crop_trials=args.max_crop_trials,
        output_size=args.resolution,
    )
    views = augmenter(image)
    print("views:", [tuple(view.shape) for view in views])
    print("stats:", augmenter.stats.as_dict())
    print("crops:", [info.__dict__ for info in augmenter.last_crop_infos])


if __name__ == "__main__":
    main()
