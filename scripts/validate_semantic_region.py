#!/usr/bin/env python
"""Validate SAR-TPT stage-two semantic region localization on one image.

This is an acceptance script for a prepared runtime environment. It loads a CLIP
TPT model, a stage-one anchor file, and a single image, then saves debug tensors
for the heatmap and mask. It is not run by Codex in this workspace.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:  # pragma: no cover - compatibility with older torchvision
    BICUBIC = Image.BICUBIC

from clip.custom_clip import get_coop
from utils.semantic_region import SemanticRegionLocator, save_semantic_region_debug
from utils.text_anchors import load_text_anchor_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SAR-TPT semantic region localization")
    parser.add_argument("--image", type=Path, required=True, help="Path to a single RGB image")
    parser.add_argument("--anchor-path", type=Path, required=True, help="Stage-one anchor .pt file")
    parser.add_argument("--dataset", required=True, help="Dataset id matching the anchor/model class names")
    parser.add_argument("--arch", default="ViT-B/16", help="CLIP architecture; stage two expects ViT")
    parser.add_argument("--gpu", type=int, default=None, help="Optional CUDA device id")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--n-ctx", type=int, default=4)
    parser.add_argument("--ctx-init", default="a_photo_of_a")
    parser.add_argument("--mask-top-ratio", type=float, default=0.3)
    parser.add_argument("--min-mask-area-ratio", type=float, default=0.01)
    parser.add_argument("--debug-prefix", type=Path, default=Path("outputs/stage2_debug/sample"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None and torch.cuda.is_available() else "cpu")

    payload = load_text_anchor_file(args.anchor_path, map_location="cpu")
    model = get_coop(args.arch, args.dataset, device="cpu", n_ctx=args.n_ctx, ctx_init=args.ctx_init)
    model.reset_classnames(payload["classnames"], args.arch)
    model.eval().to(device)

    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    preprocess = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            normalize,
        ]
    )
    image = preprocess(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)

    locator = SemanticRegionLocator(
        payload,
        mask_top_ratio=args.mask_top_ratio,
        min_mask_area_ratio=args.min_mask_area_ratio,
    )
    result = locator.locate(model, image)
    save_semantic_region_debug(image, result, args.debug_prefix)
    print(result.as_debug_dict())


if __name__ == "__main__":
    main()
