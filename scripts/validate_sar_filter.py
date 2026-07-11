#!/usr/bin/env python
"""Validate SAR-TPT stage-four filtering with synthetic tensors.

This acceptance script does not require CLIP or datasets. It verifies the stage
four tensor contract and prints selected reliable view indices.
"""

from __future__ import annotations

import argparse

import torch

from utils.sar_filter import sar_filter_and_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SAR dual-modality filtering")
    parser.add_argument("--num-views", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--reliable-ratio", type=float, default=0.5)
    parser.add_argument("--lambda-anchor", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    logits = torch.randn(args.num_views, args.num_classes, requires_grad=True)
    image_features = torch.randn(args.num_views, args.dim)
    anchors = torch.randn(args.num_classes, args.dim)
    loss, result = sar_filter_and_loss(
        logits=logits,
        image_features=image_features,
        anchors=anchors,
        reliable_ratio=args.reliable_ratio,
        lambda_anchor=args.lambda_anchor,
    )
    loss.backward()
    print("loss:", float(loss.detach()))
    print("reliable_idx:", result.reliable_idx.tolist())
    print("debug:", result.as_debug_dict())
    print("logits_grad_ok:", logits.grad is not None)


if __name__ == "__main__":
    main()
