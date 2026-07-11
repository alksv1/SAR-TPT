#!/usr/bin/env python
"""Validate SAR-TPT stage-one description JSON and encoded anchor assets.

This script is for stage acceptance in an environment where dependencies are
available. It does not build anchors; it only checks asset contracts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.text_anchors import (
    canonical_dataset_name,
    get_dataset_classnames,
    load_description_payload,
    validate_description_payload,
    load_text_anchor_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SAR-TPT text-anchor assets")
    parser.add_argument("--dataset", required=True, help="Dataset id matching the asset")
    parser.add_argument("--description-path", type=Path, required=True, help="Description JSON to validate")
    parser.add_argument("--anchor-path", type=Path, default=None, help="Optional encoded .pt anchor asset")
    parser.add_argument("--min-descriptions", type=int, default=3)
    parser.add_argument("--check-norm", action="store_true", help="Check L2 norms of encoded anchors")
    parser.add_argument("--norm-atol", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = canonical_dataset_name(args.dataset)
    classnames = get_dataset_classnames(dataset)

    desc = load_description_payload(args.description_path)
    validate_description_payload(desc, classnames, min_descriptions=args.min_descriptions)
    print(f"[validate] description OK: {args.description_path} ({len(classnames)} classes)")

    if args.anchor_path is None:
        return

    anchor = load_text_anchor_file(args.anchor_path, map_location="cpu")
    if anchor["dataset"] != dataset:
        raise ValueError(f"anchor dataset mismatch: {anchor['dataset']} != {dataset}")
    if list(anchor["classnames"]) != list(classnames):
        raise ValueError("anchor classnames do not match dataset evaluation order")

    if args.check_norm:
        import torch

        norms = anchor["anchors"].float().norm(dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=args.norm_atol):
            raise ValueError(
                f"anchor L2 norms outside tolerance {args.norm_atol}: "
                f"min={float(norms.min())}, max={float(norms.max())}"
            )
    print(f"[validate] anchor OK: {args.anchor_path} shape={tuple(anchor['anchors'].shape)}")


if __name__ == "__main__":
    main()
