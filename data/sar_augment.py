"""Region-guided multi-view augmentation for SAR-TPT stage three.

This module keeps the output contract of ``AugMixAugmenter``: calling the
augmenter on a PIL image returns ``[clean_image, view_1, ..., view_N]`` where all
items are normalized tensors. The difference is that random crops for augmented
views are accepted only when they sufficiently cover the semantic mask produced
by stage two.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import data.augmix_ops as augmentations


CropBox = Tuple[int, int, int, int]  # left, top, right, bottom in PIL coordinates
MaskProvider = Callable[[Image.Image], torch.Tensor]


@dataclass
class GuidedCropInfo:
    """Debug information for one region-guided crop."""

    box: CropBox
    coverage: float
    accepted: bool
    trials: int
    fallback_used: bool
    fallback_reason: Optional[str]


@dataclass
class SARAugmentStats:
    """Running statistics for SAR augmentation diagnostics."""

    calls: int = 0
    views: int = 0
    accepted_views: int = 0
    fallback_views: int = 0
    total_trials: int = 0
    total_coverage: float = 0.0

    def update(self, info: GuidedCropInfo) -> None:
        self.views += 1
        self.total_trials += info.trials
        self.total_coverage += info.coverage
        if info.accepted:
            self.accepted_views += 1
        if info.fallback_used:
            self.fallback_views += 1

    def as_dict(self) -> Dict[str, float]:
        return {
            "calls": float(self.calls),
            "views": float(self.views),
            "accepted_views": float(self.accepted_views),
            "fallback_views": float(self.fallback_views),
            "avg_trials": self.total_trials / max(1, self.views),
            "avg_coverage": self.total_coverage / max(1, self.views),
        }


def ensure_mask_tensor(mask: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Convert a mask to ``[H, W]`` float tensor aligned to ``size=(W,H)``."""

    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask.squeeze(0)
        elif mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        else:
            raise ValueError(f"mask must be [H,W] or single-channel, got {tuple(mask.shape)}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be [H,W], got {tuple(mask.shape)}")

    target_w, target_h = int(size[0]), int(size[1])
    mask = mask.float()
    if tuple(mask.shape) != (target_h, target_w):
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=(target_h, target_w),
            mode="nearest",
        ).squeeze(0).squeeze(0)
    return (mask > 0).float()


def crop_coverage(box: CropBox, mask: torch.Tensor, eps: float = 1e-6) -> float:
    """Return fraction of active mask pixels covered by ``box``."""

    left, top, right, bottom = box
    height, width = int(mask.shape[0]), int(mask.shape[1])
    left = max(0, min(width, int(left)))
    right = max(0, min(width, int(right)))
    top = max(0, min(height, int(top)))
    bottom = max(0, min(height, int(bottom)))
    total = float(mask.sum().item())
    if total <= eps:
        return 0.0
    if right <= left or bottom <= top:
        return 0.0
    covered = float(mask[top:bottom, left:right].sum().item())
    return covered / max(total, eps)


def mask_bounding_box(mask: torch.Tensor) -> Optional[CropBox]:
    """Return active mask bounding box as ``(left, top, right, bottom)``."""

    active = torch.nonzero(mask > 0, as_tuple=False)
    if active.numel() == 0:
        return None
    top = int(active[:, 0].min().item())
    bottom = int(active[:, 0].max().item()) + 1
    left = int(active[:, 1].min().item())
    right = int(active[:, 1].max().item()) + 1
    return left, top, right, bottom


def expand_box(box: CropBox, image_size: Tuple[int, int], padding_ratio: float = 0.15) -> CropBox:
    """Expand a crop box by a ratio and clamp to image size."""

    width, height = int(image_size[0]), int(image_size[1])
    left, top, right, bottom = box
    box_w = right - left
    box_h = bottom - top
    pad_w = int(round(box_w * padding_ratio))
    pad_h = int(round(box_h * padding_ratio))
    return (
        max(0, left - pad_w),
        max(0, top - pad_h),
        min(width, right + pad_w),
        min(height, bottom + pad_h),
    )


def center_crop_box(image_size: Tuple[int, int], scale: float = 0.875) -> CropBox:
    """Return a center crop box for ``image_size=(W,H)``."""

    width, height = int(image_size[0]), int(image_size[1])
    crop_w = max(1, int(round(width * scale)))
    crop_h = max(1, int(round(height * scale)))
    left = max(0, (width - crop_w) // 2)
    top = max(0, (height - crop_h) // 2)
    return left, top, min(width, left + crop_w), min(height, top + crop_h)


def sample_random_resized_crop_box(
    image_size: Tuple[int, int],
    scale: Tuple[float, float] = (0.5, 1.0),
    ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
) -> CropBox:
    """Sample a torchvision-style RandomResizedCrop box."""

    width, height = int(image_size[0]), int(image_size[1])
    area = height * width
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))

    for _ in range(10):
        target_area = area * random.uniform(scale[0], scale[1])
        aspect_ratio = math.exp(random.uniform(log_ratio[0], log_ratio[1]))
        crop_w = int(round(math.sqrt(target_area * aspect_ratio)))
        crop_h = int(round(math.sqrt(target_area / aspect_ratio)))
        if 0 < crop_w <= width and 0 < crop_h <= height:
            left = random.randint(0, width - crop_w)
            top = random.randint(0, height - crop_h)
            return left, top, left + crop_w, top + crop_h

    # Same spirit as torchvision fallback: centered crop respecting ratio range.
    in_ratio = width / height
    if in_ratio < min(ratio):
        crop_w = width
        crop_h = int(round(crop_w / min(ratio)))
    elif in_ratio > max(ratio):
        crop_h = height
        crop_w = int(round(crop_h * max(ratio)))
    else:
        crop_w = width
        crop_h = height
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    return left, top, left + crop_w, top + crop_h


def select_guided_crop_box(
    image_size: Tuple[int, int],
    mask: torch.Tensor,
    tau_cov: float = 0.6,
    max_trials: int = 20,
    scale: Tuple[float, float] = (0.5, 1.0),
    ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    bbox_padding_ratio: float = 0.15,
) -> GuidedCropInfo:
    """Sample an accepted crop box or a safe fallback box."""

    best_box: Optional[CropBox] = None
    best_coverage = -1.0
    for trial in range(1, max_trials + 1):
        box = sample_random_resized_crop_box(image_size, scale=scale, ratio=ratio)
        coverage = crop_coverage(box, mask)
        if coverage > best_coverage:
            best_box, best_coverage = box, coverage
        if coverage >= tau_cov:
            return GuidedCropInfo(
                box=box,
                coverage=coverage,
                accepted=True,
                trials=trial,
                fallback_used=False,
                fallback_reason=None,
            )

    bbox = mask_bounding_box(mask)
    if bbox is not None:
        box = expand_box(bbox, image_size, padding_ratio=bbox_padding_ratio)
        coverage = crop_coverage(box, mask)
        return GuidedCropInfo(
            box=box,
            coverage=coverage,
            accepted=coverage >= tau_cov,
            trials=max_trials,
            fallback_used=True,
            fallback_reason="mask_bbox_padding",
        )

    if best_box is not None and best_coverage > 0:
        return GuidedCropInfo(
            box=best_box,
            coverage=best_coverage,
            accepted=False,
            trials=max_trials,
            fallback_used=True,
            fallback_reason="best_effort_random_crop",
        )

    box = center_crop_box(image_size)
    return GuidedCropInfo(
        box=box,
        coverage=crop_coverage(box, mask),
        accepted=False,
        trials=max_trials,
        fallback_used=True,
        fallback_reason="center_crop",
    )


def apply_crop_resize_flip(
    image: Image.Image,
    box: CropBox,
    output_size: int,
    interpolation=Image.BICUBIC,
    hflip_p: float = 0.5,
) -> Image.Image:
    """Crop, resize, and optionally flip a PIL image."""

    cropped = image.crop(box).resize((output_size, output_size), interpolation)
    if random.random() < hflip_p:
        cropped = TF.hflip(cropped)
    return cropped


def apply_augmix_ops(
    image: Image.Image,
    preprocess: Callable[[Image.Image], torch.Tensor],
    aug_list: Sequence[Callable[[Image.Image, int], Image.Image]],
    severity: int = 1,
) -> torch.Tensor:
    """Apply the same AugMix-style color/geometry perturbations as TPT."""

    x_processed = preprocess(image)
    if len(aug_list) == 0:
        return x_processed

    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))
    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = image.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    return m * x_processed + (1 - m) * mix


class SARAugMixAugmenter(object):
    """Semantic-region-guided multi-view augmenter.

    Args:
        base_transform: Stable transform producing the clean PIL view coordinate
            system, usually Resize + CenterCrop.
        preprocess: Tensor conversion and CLIP normalization.
        semantic_locator: Optional stage-two ``SemanticRegionLocator``.
        model: Model passed to ``semantic_locator.locate``. If omitted, provide
            ``mask_provider``.
        mask_provider: Optional callable returning a mask for the base PIL view.
    """

    def __init__(
        self,
        base_transform: Callable[[Image.Image], Image.Image],
        preprocess: Callable[[Image.Image], torch.Tensor],
        n_views: int = 2,
        augmix: bool = False,
        severity: int = 1,
        semantic_locator: Optional[Any] = None,
        model: Optional[Any] = None,
        mask_provider: Optional[MaskProvider] = None,
        device: Optional[torch.device] = None,
        tau_cov: float = 0.6,
        max_crop_trials: int = 20,
        crop_scale: Tuple[float, float] = (0.5, 1.0),
        crop_ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        hflip_p: float = 0.5,
        output_size: int = 224,
        bbox_padding_ratio: float = 0.15,
    ) -> None:
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        self.aug_list = augmentations.augmentations if augmix else []
        self.severity = severity
        self.semantic_locator = semantic_locator
        self.model = model
        self.mask_provider = mask_provider
        self.device = device
        self.tau_cov = tau_cov
        self.max_crop_trials = max_crop_trials
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        self.hflip_p = hflip_p
        self.output_size = output_size
        augmentations.IMAGE_SIZE = output_size
        self.bbox_padding_ratio = bbox_padding_ratio
        self.stats = SARAugmentStats()
        self.last_crop_infos: List[GuidedCropInfo] = []
        self.last_locator_debug: Optional[Dict[str, Any]] = None

    def _locate_mask(self, base_image: Image.Image, clean_tensor: torch.Tensor) -> torch.Tensor:
        if self.mask_provider is not None:
            return ensure_mask_tensor(self.mask_provider(base_image), base_image.size)

        if self.semantic_locator is None or self.model is None:
            # Full mask preserves original random-crop behavior while satisfying
            # the output contract when SAR localization is not attached yet.
            width, height = base_image.size
            return torch.ones((height, width), dtype=torch.float32)

        image_tensor = clean_tensor.unsqueeze(0)
        if self.device is not None:
            image_tensor = image_tensor.to(self.device)
        result = self.semantic_locator.locate(self.model, image_tensor)
        self.last_locator_debug = result.as_debug_dict()
        return ensure_mask_tensor(result.mask, base_image.size)

    def __call__(self, x: Image.Image) -> List[torch.Tensor]:
        base_image = self.base_transform(x)
        clean_tensor = self.preprocess(base_image)
        mask = self._locate_mask(base_image, clean_tensor)

        self.stats.calls += 1
        self.last_crop_infos = []
        views: List[torch.Tensor] = []
        for _ in range(self.n_views):
            info = select_guided_crop_box(
                image_size=base_image.size,
                mask=mask,
                tau_cov=self.tau_cov,
                max_trials=self.max_crop_trials,
                scale=self.crop_scale,
                ratio=self.crop_ratio,
                bbox_padding_ratio=self.bbox_padding_ratio,
            )
            view_image = apply_crop_resize_flip(
                base_image,
                info.box,
                output_size=self.output_size,
                hflip_p=self.hflip_p,
            )
            view_tensor = apply_augmix_ops(view_image, self.preprocess, self.aug_list, self.severity)
            views.append(view_tensor)
            self.last_crop_infos.append(info)
            self.stats.update(info)

        return [clean_tensor] + views
