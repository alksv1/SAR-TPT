"""Semantic-aware region localization for SAR-TPT stage two.

The locator is deliberately forward-only: it never calls ``backward`` and it
expects a CLIP ViT visual encoder that can expose projected patch tokens. The
output heatmap/mask is used by stage three to constrain random crops.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from utils.text_anchors import load_text_anchor_file, validate_text_anchor_payload


@dataclass(frozen=True)
class SemanticRegionResult:
    """Output of forward-only semantic region localization."""

    target_idx: int
    heatmap: torch.Tensor  # [H, W], float32 in [0, 1]
    mask: torch.Tensor  # [H, W], bool
    patch_similarity: torch.Tensor  # [Gh, Gw], float32 before upsampling
    initial_probs: torch.Tensor  # [K], float32
    mask_area_ratio: float
    fallback_used: bool
    fallback_reason: Optional[str]

    def as_debug_dict(self) -> Dict[str, Any]:
        return {
            "target_idx": self.target_idx,
            "mask_area_ratio": self.mask_area_ratio,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "heatmap_shape": tuple(self.heatmap.shape),
            "mask_shape": tuple(self.mask.shape),
            "patch_similarity_shape": tuple(self.patch_similarity.shape),
        }


def load_anchor_tensor(anchor_path: Union[str, Path], map_location: str = "cpu") -> torch.Tensor:
    """Load only the anchor tensor from a stage-one ``*.pt`` file."""

    payload = load_text_anchor_file(anchor_path, map_location=map_location)
    return payload["anchors"]


def normalize_heatmap(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Min-max normalize a heatmap-like tensor to [0, 1]."""

    values = values.float()
    min_value = values.min()
    max_value = values.max()
    denom = max_value - min_value
    if torch.abs(denom) <= eps:
        return torch.zeros_like(values, dtype=torch.float32)
    return (values - min_value) / denom.clamp_min(eps)


def make_center_mask(height: int, width: int, ratio: float, device: torch.device) -> torch.Tensor:
    """Create a rectangular center fallback mask with approximately ``ratio`` area."""

    ratio = float(max(0.0, min(1.0, ratio)))
    if ratio <= 0:
        return torch.zeros((height, width), dtype=torch.bool, device=device)
    side = ratio ** 0.5
    box_h = max(1, int(round(height * side)))
    box_w = max(1, int(round(width * side)))
    top = max(0, (height - box_h) // 2)
    left = max(0, (width - box_w) // 2)
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    mask[top : top + box_h, left : left + box_w] = True
    return mask


def mask_from_heatmap(
    heatmap: torch.Tensor,
    top_ratio: float = 0.3,
    min_area_ratio: float = 0.01,
    fallback: str = "center",
) -> Tuple[torch.Tensor, bool, Optional[str]]:
    """Threshold a normalized heatmap and produce a safe binary mask.

    Args:
        heatmap: [H, W] tensor, expected but not required to be normalized.
        top_ratio: Fraction of highest-response pixels to keep.
        min_area_ratio: Minimum accepted active area ratio.
        fallback: One of ``center``, ``full`` or ``none``.

    Returns:
        ``(mask, fallback_used, fallback_reason)``.
    """

    if heatmap.ndim != 2:
        raise ValueError(f"heatmap must be [H, W], got shape {tuple(heatmap.shape)}")
    height, width = int(heatmap.shape[0]), int(heatmap.shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("heatmap spatial size must be positive")

    top_ratio = float(max(0.0, min(1.0, top_ratio)))
    if top_ratio <= 0:
        mask = torch.zeros_like(heatmap, dtype=torch.bool)
    elif top_ratio >= 1:
        mask = torch.ones_like(heatmap, dtype=torch.bool)
    else:
        threshold = torch.quantile(heatmap.float().flatten(), 1.0 - top_ratio)
        mask = heatmap >= threshold

    area_ratio = float(mask.float().mean().item())
    if mask.any() and area_ratio >= min_area_ratio:
        return mask.bool(), False, None

    reason = "empty_or_too_small_mask"
    if fallback == "center":
        fallback_ratio = max(float(min_area_ratio), min(float(top_ratio), 1.0))
        return make_center_mask(height, width, fallback_ratio, heatmap.device), True, reason
    if fallback == "full":
        return torch.ones_like(heatmap, dtype=torch.bool), True, reason
    if fallback == "none":
        return mask.bool(), True, reason
    raise ValueError(f"Unsupported mask fallback policy: {fallback}")


def patch_similarity_to_heatmap(
    patch_similarity: torch.Tensor,
    output_size: Tuple[int, int],
) -> torch.Tensor:
    """Upsample patch-grid similarity to image resolution and normalize it."""

    if patch_similarity.ndim != 2:
        raise ValueError(
            f"patch_similarity must be [Gh, Gw], got shape {tuple(patch_similarity.shape)}"
        )
    heatmap = patch_similarity.float().unsqueeze(0).unsqueeze(0)
    heatmap = F.interpolate(heatmap, size=output_size, mode="bilinear", align_corners=False)
    return normalize_heatmap(heatmap.squeeze(0).squeeze(0))


class SemanticRegionLocator:
    """Forward-only semantic localizer using CLIP ViT spatial tokens and text anchors."""

    def __init__(
        self,
        anchors: Union[torch.Tensor, Mapping[str, Any]],
        mask_top_ratio: float = 0.3,
        min_mask_area_ratio: float = 0.01,
        fallback: str = "center",
        logit_scale: Optional[Union[torch.Tensor, float]] = None,
        allow_non_vit_fallback: bool = False,
    ) -> None:
        if isinstance(anchors, Mapping):
            validate_text_anchor_payload(anchors)
            anchors = anchors["anchors"]
        if anchors.ndim != 2:
            raise ValueError(f"anchors must be [K, D], got shape {tuple(anchors.shape)}")
        self.anchors = F.normalize(anchors.detach().float(), dim=-1)
        self.mask_top_ratio = mask_top_ratio
        self.min_mask_area_ratio = min_mask_area_ratio
        self.fallback = fallback
        self.logit_scale = logit_scale
        self.allow_non_vit_fallback = allow_non_vit_fallback

    @classmethod
    def from_anchor_path(cls, anchor_path: Union[str, Path], **kwargs: Any) -> "SemanticRegionLocator":
        return cls(load_anchor_tensor(anchor_path), **kwargs)

    def _resolve_logit_scale(self, model: Any, device: torch.device) -> torch.Tensor:
        if self.logit_scale is not None:
            value = self.logit_scale
        elif hasattr(model, "logit_scale"):
            value = model.logit_scale
        else:
            value = torch.tensor(1.0)
        if isinstance(value, torch.Tensor):
            return value.detach().float().to(device).exp() if value.ndim == 0 else value.detach().float().to(device)
        return torch.tensor(float(value), device=device, dtype=torch.float32)

    def _fallback_result(
        self,
        image: torch.Tensor,
        reason: str,
        device: torch.device,
    ) -> SemanticRegionResult:
        height, width = int(image.shape[-2]), int(image.shape[-1])
        heatmap = torch.zeros((height, width), dtype=torch.float32, device=device)
        mask, _, _ = mask_from_heatmap(
            heatmap,
            top_ratio=self.mask_top_ratio,
            min_area_ratio=self.min_mask_area_ratio,
            fallback=self.fallback,
        )
        probs = torch.full((self.anchors.shape[0],), 1.0 / self.anchors.shape[0], device=device)
        return SemanticRegionResult(
            target_idx=0,
            heatmap=heatmap.detach().cpu(),
            mask=mask.detach().cpu(),
            patch_similarity=torch.zeros((1, 1), dtype=torch.float32),
            initial_probs=probs.detach().cpu(),
            mask_area_ratio=float(mask.float().mean().item()),
            fallback_used=True,
            fallback_reason=reason,
        )

    def locate(self, model: Any, image: torch.Tensor) -> SemanticRegionResult:
        """Locate semantic region for a single image tensor.

        Args:
            model: ``ClipTestTimeTuning`` or any object implementing
                ``encode_image_with_spatial_tokens(image)`` and optionally
                ``logit_scale``.
            image: Tensor shaped ``[C, H, W]`` or ``[1, C, H, W]`` in CLIP input
                space. Batch size must be one.
        """

        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError(f"image must be [C,H,W] or [1,C,H,W], got {tuple(image.shape)}")

        device = image.device
        if not hasattr(model, "encode_image_with_spatial_tokens"):
            if self.allow_non_vit_fallback:
                return self._fallback_result(image, "model_missing_spatial_token_api", device)
            raise TypeError("model must provide encode_image_with_spatial_tokens(image)")

        anchors = self.anchors.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            try:
                cls_feature, spatial_tokens, grid_size = model.encode_image_with_spatial_tokens(image)
            except NotImplementedError:
                if self.allow_non_vit_fallback:
                    return self._fallback_result(image, "non_vit_visual_encoder", device)
                raise

            cls_feature = F.normalize(cls_feature.float(), dim=-1)
            spatial_tokens = F.normalize(spatial_tokens.float(), dim=-1)
            scale = self._resolve_logit_scale(model, device)
            logits = scale * cls_feature @ anchors.t()
            probs = logits.softmax(dim=-1).squeeze(0)
            target_idx = int(torch.argmax(probs).item())
            target_anchor = anchors[target_idx : target_idx + 1]
            similarity = (spatial_tokens.squeeze(0) @ target_anchor.squeeze(0)).float()

            grid_h, grid_w = int(grid_size[0]), int(grid_size[1])
            if similarity.numel() != grid_h * grid_w:
                raise ValueError(
                    f"spatial token count {similarity.numel()} does not match grid {grid_h}x{grid_w}"
                )
            patch_similarity = similarity.reshape(grid_h, grid_w)
            output_size = (int(image.shape[-2]), int(image.shape[-1]))
            heatmap = patch_similarity_to_heatmap(patch_similarity, output_size=output_size)
            mask, fallback_used, fallback_reason = mask_from_heatmap(
                heatmap,
                top_ratio=self.mask_top_ratio,
                min_area_ratio=self.min_mask_area_ratio,
                fallback=self.fallback,
            )

        return SemanticRegionResult(
            target_idx=target_idx,
            heatmap=heatmap.detach().cpu(),
            mask=mask.detach().cpu(),
            patch_similarity=patch_similarity.detach().cpu(),
            initial_probs=probs.detach().cpu(),
            mask_area_ratio=float(mask.float().mean().item()),
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )


def save_semantic_region_debug(
    image: torch.Tensor,
    result: SemanticRegionResult,
    output_prefix: Union[str, Path],
) -> None:
    """Save lightweight debug tensors for visual inspection.

    This intentionally avoids adding plotting dependencies. Files are saved as
    ``*_heatmap.pt`` and ``*_mask.pt``; callers can render them however they like.
    """

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    torch.save(image.detach().cpu(), prefix.with_name(prefix.name + "_image.pt"))
    torch.save(result.heatmap.detach().cpu(), prefix.with_name(prefix.name + "_heatmap.pt"))
    torch.save(result.mask.detach().cpu(), prefix.with_name(prefix.name + "_mask.pt"))
    torch.save(result.as_debug_dict(), prefix.with_name(prefix.name + "_meta.pt"))
