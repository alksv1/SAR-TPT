"""Dual-modality consistency filtering and SAR entropy loss.

Stage four of SAR-TPT filters augmented views using both prediction confidence
(logits entropy) and feature-space alignment to the strong text anchor selected
for the current test image. The final loss is the marginal entropy over reliable
views only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SARFilterResult:
    """Result of SAR view filtering."""

    reliable_idx: torch.Tensor
    reliable_logits: torch.Tensor
    reliable_features: torch.Tensor
    scores: torch.Tensor
    entropies: torch.Tensor
    anchor_similarities: torch.Tensor
    target_idx: int
    fallback_used: bool
    fallback_reason: Optional[str]

    def as_debug_dict(self) -> Dict[str, float]:
        return {
            "num_views": float(self.scores.numel()),
            "num_reliable": float(self.reliable_idx.numel()),
            "target_idx": float(self.target_idx),
            "fallback_used": float(self.fallback_used),
            "mean_score": float(self.scores.detach().float().mean().cpu()),
            "mean_entropy": float(self.entropies.detach().float().mean().cpu()),
            "mean_anchor_similarity": float(self.anchor_similarities.detach().float().mean().cpu()),
        }


def prediction_entropy(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-view Shannon entropy from logits."""

    probs = logits.softmax(dim=-1)
    log_probs = torch.log(probs.clamp_min(eps))
    return -(probs * log_probs).sum(dim=-1)


def marginal_entropy_loss(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Marginal entropy loss over selected view logits."""

    probs = logits.softmax(dim=-1)
    avg_probs = probs.mean(dim=0)
    return -(avg_probs * torch.log(avg_probs.clamp_min(eps))).sum()


def compute_anchor_target(
    image_features: torch.Tensor,
    anchors: torch.Tensor,
    logit_scale: Optional[torch.Tensor] = None,
) -> Tuple[int, torch.Tensor]:
    """Select pseudo target class from the clean/original view.

    The first feature in ``image_features`` is treated as the clean view.
    """

    features = F.normalize(image_features.detach().float(), dim=-1)
    anchors = F.normalize(anchors.detach().float().to(features.device), dim=-1)
    scale = 1.0 if logit_scale is None else logit_scale.detach().float().to(features.device)
    logits = scale * features[:1] @ anchors.t()
    probs = logits.softmax(dim=-1).squeeze(0)
    return int(torch.argmax(probs).item()), probs


def dual_modality_scores(
    logits: torch.Tensor,
    image_features: torch.Tensor,
    anchors: torch.Tensor,
    target_idx: int,
    lambda_anchor: float = 0.5,
    entropy_scale: float = 1.0,
    disable_anchor_filter: bool = False,
    disable_entropy_filter: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute SAR reliability scores for all views."""

    entropies = prediction_entropy(logits).detach().float()
    features = F.normalize(image_features.detach().float(), dim=-1)
    anchors = F.normalize(anchors.detach().float().to(features.device), dim=-1)
    target_anchor = anchors[target_idx]
    anchor_similarities = (features @ target_anchor).detach().float()

    if disable_anchor_filter and disable_entropy_filter:
        scores = torch.zeros_like(entropies)
    elif disable_anchor_filter:
        scores = -entropy_scale * entropies
    elif disable_entropy_filter:
        scores = anchor_similarities
    else:
        lam = float(max(0.0, min(1.0, lambda_anchor)))
        scores = lam * anchor_similarities - (1.0 - lam) * float(entropy_scale) * entropies
    return scores, entropies, anchor_similarities


def reliable_count(
    num_views: int,
    reliable_ratio: float = 0.5,
    reliable_top_k: Optional[int] = None,
    min_reliable_views: int = 1,
) -> int:
    """Resolve how many views to keep."""

    if num_views <= 0:
        raise ValueError("num_views must be positive")
    if reliable_top_k is not None and reliable_top_k > 0:
        count = int(reliable_top_k)
    else:
        count = int(round(num_views * float(reliable_ratio)))
    count = max(int(min_reliable_views), count)
    return max(1, min(num_views, count))


def select_reliable_views(
    logits: torch.Tensor,
    image_features: torch.Tensor,
    anchors: torch.Tensor,
    target_idx: int,
    lambda_anchor: float = 0.5,
    entropy_scale: float = 1.0,
    reliable_ratio: float = 0.5,
    reliable_top_k: Optional[int] = None,
    min_reliable_views: int = 1,
    disable_anchor_filter: bool = False,
    disable_entropy_filter: bool = False,
) -> SARFilterResult:
    """Select reliable augmented views by dual-modality score."""

    if logits.ndim != 2:
        raise ValueError(f"logits must be [N,K], got {tuple(logits.shape)}")
    if image_features.ndim != 2:
        raise ValueError(f"image_features must be [N,D], got {tuple(image_features.shape)}")
    if logits.shape[0] != image_features.shape[0]:
        raise ValueError("logits and image_features must have the same number of views")
    if anchors.ndim != 2:
        raise ValueError(f"anchors must be [K,D], got {tuple(anchors.shape)}")
    if logits.shape[1] != anchors.shape[0]:
        raise ValueError("logit class count and anchor class count do not match")
    if image_features.shape[1] != anchors.shape[1]:
        raise ValueError("image feature dim and anchor dim do not match")

    scores, entropies, anchor_similarities = dual_modality_scores(
        logits=logits,
        image_features=image_features,
        anchors=anchors,
        target_idx=target_idx,
        lambda_anchor=lambda_anchor,
        entropy_scale=entropy_scale,
        disable_anchor_filter=disable_anchor_filter,
        disable_entropy_filter=disable_entropy_filter,
    )

    fallback_used = False
    fallback_reason = None
    if not torch.isfinite(scores).all():
        scores = -entropies
        fallback_used = True
        fallback_reason = "non_finite_scores_entropy_fallback"

    count = reliable_count(
        num_views=logits.shape[0],
        reliable_ratio=reliable_ratio,
        reliable_top_k=reliable_top_k,
        min_reliable_views=min_reliable_views,
    )
    reliable_idx = torch.argsort(scores, descending=True)[:count]
    if reliable_idx.numel() == 0:
        reliable_idx = torch.argsort(entropies, descending=False)[:1]
        fallback_used = True
        fallback_reason = "empty_selection_entropy_fallback"

    return SARFilterResult(
        reliable_idx=reliable_idx,
        reliable_logits=logits[reliable_idx],
        reliable_features=image_features[reliable_idx],
        scores=scores,
        entropies=entropies,
        anchor_similarities=anchor_similarities,
        target_idx=target_idx,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
    )


def sar_filter_and_loss(
    logits: torch.Tensor,
    image_features: torch.Tensor,
    anchors: torch.Tensor,
    target_idx: Optional[int] = None,
    logit_scale: Optional[torch.Tensor] = None,
    lambda_anchor: float = 0.5,
    entropy_scale: float = 1.0,
    reliable_ratio: float = 0.5,
    reliable_top_k: Optional[int] = None,
    min_reliable_views: int = 1,
    disable_anchor_filter: bool = False,
    disable_entropy_filter: bool = False,
) -> Tuple[torch.Tensor, SARFilterResult]:
    """Compute SAR loss after reliable-view selection."""

    anchors = anchors.to(device=logits.device, dtype=torch.float32)
    if target_idx is None:
        target_idx, _ = compute_anchor_target(image_features, anchors, logit_scale=logit_scale)
    result = select_reliable_views(
        logits=logits,
        image_features=image_features,
        anchors=anchors,
        target_idx=target_idx,
        lambda_anchor=lambda_anchor,
        entropy_scale=entropy_scale,
        reliable_ratio=reliable_ratio,
        reliable_top_k=reliable_top_k,
        min_reliable_views=min_reliable_views,
        disable_anchor_filter=disable_anchor_filter,
        disable_entropy_filter=disable_entropy_filter,
    )
    loss = marginal_entropy_loss(result.reliable_logits)
    return loss, result
