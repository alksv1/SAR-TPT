"""Pure PyTorch tests for SAR-TPT stage-four filtering utilities.

These tests do not load CLIP or datasets. They validate reliable-view selection,
SAR loss construction, and ablation switches.

Run in a prepared environment with:
    python -m unittest tests/test_stage4_sar_filter.py
"""

import unittest

import torch

from utils.sar_filter import (
    compute_anchor_target,
    dual_modality_scores,
    marginal_entropy_loss,
    prediction_entropy,
    reliable_count,
    sar_filter_and_loss,
    select_reliable_views,
)


class Stage4SARFilterTests(unittest.TestCase):
    def test_prediction_entropy_shape(self):
        logits = torch.tensor([[3.0, 0.0], [0.0, 0.0]])
        entropy = prediction_entropy(logits)
        self.assertEqual(tuple(entropy.shape), (2,))
        self.assertLess(float(entropy[0]), float(entropy[1]))

    def test_compute_anchor_target_uses_clean_view(self):
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        target_idx, probs = compute_anchor_target(features, anchors)
        self.assertEqual(target_idx, 0)
        self.assertEqual(tuple(probs.shape), (2,))

    def test_dual_modality_scores_anchor_only(self):
        logits = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        scores, entropies, sims = dual_modality_scores(
            logits, features, anchors, target_idx=0, disable_entropy_filter=True
        )
        self.assertGreater(float(scores[0]), float(scores[1]))
        self.assertTrue(torch.allclose(scores, sims))
        self.assertEqual(tuple(entropies.shape), (2,))

    def test_reliable_count_bounds(self):
        self.assertEqual(reliable_count(8, reliable_ratio=0.5), 4)
        self.assertEqual(reliable_count(8, reliable_top_k=3), 3)
        self.assertEqual(reliable_count(2, reliable_ratio=0.1, min_reliable_views=1), 1)

    def test_select_reliable_views_and_loss(self):
        logits = torch.tensor(
            [[4.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, 4.0]],
            requires_grad=True,
        )
        features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
        anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        result = select_reliable_views(
            logits, features, anchors, target_idx=0, reliable_top_k=2, disable_entropy_filter=True
        )
        self.assertEqual(result.reliable_idx.numel(), 2)
        self.assertTrue(torch.all(result.reliable_idx == torch.tensor([0, 1])))

        loss, result = sar_filter_and_loss(
            logits, features, anchors, target_idx=0, reliable_top_k=2, disable_entropy_filter=True
        )
        self.assertEqual(result.reliable_logits.shape[0], 2)
        self.assertGreaterEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_marginal_entropy_loss_scalar(self):
        logits = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
        loss = marginal_entropy_loss(logits)
        self.assertEqual(loss.ndim, 0)


if __name__ == "__main__":
    unittest.main()
