"""Pure-Python/PyTorch contract tests for SAR-TPT stage-two utilities.

These tests do not load CLIP weights or datasets. They validate heatmap/mask
logic and the forward-only locator contract with a tiny fake model.

Run in a prepared environment with:
    python -m unittest tests/test_stage2_semantic_region.py
"""

import unittest

import torch

from utils.semantic_region import (
    SemanticRegionLocator,
    mask_from_heatmap,
    normalize_heatmap,
    patch_similarity_to_heatmap,
)


class FakeSpatialModel:
    def __init__(self):
        self.logit_scale = torch.tensor(1.0)

    def encode_image_with_spatial_tokens(self, image):
        cls_feature = torch.tensor([[1.0, 0.0]], device=image.device)
        spatial_tokens = torch.tensor(
            [[[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]]],
            device=image.device,
        )
        return cls_feature, spatial_tokens, (2, 2)


class Stage2SemanticRegionTests(unittest.TestCase):
    def test_normalize_heatmap_constant_is_safe(self):
        heatmap = normalize_heatmap(torch.ones(2, 3))
        self.assertEqual(tuple(heatmap.shape), (2, 3))
        self.assertTrue(torch.all(heatmap == 0))

    def test_mask_from_heatmap_keeps_top_region(self):
        heatmap = torch.tensor([[0.0, 0.1], [0.9, 1.0]])
        mask, fallback_used, reason = mask_from_heatmap(heatmap, top_ratio=0.5)
        self.assertFalse(fallback_used)
        self.assertIsNone(reason)
        self.assertEqual(int(mask.sum().item()), 2)
        self.assertTrue(bool(mask[1, 0]))
        self.assertTrue(bool(mask[1, 1]))

    def test_patch_similarity_upsamples(self):
        patch = torch.tensor([[0.0, 1.0], [0.5, 0.25]])
        heatmap = patch_similarity_to_heatmap(patch, output_size=(8, 8))
        self.assertEqual(tuple(heatmap.shape), (8, 8))
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)

    def test_locator_contract_with_fake_model(self):
        anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        locator = SemanticRegionLocator(anchors, mask_top_ratio=0.5, min_mask_area_ratio=0.01)
        result = locator.locate(FakeSpatialModel(), torch.zeros(1, 3, 8, 8))
        self.assertEqual(result.target_idx, 0)
        self.assertEqual(tuple(result.heatmap.shape), (8, 8))
        self.assertEqual(tuple(result.mask.shape), (8, 8))
        self.assertEqual(tuple(result.patch_similarity.shape), (2, 2))
        self.assertFalse(result.fallback_used)
        self.assertGreater(result.mask_area_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
