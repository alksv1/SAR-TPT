"""Pure utility tests for SAR-TPT stage-three region-guided augmentation.

These tests avoid CLIP and datasets. They validate coverage math, fallback crop
selection, and the AugMixAugmenter-compatible output contract.

Run in a prepared environment with:
    python -m unittest tests/test_stage3_sar_augment.py
"""

import unittest

import torch
from PIL import Image
import torchvision.transforms as transforms

from data.sar_augment import (
    SARAugMixAugmenter,
    crop_coverage,
    ensure_mask_tensor,
    mask_bounding_box,
    select_guided_crop_box,
)


class Stage3SARAugmentTests(unittest.TestCase):
    def test_crop_coverage(self):
        mask = torch.zeros(10, 10)
        mask[2:6, 2:6] = 1
        self.assertAlmostEqual(crop_coverage((2, 2, 6, 6), mask), 1.0)
        self.assertAlmostEqual(crop_coverage((0, 0, 4, 4), mask), 0.25)

    def test_mask_bounding_box(self):
        mask = torch.zeros(8, 9)
        mask[2:5, 3:7] = 1
        self.assertEqual(mask_bounding_box(mask), (3, 2, 7, 5))

    def test_select_guided_crop_box_with_bbox_fallback(self):
        mask = torch.zeros(32, 32)
        mask[12:20, 12:20] = 1
        info = select_guided_crop_box(
            image_size=(32, 32),
            mask=mask,
            tau_cov=1.1,  # impossible threshold forces fallback
            max_trials=1,
        )
        self.assertTrue(info.fallback_used)
        self.assertGreater(info.coverage, 0.0)

    def test_augmenter_output_contract_with_mask_provider(self):
        image = Image.new("RGB", (64, 64), color=(127, 127, 127))

        def mask_provider(_base_image):
            mask = torch.zeros(32, 32)
            mask[8:24, 8:24] = 1
            return mask

        augmenter = SARAugMixAugmenter(
            base_transform=transforms.Resize((32, 32)),
            preprocess=transforms.ToTensor(),
            n_views=3,
            augmix=False,
            mask_provider=mask_provider,
            tau_cov=0.5,
            max_crop_trials=5,
            output_size=32,
        )
        views = augmenter(image)
        self.assertEqual(len(views), 4)
        self.assertTrue(all(isinstance(view, torch.Tensor) for view in views))
        self.assertTrue(all(tuple(view.shape) == (3, 32, 32) for view in views))
        self.assertEqual(augmenter.stats.views, 3)

    def test_ensure_mask_tensor_resizes(self):
        mask = torch.ones(4, 4)
        resized = ensure_mask_tensor(mask, size=(8, 8))
        self.assertEqual(tuple(resized.shape), (8, 8))
        self.assertTrue(torch.all(resized == 1))


if __name__ == "__main__":
    unittest.main()
