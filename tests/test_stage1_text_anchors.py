"""Pure-Python tests for SAR-TPT stage-one text-anchor helpers.

These tests avoid loading CLIP and do not require dataset files. They validate the
asset contract that later encoding/inference code depends on.

Run in a prepared environment with:
    python -m unittest tests/test_stage1_text_anchors.py
"""

import tempfile
import unittest
from pathlib import Path

from utils.text_anchors import (
    build_description_payload,
    canonical_dataset_name,
    default_visual_descriptions,
    save_description_payload,
    extract_json_array,
    load_description_payload,
    normalize_chat_completions_url,
    validate_description_payload,
)


class Stage1TextAnchorHelperTests(unittest.TestCase):
    def test_dataset_aliases(self):
        self.assertEqual(canonical_dataset_name("pets"), "Pets")
        self.assertEqual(canonical_dataset_name("imagenet-a"), "A")
        self.assertEqual(canonical_dataset_name("FGVC_Aircraft"), "Aircraft")

    def test_default_descriptions_are_deterministic_and_visual(self):
        descriptions = default_visual_descriptions("american_bulldog", count=3)
        self.assertEqual(len(descriptions), 3)
        self.assertTrue(all("american bulldog" in item for item in descriptions))
        self.assertTrue(any("localized" in item for item in descriptions))


    def test_openai_compatible_url_normalization(self):
        self.assertEqual(
            normalize_chat_completions_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1/chat/completions",
        )
        self.assertEqual(
            normalize_chat_completions_url("https://example.com/v1/chat/completions"),
            "https://example.com/v1/chat/completions",
        )

    def test_llm_json_array_extraction(self):
        self.assertEqual(extract_json_array('["a", "b"]'), ["a", "b"])
        self.assertEqual(extract_json_array('```json\n["a", "b"]\n```'), ["a", "b"])
        self.assertEqual(extract_json_array('Here is JSON: ["a", "b"]'), ["a", "b"])

    def test_payload_round_trip_and_validation(self):
        classnames = ["class_a", "class_b"]
        payload = build_description_payload(
            dataset="Pets",
            classnames=classnames,
            descriptions_per_class=3,
        )
        validate_description_payload(payload, classnames, min_descriptions=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "descriptions.json"
            save_description_payload(payload, path)
            loaded = load_description_payload(path)
        self.assertEqual(loaded["dataset"], "Pets")
        self.assertEqual(list(loaded["classes"].keys()), classnames)

    def test_validation_rejects_missing_class(self):
        classnames = ["class_a", "class_b"]
        payload = build_description_payload("Pets", ["class_a"], descriptions_per_class=3)
        with self.assertRaises(ValueError):
            validate_description_payload(payload, classnames, min_descriptions=3)


if __name__ == "__main__":
    unittest.main()
