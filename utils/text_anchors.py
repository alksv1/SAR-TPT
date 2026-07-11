"""Utilities for SAR-TPT stage 1 text-anchor assets.

This module intentionally keeps runtime responsibilities small:
- resolve dataset class names in the exact order used by evaluation;
- validate/load description JSON files;
- validate/load encoded anchor tensors produced offline.

Encoding CLIP features is implemented in ``scripts/build_text_anchors.py`` so that
normal inference code can import this module without eagerly constructing CLIP.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Union


DATASET_ALIASES: Dict[str, str] = {
    "i": "I",
    "imagenet": "I",
    "imagenet1k": "I",
    "a": "A",
    "imagenet-a": "A",
    "imagenet_a": "A",
    "r": "R",
    "imagenet-r": "R",
    "imagenet_r": "R",
    "v": "V",
    "imagenet-v2": "V",
    "imagenet_v2": "V",
    "k": "K",
    "imagenet-sketch": "K",
    "imagenet_sketch": "K",
    "pets": "Pets",
    "oxfordpets": "Pets",
    "oxford_pets": "Pets",
    "cars": "Cars",
    "stanfordcars": "Cars",
    "stanford_cars": "Cars",
    "aircraft": "Aircraft",
    "fgvc_aircraft": "Aircraft",
    "flower102": "Flower102",
    "flowers102": "Flower102",
    "food101": "Food101",
    "dtd": "DTD",
    "sun397": "SUN397",
    "caltech101": "Caltech101",
    "ucf101": "UCF101",
    "eurosat": "eurosat",
}


def canonical_dataset_name(dataset: str) -> str:
    """Return the project canonical dataset id used by TPT/SAR-TPT."""

    key = dataset.strip()
    if not key:
        raise ValueError("dataset name must be non-empty")
    return DATASET_ALIASES.get(key.lower(), key)


def get_dataset_classnames(dataset: str) -> List[str]:
    """Resolve class names in the same order as ``tpt_classification.py``.

    Args:
        dataset: Dataset id such as ``Pets``, ``Cars``, ``Aircraft``, ``I``,
            ``A``, ``R``, ``V`` or a supported alias.

    Returns:
        Ordered class-name list matching the model logits for that dataset.
    """

    dataset = canonical_dataset_name(dataset)

    if dataset in {"I", "K"}:
        from data.imagnet_prompts import imagenet_classes

        return list(imagenet_classes)

    if dataset in {"A", "R", "V"}:
        from data.imagnet_prompts import imagenet_classes
        from data.imagenet_variants import imagenet_a_mask, imagenet_r_mask, imagenet_v_mask

        if dataset == "A":
            return [imagenet_classes[i] for i in imagenet_a_mask]
        if dataset == "V":
            return [imagenet_classes[i] for i in imagenet_v_mask]
        return [name for name, keep in zip(imagenet_classes, imagenet_r_mask) if keep]

    import data.cls_to_names as cls_to_names_module

    cls_to_names = vars(cls_to_names_module)
    attr = f"{dataset.lower()}_classes"
    if attr not in cls_to_names:
        available = sorted(
            name[: -len("_classes")]
            for name in cls_to_names
            if name.endswith("_classes")
        )
        raise KeyError(
            f"Unsupported dataset '{dataset}'. Expected one of ImageNet ids "
            f"(I/A/R/V/K) or class lists: {available}"
        )
    return list(cls_to_names[attr])


def normalize_classname(classname: str) -> str:
    """Normalize a class name for description text without changing ordering keys."""

    return " ".join(classname.replace("_", " ").split())


def default_visual_descriptions(classname: str, count: int = 3) -> List[str]:
    """Create deterministic offline fallback descriptions.

    These are not a replacement for curated LLM descriptions, but they satisfy the
    stage-one offline/cache contract and make the pipeline usable without network
    access. Users can edit the generated JSON later and re-encode anchors.
    """

    name = normalize_classname(classname)
    templates = [
        (
            "A photo of {name}, emphasizing visible localized parts, distinctive "
            "shape, color pattern, texture, markings, and fine-grained visual cues."
        ),
        (
            "The distinguishing visual characteristics of {name} include local "
            "structures, silhouettes, surface textures, color distribution, and "
            "part-level details useful for image classification."
        ),
        (
            "A close visual description of {name} focusing on identifiable parts, "
            "edges, contours, proportions, material appearance, and subtle details "
            "that separate it from similar categories."
        ),
        (
            "Fine-grained cues for {name}: inspect prominent parts, boundary shape, "
            "relative proportions, recurring patterns, and localized color or texture "
            "regions visible in the image."
        ),
        (
            "An image of {name} can be recognized by visual attributes on key local "
            "regions, including part geometry, texture, markings, color contrast, "
            "and class-specific appearance."
        ),
    ]
    if count <= len(templates):
        return [t.format(name=name) for t in templates[:count]]
    descriptions = [t.format(name=name) for t in templates]
    while len(descriptions) < count:
        descriptions.append(
            f"Additional fine-grained visual description for {name}, focusing on "
            f"localized parts, shape, texture, color, markings, and visible details "
            f"for class discrimination #{len(descriptions) + 1}."
        )
    return descriptions


def build_description_payload(
    dataset: str,
    classnames: Sequence[str],
    descriptions: Optional[Mapping[str, Sequence[str]]] = None,
    descriptions_per_class: int = 3,
    generator: str = "template-fallback",
    prompt_version: str = "sar-tpt-stage1-v1",
) -> Dict[str, Any]:
    """Build a serializable stage-one description payload."""

    classes: MutableMapping[str, List[str]] = {}
    for classname in classnames:
        provided = descriptions.get(classname) if descriptions else None
        if provided is None:
            provided = default_visual_descriptions(classname, descriptions_per_class)
        classes[classname] = [str(item).strip() for item in provided if str(item).strip()]

    payload: Dict[str, Any] = {
        "dataset": canonical_dataset_name(dataset),
        "classes": dict(classes),
        "meta": {
            "generator": generator,
            "prompt_version": prompt_version,
            "description_contract": "visible fine-grained local visual attributes only",
        },
    }
    validate_description_payload(payload, classnames, min_descriptions=1)
    return payload


def load_description_payload(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a stage-one description JSON file."""

    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Description file must contain a JSON object: {path}")
    return payload


def save_description_payload(payload: Mapping[str, Any], path: Union[str, Path]) -> None:
    """Write a description payload as pretty UTF-8 JSON."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def validate_description_payload(
    payload: Mapping[str, Any],
    classnames: Sequence[str],
    min_descriptions: int = 3,
) -> None:
    """Validate that descriptions cover all classes in exact order.

    Raises ``ValueError`` with actionable messages when the asset is incomplete.
    """

    classes = payload.get("classes")
    if not isinstance(classes, Mapping):
        raise ValueError("description payload must contain a 'classes' object")

    missing = [name for name in classnames if name not in classes]
    extra = [name for name in classes if name not in set(classnames)]
    if missing:
        raise ValueError(f"description payload is missing {len(missing)} classes: {missing[:5]}")
    if extra:
        raise ValueError(f"description payload has {len(extra)} unknown classes: {extra[:5]}")

    for classname in classnames:
        descriptions = classes[classname]
        if not isinstance(descriptions, Sequence) or isinstance(descriptions, (str, bytes)):
            raise ValueError(f"descriptions for class '{classname}' must be a list of strings")
        valid = [d for d in descriptions if isinstance(d, str) and d.strip()]
        if len(valid) < min_descriptions:
            raise ValueError(
                f"class '{classname}' has {len(valid)} descriptions, "
                f"expected at least {min_descriptions}"
            )


def normalize_chat_completions_url(base_url: str) -> str:
    """Return a Chat Completions URL for OpenAI-compatible providers."""

    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def extract_json_array(text: str) -> List[str]:
    """Parse an LLM response into a list of strings.

    Accepts a raw JSON array, a fenced JSON array, or a response containing a
    JSON array with brief surrounding prose.
    """

    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("[")
        end = stripped.rfind("]")
        if start < 0 or end <= start:
            raise ValueError(f"LLM response does not contain a JSON array: {text[:200]!r}")
        parsed = json.loads(stripped[start : end + 1])

    if not isinstance(parsed, list):
        raise ValueError("LLM response JSON must be an array of strings")
    descriptions = [str(item).strip() for item in parsed if str(item).strip()]
    if not descriptions:
        raise ValueError("LLM response JSON array is empty")
    return descriptions


def build_llm_user_prompt(classname: str, count: int) -> str:
    """Prompt for one class worth of fine-grained visual descriptions."""

    return (
        f"Generate exactly {count} distinguishing visual descriptions for the "
        f"image class: {classname!r}.\n"
        "Focus heavily on specific localized parts and visible fine-grained cues. "
        "Each description should be one sentence, useful for CLIP text encoding, "
        "and should help distinguish this class from visually similar categories.\n"
        "Return only a valid JSON array of strings. Do not wrap it in markdown."
    )


def load_text_anchor_file(path: Union[str, Path], map_location: str = "cpu") -> Dict[str, Any]:
    """Load a ``*.pt`` anchor asset produced by stage one.

    Torch is imported lazily so non-encoding tooling can still import this module
    in minimal environments.
    """

    import torch

    payload = torch.load(Path(path), map_location=map_location)
    validate_text_anchor_payload(payload)
    return payload


def validate_text_anchor_payload(payload: Mapping[str, Any]) -> None:
    """Validate metadata and tensor shape of an encoded anchor payload."""

    required = {"dataset", "arch", "classnames", "anchors", "description_count", "normalization"}
    missing = required.difference(payload.keys())
    if missing:
        raise ValueError(f"anchor payload missing keys: {sorted(missing)}")

    classnames = payload["classnames"]
    anchors = payload["anchors"]
    if len(classnames) != int(anchors.shape[0]):
        raise ValueError(
            f"anchor rows ({anchors.shape[0]}) do not match classnames ({len(classnames)})"
        )
    if anchors.ndim != 2:
        raise ValueError(f"anchors must be a rank-2 tensor [K, D], got shape {tuple(anchors.shape)}")
    if payload["normalization"] != "l2":
        raise ValueError("anchor payload normalization must be 'l2'")
