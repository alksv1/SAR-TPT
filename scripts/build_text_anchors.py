#!/usr/bin/env python
"""Build SAR-TPT stage-one strong text anchors.

This script is intentionally offline-first:
1. resolve dataset class names in the same order as evaluation;
2. create or reuse a human-editable description JSON cache;
3. encode descriptions with a frozen CLIP text encoder;
4. average and L2-normalize per-class features;
5. save a reusable ``*.pt`` asset for stages two and four.

By default no online LLM API is called. Pass ``--llm-generate`` to call an
OpenAI-compatible Chat Completions endpoint and cache generated descriptions in
``--description-path`` before CLIP encoding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from clip import load as load_clip
from clip import tokenize
from utils.text_anchors import (
    build_description_payload,
    build_llm_user_prompt,
    canonical_dataset_name,
    extract_json_array,
    get_dataset_classnames,
    load_description_payload,
    normalize_chat_completions_url,
    save_description_payload,
    validate_description_payload,
)

DEFAULT_DESCRIPTION_ROOT = Path("assets/anchors/descriptions")
DEFAULT_FEATURE_ROOT = Path("assets/anchors/features")
DEFAULT_PROMPT_VERSION = "sar-tpt-stage1-v1"
DEFAULT_LLM_SYSTEM_PROMPT = (
    "You write concise visual descriptions for fine-grained image classification. "
    "Only mention visible attributes in the image: localized parts, shape, color, "
    "texture, markings, proportions, and other visual cues. Do not mention history, "
    "geography, usage, manufacturer trivia, dataset labels, or non-visual facts."
)


def safe_arch_name(arch: str) -> str:
    """Make a CLIP architecture name safe for filenames."""

    return arch.replace("/", "-").replace("@", "_").replace(" ", "")


def default_description_path(dataset: str) -> Path:
    return DEFAULT_DESCRIPTION_ROOT / f"{canonical_dataset_name(dataset)}.json"


def default_anchor_path(dataset: str, arch: str) -> Path:
    return DEFAULT_FEATURE_ROOT / f"{canonical_dataset_name(dataset)}_{safe_arch_name(arch)}.pt"


def description_digest(payload: Mapping[str, Any]) -> str:
    """Stable digest for asset provenance."""

    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iter_batches(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def encode_texts(
    clip_model: torch.nn.Module,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Encode text descriptions with frozen CLIP and return L2-normalized features."""

    features: List[torch.Tensor] = []
    clip_model.eval()
    with torch.no_grad():
        for batch in iter_batches(list(texts), batch_size):
            tokens = tokenize(list(batch), truncate=True).to(device)
            text_features = clip_model.encode_text(tokens)
            text_features = text_features.float()
            text_features = F.normalize(text_features, dim=-1)
            features.append(text_features.cpu())
    return torch.cat(features, dim=0)




def call_openai_compatible_chat(
    base_url: str,
    api_key: str,
    model: str,
    messages: Sequence[Mapping[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> str:
    """Call an OpenAI-compatible Chat Completions endpoint using stdlib only."""

    url = normalize_chat_completions_url(base_url)
    body = json.dumps(
        {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected LLM response schema: {payload}") from exc


def generate_descriptions_for_class(
    classname: str,
    count: int,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int,
    retry_sleep: float,
    system_prompt: str,
) -> List[str]:
    """Generate descriptions for one class with simple retry handling."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_llm_user_prompt(classname, count)},
    ]
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            content = call_openai_compatible_chat(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            descriptions = extract_json_array(content)
            if len(descriptions) < count:
                raise ValueError(
                    f"LLM returned {len(descriptions)} descriptions for {classname}, expected {count}"
                )
            return descriptions[:count]
        except Exception as exc:  # noqa: BLE001 - CLI should report the final concrete error.
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep)
    raise RuntimeError(f"Failed to generate descriptions for {classname!r}: {last_error}")


def prepare_llm_description_file(
    dataset: str,
    classnames: Sequence[str],
    path: Path,
    descriptions_per_class: int,
    force: bool,
    min_descriptions: int,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int,
    retry_sleep: float,
    system_prompt: str,
    save_every: int,
) -> Dict[str, Any]:
    """Create/update descriptions by calling an OpenAI-compatible endpoint.

    Existing valid class descriptions are reused unless ``force`` is true. The
    cache is written periodically so long jobs can be resumed safely.
    """

    if path.exists() and not force:
        payload = load_description_payload(path)
        classes = dict(payload.get("classes", {}))
    else:
        payload = {
            "dataset": canonical_dataset_name(dataset),
            "classes": {},
            "meta": {},
        }
        classes = {}

    payload["dataset"] = canonical_dataset_name(dataset)
    payload["classes"] = classes
    payload["meta"] = {
        **dict(payload.get("meta", {})),
        "generator": "openai-compatible-chat-completions",
        "llm_model": model,
        "llm_base_url": base_url,
        "prompt_version": DEFAULT_PROMPT_VERSION,
        "description_contract": "visible fine-grained local visual attributes only",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    generated = 0
    for index, classname in enumerate(classnames, start=1):
        existing = classes.get(classname)
        if (
            not force
            and isinstance(existing, list)
            and len([d for d in existing if isinstance(d, str) and d.strip()]) >= min_descriptions
        ):
            continue

        descriptions = generate_descriptions_for_class(
            classname=classname,
            count=descriptions_per_class,
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=retries,
            retry_sleep=retry_sleep,
            system_prompt=system_prompt,
        )
        classes[classname] = descriptions
        generated += 1
        print(f"[stage1][llm] {index}/{len(classnames)} generated: {classname}")
        if save_every > 0 and generated % save_every == 0:
            save_description_payload(payload, path)

    save_description_payload(payload, path)
    validate_description_payload(payload, classnames, min_descriptions=min_descriptions)
    return payload


def build_anchor_features(
    description_payload: Mapping[str, Any],
    classnames: Sequence[str],
    arch: str,
    device: str,
    batch_size: int,
    download_root: Optional[str],
) -> Dict[str, Any]:
    """Encode descriptions and aggregate per-class strong text anchors."""

    resolved_device = torch.device(device)
    clip_model, _, _ = load_clip(arch, device=resolved_device, download_root=download_root)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad_(False)

    classes = description_payload["classes"]
    anchors: List[torch.Tensor] = []
    description_count: Dict[str, int] = {}

    for classname in classnames:
        descriptions = [d.strip() for d in classes[classname] if isinstance(d, str) and d.strip()]
        encoded = encode_texts(clip_model, descriptions, resolved_device, batch_size=batch_size)
        anchor = F.normalize(encoded.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
        anchors.append(anchor)
        description_count[classname] = len(descriptions)

    anchor_tensor = torch.stack(anchors, dim=0).contiguous()
    created_at = datetime.now(timezone.utc).isoformat()
    return {
        "dataset": canonical_dataset_name(str(description_payload.get("dataset", ""))),
        "arch": arch,
        "classnames": list(classnames),
        "anchors": anchor_tensor,
        "description_count": description_count,
        "normalization": "l2",
        "prompt_version": description_payload.get("meta", {}).get("prompt_version", DEFAULT_PROMPT_VERSION),
        "description_sha256": description_digest(description_payload),
        "created_at": created_at,
        "meta": {
            "source_description_generator": description_payload.get("meta", {}).get("generator"),
            "text_encoder": "CLIP.encode_text",
            "aggregation": "mean_then_l2_normalize",
            "device_used_for_encoding": str(resolved_device),
        },
    }


def prepare_description_file(
    dataset: str,
    classnames: Sequence[str],
    path: Path,
    descriptions_per_class: int,
    force: bool,
    min_descriptions: int,
) -> Dict[str, Any]:
    """Create or load the human-editable description cache."""

    if path.exists() and not force:
        payload = load_description_payload(path)
        validate_description_payload(payload, classnames, min_descriptions=min_descriptions)
        return payload

    payload = build_description_payload(
        dataset=dataset,
        classnames=classnames,
        descriptions_per_class=descriptions_per_class,
        generator="template-fallback-offline",
        prompt_version=DEFAULT_PROMPT_VERSION,
    )
    payload.setdefault("meta", {})["created_at"] = datetime.now(timezone.utc).isoformat()
    payload["meta"]["note"] = (
        "Offline template fallback. Replace class descriptions with curated LLM "
        "visual attributes when available, then rerun anchor encoding."
    )
    save_description_payload(payload, path)
    validate_description_payload(payload, classnames, min_descriptions=min_descriptions)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SAR-TPT strong text anchors")
    parser.add_argument("--dataset", required=True, help="Dataset id, e.g. Pets, Cars, Aircraft, I, A, R, V, K")
    parser.add_argument("--arch", default="ViT-B/16", help="CLIP architecture used for text encoding")
    parser.add_argument("--description-path", type=Path, default=None, help="Path to description JSON cache")
    parser.add_argument("--output", type=Path, default=None, help="Path to output .pt anchor file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Encoding device")
    parser.add_argument("--batch-size", type=int, default=128, help="Text encoding batch size")
    parser.add_argument("--descriptions-per-class", type=int, default=3, help="Fallback descriptions per class")
    parser.add_argument("--min-descriptions", type=int, default=3, help="Minimum required descriptions per class")
    parser.add_argument("--force-description", action="store_true", help="Overwrite existing description JSON")
    parser.add_argument("--force-output", action="store_true", help="Overwrite existing .pt anchor output")
    parser.add_argument("--skip-encode", action="store_true", help="Only create/validate description JSON; do not load CLIP")
    parser.add_argument("--download-root", default=None, help="Optional CLIP download/cache root")

    # Optional online description generation through OpenAI-compatible APIs.
    parser.add_argument("--llm-generate", action="store_true", help="Generate description JSON with an OpenAI-compatible Chat Completions API")
    parser.add_argument("--llm-base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), help="OpenAI-compatible base URL or full /chat/completions URL")
    parser.add_argument("--llm-api-key", default=None, help="API key value. Prefer --llm-api-key-env for shell history safety")
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY", help="Environment variable containing the API key")
    parser.add_argument("--llm-model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), help="Chat model name for description generation")
    parser.add_argument("--llm-temperature", type=float, default=0.2, help="LLM sampling temperature")
    parser.add_argument("--llm-max-tokens", type=int, default=512, help="Max tokens per class generation call")
    parser.add_argument("--llm-timeout", type=float, default=60.0, help="HTTP timeout seconds per LLM call")
    parser.add_argument("--llm-retries", type=int, default=2, help="Retries per class on request/parse failure")
    parser.add_argument("--llm-retry-sleep", type=float, default=2.0, help="Sleep seconds between LLM retries")
    parser.add_argument("--llm-save-every", type=int, default=1, help="Save description cache after this many generated classes")
    parser.add_argument("--llm-system-prompt", default=DEFAULT_LLM_SYSTEM_PROMPT, help="System prompt for LLM description generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = canonical_dataset_name(args.dataset)
    classnames = get_dataset_classnames(dataset)
    description_path = args.description_path or default_description_path(dataset)
    output_path = args.output or default_anchor_path(dataset, args.arch)

    if args.llm_generate:
        api_key = args.llm_api_key or os.getenv(args.llm_api_key_env)
        if not api_key:
            raise ValueError(
                f"--llm-generate requires an API key via --llm-api-key or "
                f"environment variable {args.llm_api_key_env!r}"
            )
        payload = prepare_llm_description_file(
            dataset=dataset,
            classnames=classnames,
            path=description_path,
            descriptions_per_class=args.descriptions_per_class,
            force=args.force_description,
            min_descriptions=args.min_descriptions,
            base_url=args.llm_base_url,
            api_key=api_key,
            model=args.llm_model,
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens,
            timeout=args.llm_timeout,
            retries=args.llm_retries,
            retry_sleep=args.llm_retry_sleep,
            system_prompt=args.llm_system_prompt,
            save_every=args.llm_save_every,
        )
    else:
        payload = prepare_description_file(
            dataset=dataset,
            classnames=classnames,
            path=description_path,
            descriptions_per_class=args.descriptions_per_class,
            force=args.force_description,
            min_descriptions=args.min_descriptions,
        )

    print(f"[stage1] dataset={dataset} classes={len(classnames)}")
    print(f"[stage1] description JSON: {description_path}")

    if args.skip_encode:
        print("[stage1] --skip-encode set; description cache prepared and validated only.")
        return

    if output_path.exists() and not args.force_output:
        raise FileExistsError(
            f"Output anchor file already exists: {output_path}. "
            "Use --force-output to overwrite."
        )

    anchor_payload = build_anchor_features(
        description_payload=payload,
        classnames=classnames,
        arch=args.arch,
        device=args.device,
        batch_size=args.batch_size,
        download_root=args.download_root,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(anchor_payload, output_path)
    print(f"[stage1] saved anchors: {output_path}")
    print(f"[stage1] anchor shape: {tuple(anchor_payload['anchors'].shape)}")


if __name__ == "__main__":
    main()
