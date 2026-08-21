"""Shared assertions for extraction quality."""

from typing import Any, Dict, List


def flatten_extracted(extracted: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in extracted.items():
        if key in {"raw_text", "processed_text", "extraction_errors"}:
            continue
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                if inner_key != "confidence_scores":
                    flat[inner_key] = inner_value
                    flat[f"{key}_{inner_key}"] = inner_value
        else:
            flat[key] = value
    return flat


def _normalize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(_normalize(v) for v in value)
    return str(value).lower().replace(" ", "").replace("-", "").replace(".", "")


def field_hit(extracted: Dict[str, Any], candidates: List[str]) -> bool:
    blob = " | ".join(_normalize(v) for v in flatten_extracted(extracted).values())
    return any(_normalize(candidate) in blob for candidate in candidates)


def key_field_score(extracted: Dict[str, Any], expected: Dict[str, List[str]]) -> float:
    if not expected:
        return 0.0
    hits = sum(1 for candidates in expected.values() if field_hit(extracted, candidates))
    return hits / len(expected)
