"""Fill PA PDFs from extracted referral data using widget tooltips or text overlay."""

from __future__ import annotations

import os
import json
import sys
from typing import Any, Dict, List, Optional

from loguru import logger
from PyPDF2 import PdfReader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from field_mapper import flatten_extracted, map_widget_value
from utils import ensure_directory_exists, normalize_field_value, setup_logging, validate_extracted_data

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    fitz = None


AMY_OVERLAYS = {
    0: [
        ("last_name", 95, 254, "patient"),
        ("first_name", 370, 254, "patient"),
        ("mrn", 150, 312, "patient"),
        ("dob", 380, 312, "patient"),
        ("weight_kg", 160, 366, "patient"),
        ("last_name", 95, 411, "prescriber"),
        ("first_name", 370, 411, "prescriber"),
        ("npi", 120, 469, "prescriber"),
        ("phone", 120, 527, "patient"),
        ("medication", 100, 608, "clinical"),
        ("dose", 120, 624, "clinical"),
    ],
    1: [
        ("last_name", 155, 88, "patient"),
        ("first_name", 420, 88, "patient"),
    ],
    2: [
        ("last_name", 155, 85, "patient"),
        ("first_name", 420, 85, "patient"),
    ],
}


def _tooltip_map(pdf_path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    try:
        reader = PdfReader(pdf_path)
        fields = reader.get_fields() or {}
        for name, info in fields.items():
            if isinstance(info, dict):
                mapping[str(name)] = str(info.get("/TU") or name)
            else:
                mapping[str(name)] = str(name)
    except Exception as exc:
        logger.warning(f"Could not read field tooltips: {exc}")
    return mapping


def _lookup_overlay_value(extracted: Dict[str, Any], key: str, source: str) -> str:
    section = {
        "patient": extracted.get("patient_info", {}),
        "prescriber": extracted.get("prescriber_info", {}),
        "clinical": extracted.get("clinical_info", {}),
        "insurance": extracted.get("insurance_info", {}),
    }.get(source, {})
    value = section.get(key) or flatten_extracted(extracted).get(key)
    if value is None:
        return ""
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value)
    return str(value)


def _fill_widgets(pdf_path: str, extracted: Dict[str, Any], output_path: str) -> Dict[str, Any]:
    if fitz is None:
        raise RuntimeError("PyMuPDF is required for widget form filling")

    tooltips = _tooltip_map(pdf_path)
    doc = fitz.open(pdf_path)
    filled_details: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    applicable = 0

    for page in doc:
        widgets = page.widgets() or []
        for widget in widgets:
            name = widget.field_name or ""
            tooltip = tooltips.get(name) or getattr(widget, "field_label", None) or name
            ftype = str(getattr(widget, "field_type_string", "") or widget.field_type)
            value, confidence = map_widget_value(tooltip, ftype, extracted)
            if not value:
                continue

            applicable += 1
            try:
                normalized = normalize_field_value(tooltip, value) or str(value)
                if "btn" in ftype.lower() or "check" in ftype.lower() or widget.field_type == fitz.PDF_WIDGET_TYPE_CHECKBOX:
                    on_state = widget.on_state() if hasattr(widget, "on_state") else "Yes"
                    widget.field_value = on_state or True
                else:
                    widget.field_value = normalized
                widget.update()
                filled_details.append({
                    "field_name": name,
                    "tooltip": tooltip,
                    "value": normalized,
                    "confidence": confidence,
                    "page": page.number,
                })
            except Exception as exc:
                missing.append({
                    "field_name": name,
                    "tooltip": tooltip,
                    "reason": f"Fill error: {exc}",
                    "page": page.number,
                })

    ensure_directory_exists(os.path.dirname(output_path) or ".")
    doc.save(output_path, incremental=False, encryption=fitz.PDF_ENCRYPT_KEEP)
    doc.close()

    filled_count = len(filled_details)
    total = max(applicable, filled_count)
    fill_percentage = (filled_count / total * 100) if total else 0.0
    return {
        "status": "completed" if fill_percentage >= 90 else "completed_with_errors",
        "form_type": "widget_based",
        "filled_count": filled_count,
        "filled_fields": filled_count,
        "total_fields": total,
        "fill_rate": fill_percentage / 100,
        "fill_percentage": fill_percentage,
        "filled_field_details": filled_details,
        "missing_fields": missing,
        "processing_errors": [],
    }


def _fill_static_overlay(pdf_path: str, extracted: Dict[str, Any], output_path: str) -> Dict[str, Any]:
    if fitz is None:
        raise RuntimeError("PyMuPDF is required for static form overlay")

    doc = fitz.open(pdf_path)
    filled_details: List[Dict[str, Any]] = []
    applicable = 0

    for page_index, placements in AMY_OVERLAYS.items():
        if page_index >= doc.page_count:
            continue
        page = doc[page_index]
        for key, x, y, source in placements:
            value = _lookup_overlay_value(extracted, key, source)
            if not value:
                continue
            applicable += 1
            page.insert_text((x, y), str(value), fontsize=10, fontname="helv")
            filled_details.append({
                "field_name": f"{source}_{key}",
                "tooltip": key,
                "value": value,
                "confidence": 0.85,
                "page": page_index,
            })

    ensure_directory_exists(os.path.dirname(output_path) or ".")
    doc.save(output_path)
    doc.close()

    filled_count = len(filled_details)
    total = max(applicable, filled_count)
    fill_percentage = (filled_count / total * 100) if total else 0.0
    return {
        "status": "completed" if fill_percentage >= 90 else "completed_with_errors",
        "form_type": "static_overlay",
        "filled_count": filled_count,
        "filled_fields": filled_count,
        "total_fields": total,
        "fill_rate": fill_percentage / 100,
        "fill_percentage": fill_percentage,
        "filled_field_details": filled_details,
        "missing_fields": [],
        "processing_errors": [],
    }


def _has_widgets(pdf_path: str) -> bool:
    try:
        reader = PdfReader(pdf_path)
        fields = reader.get_fields()
        return bool(fields)
    except Exception:
        return False


class EnhancedPAFormFiller:
    """Fill widget-based or static PA forms from structured extraction results."""

    def fill_pa_form(self, pa_form_path: str, extracted_data: Dict[str, Any], output_path: str) -> Dict[str, Any]:
        if not os.path.exists(pa_form_path):
            return self._error("PA form file not found")

        validate_extracted_data(extracted_data)

        try:
            if _has_widgets(pa_form_path):
                result = _fill_widgets(pa_form_path, extracted_data, output_path)
            else:
                result = _fill_static_overlay(pa_form_path, extracted_data, output_path)
            result["data_validation"] = validate_extracted_data(extracted_data)
            return result
        except Exception as exc:
            logger.error(f"Form filling failed: {exc}")
            return self._error(str(exc))

    def _error(self, message: str) -> Dict[str, Any]:
        return {
            "status": "error",
            "error_message": message,
            "form_type": "unknown",
            "filled_count": 0,
            "filled_fields": 0,
            "total_fields": 0,
            "fill_rate": 0.0,
            "fill_percentage": 0.0,
            "filled_field_details": [],
            "missing_fields": [],
            "processing_errors": [message],
        }


PAFormFiller = EnhancedPAFormFiller


def fill_pa_form(pa_pdf_path: str, extracted_data: Dict[str, Any], output_path: str) -> Dict[str, Any]:
    setup_logging()
    filler = PAFormFiller()
    return filler.fill_pa_form(pa_pdf_path, extracted_data, output_path)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python fill_pa_form.py <pa_pdf_path> <extracted_data_json> <output_path>")
        sys.exit(1)

    with open(sys.argv[2], "r", encoding="utf-8") as handle:
        data = json.load(handle)
    result = fill_pa_form(sys.argv[1], data, sys.argv[3])
    print(
        f"Form filling completed: {result.get('filled_count', 0)}/"
        f"{result.get('total_fields', 0)} fields filled ({result.get('fill_percentage', 0):.1f}%)"
    )
