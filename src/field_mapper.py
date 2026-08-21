"""Map PDF widget names/tooltips onto extracted referral fields."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple


CHECKED_VALUES = {"Yes", "/Yes", True, "On"}

TOOLTIP_ALIASES = {
    "patient first name": "first_name",
    "patient last name": "last_name",
    "patient dob (mm/dd/yyyy)": "dob",
    "patient phone": "phone",
    "member id #": "member_id",
    "member id": "member_id",
    "group #": "group_number",
    "primary icd code": "icd_code",
    "carrier name": "insurance_carrier",
    "insured": "patient_name",
    "allergies": "allergies",
    "address": "address",
    "city": "city",
    "state": "state",
    "zip": "zip",
    "npi #": "npi",
    "npi": "npi",
    "dose": "dose",
    "medication": "medication",
    "first name": "prescriber_first_or_patient_first",
    "last name": "prescriber_last_or_patient_last",
    "name": "patient_name",
    "phone": "phone",
    "work phone": "prescriber_phone",
    "cell phone": "phone",
    "kgs": "weight_kg",
}


def flatten_extracted(extracted: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for category, fields in (extracted or {}).items():
        if not isinstance(fields, dict):
            continue
        if category in {"metadata"}:
            continue
        for key, value in fields.items():
            if key == "confidence_scores" or value in (None, "", [], {}):
                continue
            prefixed = f"{category}_{key}"
            flat[prefixed] = value
            if key not in flat:
                flat[key] = value
    return flat


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _lookup(flat: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = flat.get(key)
        if value not in (None, "", [], {}):
            if isinstance(value, list):
                return ", ".join(str(item) for item in value)
            return str(value)
    return ""


def _checkbox_should_check(tooltip: str, flat: Dict[str, Any]) -> bool:
    haystack = " ".join(
        str(flat.get(key, ""))
        for key in ("diagnosis", "primary_diagnosis", "medication", "medication_key", "icd_code")
    ).lower()
    tu = tooltip.lower()
    if not haystack:
        return False

    if "ritux" in tu and "ritux" in haystack:
        return True
    if "truxima" in tu and "truxima" in haystack:
        return True
    if "vyepti" in tu and "vyepti" in haystack:
        return True

    diagnosis = str(flat.get("diagnosis", "")).lower()
    if "multiple sclerosis" in diagnosis or "sclerosis" in diagnosis:
        if any(token in tu for token in ("multiple sclerosis", "relapsing-remitting ms", "rrms", "pediatric")):
            return True

    if "migraine" in diagnosis and "migraine" in tu:
        return True

    if "m.d." in tu and "md" in str(flat.get("credentials", "")).lower():
        return True
    if tu in {"n.p.", "np"} and "np" in str(flat.get("credentials", "")).lower():
        return True
    if "start of treatment" in tu:
        return True
    if tu == "home" and "home infusion" in haystack:
        return True
    return False


def map_widget_value(
    tooltip: str,
    field_type: str,
    extracted: Dict[str, Any],
) -> Tuple[Optional[str], float]:
    """Return (value, confidence) for a PDF widget given its tooltip/label."""
    flat = flatten_extracted(extracted)
    raw_tooltip = (tooltip or "").strip()
    tu = raw_tooltip.lower().rstrip(":")
    ftype = str(field_type or "").replace("/", "")

    if re.fullmatch(r"t\d+[a-z]?", raw_tooltip, flags=re.IGNORECASE) or re.fullmatch(r"cb\d+[a-z]?", raw_tooltip, flags=re.IGNORECASE):
        return "", 0.0

    if ftype.lower() in {"btn", "button", "checkbox"}:
        if _checkbox_should_check(tu, flat):
            return "Yes", 0.85
        return "", 0.0

    alias = TOOLTIP_ALIASES.get(tu)
    if alias == "prescriber_first_or_patient_first":
        value = _lookup(flat, "prescriber_info_first_name", "first_name")
        return (value, 0.8) if value else ("", 0.0)
    if alias == "prescriber_last_or_patient_last":
        value = _lookup(flat, "prescriber_info_last_name", "last_name")
        return (value, 0.8) if value else ("", 0.0)
    if alias:
        value = _lookup(flat, alias, f"patient_info_{alias}", f"clinical_info_{alias}", f"insurance_info_{alias}", f"prescriber_info_{alias}")
        if alias == "npi":
            value = _lookup(flat, "npi", "prescriber_npi")
        if alias == "insurance_carrier":
            value = _lookup(flat, "insurance_carrier")
        if alias == "member_id":
            value = _lookup(flat, "member_id")
        return (value, 0.9) if value else ("", 0.0)

    # Generic semantic matches for remaining labeled fields.
    semantic = [
        (("patient first", "first name"), ("first_name",)),
        (("patient last", "last name"), ("last_name",)),
        (("dob", "date of birth", "birth"), ("dob",)),
        (("phone",), ("phone",)),
        (("address",), ("address",)),
        (("city",), ("city",)),
        (("state",), ("state",)),
        (("zip",), ("zip",)),
        (("icd",), ("icd_code",)),
        (("carrier", "insurance"), ("insurance_carrier",)),
        (("member",), ("member_id", "mrn")),
        (("npi",), ("npi", "prescriber_npi")),
        (("allerg",), ("allergies",)),
        (("dose",), ("dose",)),
        (("medication", "drug"), ("medication",)),
        (("weight", "kgs"), ("weight_kg",)),
    ]
    for needles, keys in semantic:
        if any(needle in tu for needle in needles):
            value = _lookup(flat, *keys)
            if value:
                return value, 0.8
    return "", 0.0
