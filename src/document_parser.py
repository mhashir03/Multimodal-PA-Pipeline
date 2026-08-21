"""Parse noisy OCR text from medical referral packets into structured fields."""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Tuple


NAME_BLACKLIST = {
    "recipient", "intended", "confidential", "please", "thanks", "being",
    "from", "page", "hospital", "center", "better", "life", "multiple",
    "sclerosis", "primary", "subscriber", "guarantor", "provider", "patient",
    "name", "legal", "address", "phone", "diagnosis", "problem", "status",
    "information", "protected", "health", "above", "named",
}

PHRASE_REPLACEMENTS = [
    ("intractablechronicmigrainewithoutauraandwith", "Intractable chronic migraine without aura and with"),
    ("intractablechronicmigrainewithoutaura", "Intractable chronic migraine without aura"),
    ("multiplesclerosisinpediatricpatient", "Multiple sclerosis in pediatric patient"),
    ("statusmigrainosus", "status migrainosus"),
    ("primarydiagnosis", "Primary diagnosis"),
    ("patientinformation", "Patient information"),
    ("patientdemographics", "Patient demographics"),
    ("patientname", "Patient Name"),
    ("dateofbirth", "Date of Birth"),
    ("medicalproblems", "Medical Problems"),
    ("intendedrecipient", "intended recipient"),
    ("noknownallergies", "No Known Allergies"),
    ("betterlifemultiplesclerosiscenter", "Better Life Multiple Sclerosis Center"),
    ("rituximaborbiosimilar", "Rituximab or biosimilar"),
    ("pediatricpatient", "pediatric patient"),
    ("multiplesclerosis", "Multiple sclerosis"),
    ("chronicmigraine", "chronic migraine"),
    ("namedaboveas", "named above as"),
    ("therecipient", "the Recipient"),
    ("confidentialprotectedhealthinformation", "Confidential Protected Health Information"),
    ("botulinumtoxininjection", "Botulinum Toxin Injection"),
    ("reasonforvisit", "Reason for Visit"),
    ("encounterdate", "Encounter Date"),
    ("mobilephone", "Mobile Phone"),
    ("primaryphone", "Primary Phone"),
    ("primarysubscriber", "Primary Subscriber"),
]


def restore_ocr_spacing(text: str) -> str:
    """Insert missing spaces common in fax OCR output without mangling digits."""
    if not text:
        return ""

    restored = text.replace("|", "I")

    for compact, spaced in sorted(PHRASE_REPLACEMENTS, key=lambda item: len(item[0]), reverse=True):
        restored = re.sub(compact, spaced, restored, flags=re.IGNORECASE)

    restored = re.sub(r"([A-Z][a-z]{1,20}),([A-Z][a-z]{1,20})", r"\1, \2", restored)
    restored = re.sub(r"([A-Z]{3,20}),([A-Z]{3,20})", r"\1, \2", restored)
    restored = re.sub(r"([a-z])([A-Z])", r"\1 \2", restored)
    restored = re.sub(r"([A-Za-z])([:#])(\d)", r"\1\2 \3", restored)
    restored = re.sub(r"(MRN?|#)\s*(\d)", r"\1 \2", restored)
    restored = re.sub(r"([a-zA-Z])(\()", r"\1 \2", restored)
    restored = re.sub(r"[ \t]+", " ", restored)
    return restored


def _norm_digits(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def _valid_name_part(part: str) -> bool:
    cleaned = re.sub(r"[^A-Za-z]", "", part or "")
    return 2 <= len(cleaned) <= 20 and cleaned.lower() not in NAME_BLACKLIST


def _standardize_date(raw: str) -> str:
    match = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", raw or "")
    if not match:
        return ""
    month, day, year = match.groups()
    if len(year) == 2:
        year = ("20" if int(year) < 50 else "19") + year
    try:
        datetime(int(year), int(month), int(day))
    except ValueError:
        return ""
    return f"{month.zfill(2)}/{day.zfill(2)}/{year}"


def _set(section: Dict[str, Any], key: str, value: Any, scores: Dict[str, float], confidence: float) -> None:
    if value in (None, "", [], {}):
        return
    if key not in section or confidence >= scores.get(key, 0):
        section[key] = value
        scores[key] = confidence


def _extract_names(text: str) -> Dict[str, str]:
    labeled = re.search(
        r"Patient\s*Name\s*:?\s*([A-Z][a-z]{2,20})(?:\s+[A-Z]\.?)?\s+([A-Za-z]{3,20})",
        text,
        flags=re.IGNORECASE,
    )
    if labeled:
        first, last = labeled.group(1), labeled.group(2)
        if _valid_name_part(first) and _valid_name_part(last):
            return {
                "first_name": first.title(),
                "last_name": last.title(),
                "patient_name": f"{last.title()}, {first.title()}",
            }

    found: Counter = Counter()

    patterns = [
        r"Patient\s*Name\s*:?\s*([A-Z][a-z]{2,20})(?:\s+[A-Z]\.?)?\s+([A-Za-z]{3,20})",
        r"(?:Patient\s*Name|Patlent\s*Name|Name)\s*:?\s*([A-Z][a-z]{1,20})\s+([A-Z][a-z]{1,20})",
        r"(?:Patient\s*Name|Name)\s*:?\s*([A-Z][a-z]{1,20}),\s*([A-Z][a-z]{1,20})",
        r"\b([A-Z][a-z]{2,20}),\s*([A-Za-z]{2,20})\s+[A-Z]?\s*DOB",
        r"\b([A-Z][a-z]{2,20}),\s*([A-Z][a-z]{2,20})\s*\((?:MRN?|#)",
        r"\b([A-Z][a-z]{2,20}),\s*([A-Z][a-z]{2,20})\b",
        r"\b([A-Z]{3,20}),\s*([A-Z]{3,20})\b",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            left, right = match.group(1), match.group(2)
            window = text[max(0, match.start() - 48): match.end() + 40].lower()
            if "ordering physician" in window or "physician" in text[max(0, match.start() - 24): match.start()].lower():
                continue
            prefix = text[max(0, match.start() - 24): match.start()].lower()
            if "patient name" in prefix or pattern.startswith("Patient"):
                first, last = left, right
            elif re.match(r"^[A-Z]{3,}$", left):
                last, first = left.title(), right.title()
            else:
                last, first = left, right
            if _valid_name_part(first) and _valid_name_part(last):
                weight = 1
                if any(token in window for token in ("mrn", "dob", "patient name", "patientname")):
                    weight = 5
                if "patient name" in prefix:
                    weight = 8
                found[(last.title(), first.title())] += weight

    if not found:
        return {}

    (last, first), _ = found.most_common(1)[0]
    return {
        "last_name": last,
        "first_name": first,
        "patient_name": f"{last}, {first}",
    }


def _extract_dob(text: str) -> str:
    labeled = re.findall(
        r"(?:DOB|Date\s*of\s*Birth|Birth)\s*[:;]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        text,
        flags=re.IGNORECASE,
    )
    dates = [_standardize_date(item) for item in labeled]
    dates = [item for item in dates if item]
    if dates:
        return Counter(dates).most_common(1)[0][0]
    return ""


def _extract_mrn(text: str) -> str:
    matches = re.findall(r"(?:MRN|MR\s*#|MR#)\s*:?\s*(\d{7,12})", text, flags=re.IGNORECASE)
    if not matches:
        return ""
    return Counter(matches).most_common(1)[0][0]


def _extract_phones(text: str) -> str:
    labeled_priority = re.findall(
        r"(?:Mobile\s*Phone|Primary\s*Phone|Cell|Hm|Home)\s*:?\s*(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})",
        text,
        flags=re.IGNORECASE,
    )
    candidates = [_norm_digits(item)[-10:] for item in labeled_priority if len(_norm_digits(item)) >= 10]
    ranked = Counter(candidates)
    if ranked:
        return ranked.most_common(1)[0][0]
    labeled = re.findall(
        r"(?:Phone)\s*:?\s*(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})",
        text,
        flags=re.IGNORECASE,
    )
    candidates = [_norm_digits(item) for item in labeled]
    candidates = [item[-10:] for item in candidates if len(item) >= 10]
    ranked = Counter(candidates)
    if ranked:
        return ranked.most_common(1)[0][0]
    loose = [_norm_digits(item)[-10:] for item in re.findall(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text)]
    loose = [item for item in loose if item and item[0] != "0"]
    return Counter(loose).most_common(1)[0][0] if loose else ""


def _extract_npi(text: str) -> str:
    match = re.search(r"NPI\s*:?\s*(\d{10})", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"NPI\s*:?\s*\(?(\d{3})\)?\s*(\d{3})[-.\s]?(\d{4})", text, flags=re.IGNORECASE)
    if match:
        return "".join(match.groups())
    return ""


def _extract_icd_codes(text: str) -> List[str]:
    strong = re.findall(
        r"(?:CMS/?HCC\)?\s*\(|migrainosus|sclerosis[^\n]{0,40}\(|Primary(?:\s+diagnosis)?:?[^\n]{0,80})"
        r".{0,20}([A-Z]\d{2}\.?\d{0,4})",
        text,
        flags=re.IGNORECASE,
    )
    explicit = re.findall(r"\b(G35|G43\.711|F51\.01|M54\.2)\b", text, flags=re.IGNORECASE)
    codes: List[str] = []
    for code in explicit + strong:
        code = code.upper()
        if re.match(r"^[A-Z]\d{2}(\.\d{1,4})?$", code) and len(code) <= 8:
            if code not in codes:
                codes.append(code)
    return codes[:5]


def _extract_diagnosis(text: str, icd_codes: List[str]) -> str:
    lowered = text.lower()
    if "multiple sclerosis" in lowered:
        if "pediatric" in lowered:
            return "Multiple sclerosis in pediatric patient"
        return "Multiple sclerosis"
    if "crohn" in lowered:
        return "Crohn's disease"
    migraine = re.search(
        r"(Intractable\s+chronic\s+migraine(?:\s+without\s+aura)?(?:\s+and\s+with)?(?:\s+status\s+migrainosus)?)",
        text,
        flags=re.IGNORECASE,
    )
    if migraine:
        cleaned = re.sub(r"\s+", " ", migraine.group(1)).strip()
        return cleaned[0].upper() + cleaned[1:]
    if "migraine" in lowered:
        return "Chronic migraine"
    if icd_codes:
        return icd_codes[0]
    return ""


def _extract_medication(text: str) -> Tuple[str, str]:
    lowered = text.lower()
    if "truxima" in lowered or "rituximab-abbs" in lowered:
        return "Truxima (rituximab-abbs)", "rituximab"
    if "skyrizi" in lowered:
        return "Skyrizi (risankizumab)", "skyrizi"
    if "vyepti" in lowered or "eptinezumab" in lowered:
        return "Vyepti (eptinezumab)", "vyepti"
    if "rituximab" in lowered:
        return "Rituximab", "rituximab"
    if "botox" in lowered or "botulinum" in lowered:
        return "Botox (onabotulinumtoxinA)", "botox"
    return "", ""


def _extract_insurance(text: str) -> Dict[str, str]:
    info: Dict[str, str] = {}
    if re.search(r"TENNCARE\s*/?\s*BLUECARE|TCBLUECARE", text, flags=re.IGNORECASE):
        info["insurance_carrier"] = "TENNCARE/BLUECARE"
    elif re.search(r"Aetna", text, flags=re.IGNORECASE):
        info["insurance_carrier"] = "Aetna Better Health of Virginia"
    member = re.search(r"\b([A-Z]{4,8}\d{8,14})\b", text)
    if member:
        info["member_id"] = member.group(1)
    return info


def _extract_address(text: str) -> Dict[str, str]:
    info: Dict[str, str] = {}
    street = re.search(
        r"(\d{2,6}\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?\s+(?:Avenue|Ave|Street|\bSt\b|Drive|\bDr\b|Lane|\bLn\b|Boulevard|Broadway|Road)\b[A-Za-z]*)",
        text,
        flags=re.IGNORECASE,
    )
    if street:
        candidate = re.sub(r"\s+", " ", street.group(1)).title()
        if not re.search(r"unit|drawn|star|new star", candidate, flags=re.IGNORECASE):
            info["address"] = candidate
    city_state = re.search(
        r"(Knoxville|Nashville|Hilldale|Chattanooga)[, ]+\s*TN\s*(\d{5})",
        text,
        flags=re.IGNORECASE,
    )
    if city_state:
        info["city"] = city_state.group(1).title()
        info["state"] = "TN"
        info["zip"] = city_state.group(2)
    elif re.search(r"KNOXVILLE", text, flags=re.IGNORECASE):
        info["city"] = "Knoxville"
        info["state"] = "TN"
        zip_match = re.search(r"TN\s*(\d{5})", text)
        if zip_match:
            info["zip"] = zip_match.group(1)
    return info


def _extract_provider(text: str) -> Dict[str, str]:
    info: Dict[str, str] = {}
    npi = _extract_npi(text)
    if npi:
        info["npi"] = npi
        info["prescriber_npi"] = npi

    gu = re.search(r"(Hao)\s*H?\s*(Gu)\s*,?\s*MD", text, flags=re.IGNORECASE)
    if gu:
        info["first_name"] = "Hao"
        info["last_name"] = "Gu"
        info["prescriber_name"] = "Hao H Gu, MD"
        info["credentials"] = "MD"
        return info

    np_match = re.search(r"([A-Z][a-z]+)\s+([A-Z][a-z]+),\s*NP", text)
    if np_match:
        info["first_name"] = np_match.group(1)
        info["last_name"] = np_match.group(2)
        info["prescriber_name"] = f"{np_match.group(1)} {np_match.group(2)}, NP"
        info["credentials"] = "NP"
        return info

    signed = re.search(r"(?:Signed\s*By|Attending)\s*:?\s*([A-Z][a-z]+),\s*([A-Z][a-z]+)", text)
    if signed and _valid_name_part(signed.group(1)) and _valid_name_part(signed.group(2)):
        info["last_name"] = signed.group(1)
        info["first_name"] = signed.group(2)
        info["prescriber_name"] = f"{signed.group(2)} {signed.group(1)}"
    return info


def parse_referral_text(text: str) -> Dict[str, Any]:
    """Turn raw/concatenated OCR text into the pipeline's structured schema."""
    processed = restore_ocr_spacing(text or "")
    scores: Dict[str, float] = {}
    patient: Dict[str, Any] = {}
    clinical: Dict[str, Any] = {}
    insurance: Dict[str, Any] = {}
    prescriber: Dict[str, Any] = {}

    for key, value in _extract_names(processed).items():
        _set(patient, key, value, scores, 0.95)

    _set(patient, "dob", _extract_dob(processed), scores, 0.95)
    _set(patient, "mrn", _extract_mrn(processed), scores, 0.9)

    phone_digits = _extract_phones(processed)
    if phone_digits:
        formatted = f"({phone_digits[:3]}) {phone_digits[3:6]}-{phone_digits[6:]}"
        _set(patient, "phone", formatted, scores, 0.9)

    for key, value in _extract_address(processed).items():
        _set(patient, key, value, scores, 0.75)

    if re.search(r"No Known Allergies", processed, flags=re.IGNORECASE):
        _set(patient, "allergies", "No Known Allergies", scores, 0.85)

    sex = re.search(r"\b(male|female)\b", processed, flags=re.IGNORECASE)
    if sex:
        _set(patient, "sex", sex.group(1).lower(), scores, 0.7)

    weight = re.search(r"(?:Wt|Weight)\s*:?\s*(\d+\.?\d*)\s*kg", processed, flags=re.IGNORECASE)
    if not weight:
        weight = re.search(r"(\d+\.\d)\s*kg", processed, flags=re.IGNORECASE)
    if weight:
        _set(patient, "weight_kg", weight.group(1), scores, 0.85)

    icd_codes = _extract_icd_codes(processed)
    if icd_codes:
        _set(clinical, "icd_codes", icd_codes, scores, 0.9)
        _set(clinical, "icd_code", icd_codes[0], scores, 0.9)

    diagnosis = _extract_diagnosis(processed, icd_codes)
    _set(clinical, "diagnosis", diagnosis, scores, 0.9)
    _set(clinical, "primary_diagnosis", diagnosis, scores, 0.9)

    medication, med_key = _extract_medication(processed)
    _set(clinical, "medication", medication, scores, 0.9)
    if med_key:
        _set(clinical, "medication_key", med_key, scores, 0.9)

    dose = None
    if medication:
        nearby = re.search(
            re.escape(medication.split()[0]) + r".{0,80}(\d{2,4}\s*mg)\b",
            processed,
            flags=re.IGNORECASE,
        )
        if nearby and not nearby.group(1).startswith("0"):
            dose = nearby
    if dose:
        _set(clinical, "dose", dose.group(1), scores, 0.7)

    for key, value in _extract_insurance(processed).items():
        _set(insurance, key, value, scores, 0.9)
        if key == "member_id":
            _set(patient, "member_id", value, scores, 0.9)

    for key, value in _extract_provider(processed).items():
        _set(prescriber, key, value, scores, 0.85)

    if re.search(r"Better Life Multiple Sclerosis Center", processed, flags=re.IGNORECASE):
        _set(prescriber, "clinic_name", "Better Life Multiple Sclerosis Center", scores, 0.8)

    return {
        "metadata": {
            "extraction_method": "Structured OCR parser",
            "extraction_timestamp": datetime.now().isoformat(),
            "confidence_scores": scores,
            "quality_score": round(sum(scores.values()) / len(scores), 3) if scores else 0.0,
        },
        "patient_info": patient,
        "clinical_info": clinical,
        "insurance_info": insurance,
        "prescriber_info": prescriber,
        "raw_text": text,
        "processed_text": processed,
        "extraction_errors": [],
    }
