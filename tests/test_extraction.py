"""Extraction tests against real concatenated OCR from the sample packets."""

from document_parser import parse_referral_text

from helpers import key_field_score

ADBULLA_EXPECTED = {
    "last_name": ["Abdulla"],
    "first_name": ["Shakh"],
    "dob": ["04/01/2001", "4/1/2001"],
    "mrn": ["041152153"],
    "diagnosis": ["multiple sclerosis", "sclerosis"],
    "icd_code": ["G35"],
    "medication": ["rituximab", "truxima"],
    "phone": ["865-839-7458", "(865) 839-7458", "8658397458"],
    "insurance": ["TENNCARE", "BLUECARE"],
    "npi": ["1154611523"],
    "member_id": ["LAJIM14345116"],
}

AMY_EXPECTED = {
    "last_name": ["Chen"],
    "first_name": ["Amy"],
    "dob": ["05/23/1983", "5/23/1983"],
    "mrn": ["01051001"],
    "diagnosis": ["migraine"],
    "icd_code": ["G43.711"],
    "medication": ["Vyepti", "eptinezumab", "Botox"],
    "phone": ["615-593-1048", "(615) 593-1048", "6155931048"],
    "weight": ["50.9"],
}

AKSHAY_EXPECTED = {
    "last_name": ["Chaudhari", "Chauchari"],
    "first_name": ["Akshay"],
    "dob": ["02/17/1981", "2/17/1981"],
    "diagnosis": ["Crohn"],
    "medication": ["Skyrizi"],
    "insurance": ["Aetna"],
}


def test_adbulla_key_fields_from_noisy_ocr(adbulla_ocr):
    extracted = parse_referral_text(adbulla_ocr)
    score = key_field_score(extracted, ADBULLA_EXPECTED)
    assert score >= 0.90, f"Adbulla key-field score {score:.0%} < 90%: {extracted}"


def test_amy_key_fields_from_noisy_ocr(amy_ocr):
    extracted = parse_referral_text(amy_ocr)
    score = key_field_score(extracted, AMY_EXPECTED)
    assert score >= 0.90, f"Amy key-field score {score:.0%} < 90%: {extracted}"


def test_does_not_treat_confidentiality_notice_as_patient_name(amy_ocr, adbulla_ocr):
    for text in (amy_ocr, adbulla_ocr):
        extracted = parse_referral_text(text)
        name = str(extracted.get("patient_info", {}).get("patient_name", "")).lower()
        assert "recipient" not in name
        assert "intended" not in name
        assert "thanks" not in name


def test_does_not_crash_on_optional_regex_groups(adbulla_ocr):
    extracted = parse_referral_text(adbulla_ocr)
    assert extracted.get("extraction_errors") in (None, [])


def test_icd_codes_are_not_fax_header_garbage(adbulla_ocr):
    extracted = parse_referral_text(adbulla_ocr)
    codes = extracted.get("clinical_info", {}).get("icd_codes", [])
    if isinstance(codes, str):
        codes = [codes]
    codes = [str(c).upper() for c in codes]
    assert "G35" in codes
    assert "F615" not in codes
    assert "A35000" not in codes


def test_akshay_key_fields_from_noisy_ocr(akshay_ocr):
    extracted = parse_referral_text(akshay_ocr)
    score = key_field_score(extracted, AKSHAY_EXPECTED)
    assert score >= 0.90, f"Akshay key-field score {score:.0%} < 90%: {extracted}"
    name = extracted["patient_info"]["patient_name"].lower()
    assert "akshay" in name
    assert "lafsky" not in name


def test_rejects_garbage_addresses(amy_ocr):
    extracted = parse_referral_text(amy_ocr)
    address = str(extracted.get("patient_info", {}).get("address", "")).lower()
    assert "drawn" not in address
    assert "units" not in address


def test_adbulla_name_is_split_for_form_fields(adbulla_ocr):
    extracted = parse_referral_text(adbulla_ocr)
    patient = extracted["patient_info"]
    assert patient["last_name"].lower() == "abdulla"
    assert patient["first_name"].lower() == "shakh"

