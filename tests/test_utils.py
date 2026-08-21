"""Utility and input-file tests."""

from utils import find_patient_files, normalize_field_value, validate_input_files


def test_finds_lowercase_pa_pdf(akshay_input):
    pa_path, referral_path = find_patient_files(str(akshay_input))
    assert pa_path.endswith("pa.pdf")
    assert referral_path.endswith("referral_package.pdf")


def test_validate_input_files_accepts_akshay(akshay_input):
    is_valid, missing = validate_input_files(str(akshay_input))
    assert is_valid
    assert missing == []


def test_normalize_handles_none_and_lists():
    assert normalize_field_value("dob", None) == ""
    assert normalize_field_value("icd_code", ["G35", "G43.711"]) == "G35, G43.711"
    assert normalize_field_value("phone", "8658397458") == "(865) 839-7458"
    assert "Abdulla" in normalize_field_value("patient_name", "Abdulla, Shakh")
