"""Tests for OCR spacing restoration and sanitization."""

from document_parser import restore_ocr_spacing


def test_splits_concatenated_last_first_name():
    restored = restore_ocr_spacing("Abdulla,Shakh(MR#041162163)DOB:04/01/2001")
    assert "Abdulla, Shakh" in restored
    assert "DOB" in restored
    assert "04/01/2001" in restored


def test_splits_camel_case_medical_terms():
    restored = restore_ocr_spacing("Multiplesclerosisinpediatricpatient(CMS/HCC)(G35)")
    lowered = restored.lower()
    assert "multiple" in lowered
    assert "sclerosis" in lowered
    assert "G35" in restored


def test_does_not_turn_zeros_into_letter_o():
    restored = restore_ocr_spacing("DOB:04/01/2001 MRN:041152153")
    assert "04/01/2001" in restored
    assert "041152153" in restored
    assert "O4/O1/2OO1" not in restored


def test_rejects_confidentiality_notice_as_readable_name_context():
    restored = restore_ocr_spacing(
        "namedaboveastheRecipient.Ifyouarenottheintendedrecipient"
    )
    lowered = restored.lower()
    assert "recipient" in lowered
    assert "intended" in lowered
