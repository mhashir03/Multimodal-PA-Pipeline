"""Form filling tests against the real sample PA PDFs."""

from pathlib import Path

from PyPDF2 import PdfReader

from document_parser import parse_referral_text
from fill_pa_form import fill_pa_form


def _filled_names(result: dict) -> set:
    details = result.get("filled_field_details") or []
    names = set()
    for item in details:
        if isinstance(item, dict):
            names.add(str(item.get("field_name", "")).lower())
            names.add(str(item.get("tooltip", "")).lower())
            names.add(str(item.get("value", "")).lower())
        else:
            names.add(str(item).lower())
    return names


def test_adbulla_widget_form_fills_key_patient_fields(adbulla_ocr, adbulla_input, tmp_path):
    extracted = parse_referral_text(adbulla_ocr)
    output_pdf = tmp_path / "PA_filled.pdf"
    result = fill_pa_form(str(adbulla_input / "PA.pdf"), extracted, str(output_pdf))

    assert output_pdf.exists()
    assert result["fill_percentage"] >= 90
    assert result["status"] in {"completed", "success", "SUCCESS"}

    blob = " ".join(_filled_names(result))
    assert "shakh" in blob
    assert "abdulla" in blob
    assert "04/01/2001" in blob or "4/1/2001" in blob
    assert "g35" in blob

    reader = PdfReader(str(output_pdf))
    fields = reader.get_form_text_fields() or {}
    values = " ".join(str(v) for v in fields.values() if v).lower()
    assert "shakh" in values
    assert "abdulla" in values


def test_amy_static_form_writes_overlay_pdf(amy_ocr, amy_input, tmp_path):
    extracted = parse_referral_text(amy_ocr)
    output_pdf = tmp_path / "PA_filled.pdf"
    result = fill_pa_form(str(amy_input / "PA.pdf"), extracted, str(output_pdf))

    assert output_pdf.exists()
    assert output_pdf.stat().st_size > 0
    assert result["fill_percentage"] >= 90
    blob = " ".join(_filled_names(result))
    assert "chen" in blob
    assert "amy" in blob
    assert "vyepti" in blob or "eptinezumab" in blob
