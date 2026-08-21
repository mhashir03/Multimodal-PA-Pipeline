"""Tooltip-based form field mapping tests."""

from field_mapper import map_widget_value


SAMPLE_DATA = {
    "patient_info": {
        "patient_name": "Abdulla, Shakh",
        "first_name": "Shakh",
        "last_name": "Abdulla",
        "dob": "04/01/2001",
        "phone": "(865) 839-7458",
        "member_id": "LAJIM14345116",
        "address": "8327 Broadway Ln APT D",
        "city": "Knoxville",
        "state": "TN",
        "zip": "37923",
        "allergies": "No Known Allergies",
    },
    "clinical_info": {
        "diagnosis": "Multiple sclerosis",
        "icd_code": "G35",
        "medication": "Truxima (rituximab-abbs)",
        "dose": "694 mg",
    },
    "insurance_info": {
        "insurance_carrier": "TENNCARE/BLUECARE",
        "member_id": "LAJIM14345116",
    },
    "prescriber_info": {
        "prescriber_name": "Hao H Gu, MD",
        "first_name": "Hao",
        "last_name": "Gu",
        "npi": "1154611523",
        "credentials": "MD",
    },
}


def test_maps_patient_tooltips():
    assert map_widget_value("Patient First Name", "Tx", SAMPLE_DATA)[0] == "Shakh"
    assert map_widget_value("Patient Last Name", "Tx", SAMPLE_DATA)[0] == "Abdulla"
    assert map_widget_value("Patient DOB (MM/DD/YYYY)", "Tx", SAMPLE_DATA)[0] == "04/01/2001"
    assert "865" in map_widget_value("Patient Phone", "Tx", SAMPLE_DATA)[0]


def test_maps_clinical_and_insurance_tooltips():
    assert map_widget_value("Primary ICD Code:", "Tx", SAMPLE_DATA)[0] == "G35"
    assert "TENNCARE" in map_widget_value("Carrier Name:", "Tx", SAMPLE_DATA)[0]
    assert map_widget_value("Member ID #:", "Tx", SAMPLE_DATA)[0] == "LAJIM14345116"


def test_checks_matching_diagnosis_and_drug_boxes():
    value, conf = map_widget_value("Truxima (rituximab-abbs)", "Btn", SAMPLE_DATA)
    assert value in {"Yes", "/Yes", True, "On"}
    assert conf >= 0.7

    value, conf = map_widget_value("Rituxan (rituximab)", "Btn", SAMPLE_DATA)
    assert value in {"Yes", "/Yes", True, "On"}

    value, _ = map_widget_value("pemphigus vulgaris", "Btn", SAMPLE_DATA)
    assert not value


def test_does_not_fuzzy_match_unnamed_widgets():
    value, conf = map_widget_value("T2", "Tx", SAMPLE_DATA)
    assert value in ("", None)
    assert conf < 0.5
