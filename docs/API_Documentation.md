# API Documentation

## Overview

This document provides detailed API documentation for the Prior Authorization (PA) Form Filling Pipeline. The pipeline consists of four main modules that work together to process medical referral packages and fill PA forms.

## Table of Contents

1. [extract_referral_data.py](#extract_referral_data)
2. [fill_pa_form.py](#fill_pa_form) 
3. [generate_report.py](#generate_report)
4. [utils.py](#utils)
5. [main.py](#main)

---

## extract_referral_data

### `extract_referral_data(referral_pdf_path, output_dir, use_advanced_ocr=True)`

**Purpose:** Extract structured data from unstructured medical referral PDF documents.

**Parameters:**
- `referral_pdf_path` (str): Path to the referral package PDF file
- `output_dir` (str): Directory to save extraction results
- `use_advanced_ocr` (bool, optional): Enable advanced OCR preprocessing. Default: True

**Returns:**
- `dict`: Extraction results containing:
  - `metadata`: Processing metadata (timestamp, file info, processing time)
  - `patient_info`: Patient demographics and contact information
  - `clinical_info`: Medical diagnosis, procedures, medications
  - `insurance_info`: Insurance carrier and policy details
  - `prescriber_info`: Healthcare provider information
  - `raw_text`: Full extracted text from the document
  - `extraction_errors`: List of any errors encountered

**Example:**
```python
from src.extract_referral_data import extract_referral_data

results = extract_referral_data(
    referral_pdf_path="input_data/Patient A/referral_package.pdf",
    output_dir="output/Patient A"
)

print(f"Patient name: {results['patient_info'].get('name', 'Not found')}")
print(f"Diagnosis: {results['clinical_info'].get('diagnosis', 'Not found')}")
```

### `ReferralDataExtractor` Class

#### `__init__(self, use_advanced_ocr=True)`

Initialize the extractor with OCR configuration.

#### `extract_data(self, pdf_path, output_dir)`

Main extraction method that coordinates the entire process.

**Internal Methods:**
- `_extract_text_from_pdf(pdf_path)`: Direct PDF text extraction
- `_extract_text_via_ocr(pdf_path)`: OCR-based text extraction with preprocessing
- `_preprocess_image(image)`: Image enhancement for better OCR
- `_process_extracted_text(text)`: Pattern matching and data structuring
- `_validate_and_clean_data(data)`: Data validation and formatting

---

## fill_pa_form

### `fill_pa_form(pa_form_path, extracted_data, output_path)`

**Purpose:** Fill a PA PDF form with extracted referral data.

**Parameters:**
- `pa_form_path` (str): Path to the blank PA form PDF
- `extracted_data` (dict): Structured data from referral extraction
- `output_path` (str): Path for the filled PA form output

**Returns:**
- `dict`: Filling results containing:
  - `success` (bool): Whether filling completed successfully
  - `filled_count` (int): Number of fields successfully filled
  - `total_fields` (int): Total number of form fields detected
  - `missing_fields` (list): Fields that couldn't be filled
  - `filled_fields` (list): Successfully filled fields with values
  - `errors` (list): Any errors encountered during filling

**Example:**
```python
from src.fill_pa_form import fill_pa_form

result = fill_pa_form(
    pa_form_path="input_data/Patient A/PA.pdf",
    extracted_data=extraction_results,
    output_path="output/Patient A/PA_filled.pdf"
)

print(f"Success: {result['success']}")
print(f"Filled {result['filled_count']}/{result['total_fields']} fields")
```

### `PAFormFiller` Class

#### `__init__(self)`

Initialize the form filler with field matching capabilities.

#### `fill_pa_form(self, pa_form_path, extracted_data, output_path)`

Main form filling method.

**Internal Methods:**
- `_analyze_form_structure(pa_form_path)`: Detect form fields and structure
- `_fill_widget_based_form(reader, writer, flat_data)`: Fill forms with AcroForm fields
- `_fill_non_widget_form(pa_form_path, flat_data, output_path)`: Text overlay for static PDFs
- `_process_form_field(field, flat_data, writer)`: Process individual form fields
- `_fill_text_field(field, value, writer)`: Handle text input fields
- `_fill_checkbox_field(field, value, writer)`: Handle checkbox fields
- `_fill_choice_field(field, value, writer)`: Handle dropdown/radio fields

---

## generate_report

### `generate_comprehensive_report(patient_id, extraction_results, filling_results, pa_form_path, referral_path, output_path)`

**Purpose:** Generate a detailed processing report in markdown format.

**Parameters:**
- `patient_id` (str): Patient identifier
- `extraction_results` (dict): Results from referral data extraction
- `filling_results` (dict): Results from PA form filling
- `pa_form_path` (str): Path to original PA form
- `referral_path` (str): Path to referral package
- `output_path` (str): Path for the output report

**Returns:**
- `bool`: True if report generation succeeded

### `generate_missing_fields_report(missing_fields, output_path, patient_id=None)`

**Purpose:** Generate a focused report on missing fields.

**Parameters:**
- `missing_fields` (list): List of fields that couldn't be filled
- `output_path` (str): Path for the output report
- `patient_id` (str, optional): Patient identifier

**Returns:**
- `bool`: True if report generation succeeded

**Example:**
```python
from src.generate_report import generate_comprehensive_report, generate_missing_fields_report

# Generate comprehensive report
success = generate_comprehensive_report(
    patient_id="Patient A",
    extraction_results=extraction_data,
    filling_results=form_data,
    pa_form_path="input_data/Patient A/PA.pdf",
    referral_path="input_data/Patient A/referral_package.pdf",
    output_path="output/Patient A/processing_report.md"
)

# Generate missing fields report
generate_missing_fields_report(
    missing_fields=form_data['missing_fields'],
    output_path="output/Patient A/missing_fields.txt",
    patient_id="Patient A"
)
```

### `ReportGenerator` Class

#### `__init__(self)`

Initialize the report generator with templates.

**Internal Methods:**
- `_prepare_report_data(extraction_results, filling_results)`: Process data for reporting
- `_format_report(template, data)`: Apply data to report template
- `_format_extracted_data_summary(data)`: Format extraction results section
- `_format_confidence_scores(data)`: Format confidence analysis
- `_format_filled_fields_summary(data)`: Format form filling results
- `_format_missing_fields_analysis(data)`: Analyze missing fields
- `_format_recommendations(data)`: Generate actionable recommendations

---

## utils

### Core Utility Functions

#### `setup_logging(log_file=None)`

**Purpose:** Configure logging for the pipeline.

**Parameters:**
- `log_file` (str, optional): Path to log file. If None, logs to console only.

**Returns:**
- `logging.Logger`: Configured logger instance

#### `validate_input_files(patient_dir)`

**Purpose:** Validate that required input files exist for a patient.

**Parameters:**
- `patient_dir` (str): Path to patient directory

**Returns:**
- `tuple`: (is_valid: bool, missing_files: list)

#### `preprocess_image(image)`

**Purpose:** Enhance image quality for better OCR results.

**Parameters:**
- `image` (np.ndarray): Input image array

**Returns:**
- `np.ndarray`: Preprocessed image

#### `extract_patient_info_patterns()`

**Purpose:** Get regex patterns for extracting patient information.

**Returns:**
- `dict`: Dictionary of field names to regex patterns

#### `normalize_field_value(field_name, value)`

**Purpose:** Normalize extracted field values based on their type.

**Parameters:**
- `field_name` (str): Name of the field being normalized
- `value` (str): Raw extracted value

**Returns:**
- `str`: Normalized value

#### `calculate_confidence_score(extracted_text, field_pattern)`

**Purpose:** Calculate confidence score for extracted data.

**Parameters:**
- `extracted_text` (str): The extracted text
- `field_pattern` (str): Regex pattern used for extraction

**Returns:**
- `float`: Confidence score between 0.0 and 1.0

### File Operations

#### `load_json_safely(file_path)`

**Purpose:** Safely load JSON data with error handling.

**Parameters:**
- `file_path` (str): Path to JSON file

**Returns:**
- `dict`: Loaded JSON data or empty dict on error

#### `save_json_safely(data, file_path)`

**Purpose:** Safely save data to JSON file.

**Parameters:**
- `data` (dict): Data to save
- `file_path` (str): Output file path

**Returns:**
- `bool`: True if save succeeded

#### `get_patient_directories(input_dir)`

**Purpose:** Get list of patient directories in input folder.

**Parameters:**
- `input_dir` (str): Input directory path

**Returns:**
- `list`: List of patient directory paths

### `FieldMatcher` Class

#### `__init__(self)`

Initialize field matcher with semantic capabilities.

#### `find_best_match(self, form_field_name, extracted_data)`

**Purpose:** Find the best matching extracted data for a form field.

**Parameters:**
- `form_field_name` (str): Name of the form field
- `extracted_data` (dict): Extracted referral data

**Returns:**
- `tuple`: (matched_value: str, confidence: float)

**Internal Methods:**
- `_direct_key_match(field_name, data)`: Direct key matching
- `_fuzzy_key_match(field_name, data)`: Fuzzy string matching
- `_semantic_match(field_name, data)`: Semantic similarity matching
- `_calculate_match_confidence(field_name, key, value)`: Confidence calculation

---

## main

### `PAProcessingPipeline` Class

#### `__init__(self, config=None)`

**Purpose:** Initialize the main pipeline orchestrator.

**Parameters:**
- `config` (dict, optional): Configuration dictionary

#### `process_single_patient(self, patient_dir, output_dir)`

**Purpose:** Process a single patient's PA form and referral package.

**Parameters:**
- `patient_dir` (str): Path to patient's input directory
- `output_dir` (str): Path to output directory

**Returns:**
- `dict`: Processing results containing:
  - `success` (bool): Overall processing success
  - `patient_id` (str): Patient identifier
  - `extraction_results` (dict): Data extraction results
  - `filling_results` (dict): Form filling results
  - `reports_generated` (list): List of generated reports
  - `processing_time` (float): Total processing time
  - `errors` (list): Any errors encountered

#### `process_batch(self, input_dir, output_dir)`

**Purpose:** Process all patients in an input directory.

**Parameters:**
- `input_dir` (str): Directory containing patient folders
- `output_dir` (str): Output directory for processed results

**Returns:**
- `dict`: Batch processing summary containing:
  - `total_patients` (int): Number of patients processed
  - `successful_patients` (int): Number successfully processed
  - `failed_patients` (int): Number that failed processing
  - `processing_time` (float): Total batch processing time
  - `patient_results` (list): Individual patient results
  - `summary_stats` (dict): Overall statistics

**Example:**
```python
from src.main import PAProcessingPipeline

# Initialize pipeline
pipeline = PAProcessingPipeline()

# Process single patient
result = pipeline.process_single_patient(
    patient_dir="input_data/Patient A",
    output_dir="output"
)

# Process all patients
batch_results = pipeline.process_batch(
    input_dir="input_data",
    output_dir="output"
)

print(f"Processed {batch_results['successful_patients']}/{batch_results['total_patients']} patients")
```

---

## Error Handling

All functions implement comprehensive error handling:

- **FileNotFoundError**: When input files don't exist
- **PermissionError**: When output directories aren't writable
- **PDFError**: When PDF files are corrupted or unreadable
- **OCRError**: When OCR processing fails
- **ValidationError**: When data validation fails

Error information is always included in return dictionaries under the `errors` key.

---

## Configuration Options

The pipeline supports extensive configuration through the `config.json` file or configuration dictionary:

### Core Settings
- `confidence_threshold`: Minimum confidence for accepting extracted data
- `use_advanced_ocr`: Enable image preprocessing for OCR
- `generate_comprehensive_reports`: Create detailed markdown reports
- `generate_missing_fields_reports`: Create focused missing fields reports
- `save_intermediate_files`: Save extraction results as JSON

### OCR Settings
- `tesseract_config`: Tesseract OCR configuration string
- `ocr_preprocessing`: Image enhancement settings

### Field Matching
- `fuzzy_match_threshold`: Threshold for fuzzy string matching
- `semantic_match_threshold`: Threshold for semantic matching
- `use_semantic_matching`: Enable semantic field matching

### Form Filling
- `validate_field_types`: Enable field type validation
- `handle_checkboxes`: Process checkbox fields
- `process_conditional_fields`: Handle conditional form sections
- `overlay_font_size`: Font size for text overlays
- `overlay_font_color`: Color for text overlays

---

## Return Value Specifications

### Extraction Results Structure
```python
{
    "metadata": {
        "timestamp": "2024-01-15T10:30:00",
        "file_path": "/path/to/referral.pdf",
        "file_size": 1024000,
        "processing_time": 15.7,
        "extraction_method": "ocr"
    },
    "patient_info": {
        "name": "John Doe",
        "date_of_birth": "1985-03-15",
        "ssn": "123-45-6789",
        "phone": "(555) 123-4567",
        "address": "123 Main St, City, ST 12345"
    },
    "clinical_info": {
        "diagnosis": "Type 2 Diabetes",
        "icd_codes": ["E11.9"],
        "medications": ["Metformin 500mg"],
        "dosage": "Twice daily",
        "procedure_codes": ["99213"]
    },
    "insurance_info": {
        "primary_carrier": "Blue Cross Blue Shield",
        "policy_number": "ABC123456789",
        "group_number": "GRP789",
        "subscriber_id": "SUB456"
    },
    "prescriber_info": {
        "name": "Dr. Jane Smith",
        "npi": "1234567890",
        "phone": "(555) 987-6543",
        "address": "456 Medical Dr, City, ST 12345"
    },
    "raw_text": "Full extracted text...",
    "extraction_errors": []
}
```

### Form Filling Results Structure
```python
{
    "success": True,
    "filled_count": 15,
    "total_fields": 20,
    "missing_fields": [
        {
            "field_name": "member_id",
            "reason": "No matching data found",
            "confidence": 0.0
        }
    ],
    "filled_fields": [
        {
            "field_name": "patient_name",
            "value": "John Doe",
            "confidence": 0.95,
            "source": "patient_info.name"
        }
    ],
    "errors": []
}
```

This API documentation provides comprehensive information for developers working with the PA pipeline, including function signatures, parameters, return values, and usage examples. 