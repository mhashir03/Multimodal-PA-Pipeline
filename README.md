# Prior Authorization (PA) Form Filling Pipeline

A Python-based automated pipeline that extracts information from unstructured medical referral packages and fills out structured PA PDF forms for Mandolin.

## 🎯 Overview

This pipeline automates the Prior Authorization form filling workflow by:

1. **Extracting** critical information from unstructured, high-resolution medical referral packets using OCR and NLP
2. **Filling** structured, widget-based PA PDF forms with extracted data
3. **Generating** comprehensive reports including missing fields analysis

## 🏗️ Architecture

```
📁 src/
├── extract_referral_data.py   # OCR + NLP to extract structured fields
├── fill_pa_form.py            # Populate PA.pdf with extracted data
├── generate_report.py         # Create missing fields report
├── utils.py                   # Reusable helpers
└── main.py                    # Pipeline orchestrator

📁 input_data/
├── Patient A/
│   ├── PA.pdf                 # Structured PA form to fill
│   └── referral_package.pdf   # Unstructured referral documents
└── Patient B/
    ├── PA.pdf
    └── referral_package.pdf

📁 output/
├── Patient A/
│   ├── PA_filled.pdf          # Completed PA form
│   ├── missing_fields.txt     # Missing fields report
│   ├── processing_report.md   # Comprehensive processing report
│   └── extracted_referral_data.json  # Raw extraction results
└── batch_summary.json         # Batch processing summary
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ 
- Tesseract OCR installed on your system
- Required Python packages (see requirements.txt)

**Note for macOS users:** Use `python3` instead of `python` in all commands.

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Multimodal-PA-Pipeline
   ```

2. **Install dependencies:**
   ```bash
   # For macOS, use python3
   pip3 install -r requirements.txt
   
   # For other systems
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR:**
   
   **macOS:**
   ```bash
   brew install tesseract
   ```
   
   **Ubuntu/Debian:**
   ```bash
   sudo apt-get install tesseract-ocr
   ```
   
   **Windows:**
   Download from: https://github.com/UB-Mannheim/tesseract/wiki

4. **Set up directory structure:**
   ```bash
   mkdir -p input_data output logs
   ```

### Basic Usage

1. **Place your files in the input directory:**
   ```
   input_data/
   ├── Patient A/
   │   ├── PA.pdf
   │   └── referral_package.pdf
   ```

2. **Run the pipeline:**
   ```bash
   # For macOS, use python3
   # Process all patients
   python3 src/main.py --input input_data --output output
   
   # Process a single patient
   python3 src/main.py --patient "input_data/Patient A" --output output
   
   # For other systems, use python
   python src/main.py --input input_data --output output
   ```

3. **Check results:**
   ```
   output/
   ├── Patient A/
   │   ├── PA_filled.pdf          # ✅ Filled PA form
   │   ├── missing_fields.txt     # 📋 Missing fields report
   │   └── processing_report.md   # 📊 Detailed analysis
   ```

## 📖 Detailed Usage

### Command Line Options

```bash
python src/main.py [OPTIONS]

Options:
  -i, --input PATH     Input directory containing patient folders
  -o, --output PATH    Output directory for processed files
  -p, --patient PATH   Process a single patient directory
  -c, --config PATH    Path to configuration JSON file
  -l, --log-file PATH  Path to log file
  -v, --verbose        Enable verbose logging
  -h, --help          Show help message
```

### Examples

**Process all patients with custom output:**
```bash
python src/main.py --input data/referrals --output results/processed
```

**Process single patient with verbose logging:**
```bash
python src/main.py --patient "input_data/John Doe" --output output --verbose
```

**Use custom configuration:**
```bash
python src/main.py --input input_data --config config.json
```

### Configuration File

Create a `config.json` file to customize pipeline behavior:

```json
{
  "confidence_threshold": 0.6,
  "use_advanced_ocr": true,
  "generate_comprehensive_reports": true,
  "generate_missing_fields_reports": true,
  "save_intermediate_files": true,
  "log_file": "logs/custom.log"
}
```

## 🔧 Core Features

### 1. Data Extraction (`extract_referral_data.py`)

**Capabilities:**
- **Dual-mode extraction:** Direct PDF text extraction + OCR fallback
- **Advanced OCR:** Image preprocessing for better accuracy
- **Pattern matching:** Regex patterns for common medical fields
- **Confidence scoring:** Quality assessment for extracted data
- **Error handling:** Graceful degradation with detailed error reporting

**Extracted Fields:**
- Patient information (name, DOB, SSN, phone, address)
- Clinical information (diagnosis, ICD codes, medications, dosage)
- Insurance information (carrier, policy numbers)
- Prescriber information (provider name, NPI, contact details)

**Usage:**
```python
from src.extract_referral_data import extract_referral_data

results = extract_referral_data('path/to/referral.pdf', 'output/dir')
print(f"Extracted {len(results['patient_info'])} patient fields")
```

### 2. Form Filling (`fill_pa_form.py`)

**Capabilities:**
- **Widget-based forms:** Full support for PDF form fields
- **Non-widget forms:** Text overlay for static PDFs
- **Intelligent matching:** Field name matching with confidence scoring
- **Checkbox logic:** Smart checkbox selection based on context
- **Choice fields:** Dropdown/radio button handling with fuzzy matching
- **Validation:** Data type validation and format checking

**Form Types Supported:**
- ✅ Widget-based PDFs (AcroForm fields)
- ✅ Non-widget PDFs (text overlay)
- ✅ Checkboxes with mutually exclusive logic
- ✅ Dropdown/choice fields
- ✅ Text fields with validation

**Usage:**
```python
from src.fill_pa_form import fill_pa_form

result = fill_pa_form('PA.pdf', extracted_data, 'PA_filled.pdf')
print(f"Filled {result['filled_count']}/{result['total_fields']} fields")
```

### 3. Report Generation (`generate_report.py`)

**Report Types:**

**Comprehensive Report (`processing_report.md`):**
- Executive summary with key metrics
- Detailed extraction results with confidence scores
- Form filling analysis
- Missing fields breakdown
- Actionable recommendations
- Data quality assessment
- Technical details and error logs

**Missing Fields Report (`missing_fields.txt`):**
- Categorized missing fields
- Specific reasons for each missing field
- Targeted recommendations for resolution

**Usage:**
```python
from src.generate_report import generate_comprehensive_report

success = generate_comprehensive_report(
    patient_id="Patient A",
    extraction_results=extraction_data,
    filling_results=form_data,
    pa_form_path="PA.pdf",
    referral_path="referral.pdf",
    output_path="report.md"
)
```

## 🎛️ Advanced Configuration

### OCR Optimization

The pipeline includes advanced OCR preprocessing:

```python
# In extract_referral_data.py
extractor = ReferralDataExtractor(use_advanced_ocr=True)
```

**Image preprocessing steps:**
1. Grayscale conversion
2. Noise reduction
3. Adaptive thresholding  
4. Morphological operations

### Custom Field Patterns

Add custom extraction patterns in `utils.py`:

```python
def extract_patient_info_patterns():
    patterns = {
        "custom_field": r"(?:custom_label)[\s:]*([^\n]+)",
        # Add more patterns...
    }
    return patterns
```

### Field Matching Logic

Customize field matching in `utils.py`:

```python
class FieldMatcher:
    def find_best_match(self, form_field_name, extracted_data):
        # Custom matching logic
        pass
```

## 📊 Output Analysis

### Success Metrics

The pipeline tracks several key metrics:

- **Extraction Success Rate:** % of referrals successfully processed
- **Field Fill Rate:** % of PA form fields populated
- **Confidence Scores:** Quality assessment of extracted data
- **Processing Time:** Performance metrics per patient

### Understanding Reports

**Confidence Levels:**
- 🟢 **High (≥0.8):** Reliable data, likely accurate
- 🟡 **Medium (0.6-0.8):** Good data, may need review  
- 🔴 **Low (<0.6):** Uncertain data, manual review recommended

**Missing Field Reasons:**
- `No matching data found`: Information not present in referral
- `Low confidence (X.XX)`: Data found but quality uncertain
- `No matching option for 'value'`: Choice field value not in options

## 🚨 Troubleshooting

### Common Issues

**1. OCR Quality Issues**
```
Problem: Poor text extraction from referral PDFs
Solution: 
- Ensure high-resolution input documents (300+ DPI)
- Check Tesseract installation: tesseract --version
- Enable advanced OCR preprocessing
```

**2. Form Field Detection**
```
Problem: PA form fields not detected
Solution:
- Verify PDF has fillable form fields (AcroForm)
- Check PDF with: pdfplumber or Adobe Reader
- Use non-widget mode for static PDFs
```

**3. Low Confidence Scores**
```
Problem: Many fields have low confidence
Solution:
- Review extraction patterns in utils.py
- Improve referral document quality
- Manually validate uncertain fields
```

**4. Permission Errors**
```
Problem: Cannot write output files
Solution:
- Check output directory permissions
- Ensure sufficient disk space
- Run with appropriate user permissions
```

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
python src/main.py --input input_data --verbose --log-file debug.log
```

### Testing Individual Components

**Test extraction only:**
```bash
python src/extract_referral_data.py referral_package.pdf output_dir
```

**Test form filling only:**
```bash
python src/fill_pa_form.py PA.pdf extracted_data.json PA_filled.pdf
```

**Test report generation:**
```bash
python src/generate_report.py missing_fields.json report.md "Patient A"
```

## 🔮 Future Enhancements

### Planned Features

1. **LLM Integration:**
   - Mistral API integration for improved extraction
   - Context-aware field matching
   - Natural language processing for complex medical terms

2. **Machine Learning:**
   - Training data collection from successful extractions
   - Custom models for medical document understanding
   - Continuous improvement through feedback loops

3. **Web Interface:**
   - Browser-based upload and processing
   - Real-time progress tracking
   - Interactive field validation

4. **API Integration:**
   - RESTful API for external system integration
   - Webhook support for automated workflows
   - Bulk processing capabilities

### Stretch Goals Implemented

- ✅ **Non-widget PDF support:** Text overlay for static forms
- ✅ **Advanced OCR:** Image preprocessing pipeline
- ✅ **Comprehensive reporting:** Detailed analysis and recommendations
- ✅ **Batch processing:** Multiple patients at once
- ✅ **Confidence scoring:** Data quality assessment

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make changes and add tests
5. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function parameters
- Include docstrings for all public functions
- Add unit tests for new features

### Testing

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Check code coverage
coverage run -m pytest && coverage report
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions, issues, or contributions:

1. **Issues:** Create a GitHub issue with detailed description
2. **Documentation:** Check this README and inline code comments
3. **Logs:** Include relevant log files with issue reports

## 🔄 Version History

- **v1.0.0** - Initial release with core functionality
  - OCR-based data extraction
  - Widget and non-widget PDF form filling
  - Comprehensive reporting system
  - Batch processing capabilities

---

*Built with ❤️ for Mandolin's Prior Authorization automation.* 