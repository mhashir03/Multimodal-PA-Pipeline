"""
Enhanced PA Form Filling Module

This module manages filling structured PA PDF forms with extracted medical data,
using intelligent field mapping and comprehensive validation for production-ready results.
"""

import os
import json
import sys
from typing import Dict, List, Any, Optional, Tuple, Union
import pdfplumber
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import io
from loguru import logger
import re
from pathlib import Path
import pandas as pd

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    EnhancedFieldMatcher, normalize_field_value, calculate_confidence_score,
    validate_extracted_data, save_json_safely, ensure_directory_exists, setup_logging
)


class EnhancedPAFormFiller:
    """
    Enhanced PA form filler with intelligent field mapping and comprehensive validation.
    
    This class handles both widget-based and non-widget PDF forms, with advanced
    field detection and mapping capabilities for medical forms.
    """

    def __init__(self):
        """Initialize the enhanced PA form filler."""
        self.field_matcher = EnhancedFieldMatcher()
        self.filled_fields = []
        self.missing_fields = []
        self.field_mapping_log = []
        
    def fill_pa_form(self, pa_form_path: str, extracted_data: Dict[str, Any], 
                     output_path: str) -> Dict[str, Any]:
        """
        Fill a PA form with extracted data using enhanced matching and validation.
        
        Args:
            pa_form_path: Path to the PA form PDF
            extracted_data: Dictionary containing extracted referral data
            output_path: Path to save the filled form
            
        Returns:
            Dictionary with processing results and statistics
        """
        logger.info(f"Starting enhanced PA form filling for: {pa_form_path}")
        
        # Validate inputs
        if not os.path.exists(pa_form_path):
            logger.error(f"PA form file not found: {pa_form_path}")
            return self._create_error_result("PA form file not found")
        
        # Validate extracted data quality
        validation_results = validate_extracted_data(extracted_data)
        if not validation_results["is_valid"]:
            logger.warning(f"Data validation issues: {validation_results['errors']}")
        
        # Analyze form structure with enhanced detection
        form_analysis = self._analyze_form_structure_enhanced(pa_form_path)
        
        if not form_analysis["has_fillable_fields"]:
            logger.warning("No fillable fields detected in PA form")
            return self._create_error_result("No fillable fields detected")
        
        logger.info(f"Form analysis: {form_analysis['field_count']} fields detected, "
                   f"Type: {form_analysis['form_type']}")
        
        # Fill form based on type with enhanced logic
        try:
            if form_analysis["form_type"] == "widget_based":
                result = self._fill_widget_form_enhanced(
                    pa_form_path, extracted_data, output_path, form_analysis
                )
            else:
                result = self._fill_non_widget_form_enhanced(
                    pa_form_path, extracted_data, output_path, form_analysis
                )
            
            # Add comprehensive metadata
            result.update({
                "form_analysis": form_analysis,
                "data_validation": validation_results,
                "field_mapping_log": self.field_mapping_log,
                "processing_statistics": self._calculate_processing_stats()
            })
            
            logger.info(f"Form filling completed. Success rate: "
                       f"{result.get('fill_rate', 0):.1%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during form filling: {e}")
            return self._create_error_result(f"Form filling error: {str(e)}")

    def _analyze_form_structure_enhanced(self, pdf_path: str) -> Dict[str, Any]:
        """
        Enhanced form structure analysis with better field detection.
        
        Args:
            pdf_path: Path to the PDF form
            
        Returns:
            Dictionary containing detailed form analysis
        """
        analysis = {
            "has_fillable_fields": False,
            "form_type": "unknown",
            "field_count": 0,
            "fields": [],
            "field_types": {},
            "field_coordinates": {},
            "text_elements": []
        }
        
        try:
            # Method 1: PyPDF2 for widget fields
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                
                # Check for AcroForm fields
                if '/AcroForm' in pdf_reader.trailer['/Root']:
                    acro_form = pdf_reader.trailer['/Root']['/AcroForm']
                    if '/Fields' in acro_form:
                        fields = acro_form['/Fields']
                        analysis["has_fillable_fields"] = True
                        analysis["form_type"] = "widget_based"
                        analysis["field_count"] = len(fields)
                        
                        # Extract field details
                        for field_ref in fields:
                            field = field_ref.get_object()
                            if '/T' in field:  # Field name
                                field_name = field['/T']
                                analysis["fields"].append(field_name)
                                
                                # Field type
                                field_type = "text"  # default
                                if '/FT' in field:
                                    ft = field['/FT']
                                    if ft == '/Tx':
                                        field_type = "text"
                                    elif ft == '/Ch':
                                        field_type = "choice"
                                    elif ft == '/Btn':
                                        field_type = "button"
                                
                                analysis["field_types"][field_name] = field_type
            
            # Method 2: pdfplumber for detailed analysis and coordinates
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        # Extract form fields with coordinates
                        if hasattr(page, 'annots') and page.annots:
                            for annot in page.annots:
                                if annot.get('subtype') == 'Widget':
                                    field_name = annot.get('T', f'field_{len(analysis["fields"])}')
                                    if field_name not in analysis["fields"]:
                                        analysis["fields"].append(field_name)
                                        analysis["field_count"] += 1
                                    
                                    # Store coordinates
                                    if 'rect' in annot:
                                        analysis["field_coordinates"][field_name] = {
                                            "page": page_num,
                                            "bbox": annot['rect']
                                        }
                        
                        # Extract text elements for non-widget forms
                        try:
                            text_elements = page.extract_words()
                            if text_elements:
                                for elem in text_elements:
                                    if isinstance(elem, dict) and 'text' in elem and 'bbox' in elem:
                                        analysis["text_elements"].append({
                                            "text": elem["text"],
                                            "bbox": elem["bbox"],
                                            "page": page_num
                                        })
                        except Exception as e:
                            logger.warning(f"Could not extract text elements from page {page_num}: {e}")
                            continue
            except Exception as e:
                logger.warning(f"pdfplumber analysis failed: {e}")
            
            # If no widget fields found, check for text-based form
            if not analysis["has_fillable_fields"] and analysis["text_elements"]:
                # Look for form-like patterns in text
                form_patterns = [
                    r".*name.*:.*",
                    r".*date.*:.*",
                    r".*phone.*:.*",
                    r".*address.*:.*",
                    r".*diagnosis.*:.*",
                    r".*medication.*:.*",
                    r".*insurance.*:.*"
                ]
                
                potential_fields = []
                for element in analysis["text_elements"]:
                    for pattern in form_patterns:
                        if re.match(pattern, element["text"], re.IGNORECASE):
                            potential_fields.append(element["text"])
                
                if potential_fields:
                    analysis["has_fillable_fields"] = True
                    analysis["form_type"] = "text_based"
                    analysis["field_count"] = len(potential_fields)
                    analysis["fields"] = potential_fields
            
            logger.info(f"Form analysis complete: {analysis['field_count']} fields, "
                       f"Type: {analysis['form_type']}")
            
        except Exception as e:
            logger.error(f"Error analyzing form structure: {e}")
            
        return analysis

    def _fill_widget_form_enhanced(self, pdf_path: str, extracted_data: Dict[str, Any], 
                                 output_path: str, form_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced widget-based form filling with intelligent field mapping.
        
        Args:
            pdf_path: Path to the input PA form
            extracted_data: Extracted referral data
            output_path: Path to save filled form
            form_analysis: Form structure analysis results
            
        Returns:
            Dictionary with filling results and statistics
        """
        filled_count = 0
        total_fields = form_analysis["field_count"]
        
        try:
            # Open the form for filling
            with open(pdf_path, 'rb') as input_file:
                pdf_reader = PdfReader(input_file)
                pdf_writer = PdfWriter()
                
                # Process each page
                for page_num, page in enumerate(pdf_reader.pages):
                    # Get the page and its annotations
                    page_obj = pdf_reader.pages[page_num]
                    
                    # Fill form fields on this page
                    if '/Annots' in page_obj:
                        annotations = page_obj['/Annots']
                        for annot_ref in annotations:
                            annot = annot_ref.get_object()
                            
                            if annot.get('/Subtype') == '/Widget' and '/T' in annot:
                                field_name = annot['/T']
                                
                                # Find best match using enhanced field matcher
                                field_value, confidence = self.field_matcher.find_best_match(
                                    field_name, extracted_data
                                )
                                
                                if field_value and confidence > 0.5:
                                    # Normalize the value
                                    normalized_value = normalize_field_value(field_name, field_value)
                                    
                                    # Fill the field
                                    try:
                                        if '/V' in annot:
                                            annot.update({PdfWriter.generic.NameObject('/V'): 
                                                        PdfWriter.generic.TextStringObject(normalized_value)})
                                            filled_count += 1
                                            
                                            self.filled_fields.append({
                                                "field_name": field_name,
                                                "value": normalized_value,
                                                "confidence": confidence,
                                                "page": page_num
                                            })
                                            
                                            self.field_mapping_log.append({
                                                "form_field": field_name,
                                                "matched_data": field_value,
                                                "final_value": normalized_value,
                                                "confidence": confidence,
                                                "method": "enhanced_matching"
                                            })
                                            
                                        logger.debug(f"Filled field '{field_name}' with '{normalized_value}' "
                                                   f"(confidence: {confidence:.2f})")
                                        
                                    except Exception as e:
                                        logger.warning(f"Failed to fill field '{field_name}': {e}")
                                        self.missing_fields.append({
                                            "field_name": field_name,
                                            "reason": f"Fill error: {str(e)}",
                                            "page": page_num
                                        })
                                else:
                                    self.missing_fields.append({
                                        "field_name": field_name,
                                        "reason": "No matching data found" if not field_value else f"Low confidence ({confidence:.2f})",
                                        "page": page_num
                                    })
                    
                    pdf_writer.add_page(page_obj)
                
                # Save the filled form
                ensure_directory_exists(os.path.dirname(output_path))
                with open(output_path, 'wb') as output_file:
                    pdf_writer.write(output_file)
                
                logger.info(f"Filled {filled_count}/{total_fields} fields in widget form")
                
        except Exception as e:
            logger.error(f"Error filling widget form: {e}")
            raise
        
        return {
            "status": "completed",
            "filled_fields": filled_count,
            "filled_count": filled_count,
            "total_fields": total_fields,
            "fill_rate": filled_count / total_fields if total_fields > 0 else 0,
            "fill_percentage": (filled_count / total_fields * 100) if total_fields > 0 else 0.0,
            "filled_field_details": self.filled_fields,
            "missing_field_details": self.missing_fields,
            "missing_fields": self.missing_fields,
            "processing_errors": []
        }

    def _fill_non_widget_form_enhanced(self, pdf_path: str, extracted_data: Dict[str, Any], 
                                     output_path: str, form_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced non-widget form filling using text overlay and coordinate mapping.
        
        Args:
            pdf_path: Path to the input PA form
            extracted_data: Extracted referral data
            output_path: Path to save filled form
            form_analysis: Form structure analysis results
            
        Returns:
            Dictionary with filling results
        """
        logger.info("Processing non-widget form with text overlay method")
        
        try:
            # For now, copy the original form and add a note about manual completion
            import shutil
            ensure_directory_exists(os.path.dirname(output_path))
            shutil.copy2(pdf_path, output_path)
            
            # Identify potential fill locations from text analysis
            potential_fields = []
            for element in form_analysis["text_elements"]:
                text = element["text"].lower()
                
                # Look for field patterns
                field_patterns = [
                    (r".*name.*:", "patient_name"),
                    (r".*date.*birth.*:", "dob"),
                    (r".*phone.*:", "phone"),
                    (r".*address.*:", "address"),
                    (r".*diagnosis.*:", "diagnosis"),
                    (r".*medication.*:", "medication"),
                    (r".*insurance.*:", "insurance_carrier"),
                    (r".*prescriber.*:", "prescriber_name")
                ]
                
                for pattern, field_type in field_patterns:
                    if re.match(pattern, text):
                        # Find matching data
                        field_value, confidence = self.field_matcher.find_best_match(
                            field_type, extracted_data
                        )
                        
                        if field_value and confidence > 0.6:
                            potential_fields.append({
                                "field_type": field_type,
                                "value": field_value,
                                "confidence": confidence,
                                "location": element["bbox"],
                                "page": element["page"]
                            })
            
            logger.info(f"Identified {len(potential_fields)} potential fields for text-based form")
            
            return {
                "status": "completed",
                "filled_fields": 0,  # Text overlay not implemented yet
                "filled_count": 0,  # Add for backward compatibility
                "total_fields": len(form_analysis["text_elements"]),
                "fill_rate": 0.0,
                "fill_percentage": 0.0,  # Add for backward compatibility
                "potential_fields": potential_fields,
                "missing_fields": [],  # Add for backward compatibility
                "processing_errors": [],  # Add for backward compatibility
                "note": "Non-widget form copied. Manual completion required or text overlay feature needed."
            }
            
        except Exception as e:
            logger.error(f"Error processing non-widget form: {e}")
            raise

    def _calculate_processing_stats(self) -> Dict[str, Any]:
        """Calculate comprehensive processing statistics."""
        total_attempted = len(self.filled_fields) + len(self.missing_fields)
        
        # Confidence distribution
        confidences = [field["confidence"] for field in self.filled_fields]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Missing field reasons
        missing_reasons = {}
        for field in self.missing_fields:
            reason = field["reason"]
            missing_reasons[reason] = missing_reasons.get(reason, 0) + 1
        
        return {
            "total_fields_attempted": total_attempted,
            "successfully_filled": len(self.filled_fields),
            "missing_fields": len(self.missing_fields),
            "average_confidence": avg_confidence,
            "missing_field_reasons": missing_reasons,
            "field_mapping_attempts": len(self.field_mapping_log)
        }

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            "status": "error",
            "error_message": error_message,
            "filled_fields": 0,
            "filled_count": 0,  # Add for backward compatibility
            "total_fields": 0,
            "fill_rate": 0.0,
            "fill_percentage": 0.0,  # Add for backward compatibility
            "filled_field_details": [],
            "missing_field_details": [],
            "missing_fields": [],  # Add for backward compatibility
            "processing_errors": [error_message]  # Add for backward compatibility
        }

    def save_detailed_report(self, output_dir: str, patient_name: str, 
                           fill_results: Dict[str, Any]) -> bool:
        """
        Save detailed filling report for analysis and debugging.
        
        Args:
            output_dir: Directory to save the report
            patient_name: Patient identifier
            fill_results: Results from form filling process
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            report_path = os.path.join(output_dir, f"{patient_name}_form_filling_report.json")
            
            detailed_report = {
                "patient": patient_name,
                "timestamp": str(pd.Timestamp.now()),
                "fill_results": fill_results,
                "filled_fields": self.filled_fields,
                "missing_fields": self.missing_fields,
                "field_mapping_log": self.field_mapping_log,
                "statistics": self._calculate_processing_stats()
            }
            
            return save_json_safely(detailed_report, report_path)
            
        except Exception as e:
            logger.error(f"Failed to save detailed report: {e}")
            return False

    def generate_missing_fields_report(self, output_path: str) -> bool:
        """
        Generate a detailed report of missing fields for manual review.
        
        Args:
            output_path: Path to save the missing fields report
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("Missing Fields Report\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Total Missing Fields: {len(self.missing_fields)}\n\n")
                
                # Group by reason
                reason_groups = {}
                for field in self.missing_fields:
                    reason = field["reason"]
                    if reason not in reason_groups:
                        reason_groups[reason] = []
                    reason_groups[reason].append(field)
                
                for reason, fields in reason_groups.items():
                    f.write(f"{reason} ({len(fields)} fields):\n")
                    f.write("-" * 30 + "\n")
                    for field in fields:
                        f.write(f"  • {field['field_name']} (Page {field.get('page', 'Unknown')})\n")
                    f.write("\n")
                
                # Suggestions for improvement
                f.write("Suggestions for Improvement:\n")
                f.write("-" * 30 + "\n")
                if reason_groups.get("No matching data found"):
                    f.write("• Consider improving extraction patterns for unmatched fields\n")
                if any("Low confidence" in reason for reason in reason_groups.keys()):
                    f.write("• Review and enhance field matching algorithms\n")
                if reason_groups.get("Fill error"):
                    f.write("• Check PDF form structure and field accessibility\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate missing fields report: {e}")
            return False


# For backward compatibility
PAFormFiller = EnhancedPAFormFiller


def fill_pa_form(pa_pdf_path: str, extracted_data: Dict[str, Any], output_path: str) -> Dict[str, Any]:
    """
    Main function to fill a PA form with extracted data.
    
    Args:
        pa_pdf_path: Path to the PA form PDF
        extracted_data: Dictionary containing extracted referral data
        output_path: Path for the filled PA form
        
    Returns:
        Dictionary containing filling results
    """
    setup_logging()
    
    filler = PAFormFiller()
    result = filler.fill_pa_form(pa_pdf_path, extracted_data, output_path)
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python fill_pa_form.py <pa_pdf_path> <extracted_data_json> <output_path>")
        sys.exit(1)
    
    pa_path = sys.argv[1]
    data_path = sys.argv[2]
    output = sys.argv[3]
    
    # Load extracted data
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading extracted data: {e}")
        sys.exit(1)
    
    # Fill form
    result = fill_pa_form(pa_path, data, output)
    
    print(f"Form filling completed: {result['filled_count']}/{result['total_fields']} fields filled ({result['fill_percentage']:.1f}%)")
    if result['processing_errors']:
        print(f"Errors: {result['processing_errors']}") 