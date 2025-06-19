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
            elif form_analysis["form_type"] == "static_overlay":
                result = self._fill_static_form_with_overlay(
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
            "text_elements": [],
            "is_static_form": False,
            "form_text": ""
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
                    all_text = ""
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
                        
                        # Extract full text for static form detection
                        page_text = page.extract_text()
                        if page_text:
                            all_text += page_text + "\n"
                    
                    analysis["form_text"] = all_text
                    
                    # If no fillable fields found but text exists, it's likely a static form
                    if not analysis["has_fillable_fields"] and all_text:
                        analysis["is_static_form"] = True
                        analysis["form_type"] = "static_overlay"
                        # Detect potential form fields from text patterns
                        self._detect_static_form_fields(analysis)
                        
            except Exception as e:
                logger.warning(f"pdfplumber analysis failed: {e}")
            
            # Final determination
            if not analysis["has_fillable_fields"] and not analysis["is_static_form"]:
                analysis["form_type"] = "unknown"
            
            logger.info(f"Form analysis complete: {analysis['field_count']} fields, Type: {analysis['form_type']}")
            
        except Exception as e:
            logger.error(f"Form structure analysis failed: {e}")
        
        return analysis
    
    def _detect_static_form_fields(self, analysis: Dict[str, Any]) -> None:
        """
        Detect potential form fields in static PDF forms by analyzing text patterns.
        """
        form_text = analysis.get("form_text", "")
        if not form_text:
            return
        
        # Common form field patterns
        field_patterns = {
            "patient_name": [
                r"Patient\s*Name[:\s]*_+",
                r"Name[:\s]*_+",
                r"Patient[:\s]*_+"
            ],
            "date_of_birth": [
                r"Date\s*of\s*Birth[:\s]*_+",
                r"DOB[:\s]*_+",
                r"Birth\s*Date[:\s]*_+"
            ],
            "diagnosis": [
                r"Diagnosis[:\s]*_+",
                r"Primary\s*Diagnosis[:\s]*_+",
                r"Condition[:\s]*_+"
            ],
            "prescriber_name": [
                r"Prescriber[:\s]*Name[:\s]*_+",
                r"Physician[:\s]*Name[:\s]*_+",
                r"Doctor[:\s]*_+"
            ],
            "phone": [
                r"Phone[:\s]*_+",
                r"Telephone[:\s]*_+",
                r"Contact[:\s]*Number[:\s]*_+"
            ],
            "insurance": [
                r"Insurance[:\s]*_+",
                r"Plan[:\s]*Name[:\s]*_+",
                r"Carrier[:\s]*_+"
            ]
        }
        
        detected_fields = []
        for field_name, patterns in field_patterns.items():
            for pattern in patterns:
                if re.search(pattern, form_text, re.IGNORECASE):
                    detected_fields.append(field_name)
                    break
        
        analysis["fields"] = detected_fields
        analysis["field_count"] = len(detected_fields)
        analysis["has_fillable_fields"] = len(detected_fields) > 0

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
            # Use pdfplumber to get field information and PyPDF2 to fill
            import fitz  # PyMuPDF as fallback
            
            # Try PyPDF2 first for form filling
            try:
                from PyPDF2 import PdfReader, PdfWriter
                from PyPDF2.generic import BooleanObject, NameObject, TextStringObject
                
                reader = PdfReader(pdf_path)
                writer = PdfWriter()
                
                # Get form fields from the reader
                if reader.get_form_text_fields():
                    form_fields = reader.get_form_text_fields()
                    logger.info(f"Found {len(form_fields)} fillable fields using PyPDF2")
                    
                    # Fill each field
                    for field_name, current_value in form_fields.items():
                        # Find best match using enhanced field matcher
                        field_value, confidence = self.field_matcher.find_best_match(
                            field_name, extracted_data
                        )
                        
                        if field_value and confidence > 0.5:
                            # Normalize the value
                            normalized_value = normalize_field_value(field_name, field_value)
                            
                            # Update the form field
                            writer.update_page_form_field_values(
                                writer.pages[0], {field_name: normalized_value}
                            )
                            filled_count += 1
                            
                            self.filled_fields.append({
                                "field_name": field_name,
                                "value": normalized_value,
                                "confidence": confidence,
                                "page": 0
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
                        else:
                            self.missing_fields.append({
                                "field_name": field_name,
                                "reason": "No matching data found" if not field_value else f"Low confidence ({confidence:.2f})",
                                "page": 0
                            })
                    
                    # Copy all pages to writer
                    for page in reader.pages:
                        writer.add_page(page)
                    
                    # Save the filled form
                    ensure_directory_exists(os.path.dirname(output_path))
                    with open(output_path, 'wb') as output_file:
                        writer.write(output_file)
                    
                    logger.info(f"Filled {filled_count}/{total_fields} fields using PyPDF2")
                    
                else:
                    # Fallback to manual field detection and filling
                    logger.info("No form fields detected by PyPDF2, trying manual approach")
                    return self._fill_form_manual_approach(pdf_path, extracted_data, output_path, form_analysis)
                    
            except Exception as e:
                logger.warning(f"PyPDF2 form filling failed: {e}")
                # Try alternative approach with PyMuPDF
                return self._fill_form_with_pymupdf(pdf_path, extracted_data, output_path, form_analysis)
                
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

    def _fill_form_with_pymupdf(self, pdf_path: str, extracted_data: Dict[str, Any], 
                               output_path: str, form_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fill form using PyMuPDF (fitz) as alternative approach.
        """
        try:
            import fitz
            
            doc = fitz.open(pdf_path)
            filled_count = 0
            total_fields = 0
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                widgets = page.widgets()
                
                for widget in widgets:
                    total_fields += 1
                    field_name = widget.field_name
                    
                    if field_name:
                        # Find best match using enhanced field matcher
                        field_value, confidence = self.field_matcher.find_best_match(
                            field_name, extracted_data
                        )
                        
                        if field_value and confidence > 0.5:
                            # Normalize the value
                            normalized_value = normalize_field_value(field_name, field_value)
                            
                            # Fill the widget
                            widget.field_value = normalized_value
                            widget.update()
                            filled_count += 1
                            
                            self.filled_fields.append({
                                "field_name": field_name,
                                "value": normalized_value,
                                "confidence": confidence,
                                "page": page_num
                            })
                            
                            logger.debug(f"Filled field '{field_name}' with '{normalized_value}' "
                                       f"(confidence: {confidence:.2f})")
                        else:
                            self.missing_fields.append({
                                "field_name": field_name,
                                "reason": "No matching data found" if not field_value else f"Low confidence ({confidence:.2f})",
                                "page": page_num
                            })
            
            # Save the filled form
            ensure_directory_exists(os.path.dirname(output_path))
            doc.save(output_path)
            doc.close()
            
            logger.info(f"Filled {filled_count}/{total_fields} fields using PyMuPDF")
            
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
            
        except ImportError:
            logger.warning("PyMuPDF not available, falling back to manual approach")
            return self._fill_form_manual_approach(pdf_path, extracted_data, output_path, form_analysis)
        except Exception as e:
            logger.error(f"PyMuPDF form filling failed: {e}")
            return self._fill_form_manual_approach(pdf_path, extracted_data, output_path, form_analysis)

    def _fill_form_manual_approach(self, pdf_path: str, extracted_data: Dict[str, Any], 
                                  output_path: str, form_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Manual approach to fill forms when automatic methods fail.
        """
        logger.info("Using manual form filling approach")
        
        try:
            # For now, copy the original and create a report of what would be filled
            import shutil
            ensure_directory_exists(os.path.dirname(output_path))
            shutil.copy2(pdf_path, output_path)
            
            # Create a mapping report
            potential_fills = []
            for field_name in form_analysis.get("fields", []):
                field_value, confidence = self.field_matcher.find_best_match(
                    field_name, extracted_data
                )
                
                if field_value and confidence > 0.5:
                    normalized_value = normalize_field_value(field_name, field_value)
                    potential_fills.append({
                        "field_name": field_name,
                        "value": normalized_value,
                        "confidence": confidence
                    })
            
            logger.info(f"Manual approach identified {len(potential_fills)} potential field fills")
            
            return {
                "status": "completed",
                "filled_fields": 0,  # Manual approach doesn't actually fill
                "filled_count": 0,
                "total_fields": len(form_analysis.get("fields", [])),
                "fill_rate": 0.0,
                "fill_percentage": 0.0,
                "filled_field_details": [],
                "missing_field_details": self.missing_fields,
                "missing_fields": self.missing_fields,
                "processing_errors": [],
                "potential_fills": potential_fills,
                "note": "Manual approach used - form copied with potential fill data identified"
            }
            
        except Exception as e:
            logger.error(f"Manual form filling approach failed: {e}")
            raise

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

    def _fill_static_form_with_overlay(self, pa_form_path, extracted_data, output_path, form_analysis):
        """Fill static PDF forms by overlaying text at detected field positions."""
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            from PyPDF2 import PdfReader, PdfWriter
            import io
            
            logger.info(f"Filling static form with overlay: {pa_form_path}")
            
            # Read the original PDF
            reader = PdfReader(pa_form_path)
            writer = PdfWriter()
            
            # Process each page
            for page_num, page in enumerate(reader.pages):
                # Create overlay for this page
                overlay_buffer = io.BytesIO()
                overlay_canvas = canvas.Canvas(overlay_buffer, pagesize=letter)
                
                # Get page dimensions
                page_width = float(page.mediabox.width)
                page_height = float(page.mediabox.height)
                
                # Set font
                overlay_canvas.setFont("Helvetica", 10)
                
                # Add text overlays based on detected fields
                fields_filled = 0
                for field_info in form_analysis.get("field_coordinates", []):
                    if field_info["page"] == page_num:
                        field_name = field_info["name"]
                        x, y = field_info["x"], field_info["y"]
                        
                        # Map field name to extracted data
                        value = self._map_field_to_data(field_name, extracted_data)
                        if value:
                            # Adjust coordinates for PDF coordinate system
                            pdf_x = x
                            pdf_y = page_height - y  # Flip Y coordinate
                            
                            # Add text to overlay
                            overlay_canvas.drawString(pdf_x, pdf_y, str(value))
                            fields_filled += 1
                            logger.debug(f"Added overlay text '{value}' at ({pdf_x}, {pdf_y}) for field '{field_name}'")
                
                # If no specific coordinates found, use fallback positioning
                if fields_filled == 0:
                    logger.info("No specific field coordinates found, using fallback positioning")
                    self._add_fallback_text_overlay(overlay_canvas, extracted_data, page_width, page_height)
                    fields_filled = len([v for v in extracted_data.values() if v])
                
                # Finalize overlay
                overlay_canvas.save()
                overlay_buffer.seek(0)
                
                # Merge overlay with original page
                overlay_reader = PdfReader(overlay_buffer)
                if overlay_reader.pages:
                    page.merge_page(overlay_reader.pages[0])
                
                writer.add_page(page)
            
            # Write the result
            with open(output_path, 'wb') as output_file:
                writer.write(output_file)
            
            logger.info(f"Static form filled successfully with {fields_filled} fields")
            return {
                "success": True,
                "fields_filled": fields_filled,
                "method": "static_overlay",
                "output_path": output_path
            }
            
        except Exception as e:
            logger.error(f"Error filling static form with overlay: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "method": "static_overlay"
            }
    
    def _add_fallback_text_overlay(self, canvas, extracted_data, page_width, page_height):
        """Add text overlay using fallback positioning when specific coordinates aren't available."""
        # Define common field positions for medical forms
        field_positions = {
            "patient_name": (100, page_height - 150),
            "date_of_birth": (100, page_height - 180),
            "medical_record_number": (100, page_height - 210),
            "phone_number": (100, page_height - 240),
            "diagnosis": (100, page_height - 300),
            "primary_diagnosis": (100, page_height - 330),
            "provider_name": (100, page_height - 400),
            "provider_phone": (100, page_height - 430),
            "insurance_id": (100, page_height - 460),
            "group_number": (100, page_height - 490)
        }
        
        for field_name, (x, y) in field_positions.items():
            value = extracted_data.get(field_name)
            if value:
                canvas.drawString(x, y, f"{field_name.replace('_', ' ').title()}: {value}")
    
    def _map_field_to_data(self, field_name, extracted_data):
        """Map form field names to extracted data keys."""
        # Create mapping between form field names and extracted data keys
        field_mapping = {
            # Patient information
            "patient_name": ["patient_name", "name", "patient"],
            "first_name": ["first_name", "patient_name"],
            "last_name": ["last_name", "patient_name"],
            "date_of_birth": ["date_of_birth", "dob", "birth_date"],
            "medical_record_number": ["medical_record_number", "mrn", "record_number"],
            "phone_number": ["phone_number", "phone", "contact_number"],
            
            # Medical information
            "diagnosis": ["diagnosis", "primary_diagnosis", "condition"],
            "primary_diagnosis": ["primary_diagnosis", "diagnosis"],
            "icd_code": ["icd_code", "diagnosis_code"],
            
            # Provider information
            "provider_name": ["provider_name", "doctor_name", "physician"],
            "provider_phone": ["provider_phone", "doctor_phone"],
            "provider_npi": ["provider_npi", "npi"],
            
            # Insurance information
            "insurance_id": ["insurance_id", "member_id", "policy_number"],
            "group_number": ["group_number", "group_id"],
            "insurance_name": ["insurance_name", "insurance_company"]
        }
        
        # Normalize field name
        field_key = field_name.lower().replace(" ", "_").replace("-", "_")
        
        # Try direct match first
        if field_key in extracted_data:
            return extracted_data[field_key]
        
        # Try mapped keys
        if field_key in field_mapping:
            for mapped_key in field_mapping[field_key]:
                if mapped_key in extracted_data and extracted_data[mapped_key]:
                    return extracted_data[mapped_key]
        
        # Try partial matches
        for data_key, value in extracted_data.items():
            if value and (field_key in data_key or data_key in field_key):
                return value
        
        return None


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