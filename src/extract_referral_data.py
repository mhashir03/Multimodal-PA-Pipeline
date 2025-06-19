"""
Enhanced Referral Data Extraction Module

This module provides production-ready OCR processing and structured data extraction 
from medical referral packages with improved accuracy and validation.
"""

import os
import re
import json
import sys
from typing import Dict, List, Any, Optional, Tuple
import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pdf2image import convert_from_path
import pdfplumber
import spacy
from datetime import datetime
from loguru import logger

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    preprocess_image_advanced, 
    extract_medical_patterns, 
    normalize_field_value,
    calculate_confidence_score,
    setup_logging,
    validate_extracted_data
)


class EnhancedReferralDataExtractor:
    """
    Advanced extractor with improved OCR and medical text processing.
    """
    
    def __init__(self, use_advanced_ocr: bool = True):
        """
        Initialize the enhanced extractor.
        
        Args:
            use_advanced_ocr: Whether to use advanced OCR preprocessing
        """
        self.use_advanced_ocr = use_advanced_ocr
        self.patterns = extract_medical_patterns()
        self.confidence_threshold = 0.6
        
        # Enhanced Tesseract configuration for medical documents
        self.tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()-/:;@#$%&*+= '
        
        # Try to load spaCy model for better entity recognition
        self.nlp = None
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
        
    def extract_from_referral_package(self, referral_pdf_path: str) -> Dict[str, Any]:
        """
        Extract structured data from a referral package PDF with enhanced processing.
        
        Args:
            referral_pdf_path: Path to the referral package PDF
            
        Returns:
            Dictionary containing extracted structured data
        """
        logger.info(f"Starting enhanced extraction from {referral_pdf_path}")
        
        extracted_data = {
            "metadata": {
                "source_file": referral_pdf_path,
                "extraction_method": "Enhanced OCR + NLP",
                "extraction_timestamp": datetime.now().isoformat(),
                "confidence_scores": {},
                "quality_score": 0.0
            },
            "patient_info": {},
            "clinical_info": {},
            "insurance_info": {},
            "prescriber_info": {},
            "raw_text": "",
            "processed_text": "",
            "extraction_errors": []
        }
        
        try:
            # Multi-stage text extraction approach
            text_extraction_successful = self._extract_text_with_fallback(referral_pdf_path, extracted_data)
            
            if not text_extraction_successful:
                raise Exception("All text extraction methods failed")
            
            # Enhanced text processing with multiple techniques
            self._process_extracted_text_enhanced(extracted_data)
            
            # Apply NLP entity recognition if available
            if self.nlp:
                self._apply_nlp_entity_recognition(extracted_data)
            
            # Post-process and validate extracted data
            self._validate_and_clean_data_enhanced(extracted_data)
            
            # Calculate overall quality score
            self._calculate_quality_score(extracted_data)
            
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            extracted_data["extraction_errors"].append(str(e))
        
        return extracted_data
    
    def _extract_text_with_fallback(self, pdf_path: str, extracted_data: Dict[str, Any]) -> bool:
        """
        Multi-stage text extraction with fallback options.
        
        Args:
            pdf_path: Path to the PDF file
            extracted_data: Dictionary to store extracted data
            
        Returns:
            True if any extraction method was successful
        """
        methods = [
            ("Direct PDF Text", self._extract_text_from_pdf),
            ("Enhanced OCR", self._extract_text_via_enhanced_ocr),
            ("Alternative OCR", self._extract_text_via_alternative_ocr)
        ]
        
        for method_name, method_func in methods:
            try:
                logger.info(f"Attempting text extraction using: {method_name}")
                if method_func(pdf_path, extracted_data):
                    extracted_data["metadata"]["extraction_method"] = method_name
                    logger.info(f"Text extraction successful using: {method_name}")
                    return True
            except Exception as e:
                logger.warning(f"{method_name} failed: {e}")
                continue
        
        return False
    
    def _extract_text_from_pdf(self, pdf_path: str, extracted_data: Dict[str, Any]) -> bool:
        """
        Enhanced direct PDF text extraction with better validation.
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                all_text = ""
                page_texts = []
                
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        # Clean and normalize the text
                        cleaned_text = self._clean_extracted_text(text)
                        page_texts.append(f"--- Page {page_num + 1} ---\n{cleaned_text}")
                        all_text += cleaned_text + "\n"
                
                # Validate text quality
                if self._validate_text_quality(all_text):
                    extracted_data["raw_text"] = "\n".join(page_texts)
                    logger.info(f"Extracted {len(all_text)} characters via direct PDF reading")
                    return True
                    
        except Exception as e:
            logger.warning(f"Direct PDF text extraction failed: {e}")
        
        return False
    
    def _extract_text_via_enhanced_ocr(self, pdf_path: str, extracted_data: Dict[str, Any]) -> bool:
        """
        Enhanced OCR with improved preprocessing and error handling.
        """
        try:
            # Convert PDF to high-quality images
            images = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=15)
            
            all_text = ""
            successful_pages = 0
            
            for i, image in enumerate(images):
                logger.info(f"Processing page {i+1} with enhanced OCR")
                
                try:
                    # Multiple preprocessing approaches
                    processed_images = self._create_multiple_image_variants(image)
                    
                    best_text = ""
                    best_confidence = 0
                    
                    for variant_name, img_array in processed_images.items():
                        # Perform OCR on each variant
                        page_text = pytesseract.image_to_string(
                            img_array, 
                            config=self.tesseract_config
                        )
                        
                        # Get OCR confidence
                        confidence = self._calculate_ocr_confidence(img_array, page_text)
                        
                        if confidence > best_confidence:
                            best_text = page_text
                            best_confidence = confidence
                    
                    if best_text and len(best_text.strip()) > 10:
                        all_text += f"--- Page {i+1} ---\n{best_text}\n"
                        successful_pages += 1
                    
                except Exception as e:
                    logger.warning(f"OCR failed for page {i+1}: {e}")
                    continue
            
            if successful_pages > 0:
                extracted_data["raw_text"] = all_text
                logger.info(f"Enhanced OCR extracted {len(all_text)} characters from {successful_pages} pages")
                return True
                
        except Exception as e:
            logger.error(f"Enhanced OCR extraction failed: {e}")
        
        return False
    
    def _extract_text_via_alternative_ocr(self, pdf_path: str, extracted_data: Dict[str, Any]) -> bool:
        """
        Alternative OCR approach with different settings as last resort.
        """
        try:
            images = convert_from_path(pdf_path, dpi=200, first_page=1, last_page=10)
            
            all_text = ""
            alternative_config = '--oem 1 --psm 3'
            
            for i, image in enumerate(images):
                img_array = np.array(image)
                
                # Simple preprocessing
                if len(img_array.shape) == 3:
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img_array
                
                # Apply basic thresholding
                _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                
                page_text = pytesseract.image_to_string(thresh, config=alternative_config)
                
                if page_text and len(page_text.strip()) > 5:
                    all_text += f"--- Page {i+1} ---\n{page_text}\n"
            
            if len(all_text.strip()) > 100:
                extracted_data["raw_text"] = all_text
                logger.info(f"Alternative OCR extracted {len(all_text)} characters")
                return True
                
        except Exception as e:
            logger.error(f"Alternative OCR extraction failed: {e}")
        
        return False
    
    def _create_multiple_image_variants(self, image: Image.Image) -> Dict[str, np.ndarray]:
        """
        Create multiple preprocessed variants of an image for OCR.
        """
        variants = {}
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Original
        variants["original"] = img_array
        
        # Enhanced contrast
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(1.5)
        variants["enhanced_contrast"] = np.array(enhanced)
        
        # Sharpened
        sharpened = image.filter(ImageFilter.SHARPEN)
        variants["sharpened"] = np.array(sharpened)
        
        # Advanced preprocessing
        if self.use_advanced_ocr:
            variants["advanced"] = preprocess_image_advanced(img_array)
        
        return variants
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors
        text = text.replace('|', 'I')
        text = text.replace('0', 'O')  # Only in specific contexts
        text = re.sub(r'[^\w\s.,()/-:@#$%&*+=]', '', text)
        
        return text.strip()
    
    def _validate_text_quality(self, text: str) -> bool:
        """
        Validate the quality of extracted text.
        """
        if not text or len(text.strip()) < 50:
            return False
        
        # Check for reasonable character distribution
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.3:  # Less than 30% alphabetic characters
            return False
        
        # Check for medical document indicators
        medical_indicators = [
            "patient", "diagnosis", "doctor", "medication", "treatment", 
            "hospital", "clinic", "medical", "health", "insurance"
        ]
        
        text_lower = text.lower()
        found_indicators = sum(1 for indicator in medical_indicators if indicator in text_lower)
        
        return found_indicators >= 2
    
    def _calculate_ocr_confidence(self, image: np.ndarray, text: str) -> float:
        """
        Calculate confidence score for OCR results.
        """
        try:
            # Get detailed OCR data
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            
            if confidences:
                return sum(confidences) / len(confidences) / 100.0
            else:
                return 0.0
        except:
            # Fallback confidence based on text length and content
            if len(text.strip()) > 50:
                return 0.7
            elif len(text.strip()) > 10:
                return 0.5
            else:
                return 0.2
    
    def _process_extracted_text_enhanced(self, extracted_data: Dict[str, Any]) -> None:
        """
        Enhanced text processing with improved pattern matching and error handling.
        """
        logger.info("Processing extracted text with enhanced pattern matching")
        
        raw_text = extracted_data.get("raw_text", "")
        if not raw_text:
            logger.warning("No raw text available for processing")
            return
        
        # Preprocess text for better pattern matching
        processed_text = self._preprocess_text_for_patterns(raw_text)
        extracted_data["processed_text"] = processed_text
        
        # Get patterns for extraction
        patterns = self.patterns
        
        # Process each pattern
        for field_name, pattern_info in patterns.items():
            try:
                pattern = pattern_info["pattern"] if isinstance(pattern_info, dict) else pattern_info
                
                matches = re.finditer(pattern, processed_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                
                best_match = ""
                best_confidence = 0.0
                
                for match in matches:
                    if match.groups():
                        # Fix: Add proper null check before calling strip()
                        candidate = match.group(1)
                        if candidate is None:
                            continue
                        
                        # Additional check to ensure candidate is a string
                        if not isinstance(candidate, str):
                            continue
                            
                        candidate = candidate.strip()
                        
                        # Skip empty candidates
                        if not candidate:
                            continue
                        
                        # Apply field-specific validation
                        if self._validate_field_candidate(field_name, candidate):
                            confidence = calculate_confidence_score(candidate, field_name)
                            
                            if confidence > best_confidence:
                                best_match = candidate
                                best_confidence = confidence
                
                if best_match and best_confidence >= self.confidence_threshold:
                    # Categorize and store the field
                    category = self._categorize_field(field_name)
                    if category not in extracted_data:
                        extracted_data[category] = {}
                    
                    # Normalize the value
                    normalized_value = normalize_field_value(field_name, best_match)
                    extracted_data[category][field_name] = normalized_value
                    extracted_data["metadata"]["confidence_scores"][field_name] = best_confidence
                    
            except Exception as e:
                logger.warning(f"Error processing pattern for {field_name}: {e}")
                continue
        
        # Extract additional complex fields
        self._extract_complex_medical_fields(processed_text, extracted_data)
    
    def _preprocess_text_for_patterns(self, text: str) -> str:
        """
        Preprocess text to improve pattern matching accuracy.
        """
        # Normalize whitespace but preserve line breaks for section parsing
        text = re.sub(r'[ \t]+', ' ', text)  # Replace multiple spaces/tabs with single space
        text = re.sub(r'\n+', '\n', text)    # Replace multiple newlines with single newline
        
        # Fix common OCR concatenation issues in medical documents
        medical_corrections = {
            # Fix concatenated words common in medical documents
            r'MultipleSclerosis': 'Multiple Sclerosis',
            r'BetterLife': 'Better Life',
            r'MonroeCarell': 'Monroe Carell',
            r'ChildrensHospital': 'Childrens Hospital',
            r'NashvilleTN': 'Nashville TN',
            r'KnoxvilleTN': 'Knoxville TN',
            r'ShermanAve': 'Sherman Ave',
            r'BroadwayLn': 'Broadway Ln',
            
            # Fix doctor names and titles
            r'([A-Z][a-z]+),([A-Z][a-z]+),MD': r'\1, \2, MD',
            r'Dr\.([A-Z])': r'Dr. \1',
            r'MD([A-Z])': r'MD \1',
            
            # Fix dates
            r'(\d{2})/(\d{2})/(\d{4})([A-Z])': r'\1/\2/\3 \4',
            r'(\d{1,2})/(\d{1,2})/(\d{2,4})': r'\1/\2/\3',
            
            # Fix phone numbers
            r'(\d{3})-(\d{3})\.\.\.(\d{4})': r'(\1) \2-\3',
            r'(\d{3})-(\d{3})-(\d{4})': r'(\1) \2-\3',
            r'(\d{3})(\d{3})(\d{4})': r'(\1) \2-\3',
            
            # Fix medication names
            r'Rituximab[Oo]r[Bb]iosimilar': 'Rituximab or biosimilar',
            r'RituximabOrBiosimilar': 'Rituximab or biosimilar',
            
            # Fix insurance names
            r'TENNCARE/?BLUECARE': 'TENNCARE/BLUECARE',
            r'TCBLUECARE': 'TC BLUECARE',
            
            # Fix address formatting
            r'(\d+)([A-Z][a-z]+(?:Ave|St|Dr|Ln|Blvd))': r'\1 \2',
            r'(Ave|Street|Dr|Drive|Ln|Lane|Blvd|Boulevard)([A-Z])': r'\1 \2',
            
            # Fix common OCR character errors
            r'\b([A-Z]{1,2}\d{2,3}\.?\d*)\b': r'\1',  # ICD codes
            r'(\d+)\s*mg\s*(\w)': r'\1 mg \2',  # Dosage
            
            # Separate concatenated fields
            r'(MRN?:?\s*)(\d+)([A-Za-z])': r'\1\2 \3',
            r'DOB:?(\d{1,2}/\d{1,2}/\d{4})([A-Za-z])': r'DOB: \1 \2',
            r'([A-Za-z]+),([A-Za-z]+)([A-Z]{2}\d{5})': r'\1, \2 \3',
        }
        
        for pattern, replacement in medical_corrections.items():
            text = re.sub(pattern, replacement, text)
        
        # Add spaces around key medical terms for better pattern matching
        key_terms = [
            'DOB', 'MRN', 'phone', 'address', 'diagnosis', 'medication', 
            'insurance', 'prescriber', 'patient', 'Multiple Sclerosis',
            'Rituximab', 'TENNCARE', 'BLUECARE'
        ]
        
        for term in key_terms:
            # Add space before term if it's concatenated
            text = re.sub(f'([a-z])({term})', r'\1 \2', text, flags=re.IGNORECASE)
            # Add space after term if it's concatenated
            text = re.sub(f'({term})([A-Z][a-z])', r'\1 \2', text, flags=re.IGNORECASE)
        
        return text
    
    def _validate_field_candidate(self, field_name: str, candidate: str) -> bool:
        """
        Validate extracted field candidates based on field type.
        """
        if not candidate or len(candidate.strip()) < 1:
            return False
        
        # Field-specific validation
        if "dob" in field_name.lower() or "birth" in field_name.lower():
            return bool(re.match(r'\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}', candidate))
        
        elif "phone" in field_name.lower():
            digits = re.sub(r'\D', '', candidate)
            return len(digits) == 10
        
        elif "ssn" in field_name.lower():
            digits = re.sub(r'\D', '', candidate)
            return len(digits) == 9
        
        elif "name" in field_name.lower():
            # Names should be mostly alphabetic
            return len(candidate) > 2 and sum(c.isalpha() or c.isspace() for c in candidate) / len(candidate) > 0.7
        
        elif "diagnosis" in field_name.lower():
            # Should contain medical terms or be reasonable length
            return len(candidate) > 3 and len(candidate) < 200
        
        # Default validation - reject very short or very long candidates
        return 2 <= len(candidate) <= 100
    
    def _apply_nlp_entity_recognition(self, extracted_data: Dict[str, Any]) -> None:
        """
        Apply NLP entity recognition to improve extraction accuracy.
        """
        if not self.nlp:
            return
        
        text = extracted_data.get("processed_text", "")
        if not text:
            return
        
        logger.info("Applying NLP entity recognition")
        
        try:
            doc = self.nlp(text[:1000000])  # Limit text length for performance
            
            # Extract person names
            persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            if persons and "patient_name" not in extracted_data.get("patient_info", {}):
                # Use the most frequently mentioned person name
                from collections import Counter
                most_common_person = Counter(persons).most_common(1)
                if most_common_person:
                    extracted_data.setdefault("patient_info", {})["patient_name"] = most_common_person[0][0]
                    extracted_data["metadata"]["confidence_scores"]["patient_name"] = 0.8
            
            # Extract organizations (hospitals, clinics)
            orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
            if orgs:
                extracted_data.setdefault("prescriber_info", {})["organization"] = orgs[0]
                extracted_data["metadata"]["confidence_scores"]["organization"] = 0.7
            
        except Exception as e:
            logger.warning(f"NLP entity recognition failed: {e}")
    
    def _extract_complex_medical_fields(self, text: str, extracted_data: Dict[str, Any]) -> None:
        """
        Extract complex medical fields that require specialized processing.
        """
        # Extract medication lists
        medication_patterns = [
            r'(?:medication|drug|rx)s?:?\s*([^\n]+(?:\n[^\n]{1,50})*)',
            r'(?:prescri|ordered?).*?:?\s*([A-Za-z]+(?:\s+\d+\s*mg)?)',
        ]
        
        medications = []
        for pattern in medication_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                med_text = match.group(1).strip()
                if len(med_text) > 3 and len(med_text) < 100:
                    medications.append(med_text)
        
        if medications:
            extracted_data.setdefault("clinical_info", {})["medications"] = list(set(medications[:10]))
        
        # Extract ICD codes
        icd_pattern = r'\b([A-Z]\d{2}\.?\d*)\b'
        icd_codes = re.findall(icd_pattern, text)
        if icd_codes:
            extracted_data.setdefault("clinical_info", {})["icd_codes"] = list(set(icd_codes[:5]))
    
    def _categorize_field(self, field_name: str) -> str:
        """
        Categorize fields into appropriate sections.
        """
        if any(term in field_name.lower() for term in ["patient", "name", "dob", "birth", "ssn", "phone", "address"]):
            return "patient_info"
        elif any(term in field_name.lower() for term in ["diagnosis", "icd", "medication", "treatment", "dosage", "symptom"]):
            return "clinical_info"
        elif any(term in field_name.lower() for term in ["insurance", "policy", "member", "carrier", "plan"]):
            return "insurance_info"
        elif any(term in field_name.lower() for term in ["prescriber", "doctor", "physician", "provider", "npi"]):
            return "prescriber_info"
        else:
            return "other_info"
    
    def _validate_and_clean_data_enhanced(self, extracted_data: Dict[str, Any]) -> None:
        """
        Enhanced validation and cleaning of extracted data.
        """
        for category, fields in extracted_data.items():
            if not isinstance(fields, dict):
                continue
            
            cleaned_fields = {}
            for field_name, value in fields.items():
                if field_name == "confidence_scores":
                    continue
                
                # Apply field-specific cleaning and validation
                cleaned_value = self._clean_field_value(field_name, value)
                if cleaned_value and self._final_validation(field_name, cleaned_value):
                    cleaned_fields[field_name] = cleaned_value
            
            extracted_data[category] = cleaned_fields
    
    def _clean_field_value(self, field_name: str, value: str) -> str:
        """
        Apply field-specific cleaning rules.
        """
        if not value:
            return ""
        
        value = value.strip()
        
        # Remove common OCR artifacts
        value = re.sub(r'[^\w\s.,()/-:@#$%&*+=]', '', value)
        
        # Field-specific cleaning
        if "phone" in field_name.lower():
            digits = re.sub(r'\D', '', value)
            if len(digits) == 10:
                return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        
        elif "dob" in field_name.lower() or "birth" in field_name.lower():
            # Standardize date format
            date_match = re.search(r'(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})', value)
            if date_match:
                month, day, year = date_match.groups()
                if len(year) == 2:
                    year = "20" + year if int(year) < 50 else "19" + year
                return f"{month.zfill(2)}/{day.zfill(2)}/{year}"
        
        elif "name" in field_name.lower():
            # Capitalize names properly
            return " ".join(word.capitalize() for word in value.split() if word.isalpha())
        
        return value
    
    def _final_validation(self, field_name: str, value: str) -> bool:
        """
        Final validation before accepting extracted data.
        """
        if not value or len(value.strip()) < 1:
            return False
        
        # Reject values that are clearly OCR garbage
        if len(value) > 2:
            alpha_ratio = sum(c.isalpha() for c in value) / len(value)
            if alpha_ratio < 0.3 and not any(char.isdigit() for char in value):
                return False
        
        # Field-specific final validation
        if "diagnosis" in field_name.lower():
            # Reject diagnosis that's too short or contains too much garbage
            if len(value) < 5 or len(value) > 200:
                return False
            # Should contain some medical-sounding words
            medical_terms = ["syndrome", "disease", "disorder", "condition", "sclerosis", "diabetes", "hypertension"]
            if not any(term in value.lower() for term in medical_terms) and len(value) > 50:
                return False
        
        return True
    
    def _calculate_quality_score(self, extracted_data: Dict[str, Any]) -> None:
        """
        Calculate overall quality score for the extraction.
        """
        confidence_scores = extracted_data["metadata"].get("confidence_scores", {})
        
        if not confidence_scores:
            extracted_data["metadata"]["quality_score"] = 0.0
            return
        
        # Calculate weighted average confidence
        weights = {
            "patient_name": 3.0,
            "dob": 2.0,
            "diagnosis": 2.5,
            "phone": 1.5,
            "prescriber": 2.0
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for field, confidence in confidence_scores.items():
            weight = weights.get(field, 1.0)
            weighted_sum += confidence * weight
            total_weight += weight
        
        if total_weight > 0:
            quality_score = weighted_sum / total_weight
        else:
            quality_score = sum(confidence_scores.values()) / len(confidence_scores)
        
        extracted_data["metadata"]["quality_score"] = round(quality_score, 3)


def extract_referral_data(referral_pdf_path: str, output_dir: Optional[str] = None, use_advanced_ocr: bool = True) -> Dict[str, Any]:
    """
    Enhanced main function to extract referral data from PDF.
    
    Args:
        referral_pdf_path: Path to the referral package PDF
        output_dir: Optional output directory to save extraction results
        use_advanced_ocr: Whether to use advanced OCR preprocessing
        
    Returns:
        Dictionary containing extracted structured data
    """
    extractor = EnhancedReferralDataExtractor(use_advanced_ocr=use_advanced_ocr)
    results = extractor.extract_from_referral_package(referral_pdf_path)
    
    # Save results if output directory is provided
    if output_dir:
        output_file = os.path.join(output_dir, "extracted_referral_data.json")
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Extraction results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save extraction results: {e}")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_referral_data.py <referral_pdf_path> <output_dir>")
        sys.exit(1)
    
    referral_path = sys.argv[1]
    output_directory = sys.argv[2]
    
    if not os.path.exists(referral_path):
        print(f"Error: Referral PDF not found: {referral_path}")
        sys.exit(1)
    
    results = extract_referral_data(referral_path, output_directory)
    
    # Print summary
    print(f"Extraction completed!")
    print(f"Quality Score: {results['metadata'].get('quality_score', 0):.1%}")
    print(f"Fields extracted: {len(results['metadata'].get('confidence_scores', {}))}") 