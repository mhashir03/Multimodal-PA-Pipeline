"""
Enhanced Utility functions for the Prior Authorization (PA) form filling pipeline.

This module provides production-ready helper functions for medical document processing,
advanced OCR preprocessing, and intelligent field matching.
"""

import os
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
from loguru import logger
import cv2
import numpy as np
from PIL import Image
from difflib import SequenceMatcher
from collections import Counter


def setup_logging(log_file: Optional[str] = None):
    """
    Set up logging configuration for the pipeline.
    
    Args:
        log_file: Optional path to log file. If None, logs only to console.
        
    Returns:
        Logger instance
    """
    logger.remove()  # Remove default handler
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    if log_file:
        ensure_directory_exists(os.path.dirname(log_file) if os.path.dirname(log_file) else ".")
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB"
        )
    
    return logger


def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory to create
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def validate_input_files(patient_dir: str) -> Tuple[bool, List[str]]:
    """
    Validate that required input files exist for a patient.
    
    Args:
        patient_dir: Path to the patient's input directory
        
    Returns:
        Tuple of (is_valid, list_of_missing_files)
    """
    required_files = ["PA.pdf", "referral_package.pdf"]
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(patient_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    return len(missing_files) == 0, missing_files


def preprocess_image_advanced(image: np.ndarray) -> np.ndarray:
    """
    Advanced image preprocessing for better OCR results.
    """
    if len(image.shape) == 3:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Noise reduction using bilateral filter
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(enhanced, (1, 1), 0)
    
    # Adaptive thresholding for better text/background separation
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Morphological operations to clean up the image
    kernel = np.ones((1, 1), np.uint8)
    
    # Opening operation (erosion followed by dilation) to remove noise
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Closing operation (dilation followed by erosion) to fill gaps
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    # Edge enhancement for sharper text
    kernel_sharpen = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
    sharpened = cv2.filter2D(closed, -1, kernel_sharpen)
    
    # Ensure the image is binary (0 or 255)
    _, final = cv2.threshold(sharpened, 127, 255, cv2.THRESH_BINARY)
    
    return final


def extract_medical_patterns() -> Dict[str, Any]:
    """
    Return enhanced regex patterns for extracting medical information.
    
    Returns:
        Dictionary of field names to pattern information
    """
    patterns = {
        # Patient Information - Enhanced patterns based on actual medical documents
        "patient_name": {
            "pattern": r"(?:patient[:\s]*name[:\s]*|patient[:\s]*|name[:\s]*)[:]?\s*([A-Za-z]+[,]?\s*[A-Za-z]+(?:[A-Za-z\s,.-]{0,30})?)",
            "weight": 3.0
        },
        "dob": {
            "pattern": r"(?:DOB|date[:\s]*of[:\s]*birth|birth[:\s]*date|born)[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            "weight": 2.5
        },
        "patient_dob": {
            "pattern": r"(?:MR[#\s]*\d+[)]?)DOB[:\s]*(\d{1,2}/\d{1,2}/\d{4})",
            "weight": 3.0
        },
        "mrn": {
            "pattern": r"(?:MRN|MR[#\s]*|Medical[:\s]*Record)[:\s]*(\d{6,12})",
            "weight": 2.5
        },
        "ssn": {
            "pattern": r"(?:SSN|social[:\s]*security)[:\s]*(\d{3}[-\s]?\d{2}[-\s]?\d{4})",
            "weight": 2.0
        },
        "phone": {
            "pattern": r"(?:phone|tel|telephone|cell|mobile|primary[:\s]*phone)[:\s]*(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})",
            "weight": 1.5
        },
        "address": {
            "pattern": r"(?:address|addr)[:\s]*(\d+\s+[A-Za-z\s]+(?:St|Street|Ave|Avenue|Rd|Road|Dr|Drive|Blvd|Boulevard|Ln|Lane)[A-Za-z\s,\d]*)",
            "weight": 1.5
        },
        "patient_address": {
            "pattern": r"(\d+\s*[A-Za-z\s]+(?:Ave|Avenue|St|Street|Dr|Drive|Ln|Lane|Blvd)(?:\s*APT\s*[A-Z\d]+)?)\s*(?:City:|[A-Za-z]+,?\s*[A-Z]{2}\s*\d{5})",
            "weight": 2.0
        },
        
        # Clinical Information - Enhanced for medical records
        "diagnosis": {
            "pattern": r"(?:diagnosis|dx|condition|disease|principal[:\s]*problem)[:\s]*([A-Za-z][A-Za-z\s,.-]{5,100}?)(?:\n|$|\.|\()",
            "weight": 3.0
        },
        "primary_diagnosis": {
            "pattern": r"(?:primary[:\s]*diagnosis|main[:\s]*diagnosis|principal[:\s]*problem)[:\s]*([A-Za-z][A-Za-z\s,.-]{5,100}?)(?:\n|$|\.|\()",
            "weight": 3.0
        },
        "multiple_sclerosis": {
            "pattern": r"(Multiple\s*sclerosis(?:\s*in\s*pediatric\s*patient)?(?:\s*\([^)]+\))?)",
            "weight": 3.5
        },
        "icd_code": {
            "pattern": r"(?:ICD|code)[:\s-]*([A-Z]\d{2}\.?\d*)|(?:\(CMS/HCC\)\s*\()([A-Z]\d{2,3}\.?\d*)\)|([A-Z]\d{2,3}\.?\d*)\s*(?=\s*[A-Z][A-Z]|$)",
            "weight": 2.0
        },
        "medication": {
            "pattern": r"(?:medication|drug|prescription|rx)[:\s]*([A-Za-z][A-Za-z\d\s,.-]{2,50}?)(?:\n|$|,)|(?:Rituximab(?:\s*or\s*biosimilar)?(?:\s*days?\s*\d+)?)",
            "weight": 2.5
        },
        "rituximab": {
            "pattern": r"(Rituximab(?:\s*[Oo]r\s*[Bb]iosimilar)?(?:\s*[Dd]ays?\s*\d+,?\s*\d*)?(?:\s*[Ll]oad\s*[Tt]hen\s*[Oo]nce\s*[Ee]very\s*\d+\s*[Ww]eeks)?)",
            "weight": 3.0
        },
        "dosage": {
            "pattern": r"(?:dosage|dose|strength)[:\s]*(\d+\s*(?:mg|mcg|g|ml|units?)[A-Za-z\s\d,.-]*?)(?:\n|$|,)",
            "weight": 2.0
        },
        "frequency": {
            "pattern": r"(?:frequency|freq|take)[:\s]*(\d+\s*(?:times?|x)\s*(?:daily|day|week|month)|(?:daily|weekly|monthly|BID|TID|QID))",
            "weight": 1.5
        },
        
        # Insurance Information - Enhanced for actual medical forms
        "insurance_carrier": {
            "pattern": r"(?:insurance|carrier|plan|payor)[:\s]*([A-Za-z][A-Za-z\s&,.-]{2,50}?)(?:\n|$|,)",
            "weight": 2.0
        },
        "tenncare": {
            "pattern": r"(TENNCARE[/]?BLUECARE(?:\s*NO\s*COPAY)?)",
            "weight": 2.5
        },
        "policy_number": {
            "pattern": r"(?:policy|member[:\s]*id|plan[:\s]*id|subscriber[:\s]*id)[:\s]*([A-Za-z0-9-]{5,20})",
            "weight": 2.0
        },
        "subscriber_id": {
            "pattern": r"(?:ID\s+Name|Subscriber\s*ID)[:\s]*([A-Z0-9]{8,15})",
            "weight": 2.0
        },
        "group_number": {
            "pattern": r"(?:group[:\s]*number|group[:\s]*id)[:\s]*([A-Za-z0-9-]{3,20})",
            "weight": 1.5
        },
        
        # Prescriber Information - Enhanced for medical records
        "prescriber_name": {
            "pattern": r"(?:prescriber|doctor|physician|provider|dr\.?|attending|signed[:\s]*by)[:\s]*([A-Za-z][A-Za-z\s,.-]{5,40}?)(?:\n|$|,|MD|DO)",
            "weight": 2.5
        },
        "attending_physician": {
            "pattern": r"(?:Attending|Last\s*attending)[:\s]*([A-Za-z]+,\s*[A-Za-z]+,?\s*MD)",
            "weight": 2.5
        },
        "prescriber_npi": {
            "pattern": r"(?:NPI)[:\s]*(\d{10})",
            "weight": 2.0
        },
        "prescriber_phone": {
            "pattern": r"(?:prescriber[:\s]*phone|provider[:\s]*phone|doctor[:\s]*phone)[:\s]*(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})",
            "weight": 1.5
        },
        "clinic_name": {
            "pattern": r"(?:clinic|hospital|center|facility)[:\s]*([A-Za-z][A-Za-z\s&,.-]{5,60}?)(?:\n|$|,)",
            "weight": 1.5
        },
        "better_life_ms_center": {
            "pattern": r"(Better\s*Life\s*Multiple\s*Sclerosis\s*Center)",
            "weight": 2.5
        },
        "monroe_carell": {
            "pattern": r"(Monroe\s*Carell\s*Jr\.?\s*Children\'?s\s*Hospital)",
            "weight": 2.5
        },
        
        # Contact Information
        "emergency_contact": {
            "pattern": r"(?:emergency[:\s]*contact|contact[:\s]*name)[:\s]*([A-Za-z]+,\s*[A-Za-z]+)",
            "weight": 2.0
        },
        "relationship": {
            "pattern": r"(?:relation|relationship)[:\s]*([A-Za-z]+)",
            "weight": 1.5
        }
    }
    return patterns


def normalize_field_value(field_name: str, value: Any) -> str:
    """
    Enhanced field value normalization with medical document specifics.
    
    Args:
        field_name: Name of the field
        value: Raw extracted value (can be string, list, etc.)
        
    Returns:
        Normalized value as string
    """
    if not value:
        return ""
    
    # Handle list values (like icd_codes)
    if isinstance(value, list):
        if len(value) == 0:
            return ""
        elif len(value) == 1:
            value = str(value[0])
        else:
            # Join multiple values with comma for lists
            value = ", ".join(str(v) for v in value)
    else:
        value = str(value)
    
    value = value.strip()
    
    # Date formatting
    if "dob" in field_name.lower() or "birth" in field_name.lower():
        # Try to standardize date format
        date_match = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})", value)
        if date_match:
            month, day, year = date_match.groups()
            if len(year) == 2:
                year = "20" + year if int(year) < 50 else "19" + year
            return f"{month.zfill(2)}/{day.zfill(2)}/{year}"
    
    # Phone number formatting
    if "phone" in field_name.lower():
        digits = re.sub(r"\D", "", value)
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits[0] == "1":
            return f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
    
    # Name formatting
    if "name" in field_name.lower():
        # Capitalize each word properly
        words = value.split()
        formatted_words = []
        for word in words:
            if word.upper() in ["MD", "DO", "RN", "NP", "PA", "DDS", "DVM"]:
                formatted_words.append(word.upper())
            elif word.lower() in ["jr", "sr", "ii", "iii", "iv"]:
                formatted_words.append(word.title())
            else:
                formatted_words.append(word.capitalize())
        return " ".join(formatted_words)
    
    # ICD code formatting
    if "icd" in field_name.lower():
        # If it's a comma-separated list, keep it as is
        if "," in value:
            return value
        # Ensure proper ICD-10 format for single codes
        icd_match = re.match(r"([A-Z])(\d{2})\.?(\d*)", value.upper())
        if icd_match:
            letter, first_digits, last_digits = icd_match.groups()
            if last_digits:
                return f"{letter}{first_digits}.{last_digits}"
            else:
                return f"{letter}{first_digits}"
    
    # Medication formatting
    if "medication" in field_name.lower() or "drug" in field_name.lower():
        # Capitalize medication names properly
        return " ".join(word.capitalize() for word in value.split())
    
    # Diagnosis formatting
    if "diagnosis" in field_name.lower():
        # Remove common OCR artifacts and normalize
        cleaned = re.sub(r'[^\w\s,.-]', '', value)
        return " ".join(word.capitalize() for word in cleaned.split())
    
    return value


def calculate_confidence_score(extracted_text: str, field_name: str) -> float:
    """
    Enhanced confidence scoring with medical document specifics.
    
    Args:
        extracted_text: The extracted text
        field_name: The field name for context
        
    Returns:
        Confidence score between 0 and 1
    """
    if not extracted_text:
        return 0.0
    
    # Base confidence
    confidence = 0.5
    
    # Length-based scoring
    text_len = len(extracted_text.strip())
    if text_len > 50:
        confidence += 0.1
    elif text_len > 10:
        confidence += 0.2
    elif text_len > 5:
        confidence += 0.1
    else:
        confidence -= 0.2
    
    # Field-specific validation and scoring
    if "dob" in field_name.lower() or "birth" in field_name.lower():
        if re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$", extracted_text):
            confidence += 0.4
        else:
            confidence -= 0.3
    
    elif "phone" in field_name.lower():
        digits = re.sub(r"\D", "", extracted_text)
        if len(digits) == 10:
            confidence += 0.3
        elif len(digits) == 11 and digits[0] == "1":
            confidence += 0.3
        else:
            confidence -= 0.3
    
    elif "ssn" in field_name.lower():
        digits = re.sub(r"\D", "", extracted_text)
        if len(digits) == 9:
            confidence += 0.4
        else:
            confidence -= 0.4
    
    elif "name" in field_name.lower():
        # Names should be mostly alphabetic
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in extracted_text) / len(extracted_text)
        if alpha_ratio > 0.8:
            confidence += 0.3
        elif alpha_ratio > 0.6:
            confidence += 0.1
        else:
            confidence -= 0.2
    
    elif "diagnosis" in field_name.lower():
        # Check for medical terminology
        medical_terms = [
            "syndrome", "disease", "disorder", "condition", "sclerosis", 
            "diabetes", "hypertension", "depression", "anxiety", "cancer", 
            "arthritis", "asthma", "pneumonia", "infection", "deficiency"
        ]
        found_terms = sum(1 for term in medical_terms if term.lower() in extracted_text.lower())
        if found_terms > 0:
            confidence += 0.2 * found_terms
    
    elif "medication" in field_name.lower():
        # Check for medication-like patterns
        if re.search(r"\d+\s*mg|tablets?|capsules?|ml|units?", extracted_text.lower()):
            confidence += 0.2
    
    elif "icd" in field_name.lower():
        if re.match(r"^[A-Z]\d{2}\.?\d*$", extracted_text):
            confidence += 0.4
        else:
            confidence -= 0.3
    
    # Character quality assessment
    special_char_ratio = sum(1 for c in extracted_text if not c.isalnum() and not c.isspace()) / len(extracted_text)
    if special_char_ratio > 0.3:  # Too many special characters
        confidence -= 0.2
    
    # Return confidence clamped between 0 and 1
    return max(0.0, min(1.0, confidence))


def validate_extracted_data(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate extracted data for medical document consistency.
    
    Args:
        extracted_data: Dictionary containing extracted data
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "suggestions": []
    }
    
    # Check for required fields
    required_fields = ["patient_name", "dob"]
    for field in required_fields:
        found = False
        for category in extracted_data.values():
            if isinstance(category, dict) and field in category:
                found = True
                break
        
        if not found:
            validation_results["errors"].append(f"Required field missing: {field}")
            validation_results["is_valid"] = False
    
    # Validate data consistency
    confidence_scores = extracted_data.get("metadata", {}).get("confidence_scores", {})
    
    low_confidence_fields = [field for field, score in confidence_scores.items() if score < 0.6]
    if low_confidence_fields:
        validation_results["warnings"].extend(
            [f"Low confidence field: {field}" for field in low_confidence_fields]
        )
    
    # Check for suspicious patterns
    for category, fields in extracted_data.items():
        if not isinstance(fields, dict):
            continue
        
        for field_name, value in fields.items():
            if isinstance(value, str):
                # Check for OCR garbage
                if len(value) > 10:
                    alpha_ratio = sum(c.isalpha() for c in value) / len(value)
                    if alpha_ratio < 0.3:
                        validation_results["warnings"].append(f"Suspicious OCR result in {field_name}: {value[:50]}...")
    
    return validation_results


def load_json_safely(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Safely load JSON file with error handling.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing JSON data or None if failed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"Failed to load JSON file {file_path}: {e}")
        return None


def save_json_safely(data: Dict[str, Any], file_path: str) -> bool:
    """
    Safely save data to JSON file with error handling.
    
    Args:
        data: Dictionary to save
        file_path: Path to save the JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_directory_exists(os.path.dirname(file_path))
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON file {file_path}: {e}")
        return False


def get_patient_directories(input_dir: str) -> List[str]:
    """
    Get list of patient directories from input directory.
    
    Args:
        input_dir: Input directory containing patient folders
        
    Returns:
        List of patient directory paths
    """
    if not os.path.exists(input_dir):
        return []
    
    patient_dirs = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            # Check if it contains required files
            is_valid, _ = validate_input_files(item_path)
            if is_valid:
                patient_dirs.append(item_path)
    
    return sorted(patient_dirs)


class EnhancedFieldMatcher:
    """
    Enhanced field matching with improved semantic understanding and medical terminology.
    """
    
    def __init__(self):
        """Initialize the enhanced field matcher."""
        self.medical_synonyms = self._load_medical_synonyms()
        self.form_field_mappings = self._load_form_field_mappings()
    
    def find_best_match(self, form_field_name: str, extracted_data: Dict[str, Any]) -> Tuple[str, float]:
        """
        Find the best match for a form field in extracted data.
        
        Args:
            form_field_name: Name of the form field
            extracted_data: Dictionary containing extracted data
            
        Returns:
            Tuple of (best_value, confidence_score)
        """
        # Flatten extracted data for easier searching
        flat_data = self._flatten_extracted_data(extracted_data)
        
        best_match = ""
        best_confidence = 0.0
        
        # Try multiple matching strategies
        strategies = [
            ("exact_match", self._exact_match),
            ("semantic_match", self._semantic_match),
            ("fuzzy_match", self._fuzzy_match),
            ("pattern_match", self._pattern_match)
        ]
        
        for strategy_name, strategy_func in strategies:
            match, confidence = strategy_func(form_field_name, flat_data)
            
            if confidence > best_confidence:
                best_match = match
                best_confidence = confidence
        
        return best_match, best_confidence
    
    def _exact_match(self, form_field: str, flat_data: Dict[str, Any]) -> Tuple[str, float]:
        """Exact field name matching."""
        if form_field in flat_data:
            return str(flat_data[form_field]), 1.0
        return "", 0.0
    
    def _semantic_match(self, form_field: str, flat_data: Dict[str, Any]) -> Tuple[str, float]:
        """Semantic matching using medical terminology."""
        form_field_lower = form_field.lower()
        
        # Check direct mappings
        if form_field_lower in self.form_field_mappings:
            target_fields = self.form_field_mappings[form_field_lower]
            for target in target_fields:
                if target in flat_data:
                    return str(flat_data[target]), 0.9
        
        # Check for medical synonyms
        for field_name, value in flat_data.items():
            field_lower = field_name.lower()
            
            # Direct semantic matches
            semantic_matches = [
                ("patient", ["name", "patient_name"]),
                ("birth", ["dob", "date_of_birth"]),
                ("phone", ["phone", "telephone"]),
                ("address", ["address", "addr"]),
                ("diagnosis", ["diagnosis", "dx", "condition"]),
                ("medication", ["medication", "drug", "rx"]),
                ("insurance", ["insurance", "carrier", "plan"]),
                ("doctor", ["prescriber", "physician", "provider"]),
                ("npi", ["npi", "provider_id"])
            ]
            
            for form_keyword, field_keywords in semantic_matches:
                if form_keyword in form_field_lower:
                    if any(keyword in field_lower for keyword in field_keywords):
                        return str(value), 0.8
        
        return "", 0.0
    
    def _fuzzy_match(self, form_field: str, flat_data: Dict[str, Any]) -> Tuple[str, float]:
        """Fuzzy string matching for field names."""
        best_match = ""
        best_ratio = 0.0
        
        for field_name, value in flat_data.items():
            ratio = SequenceMatcher(None, form_field.lower(), field_name.lower()).ratio()
            if ratio > best_ratio and ratio > 0.6:  # Minimum threshold
                best_match = str(value)
                best_ratio = ratio
        
        return best_match, best_ratio * 0.7  # Reduce confidence for fuzzy matches
    
    def _pattern_match(self, form_field: str, flat_data: Dict[str, Any]) -> Tuple[str, float]:
        """Pattern-based matching for common form field patterns."""
        form_field_lower = form_field.lower()
        
        # Common form field patterns
        patterns = {
            r".*name.*": ["patient_name", "prescriber_name"],
            r".*birth.*|.*dob.*": ["dob", "date_of_birth"],
            r".*phone.*|.*tel.*": ["phone", "prescriber_phone"],
            r".*address.*": ["address", "full_address"],
            r".*diagnos.*|.*condition.*": ["diagnosis", "primary_diagnosis"],
            r".*medic.*|.*drug.*|.*rx.*": ["medication", "medications"],
            r".*insurance.*|.*carrier.*": ["insurance_carrier", "insurance"],
            r".*policy.*|.*member.*": ["policy_number", "member_id"],
            r".*prescrib.*|.*doctor.*|.*physician.*": ["prescriber_name", "prescriber"],
            r".*npi.*": ["prescriber_npi", "npi"]
        }
        
        for pattern, target_fields in patterns.items():
            if re.match(pattern, form_field_lower):
                for target in target_fields:
                    if target in flat_data:
                        return str(flat_data[target]), 0.7
        
        return "", 0.0
    
    def _flatten_extracted_data(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested extracted data for easier searching."""
        flat_data = {}
        
        for category, fields in extracted_data.items():
            if isinstance(fields, dict):
                for field_name, value in fields.items():
                    if field_name != "confidence_scores":
                        flat_data[field_name] = value
                        # Also add with category prefix
                        flat_data[f"{category}_{field_name}"] = value
        
        return flat_data
    
    def _load_medical_synonyms(self) -> Dict[str, List[str]]:
        """Load medical terminology synonyms."""
        return {
            "patient": ["patient", "client", "individual", "person"],
            "diagnosis": ["diagnosis", "condition", "disorder", "disease", "dx"],
            "medication": ["medication", "drug", "prescription", "rx", "medicine"],
            "prescriber": ["prescriber", "doctor", "physician", "provider", "md", "do"],
            "insurance": ["insurance", "carrier", "plan", "coverage"],
            "phone": ["phone", "telephone", "tel", "cell", "mobile"],
            "address": ["address", "addr", "location", "residence"]
        }
    
    def _load_form_field_mappings(self) -> Dict[str, List[str]]:
        """Load common form field name mappings."""
        return {
            # Patient info mappings
            "patientname": ["patient_name", "name"],
            "patientfirstname": ["patient_name"],
            "patientlastname": ["patient_name"],
            "dateofbirth": ["dob", "date_of_birth"],
            "phonenumber": ["phone", "telephone"],
            "patientaddress": ["address", "full_address"],
            
            # Clinical mappings
            "primarydiagnosis": ["diagnosis", "primary_diagnosis"],
            "condition": ["diagnosis", "condition"],
            "medication": ["medication", "medications"],
            "drugname": ["medication"],
            
            # Insurance mappings
            "insurancecarrier": ["insurance_carrier", "insurance"],
            "policynumber": ["policy_number"],
            "membernumber": ["policy_number", "member_id"],
            "groupnumber": ["group_number"],
            
            # Prescriber mappings
            "prescribername": ["prescriber_name", "prescriber"],
            "physicianname": ["prescriber_name"],
            "providername": ["prescriber_name"],
            "npi": ["prescriber_npi", "npi"],
            "prescriberphone": ["prescriber_phone"]
        }


# For backward compatibility
FieldMatcher = EnhancedFieldMatcher 