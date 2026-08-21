"""Extract structured referral data from scanned or native PDFs."""

from __future__ import annotations

import os
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pdfplumber
import pytesseract
from loguru import logger
from pdf2image import convert_from_path
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from document_parser import parse_referral_text, restore_ocr_spacing
from utils import setup_logging

try:
    import spacy
except ImportError:
    spacy = None


def _ocr_image(image: Image.Image) -> str:
    gray = image.convert("L")
    arr = np.array(gray)
    data = pytesseract.image_to_data(
        arr,
        output_type=pytesseract.Output.DICT,
        config="--oem 3 --psm 6",
    )
    lines = {}
    for idx, word in enumerate(data["text"]):
        if not str(word).strip():
            continue
        key = (data["block_num"][idx], data["par_num"][idx], data["line_num"][idx])
        lines.setdefault(key, []).append(str(word))
    return "\n".join(" ".join(words) for words in lines.values())


def _extract_text_from_pdf(pdf_path: str) -> str:
    chunks = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    chunks.append(f"--- Page {page_num} ---\n{text}")
    except Exception as exc:
        logger.warning(f"pdfplumber failed: {exc}")
    joined = "\n".join(chunks)
    if len(joined.strip()) > 200:
        return joined
    return ""


def _extract_text_via_ocr(pdf_path: str) -> str:
    logger.info(f"Running OCR on {pdf_path}")
    images = convert_from_path(pdf_path, dpi=250)
    pages = []
    for index, image in enumerate(images, start=1):
        logger.info(f"OCR page {index}/{len(images)}")
        page_text = _ocr_image(image)
        pages.append(f"--- Page {index} ---\n{page_text}")
    return "\n".join(pages)


class EnhancedReferralDataExtractor:
    def __init__(self, use_advanced_ocr: bool = True):
        self.use_advanced_ocr = use_advanced_ocr
        self.nlp = None
        if spacy is not None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model en_core_web_sm is not installed")

    def extract_from_referral_package(self, referral_pdf_path: str) -> Dict[str, Any]:
        logger.info(f"Starting extraction from {referral_pdf_path}")
        raw_text = _extract_text_from_pdf(referral_pdf_path)
        method = "Direct PDF Text"
        if not raw_text:
            raw_text = _extract_text_via_ocr(referral_pdf_path)
            method = "Enhanced OCR"

        result = parse_referral_text(raw_text)
        result["metadata"]["source_file"] = referral_pdf_path
        result["metadata"]["extraction_method"] = method
        result["metadata"]["extraction_timestamp"] = datetime.now().isoformat()
        result["processed_text"] = restore_ocr_spacing(raw_text)
        return result


def extract_referral_data(
    referral_pdf_path: str,
    output_dir: Optional[str] = None,
    use_advanced_ocr: bool = True,
) -> Dict[str, Any]:
    extractor = EnhancedReferralDataExtractor(use_advanced_ocr=use_advanced_ocr)
    results = extractor.extract_from_referral_package(referral_pdf_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "extracted_referral_data.json")
        with open(output_file, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, default=str)
        logger.info(f"Extraction results saved to {output_file}")
    return results


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_referral_data.py <referral_pdf_path> <output_dir>")
        sys.exit(1)
    setup_logging()
    results = extract_referral_data(sys.argv[1], sys.argv[2])
    print(f"Quality Score: {results['metadata'].get('quality_score', 0):.1%}")
    print(f"Fields extracted: {len(results['metadata'].get('confidence_scores', {}))}")
