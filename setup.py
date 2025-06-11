"""
Prior Authorization Form Filling Pipeline Setup

This file configures the installation and dependencies for the PA processing pipeline.
"""

from setuptools import setup, find_packages

# Read the README file for long description
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Prior Authorization Form Filling Pipeline"

setup(
    name="pa-form-filling-pipeline",
    version="1.0.0",
    author="PA Pipeline Team",
    author_email="support@pa-pipeline.com",
    description="Automated Prior Authorization form filling system for healthcare",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pa-form-filling-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "PyPDF2>=3.0.0",
        "pdfplumber>=0.9.0",
        "pytesseract>=0.3.10",
        "Pillow>=9.0.0",
        "opencv-python>=4.8.0",
        "reportlab>=4.0.0",
        
        # Data processing and ML
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "fuzzywuzzy[speedup]>=0.18.0",
        "python-Levenshtein>=0.20.0",
        
        # Logging and utilities
        "loguru>=0.7.0",
        "click>=8.0.0",
        "tqdm>=4.64.0",
        "colorama>=0.4.6",
        
        # Configuration and file handling
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "pathlib2>=2.3.7; python_version<'3.4'",
        
        # Optional NLP dependencies
        "spacy>=3.4.0",
        "transformers>=4.20.0",
        "torch>=1.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "nlp": [
            "spacy>=3.4.0",
            "transformers>=4.20.0",
            "torch>=1.12.0",
            "sentence-transformers>=2.2.0",
        ],
        "advanced": [
            "tensorflow>=2.9.0",
            "keras>=2.9.0",
            "nltk>=3.7",
            "textblob>=0.17.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "pa-pipeline=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["*.json", "*.yaml", "*.txt"],
    },
    zip_safe=False,
    keywords="healthcare, prior authorization, medical forms, automation, OCR, PDF processing",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/pa-form-filling-pipeline/issues",
        "Source": "https://github.com/yourusername/pa-form-filling-pipeline",
        "Documentation": "https://pa-pipeline.readthedocs.io/",
    },
) 