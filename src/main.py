#!/usr/bin/env python3
"""
Prior Authorization Form Filling Pipeline - Main Entry Point

This module serves as the main orchestrator for the PA form filling pipeline.
It handles command-line arguments, coordinates processing workflows, and manages
the overall execution flow for both single patient and batch processing modes.
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_referral_data import extract_referral_data
from fill_pa_form import fill_pa_form
from generate_report import generate_comprehensive_report, generate_missing_fields_report
from utils import setup_logging, validate_input_files, ensure_directory_exists, get_patient_directories


class PAProcessingPipeline:
    """Main pipeline orchestrator for PA form processing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the PA processing pipeline.
        
        Args:
            config: Configuration dictionary. If None, uses default settings.
        """
        self.config = config or self._get_default_config()
        self.logger = setup_logging(self.config.get('log_file'))
        
        # Initialize statistics
        self.stats = {
            'total_patients': 0,
            'successful_patients': 0,
            'failed_patients': 0,
            'processing_start_time': None,
            'processing_end_time': None
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration settings."""
        return {
            'confidence_threshold': 0.6,
            'use_advanced_ocr': True,
            'generate_comprehensive_reports': True,
            'generate_missing_fields_reports': True,
            'save_intermediate_files': True,
            'log_file': None
        }
    
    def process_single_patient(self, patient_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        Process a single patient's PA form and referral package.
        
        Args:
            patient_dir: Path to patient's input directory
            output_dir: Path to output directory
            
        Returns:
            Processing results dictionary
        """
        start_time = time.time()
        patient_id = Path(patient_dir).name
        
        self.logger.info(f"Processing patient: {patient_id}")
        
        result = {
            'patient_id': patient_id,
            'success': False,
            'processing_time': 0.0,
            'extraction_results': {},
            'filling_results': {},
            'reports_generated': [],
            'errors': []
        }
        
        try:
            # Validate input files
            is_valid, missing_files = validate_input_files(patient_dir)
            if not is_valid:
                raise FileNotFoundError(f"Missing required files: {missing_files}")
            
            # Set up patient output directory
            patient_output_dir = os.path.join(output_dir, patient_id)
            ensure_directory_exists(patient_output_dir)
            
            # Extract referral data
            referral_path = os.path.join(patient_dir, "referral_package.pdf")
            self.logger.info(f"Extracting data from referral: {referral_path}")
            
            extraction_results = extract_referral_data(
                referral_path, 
                patient_output_dir,
                use_advanced_ocr=self.config.get('use_advanced_ocr', True)
            )
            result['extraction_results'] = extraction_results
            
            # Fill PA form
            pa_form_path = os.path.join(patient_dir, "PA.pdf")
            pa_filled_path = os.path.join(patient_output_dir, "PA_filled.pdf")
            
            self.logger.info(f"Filling PA form: {pa_form_path}")
            
            filling_results = fill_pa_form(
                pa_form_path,
                extraction_results,
                pa_filled_path
            )
            result['filling_results'] = filling_results
            
            # Generate reports
            if self.config.get('generate_comprehensive_reports', True):
                report_path = os.path.join(patient_output_dir, "processing_report.md")
                success = generate_comprehensive_report(
                    patient_id=patient_id,
                    extraction_results=extraction_results,
                    filling_results=filling_results,
                    pa_form_path=pa_form_path,
                    referral_path=referral_path,
                    output_path=report_path
                )
                if success:
                    result['reports_generated'].append("processing_report.md")
            
            if self.config.get('generate_missing_fields_reports', True):
                missing_fields_path = os.path.join(patient_output_dir, "missing_fields.txt")
                success = generate_missing_fields_report(
                    missing_fields=filling_results.get('missing_fields', []),
                    output_path=missing_fields_path,
                    patient_id=patient_id
                )
                if success:
                    result['reports_generated'].append("missing_fields.txt")
            
            # Save intermediate files if requested
            if self.config.get('save_intermediate_files', True):
                extraction_json_path = os.path.join(patient_output_dir, "extracted_referral_data.json")
                with open(extraction_json_path, 'w') as f:
                    json.dump(extraction_results, f, indent=2, default=str)
                result['reports_generated'].append("extracted_referral_data.json")
            
            result['success'] = True
            self.logger.success(f"Successfully processed patient: {patient_id}")
            
        except Exception as e:
            error_msg = f"Failed to process patient {patient_id}: {str(e)}"
            self.logger.error(error_msg)
            result['errors'].append(error_msg)
            
            # Save error log
            error_log_path = os.path.join(output_dir, patient_id, "error_log.txt")
            ensure_directory_exists(os.path.dirname(error_log_path))
            with open(error_log_path, 'w') as f:
                f.write(f"Processing Error for {patient_id}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Error: {str(e)}\n")
                if hasattr(e, '__traceback__'):
                    import traceback
                    f.write(f"Traceback:\n{traceback.format_exc()}\n")
        
        finally:
            result['processing_time'] = time.time() - start_time
            
        return result
    
    def process_batch(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        Process all patients in an input directory.
        
        Args:
            input_dir: Directory containing patient folders
            output_dir: Output directory for processed results
            
        Returns:
            Batch processing summary
        """
        self.stats['processing_start_time'] = time.time()
        self.logger.info(f"Starting batch processing: {input_dir}")
        
        # Get patient directories
        patient_dirs = get_patient_directories(input_dir)
        self.stats['total_patients'] = len(patient_dirs)
        
        if not patient_dirs:
            self.logger.warning(f"No patient directories found in: {input_dir}")
            return self._create_batch_summary([])
        
        self.logger.info(f"Found {len(patient_dirs)} patients to process")
        
        # Process each patient
        patient_results = []
        for patient_dir in patient_dirs:
            result = self.process_single_patient(patient_dir, output_dir)
            patient_results.append(result)
            
            if result['success']:
                self.stats['successful_patients'] += 1
            else:
                self.stats['failed_patients'] += 1
        
        self.stats['processing_end_time'] = time.time()
        
        # Create and save batch summary
        batch_summary = self._create_batch_summary(patient_results)
        summary_path = os.path.join(output_dir, "batch_summary.json")
        self._save_batch_summary(batch_summary, summary_path)
        
        self.logger.success(
            f"Batch processing complete: {self.stats['successful_patients']}/{self.stats['total_patients']} successful"
        )
        
        return batch_summary
    
    def _create_batch_summary(self, patient_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create batch processing summary."""
        total_time = 0
        if self.stats['processing_start_time'] and self.stats['processing_end_time']:
            total_time = self.stats['processing_end_time'] - self.stats['processing_start_time']
        
        return {
            'batch_metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_processing_time': total_time,
                'pipeline_version': '1.0.0'
            },
            'statistics': {
                'total_patients': self.stats['total_patients'],
                'successful_patients': self.stats['successful_patients'],
                'failed_patients': self.stats['failed_patients'],
                'success_rate': (self.stats['successful_patients'] / max(1, self.stats['total_patients'])) * 100
            },
            'patient_results': patient_results,
            'configuration': self.config
        }
    
    def _save_batch_summary(self, summary: Dict[str, Any], output_path: str) -> bool:
        """Save batch summary to JSON file."""
        try:
            ensure_directory_exists(os.path.dirname(output_path))
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            self.logger.info(f"Batch summary saved to: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save batch summary: {e}")
            return False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {}


def main():
    """Main entry point for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Prior Authorization Form Filling Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all patients in input_data directory
  python src/main.py --input input_data --output output

  # Process a single patient
  python src/main.py --patient "input_data/Patient A" --output output

  # Use custom configuration
  python src/main.py --input input_data --output output --config config.json

  # Enable verbose logging
  python src/main.py --input input_data --output output --verbose
        """
    )
    
    # Input/Output arguments
    parser.add_argument(
        '-i', '--input',
        type=str,
        help='Input directory containing patient folders'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output directory for processed files'
    )
    
    parser.add_argument(
        '-p', '--patient',
        type=str,
        help='Process a single patient directory'
    )
    
    # Configuration arguments
    parser.add_argument(
        '-c', '--config',
        type=str,
        help='Path to configuration JSON file'
    )
    
    parser.add_argument(
        '-l', '--log-file',
        type=str,
        help='Path to log file'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if not args.input and not args.patient:
        parser.error("Must specify either --input or --patient")
    
    if args.input and args.patient:
        parser.error("Cannot specify both --input and --patient")
    
    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Override config with command line arguments
    if args.log_file:
        config['log_file'] = args.log_file
    
    if args.verbose:
        config['debug_mode'] = True
    
    # Initialize pipeline
    pipeline = PAProcessingPipeline(config)
    
    try:
        # Ensure output directory exists
        ensure_directory_exists(args.output)
        
        if args.patient:
            # Process single patient
            if not os.path.exists(args.patient):
                logger.error(f"Patient directory not found: {args.patient}")
                sys.exit(1)
            
            result = pipeline.process_single_patient(args.patient, args.output)
            
            if result['success']:
                logger.success(f"Processing completed successfully for {result['patient_id']}")
                print(f"✅ Success: {result['patient_id']}")
                print(f"⏱️  Processing time: {result['processing_time']:.1f}s")
                if result['reports_generated']:
                    print(f"📄 Reports: {', '.join(result['reports_generated'])}")
            else:
                logger.error(f"Processing failed for {result['patient_id']}")
                print(f"❌ Failed: {result['patient_id']}")
                if result['errors']:
                    print(f"🚨 Errors: {'; '.join(result['errors'])}")
                sys.exit(1)
        
        else:
            # Process batch
            if not os.path.exists(args.input):
                logger.error(f"Input directory not found: {args.input}")
                sys.exit(1)
            
            summary = pipeline.process_batch(args.input, args.output)
            
            stats = summary['statistics']
            print(f"\n📊 Batch Processing Complete")
            print(f"✅ Successful: {stats['successful_patients']}/{stats['total_patients']} patients")
            print(f"📈 Success Rate: {stats['success_rate']:.1f}%")
            print(f"⏱️  Total Time: {summary['batch_metadata']['total_processing_time']:.1f}s")
            
            if stats['failed_patients'] > 0:
                print(f"❌ Failed: {stats['failed_patients']} patients")
                print("📋 Check individual error logs for details")
    
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        print("\n⚠️  Processing interrupted")
        sys.exit(130)
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"💥 Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 