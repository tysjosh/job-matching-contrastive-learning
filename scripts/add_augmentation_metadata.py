#!/usr/bin/env python3
"""
Add augmentation metadata to simple augmentation output for Phase 1 training.

This script processes the output from simple_augmentation and adds the required
metadata that Phase 1 self-supervised training expects.
"""

import json
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_augmentation_metadata(input_file: str, output_file: str):
    """
    Add augmentation metadata to make data compatible with Phase 1 training.
    
    Args:
        input_file: Path to simple augmentation output
        output_file: Path to write enhanced data
    """
    logger.info(f"Processing {input_file} to add augmentation metadata...")
    
    total_records = 0
    augmented_records = 0
    original_records = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                record = json.loads(line)
                total_records += 1
                
                # Determine if this is an original or augmented record
                # Simple heuristic: even line numbers are originals, odd are augmented
                # (based on simple_augmentation creating 2 records per input)
                if line_num % 2 == 1:
                    # Original record
                    record['metadata'] = {
                        'augmentation_type': 'Original',
                        'source': 'original_dataset',
                        'record_type': 'original'
                    }
                    original_records += 1
                else:
                    # Augmented record
                    record['metadata'] = {
                        'augmentation_type': 'Augmented',
                        'source': 'simple_augmentation',
                        'record_type': 'augmented',
                        'augmentation_method': 'paraphrasing_and_masking'
                    }
                    augmented_records += 1
                
                # Ensure label is in string format for Phase 1 compatibility
                if 'label' in record:
                    if record['label'] == 1 or record['label'] == '1':
                        record['label'] = 'positive'
                    elif record['label'] == 0 or record['label'] == '0':
                        record['label'] = 'negative'
                
                # Write enhanced record
                json.dump(record, outfile, ensure_ascii=False)
                outfile.write('\n')
                
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing line {line_num}: {e}")
                continue
    
    logger.info(f"Processing complete:")
    logger.info(f"  Total records processed: {total_records}")
    logger.info(f"  Original records: {original_records}")
    logger.info(f"  Augmented records: {augmented_records}")
    logger.info(f"  Output written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Add augmentation metadata for Phase 1 training')
    parser.add_argument('--input', required=True, help='Input file from simple_augmentation')
    parser.add_argument('--output', required=True, help='Output file with metadata')
    
    args = parser.parse_args()
    
    try:
        add_augmentation_metadata(args.input, args.output)
        logger.info("Metadata addition completed successfully!")
    except Exception as e:
        logger.error(f"Failed to add metadata: {e}")
        raise


if __name__ == '__main__':
    main()