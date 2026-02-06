#!/usr/bin/env python3
"""
Run LLM-based negative sample generation.

This script generates hard negative samples for training by creating
semantically plausible but incorrect resume-job pairings.

For each original record, it generates 5 types of negatives:
1. Overqualified: Senior resume → Junior job
2. Underqualified: Junior resume → Senior job
3. Same Role Wrong Level: Same job title but mismatched experience
4. Adjacent Role: Similar but different role from ESCO ontology
5. Skill Gap: Resume missing critical required skills

Usage:
    python run_llm_negative_augmentation.py --input data.jsonl --output augmented.jsonl
    
    # Test with small sample
    python run_llm_negative_augmentation.py --input data.jsonl --output test.jsonl --max-records 10
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_augmentation import (
    NegativeAugmentationOrchestrator,
    run_negative_augmentation
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM-based negative samples for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full augmentation
    python run_llm_negative_augmentation.py \\
        --input preprocess/data_without_augmentation_training.jsonl \\
        --output preprocess/llm_augmented_negatives.jsonl
    
    # Test with 10 records
    python run_llm_negative_augmentation.py \\
        --input preprocess/data_without_augmentation_training.jsonl \\
        --output preprocess/test_negatives.jsonl \\
        --max-records 10
    
    # Custom number of negatives
    python run_llm_negative_augmentation.py \\
        --input data.jsonl \\
        --output augmented.jsonl \\
        --num-negatives 3
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        default="preprocess/data_without_augmentation_training.jsonl",
        help="Input JSONL file with original records"
    )
    parser.add_argument(
        "--output", "-o",
        default="preprocess/llm_augmented_with_negatives.jsonl",
        help="Output JSONL file for augmented data"
    )
    parser.add_argument(
        "--config", "-c",
        default="llm_augmentation/config.json",
        help="Path to LLM configuration JSON"
    )
    parser.add_argument(
        "--max-records", "-m",
        type=int,
        default=None,
        help="Maximum number of records to process (for testing)"
    )
    parser.add_argument(
        "--num-negatives", "-n",
        type=int,
        default=4,
        help="Number of negative samples to generate per record (default: 4)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Run augmentation
    print(f"\n{'='*60}")
    print("LLM NEGATIVE SAMPLE GENERATION")
    print(f"{'='*60}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Config: {args.config}")
    print(f"Negatives per record: {args.num_negatives}")
    if args.max_records:
        print(f"Max records: {args.max_records}")
    print(f"{'='*60}\n")
    
    try:
        stats = run_negative_augmentation(
            input_file=args.input,
            output_file=args.output,
            config_path=args.config,
            max_records=args.max_records,
            num_negatives=args.num_negatives
        )
        
        print(f"\n{'='*60}")
        print("AUGMENTATION COMPLETE")
        print(f"{'='*60}")
        print(f"Records processed:    {stats.total_records_processed}")
        print(f"Originals output:     {stats.total_originals_output}")
        print(f"Negatives generated:  {stats.total_negatives_generated}")
        print(f"Failed records:       {stats.failed_records}")
        print(f"Processing time:      {stats.processing_time_seconds:.2f}s")
        print(f"\nNegatives by type:")
        for neg_type, count in stats.negatives_by_type.items():
            print(f"  {neg_type}: {count}")
        print(f"{'='*60}")
        
        # Calculate expected output
        expected_total = stats.total_originals_output + stats.total_negatives_generated
        print(f"\nTotal output records: {expected_total}")
        print(f"Output file: {args.output}")
        
    except Exception as e:
        print(f"\nError during augmentation: {e}")
        logging.exception("Augmentation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
