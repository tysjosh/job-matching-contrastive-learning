"""
Main execution script for simple resume augmentation.
"""

import argparse
import logging
from pathlib import Path
from .pipeline import SimpleAugmentationPipeline

def main():
    """Main entry point for the augmentation system."""
    parser = argparse.ArgumentParser(description="Simple Resume Augmentation System")
    parser.add_argument(
        "--input", 
        type=str, 
        default="processed_combined_data.jsonl",
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="simple_augmentation",
        help="Output directory path"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="simple_augmentation/config.json",
        help="Configuration file path"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} does not exist")
        return 1
    
    # Initialize and run pipeline
    try:
        pipeline = SimpleAugmentationPipeline(args.config)
        results = pipeline.process_file(args.input, args.output)
        print(f"Augmentation completed successfully. Results: {results}")
        return 0
    except Exception as e:
        print(f"Error during augmentation: {e}")
        return 1

if __name__ == "__main__":
    exit(main())