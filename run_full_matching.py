#!/usr/bin/env python3
"""
Run full dataset matching - this will process all 2,770 enriched records 
with all 4,817 preprocessed records. This may take 20-30 minutes.
"""

import json
from match_datasets_fixed import DatasetMatcher


def main():
    """Run full matching without sampling."""
    # File paths
    enriched_path = "preprocess/processed_combined_enriched_data.jsonl"
    preprocessed_path = "dataset/preprocessed_resumes.jsonl"

    # Initialize matcher
    matcher = DatasetMatcher(threshold=0.3)  # Lower threshold for more matches

    # Load data
    print("Loading full datasets...")
    enriched_data = matcher.load_jsonl(enriched_path)
    preprocessed_data = matcher.load_jsonl(preprocessed_path)

    if not enriched_data or not preprocessed_data:
        print("Failed to load data. Check file paths and formats.")
        return

    print(f"Loaded {len(enriched_data)} enriched records")
    print(f"Loaded {len(preprocessed_data)} preprocessed records")
    print(
        f"This will process {len(enriched_data)} × {len(preprocessed_data)} = {len(enriched_data) * len(preprocessed_data):,} comparisons")

    # Confirm before proceeding
    response = input("This may take 20-30 minutes. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    # Find matches (top_n=1 for only best match per enriched record)
    print("\nStarting full matching process...")
    print("Progress will be shown every 50 records...")
    matches = matcher.find_best_matches(
        enriched_data, preprocessed_data, top_n=1)

    # Generate report
    matcher.generate_match_report(matches, enriched_data, preprocessed_data)

    # Save only the cleaned JSONL format
    match_pairs_output_path = "matched_datasets_pairs_full.jsonl"
    matcher.save_matches_jsonl(matches, match_pairs_output_path)

    print(f"\nFull matching complete!")
    print(f"Training-ready dataset saved to: {match_pairs_output_path}")
    print(f"Total training records: {len(matches)}")


if __name__ == "__main__":
    main()
