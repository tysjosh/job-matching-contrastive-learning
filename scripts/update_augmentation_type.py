#!/usr/bin/env python3
"""
Update augmentation_type in augmented_enriched_data_training.jsonl 
by reading career_view from processed_combined_enriched_data_times3_v4.jsonl

Assumes both files have records in the same order.
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def map_career_view_to_augmentation_type(career_view: str) -> str:
    """Map career_view values to augmentation_type values."""
    mapping = {
        'original': 'Original',
        'foundational': 'Foundational Match',
        'aspirational': 'Aspirational Match',
        'foundational_resume__aspirational_job': 'Cross-Match Negative (Foundational→Aspirational)',
        'aspirational_resume__foundational_job': 'Cross-Match Negative (Aspirational→Foundational)',
        'original_resume__foundational_job': 'Cross-Match Negative (Original→Foundational)',
        'original_resume__aspirational_job': 'Cross-Match Negative (Original→Aspirational)'
    }
    return mapping.get(career_view, 'Original')


def update_augmentation_types(
    source_file: str = 'preprocess/processed_combined_enriched_data_times3_v4.jsonl',
    target_file: str = 'preprocess/augmented_enriched_data_training.jsonl',
    output_file: str = 'preprocess/augmented_enriched_data_training_updated.jsonl'
):
    """Update augmentation_type by reading career_view from source file."""

    logger.info("Reading career_view from source and updating target...")

    total = 0
    updated = 0
    stats = {
        'Original': 0,
        'Foundational Match': 0,
        'Aspirational Match': 0,
        'Cross-Match Negative (Foundational→Aspirational)': 0,
        'Cross-Match Negative (Aspirational→Foundational)': 0,
        'Cross-Match Negative (Original→Foundational)': 0,
        'Cross-Match Negative (Original→Aspirational)': 0
    }

    with open(source_file, 'r', encoding='utf-8') as src, \
            open(target_file, 'r', encoding='utf-8') as tgt, \
            open(output_file, 'w', encoding='utf-8') as out:

        for line_num, (src_line, tgt_line) in enumerate(zip(src, tgt), 1):
            src_line = src_line.strip()
            tgt_line = tgt_line.strip()

            if not src_line or not tgt_line:
                continue

            try:
                # Read source record to get career_view
                src_record = json.loads(src_line)
                career_view = src_record.get('career_view', 'original')

                # Read target record to update
                tgt_record = json.loads(tgt_line)

                # Map career_view to augmentation_type
                augmentation_type = map_career_view_to_augmentation_type(
                    career_view)

                # Update metadata
                if 'metadata' not in tgt_record:
                    tgt_record['metadata'] = {}

                tgt_record['metadata']['augmentation_type'] = augmentation_type
                # Preserve for reference
                tgt_record['metadata']['career_view'] = career_view

                # Write updated record
                json.dump(tgt_record, out, ensure_ascii=False)
                out.write('\n')

                total += 1
                updated += 1
                stats[augmentation_type] += 1

                if line_num % 1000 == 0:
                    logger.info(f"Processed {line_num} records...")

            except json.JSONDecodeError as e:
                logger.warning(
                    f"Skipping invalid JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing line {line_num}: {e}")
                continue

    logger.info(f"\nUpdate complete!")
    logger.info(f"  Total records: {total}")
    logger.info(f"  Updated: {updated}")
    logger.info(f"  Original: {stats['Original']}")
    logger.info(f"  Foundational Match: {stats['Foundational Match']}")
    logger.info(f"  Aspirational Match: {stats['Aspirational Match']}")
    logger.info(
        f"  Cross-Match Negative (Foundational→Aspirational): {stats['Cross-Match Negative (Foundational→Aspirational)']}")
    logger.info(
        f"  Cross-Match Negative (Aspirational→Foundational): {stats['Cross-Match Negative (Aspirational→Foundational)']}")
    logger.info(
        f"  Cross-Match Negative (Original→Foundational): {stats['Cross-Match Negative (Original→Foundational)']}")
    logger.info(
        f"  Cross-Match Negative (Original→Aspirational): {stats['Cross-Match Negative (Original→Aspirational)']}")
    logger.info(f"  Output: {output_file}")

    return output_file


if __name__ == '__main__':
    try:
        output = update_augmentation_types()
        logger.info(f"\n✅ Success! Updated file: {output}")
        logger.info("To use this file for training, you can either:")
        logger.info("  1. Rename it to replace the original")
        logger.info("  2. Or point your training command to the new file")
    except Exception as e:
        logger.error(f"Failed: {e}")
        raise
