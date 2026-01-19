#!/usr/bin/env python3
"""Quick validation of the augmentation_type update."""

import json

print('Validation Report')
print('=' * 80)

# Count records by type
stats = {
    'total': 0,
    'has_career_view': 0,
    'has_augmentation_type': 0,
    'positive': 0,
    'negative': 0
}

aug_type_counts = {}
career_view_counts = {}

with open('preprocess/augmented_enriched_data_training_updated.jsonl') as f:
    for line in f:
        data = json.loads(line)
        stats['total'] += 1

        metadata = data.get('metadata', {})
        career_view = metadata.get('career_view')
        aug_type = metadata.get('augmentation_type')

        if career_view:
            stats['has_career_view'] += 1
            career_view_counts[career_view] = career_view_counts.get(
                career_view, 0) + 1

        if aug_type:
            stats['has_augmentation_type'] += 1
            aug_type_counts[aug_type] = aug_type_counts.get(aug_type, 0) + 1

        if data['label'] == 'positive':
            stats['positive'] += 1
        else:
            stats['negative'] += 1

print(f"\nTotal records: {stats['total']}")
print(f"Records with career_view: {stats['has_career_view']}")
print(f"Records with augmentation_type: {stats['has_augmentation_type']}")
print(f"Positive labels: {stats['positive']}")
print(f"Negative labels: {stats['negative']}")

print(f"\nCareer view distribution:")
for cv, count in sorted(career_view_counts.items()):
    print(f"  {cv}: {count}")

print(f"\nAugmentation type distribution:")
for at, count in sorted(aug_type_counts.items()):
    print(f"  {at}: {count}")

# Validation checks
print("\n" + "=" * 80)
print("Validation Results:")
all_good = True

if stats['total'] != 19389:
    print(f"❌ Expected 19,389 records, got {stats['total']}")
    all_good = False

if stats['has_career_view'] != stats['total']:
    print(f"❌ Not all records have career_view field")
    all_good = False

if stats['has_augmentation_type'] != stats['total']:
    print(f"❌ Not all records have augmentation_type field")
    all_good = False

if len(career_view_counts) != 7:
    print(f"❌ Expected 7 career_view types, got {len(career_view_counts)}")
    all_good = False

if len(aug_type_counts) != 7:
    print(f"❌ Expected 7 augmentation types, got {len(aug_type_counts)}")
    all_good = False

if all_good:
    print("✅ All validation checks passed!")
    print("   - All 19,389 records updated")
    print("   - All records have career_view and augmentation_type")
    print("   - All 7 augmentation types present")
else:
    print("⚠️  Some validation checks failed")
