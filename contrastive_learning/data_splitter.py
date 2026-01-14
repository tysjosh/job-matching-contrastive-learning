#!/usr/bin/env python3
"""
Data splitting utilities for train/validation/test pipeline
"""

import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for data splitting"""
    strategy: str = "sequential"  # sequential, random, stratified, user_based, temporal
    ratios: Dict[str, float] = None
    seed: int = 42
    min_samples_per_split: int = 1
    validate_splits: bool = True

    def __post_init__(self):
        if self.ratios is None:
            self.ratios = {"train": 0.7, "validation": 0.15, "test": 0.15}

        # Validate ratios sum to 1.0
        total = sum(self.ratios.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


@dataclass
class SplitResults:
    """Results from data splitting"""
    splits: Dict[str, str]  # split_name -> file_path
    statistics: Dict[str, Any]
    validation_report: Optional[Dict[str, Any]] = None


class DataSplitter:
    """Intelligent data splitting with multiple strategies"""

    def __init__(self, config: SplitConfig):
        self.config = config
        random.seed(config.seed)

    def split_dataset(self, data_path: str, output_dir: str = "data_splits") -> SplitResults:
        """Split dataset according to configuration"""

        logger.info(f"Splitting dataset: {data_path}")
        logger.info(f"Strategy: {self.config.strategy}")
        logger.info(f"Ratios: {self.config.ratios}")

        # Load data
        data = self._load_data(data_path)
        logger.info(f"Loaded {len(data)} samples")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Split data based on strategy
        if self.config.strategy == "sequential":
            splits = self._sequential_split(data)
        elif self.config.strategy == "random":
            splits = self._random_split(data)
        elif self.config.strategy == "stratified":
            splits = self._stratified_split(data)
        elif self.config.strategy == "user_based":
            splits = self._user_based_split(data)
        elif self.config.strategy == "temporal":
            splits = self._temporal_split(data)
        else:
            raise ValueError(f"Unknown split strategy: {self.config.strategy}")

        # Save splits to files
        split_files = {}
        for split_name, split_data in splits.items():
            file_path = output_path / f"{split_name}.jsonl"
            self._save_split(split_data, file_path)
            split_files[split_name] = str(file_path)
            logger.info(
                f"Saved {split_name}: {len(split_data)} samples to {file_path}")

        # Generate statistics
        statistics = self._generate_statistics(splits, data)

        # Validate splits if requested
        validation_report = None
        if self.config.validate_splits:
            validation_report = self._validate_splits(splits, data)

        return SplitResults(
            splits=split_files,
            statistics=statistics,
            validation_report=validation_report
        )

    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load JSONL data"""
        data = []
        with open(data_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    # Track original line for debugging
                    item['_line_number'] = line_num
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Skipping invalid JSON at line {line_num}: {e}")
        return data

    def _sequential_split(self, data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Sequential splitting - first 70% for train, next 15% for validation, last 15% for test"""
        logger.info("Using sequential splitting strategy")
        return self._split_by_ratios(data)

    def _random_split(self, data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Random splitting"""
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)

        return self._split_by_ratios(shuffled_data)

    def _stratified_split(self, data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Stratified splitting to maintain label distribution"""

        # Group by label
        label_groups = defaultdict(list)
        for item in data:
            label = item.get('label', 'unknown')
            label_groups[label].append(item)

        # Split each label group separately
        splits = {name: [] for name in self.config.ratios.keys()}

        for label, group_data in label_groups.items():
            random.shuffle(group_data)
            label_splits = self._split_by_ratios(group_data)

            for split_name, split_data in label_splits.items():
                splits[split_name].extend(split_data)

        # Shuffle final splits
        for split_data in splits.values():
            random.shuffle(split_data)

        return splits

    def _user_based_split(self, data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """User-based splitting to prevent data leakage"""

        # Extract user identifiers
        user_data = defaultdict(list)
        for item in data:
            user_id = self._extract_user_id(item)
            user_data[user_id].append(item)

        logger.info(f"Found {len(user_data)} unique users")

        # Split users, not individual samples
        user_ids = list(user_data.keys())
        random.shuffle(user_ids)

        user_splits = self._split_by_ratios(user_ids)

        # Convert user splits to data splits
        splits = {name: [] for name in self.config.ratios.keys()}
        for split_name, split_users in user_splits.items():
            for user_id in split_users:
                splits[split_name].extend(user_data[user_id])

        return splits

    def _temporal_split(self, data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Temporal splitting based on timestamps"""

        # Sort by timestamp
        timestamped_data = []
        for item in data:
            timestamp = self._extract_timestamp(item)
            timestamped_data.append((timestamp, item))

        timestamped_data.sort(key=lambda x: x[0])
        sorted_data = [item for _, item in timestamped_data]

        return self._split_by_ratios(sorted_data)

    def _split_by_ratios(self, data: List[Any]) -> Dict[str, List[Any]]:
        """Split data according to configured ratios"""

        n = len(data)
        splits = {}
        start_idx = 0

        for i, (split_name, ratio) in enumerate(self.config.ratios.items()):
            if i == len(self.config.ratios) - 1:  # Last split gets remaining data
                end_idx = n
            else:
                end_idx = start_idx + int(n * ratio)

            splits[split_name] = data[start_idx:end_idx]
            start_idx = end_idx

        return splits

    def _extract_user_id(self, item: Dict[str, Any]) -> str:
        """Extract user identifier from data item"""

        # Try multiple possible user ID fields
        user_fields = ['user_id', 'user', 'resume_id', 'applicant_id']
        for field in user_fields:
            if field in item:
                return str(item[field])

        # Try to extract from resume data
        if 'resume' in item and isinstance(item['resume'], dict):
            if 'user_id' in item['resume']:
                return str(item['resume']['user_id'])

        # Generate hash-based ID from resume content
        if 'resume' in item:
            resume_str = json.dumps(item['resume'], sort_keys=True)
            return hashlib.md5(resume_str.encode()).hexdigest()[:8]

        # Fallback to line number
        return f"user_{item.get('_line_number', 'unknown')}"

    def _extract_timestamp(self, item: Dict[str, Any]) -> float:
        """Extract timestamp from data item"""

        # Try multiple timestamp fields
        timestamp_fields = ['timestamp', 'created_at', 'date', 'time']
        for field in timestamp_fields:
            if field in item:
                return float(item[field])

        # Use line number as fallback ordering
        return float(item.get('_line_number', 0))

    def _save_split(self, split_data: List[Dict[str, Any]], file_path: Path):
        """Save split data to JSONL file"""
        with open(file_path, 'w') as f:
            for item in split_data:
                # Remove internal fields
                clean_item = {k: v for k,
                              v in item.items() if not k.startswith('_')}
                f.write(json.dumps(clean_item) + '\n')

    def _generate_statistics(self, splits: Dict[str, List[Dict[str, Any]]],
                             original_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics about the splits"""

        stats = {
            'total_samples': len(original_data),
            'splits': {},
            'label_distribution': {},
            'user_distribution': {}
        }

        # Split-level statistics
        for split_name, split_data in splits.items():
            stats['splits'][split_name] = {
                'count': len(split_data),
                'percentage': len(split_data) / len(original_data) * 100
            }

        # Label distribution per split
        for split_name, split_data in splits.items():
            label_counts = Counter(item.get('label', 'unknown')
                                   for item in split_data)
            stats['label_distribution'][split_name] = dict(label_counts)

        # User distribution (for user-based splits)
        if self.config.strategy == "user_based":
            for split_name, split_data in splits.items():
                user_counts = Counter(self._extract_user_id(item)
                                      for item in split_data)
                stats['user_distribution'][split_name] = {
                    'unique_users': len(user_counts),
                    'avg_samples_per_user': len(split_data) / len(user_counts) if user_counts else 0
                }

        return stats

    def _validate_splits(self, splits: Dict[str, List[Dict[str, Any]]],
                         original_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate splits for data leakage and distribution issues"""

        validation_report = {
            'data_leakage': self._check_data_leakage(splits),
            'distribution_balance': self._check_distribution_balance(splits),
            'minimum_samples': self._check_minimum_samples(splits),
            'overall_status': 'passed'
        }

        # Check if any validation failed
        for check_name, check_result in validation_report.items():
            if check_name != 'overall_status' and isinstance(check_result, dict):
                if not check_result.get('passed', True):
                    validation_report['overall_status'] = 'failed'
                    logger.warning(f"Validation check failed: {check_name}")

        return validation_report

    def _check_data_leakage(self, splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Check for data leakage between splits"""

        if self.config.strategy != "user_based":
            return {'passed': True, 'message': 'Data leakage check only applies to user-based splits'}

        # Extract users from each split
        split_users = {}
        for split_name, split_data in splits.items():
            users = set(self._extract_user_id(item) for item in split_data)
            split_users[split_name] = users

        # Check for user overlap
        overlaps = {}
        split_names = list(split_users.keys())

        for i, split1 in enumerate(split_names):
            for split2 in split_names[i+1:]:
                overlap = split_users[split1] & split_users[split2]
                if overlap:
                    overlaps[f"{split1}_{split2}"] = list(overlap)

        passed = len(overlaps) == 0

        return {
            'passed': passed,
            'overlapping_users': overlaps,
            'message': 'No user overlap found' if passed else f'Found {len(overlaps)} overlapping user groups'
        }

    def _check_distribution_balance(self, splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Check if label distributions are reasonably balanced"""

        # Calculate label distributions
        distributions = {}
        for split_name, split_data in splits.items():
            label_counts = Counter(item.get('label', 'unknown')
                                   for item in split_data)
            total = len(split_data)
            distributions[split_name] = {
                label: count/total for label, count in label_counts.items()}

        # Check balance (using Chi-square test concept)
        max_deviation = 0.0
        for label in set().union(*[d.keys() for d in distributions.values()]):
            label_props = [distributions[split].get(
                label, 0) for split in distributions.keys()]
            if label_props:
                deviation = max(label_props) - min(label_props)
                max_deviation = max(max_deviation, deviation)

        # Consider balanced if max deviation is less than 10%
        balanced = max_deviation < 0.10

        return {
            'passed': balanced,
            'max_deviation': max_deviation,
            'distributions': distributions,
            'message': f'Max label distribution deviation: {max_deviation:.3f}'
        }

    def _check_minimum_samples(self, splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Check if all splits have minimum required samples"""

        insufficient_splits = []
        for split_name, split_data in splits.items():
            if len(split_data) < self.config.min_samples_per_split:
                insufficient_splits.append({
                    'split': split_name,
                    'count': len(split_data),
                    'required': self.config.min_samples_per_split
                })

        passed = len(insufficient_splits) == 0

        return {
            'passed': passed,
            'insufficient_splits': insufficient_splits,
            'message': 'All splits have sufficient samples' if passed else f'{len(insufficient_splits)} splits have insufficient samples'
        }


def main():
    """CLI interface for data splitting"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Split dataset for train/validation/test")
    parser.add_argument("dataset", help="Path to input dataset (JSONL)")
    parser.add_argument("--strategy", choices=["sequential", "random", "stratified", "user_based", "temporal"],
                        default="sequential", help="Splitting strategy")
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15],
                        help="Train, validation, test ratios")
    parser.add_argument("--output-dir", default="data_splits",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip split validation")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')

    # Create split configuration
    config = SplitConfig(
        strategy=args.strategy,
        ratios={
            "train": args.ratios[0], "validation": args.ratios[1], "test": args.ratios[2]},
        seed=args.seed,
        validate_splits=not args.no_validate
    )

    # Create splitter and split data
    splitter = DataSplitter(config)
    results = splitter.split_dataset(args.dataset, args.output_dir)

    # Print results
    print("\n" + "="*60)
    print("📊 DATA SPLITTING RESULTS")
    print("="*60)

    print(f"\n📁 Output files:")
    for split_name, file_path in results.splits.items():
        print(f"   {split_name}: {file_path}")

    print(f"\n📈 Statistics:")
    for split_name, stats in results.statistics['splits'].items():
        print(
            f"   {split_name}: {stats['count']:,} samples ({stats['percentage']:.1f}%)")

    if results.validation_report:
        print(
            f"\n✅ Validation: {results.validation_report['overall_status'].upper()}")
        if results.validation_report['overall_status'] == 'failed':
            print("⚠️  See logs for validation details")

    print("\n🎯 Ready for pipeline training!")


if __name__ == "__main__":
    main()
