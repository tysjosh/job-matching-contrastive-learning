"""
Negative Augmentation Orchestrator for LLM-Based Data Augmentation

This orchestrator processes a dataset and generates negative samples using
the LLMNegativeGenerator. For each original record, it outputs:
1. The original record (with its original label)
2. 5 generated negative samples (label=0)

Output format: JSONL with each line containing a resume-job pair with metadata.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .negative_generator import LLMNegativeGenerator, NegativeGenerationResult
from .llm_client import create_llm_client, LLMClient
from .models import LLMAugmentationConfig, TransformationConfig
from .esco_context import ESCOContextBuilder

logger = logging.getLogger(__name__)


@dataclass
class AugmentationStats:
    """Statistics for the augmentation process."""
    total_records_processed: int = 0
    total_originals_output: int = 0
    total_negatives_generated: int = 0
    negatives_by_type: Dict[str, int] = field(default_factory=dict)
    failed_records: int = 0
    processing_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_records_processed": self.total_records_processed,
            "total_originals_output": self.total_originals_output,
            "total_negatives_generated": self.total_negatives_generated,
            "negatives_by_type": self.negatives_by_type,
            "failed_records": self.failed_records,
            "processing_time_seconds": self.processing_time_seconds,
            "average_negatives_per_record": (
                self.total_negatives_generated / self.total_records_processed
                if self.total_records_processed > 0 else 0
            )
        }


class NegativeAugmentationOrchestrator:
    """
    Orchestrates the negative sample generation process.
    
    Processes input JSONL file and generates output with:
    - Original records (preserved with original labels)
    - Generated negative samples (4 per original record, matching rule-based augmentation)
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        llm_client: Optional[LLMClient] = None,
        num_negatives: int = 4
    ):
        """
        Initialize the orchestrator.
        
        Args:
            config_path: Path to configuration JSON file
            llm_client: Optional pre-configured LLM client
            num_negatives: Number of negatives to generate per record
        """
        # Load configuration
        if config_path and Path(config_path).exists():
            self.config = LLMAugmentationConfig.load_from_file(config_path)
        else:
            self.config = self._default_config()
        
        # Initialize LLM client
        if llm_client:
            self.llm_client = llm_client
        else:
            self.llm_client = create_llm_client(self.config.llm_provider)
        
        # Initialize negative generator
        self.negative_generator = LLMNegativeGenerator(
            llm_client=self.llm_client,
            config=self.config.transformation,
            cs_skills_path=self.config.paths.cs_skills_path,
            esco_domains_path=self.config.paths.esco_domains_path,
            num_negatives=num_negatives
        )
        
        self.num_negatives = num_negatives
        self.stats = AugmentationStats()
        
        logger.info(f"NegativeAugmentationOrchestrator initialized")
        logger.info(f"  Negatives per record: {num_negatives}")
        logger.info(f"  LLM model: {self.config.llm_provider.model_name}")

    def _default_config(self) -> LLMAugmentationConfig:
        """Create default configuration."""
        from .models import (
            LLMProviderConfig, TransformationConfig, ValidationConfig,
            FallbackConfig, BatchConfig, PathsConfig, PromptsConfig
        )
        
        return LLMAugmentationConfig(
            llm_provider=LLMProviderConfig(),
            transformation=TransformationConfig(),
            validation=ValidationConfig(),
            fallback=FallbackConfig(),
            batch_processing=BatchConfig(),
            paths=PathsConfig(),
            prompts=PromptsConfig()
        )

    def _extract_sample_id(self, record: Dict[str, Any], index: int) -> str:
        """Extract or generate sample ID for a record."""
        if "sample_id" in record:
            return record["sample_id"]
        if "job_applicant_id" in record:
            return f"sample_{record['job_applicant_id']}"
        return f"sample_{index}"

    def _extract_label(self, record: Dict[str, Any]) -> int:
        """Extract label from record."""
        # Try metadata.label first
        if "metadata" in record and "label" in record["metadata"]:
            label = record["metadata"]["label"]
            if isinstance(label, int):
                return label
            if isinstance(label, str):
                return 1 if label.lower() == "positive" else 0
        
        # Try top-level label
        if "label" in record:
            label = record["label"]
            if isinstance(label, int):
                return label
            if isinstance(label, str):
                return 1 if label.lower() == "positive" else 0
        
        # Default to positive (assuming original data is positive matches)
        return 1

    def process_record(
        self,
        record: Dict[str, Any],
        index: int
    ) -> List[Dict[str, Any]]:
        """
        Process a single record and generate output records.
        
        Args:
            record: Input record with resume and job
            index: Record index for ID generation
            
        Returns:
            List of output records (original + negatives)
        """
        output_records = []
        sample_id = self._extract_sample_id(record, index)
        original_label = self._extract_label(record)
        
        # Output original record
        original_output = self.negative_generator.format_original_record(
            record=record,
            sample_id=sample_id,
            original_label=original_label
        )
        output_records.append(original_output)
        self.stats.total_originals_output += 1
        
        # Generate negatives
        result = self.negative_generator.generate_negatives(record)
        
        if result.success:
            for negative in result.negatives:
                negative_output = self.negative_generator.format_output_record(
                    original_record=record,
                    negative=negative,
                    original_sample_id=sample_id
                )
                output_records.append(negative_output)
                
                # Update stats
                self.stats.total_negatives_generated += 1
                neg_type = negative.negative_type
                self.stats.negatives_by_type[neg_type] = (
                    self.stats.negatives_by_type.get(neg_type, 0) + 1
                )
        else:
            self.stats.failed_records += 1
            logger.warning(f"Failed to generate negatives for record {sample_id}")
        
        self.stats.total_records_processed += 1
        
        return output_records

    def augment_dataset(
        self,
        input_file: str,
        output_file: str,
        max_records: Optional[int] = None,
        save_interval: int = 100
    ) -> AugmentationStats:
        """
        Process entire dataset and generate augmented output.
        
        Args:
            input_file: Path to input JSONL file
            output_file: Path to output JSONL file
            max_records: Optional limit on records to process
            save_interval: How often to log progress
            
        Returns:
            AugmentationStats with processing statistics
        """
        input_path = Path(input_file)
        output_path = Path(output_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        logger.info(f"Starting negative augmentation")
        logger.info(f"  Input: {input_file}")
        logger.info(f"  Output: {output_file}")
        
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            for index, line in enumerate(f_in):
                if max_records and index >= max_records:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    output_records = self.process_record(record, index)
                    
                    # Write output records
                    for out_record in output_records:
                        f_out.write(json.dumps(out_record) + "\n")
                    
                    # Progress logging
                    if (index + 1) % save_interval == 0:
                        logger.info(
                            f"Processed {index + 1} records, "
                            f"generated {self.stats.total_negatives_generated} negatives"
                        )
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {index + 1}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing record {index + 1}: {e}")
                    self.stats.failed_records += 1
                    continue
        
        self.stats.processing_time_seconds = time.time() - start_time
        
        # Log final statistics
        logger.info("=" * 60)
        logger.info("AUGMENTATION COMPLETE")
        logger.info(f"  Records processed: {self.stats.total_records_processed}")
        logger.info(f"  Originals output: {self.stats.total_originals_output}")
        logger.info(f"  Negatives generated: {self.stats.total_negatives_generated}")
        logger.info(f"  Failed records: {self.stats.failed_records}")
        logger.info(f"  Processing time: {self.stats.processing_time_seconds:.2f}s")
        logger.info("  Negatives by type:")
        for neg_type, count in self.stats.negatives_by_type.items():
            logger.info(f"    {neg_type}: {count}")
        logger.info("=" * 60)
        
        # Save summary
        summary_path = output_path.parent / "negative_augmentation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self.stats.to_dict(), f, indent=2)
        logger.info(f"Summary saved to: {summary_path}")
        
        return self.stats


def run_negative_augmentation(
    input_file: str = "preprocess/data_without_augmentation_training.jsonl",
    output_file: str = "preprocess/llm_augmented_with_negatives.jsonl",
    config_path: str = "llm_augmentation/config.json",
    max_records: Optional[int] = None,
    num_negatives: int = 4
) -> AugmentationStats:
    """
    Convenience function to run negative augmentation.
    
    Args:
        input_file: Path to input JSONL
        output_file: Path to output JSONL
        config_path: Path to config JSON
        max_records: Optional limit on records
        num_negatives: Number of negatives per record
        
    Returns:
        AugmentationStats
    """
    orchestrator = NegativeAugmentationOrchestrator(
        config_path=config_path,
        num_negatives=num_negatives
    )
    
    return orchestrator.augment_dataset(
        input_file=input_file,
        output_file=output_file,
        max_records=max_records
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate negative samples using LLM")
    parser.add_argument("--input", default="preprocess/data_without_augmentation_training.jsonl",
                       help="Input JSONL file")
    parser.add_argument("--output", default="preprocess/llm_augmented_with_negatives.jsonl",
                       help="Output JSONL file")
    parser.add_argument("--config", default="llm_augmentation/config.json",
                       help="Config JSON file")
    parser.add_argument("--max-records", type=int, default=None,
                       help="Maximum records to process")
    parser.add_argument("--num-negatives", type=int, default=4,
                       help="Number of negatives per record")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    stats = run_negative_augmentation(
        input_file=args.input,
        output_file=args.output,
        config_path=args.config,
        max_records=args.max_records,
        num_negatives=args.num_negatives
    )
    
    print(f"\nFinal Statistics:")
    print(json.dumps(stats.to_dict(), indent=2))
