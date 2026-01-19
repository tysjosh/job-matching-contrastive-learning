#!/usr/bin/env python3
"""
Run dataset augmentation with proper ESCO skills hierarchy and career graph loading
"""

from contrastive_learning.data_adapter import DataAdapter, DataAdapterConfig
from augmentation.dataset_augmentation_orchestrator import DatasetAugmentationOrchestrator
import sys
import os
import logging
import pandas as pd

# Add augmentation and contrastive_learning modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'augmentation'))
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), 'contrastive_learning'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))


def get_file_names(input_file, output_file, is_augmentation=True):
    """Get consistent file names for processing pipeline"""
    base_name = output_file.replace('.jsonl', '')

    return {
        'processed': f"{base_name}_processed.jsonl",
        'final': f"{base_name}_training.jsonl",
        'intermediate': f"{base_name}_raw.jsonl" if is_augmentation else None
    }


def get_data_adapter_config(processed_file, final_file, is_augmentation=True):
    """Get consistent DataAdapter configuration"""
    return DataAdapterConfig(
        augmented_data_path=processed_file,
        output_path=final_file,
        min_job_description_length=1,
        min_resume_experience_length=1,
        max_samples_per_original=20000 if not is_augmentation else 5,
        balance_labels=is_augmentation  # Balance only for augmentation
    )


def cleanup_intermediate_files(files_to_clean, preserve_file):
    """Safely clean up intermediate files"""
    logger = logging.getLogger(__name__)
    for file_path in files_to_clean:
        if file_path != preserve_file and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Removed intermediate file: {file_path}")
            except OSError as e:
                logger.warning(f"Could not remove {file_path}: {e}")


def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_esco_skills_hierarchy():
    """Load ESCO skills hierarchy from skillsHierarchy_en.csv"""
    logger = logging.getLogger(__name__)

    try:
        # Load ESCO skills hierarchy data
        skills_hierarchy_file = 'dataset/esco/skillsHierarchy_en.csv'

        if os.path.exists(skills_hierarchy_file):
            logger.info(
                "Loading ESCO skills hierarchy from skillsHierarchy_en.csv...")

            # Load skills hierarchy
            hierarchy_df = pd.read_csv(skills_hierarchy_file)

            # Build skills hierarchy structure
            skills_hierarchy = {
                'skills': {},           # skill_uri -> skill details
                'hierarchies': {},      # level -> skills at that level
                'prerequisites': {},    # skill_uri -> parent skills
                'codes': {},           # code -> skill_uri mapping
                'levels': {}           # skill_uri -> level info
            }

            # Process each row in the hierarchy
            for _, row in hierarchy_df.iterrows():
                # Extract all levels (0-3)
                levels = []
                for level in range(4):  # Level 0-3
                    uri_col = f'Level {level} URI'
                    term_col = f'Level {level} preferred term'
                    code_col = f'Level {level} code'

                    if pd.notna(row.get(uri_col)) and pd.notna(row.get(term_col)):
                        skill_info = {
                            'uri': row[uri_col],
                            'term': row[term_col],
                            'code': row.get(code_col, ''),
                            'level': level,
                            'description': row.get('Description', ''),
                            'scope_note': row.get('Scope note', '')
                        }
                        levels.append(skill_info)

                        # Store skill details
                        skills_hierarchy['skills'][skill_info['uri']
                                                   ] = skill_info

                        # Store by level
                        if level not in skills_hierarchy['hierarchies']:
                            skills_hierarchy['hierarchies'][level] = {}
                        skills_hierarchy['hierarchies'][level][skill_info['uri']] = skill_info

                        # Store code mapping
                        if skill_info['code']:
                            skills_hierarchy['codes'][skill_info['code']
                                                      ] = skill_info['uri']

                        # Store level info
                        skills_hierarchy['levels'][skill_info['uri']] = level

                # Build prerequisites (parent-child relationships)
                for i in range(1, len(levels)):
                    child_uri = levels[i]['uri']
                    parent_uri = levels[i-1]['uri']

                    if child_uri not in skills_hierarchy['prerequisites']:
                        skills_hierarchy['prerequisites'][child_uri] = []
                    skills_hierarchy['prerequisites'][child_uri].append(
                        parent_uri)

            logger.info(
                f"Loaded ESCO skills hierarchy: {len(skills_hierarchy['skills'])} skills across {len(skills_hierarchy['hierarchies'])} levels")
            logger.info(
                f"Hierarchy levels: {list(skills_hierarchy['hierarchies'].keys())}")

            return skills_hierarchy

        else:
            logger.warning(
                f"ESCO skills hierarchy file not found: {skills_hierarchy_file}")

    except Exception as e:
        logger.warning(f"Failed to load ESCO skills hierarchy: {e}")
        import traceback
        traceback.print_exc()

    # Fallback: minimal structure
    logger.info("Using minimal ESCO skills hierarchy structure")
    return {
        'skills': {},
        'hierarchies': {},
        'prerequisites': {},
        'codes': {},
        'levels': {}
    }


def load_career_graph():
    """Load career graph from GEXF file"""
    logger = logging.getLogger(__name__)

    try:
        # Check if career graph file exists - try multiple locations
        career_graph_paths = [
            'training_output/career_graph_data_driven.gexf',
            'training_output/career_graph.gexf',
            'colab_package/training_output/career_graph.gexf'
        ]

        career_graph_file = None
        for path in career_graph_paths:
            if os.path.exists(path):
                career_graph_file = path
                break

        if career_graph_file:
            logger.info("Loading career graph from GEXF file...")

            import xml.etree.ElementTree as ET

            # Parse GEXF file
            tree = ET.parse(career_graph_file)
            root = tree.getroot()

            # Define namespace
            ns = {'gexf': 'http://www.gexf.net/1.2draft'}

            career_graph = {
                'nodes': {},           # node_id -> {title, uri}
                'edges': {},           # edge_id -> {source, target}
                'pathways': {},        # source -> [targets]
                'reverse_pathways': {},  # target -> [sources]
                'occupations': {},     # title -> uri mapping
                'file_path': career_graph_file
            }

            # Parse nodes (occupations)
            nodes = root.findall('.//gexf:node', ns)
            for node in nodes:
                node_id = node.get('id')

                # Get occupation title from attributes
                title = None
                attvalues = node.findall('.//gexf:attvalue', ns)
                for attvalue in attvalues:
                    if attvalue.get('for') == '0':  # title attribute
                        title = attvalue.get('value')
                        break

                if node_id and title:
                    career_graph['nodes'][node_id] = {
                        'uri': node_id,
                        'title': title
                    }
                    career_graph['occupations'][title] = node_id

            # Parse edges (career transitions)
            edges = root.findall('.//gexf:edge', ns)
            for edge in edges:
                edge_id = edge.get('id')
                source = edge.get('source')
                target = edge.get('target')

                if edge_id and source and target:
                    career_graph['edges'][edge_id] = {
                        'source': source,
                        'target': target
                    }

                    # Build pathways (source -> targets)
                    if source not in career_graph['pathways']:
                        career_graph['pathways'][source] = []
                    career_graph['pathways'][source].append(target)

                    # Build reverse pathways (target -> sources)
                    if target not in career_graph['reverse_pathways']:
                        career_graph['reverse_pathways'][target] = []
                    career_graph['reverse_pathways'][target].append(source)

            logger.info(
                f"Loaded career graph: {len(career_graph['nodes'])} occupations, {len(career_graph['edges'])} transitions")
            logger.info(
                f"Sample occupations: {list(career_graph['occupations'].keys())[:5]}")

            return career_graph

        else:
            logger.warning(
                f"Career graph file not found in any of these locations: {career_graph_paths}")

    except Exception as e:
        logger.warning(
            f"Failed to load career graph: {e}, using minimal structure")

    # Fallback: minimal structure with all required keys
    return {
        'nodes': {},
        'edges': {},
        'pathways': {},
        'reverse_pathways': {},
        'occupations': {},
        'file_path': None
    }


def convert_without_augmentation(input_file: str, output_file: str) -> bool:
    """
    Convert original dataset format to augmented format without augmentation.

    Args:
        input_file: Path to original JSONL file (matched_datasets_pairs_full_with_uri.jsonl format)
        output_file: Path to output augmented format JSONL file

    Returns:
        bool: True if conversion succeeded, False otherwise
    """
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Converting original dataset format from {input_file}...")
        logger.info(
            "No augmentation will be performed - only format conversion")

        import json

        converted_records = []
        total_records = 0
        successful_conversions = 0

        with open(input_file, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue

                total_records += 1

                try:
                    original_record = json.loads(line)

                    # Convert label format: keep as int (0/1) for DataAdapter compatibility
                    original_label = original_record.get('label', 0)
                    # DataAdapter expects integers and converts to strings internally
                    label_for_output = original_label  # Keep as integer

                    # Convert resume structure
                    original_resume = original_record.get('resume', {})
                    converted_resume = {
                        'role': original_resume.get('role', ''),
                        'experience': [{
                            'description': original_resume.get('experience', ''),
                            'responsibilities': {
                                'action_verbs': [],
                                'technical_terms': []
                            }
                        }],
                        'experience_level': original_resume.get('experience_level', ''),
                        'skills': original_resume.get('skills', []),
                        'keywords': original_resume.get('keywords', [])
                    }

                    # Keep job structure (already compatible)
                    job = original_record.get('job', {})

                    # Create metadata
                    job_applicant_id = original_record.get(
                        'job_applicant_id', 'unknown')
                    metadata = {
                        'original_record_id': str(job_applicant_id),
                        'job_applicant_id': job_applicant_id,
                        'augmentation_type': 'Original',
                        'label': original_label,
                        'original_label': 'positive' if original_label == 1 else 'negative'
                    }

                    # Create sample_id
                    sample_id = f"sample_{job_applicant_id}_{line_num}"

                    # Create converted record in augmented format
                    converted_record = {
                        'resume': converted_resume,
                        'job': job,
                        'label': label_for_output,  # Keep as integer for DataAdapter
                        'sample_id': sample_id,
                        'metadata': metadata,
                        'view_type': 'original'  # For DataAdapter compatibility
                    }

                    converted_records.append(converted_record)
                    successful_conversions += 1

                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Skipping invalid JSON on line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.warning(
                        f"Error converting record on line {line_num}: {e}")
                    continue

        # Write converted records
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for record in converted_records:
                json.dump(record, outfile, ensure_ascii=False)
                outfile.write('\n')

        logger.info(f"Format conversion completed:")
        logger.info(f"  Total records processed: {total_records}")
        logger.info(f"  Successfully converted: {successful_conversions}")
        logger.info(f"  Output file: {output_file}")

        return successful_conversions > 0

    except Exception as e:
        logger.error(f"Failed to convert original format: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_to_training_format(input_file: str, output_file: str) -> bool:
    """
    Convert augmented data to training format using DataAdapter.

    Args:
        input_file: Path to augmented JSONL file
        output_file: Path to output training JSONL file

    Returns:
        bool: True if conversion succeeded, False otherwise
    """
    logger = logging.getLogger(__name__)

    try:
        # Configure DataAdapter with relaxed validation criteria
        config = DataAdapterConfig(
            augmented_data_path=input_file,
            output_path=output_file,
            min_job_description_length=10,   # Very low threshold
            min_resume_experience_length=5,   # Very low threshold
            max_samples_per_original=20,      # Allow more views per original
            balance_labels=True               # Balance positive/negative samples
        )

        # Initialize adapter
        logger.info("Initializing DataAdapter...")
        adapter = DataAdapter(config)

        # Load and convert data
        logger.info(f"Loading augmented data from {input_file}...")
        adapter.load_data()

        # Get and log statistics
        stats = adapter.get_data_statistics()
        logger.info("Augmented data statistics:")
        logger.info(f"  Total samples: {stats['total_samples']}")
        logger.info(f"  Positive samples: {stats['positive_samples']}")
        logger.info(f"  Negative samples: {stats['negative_samples']}")
        logger.info(
            f"  Unique original records: {stats['unique_original_records']}")
        logger.info(
            f"  Average views per original: {stats['avg_views_per_original']:.2f}")

        # Convert and save
        logger.info(
            f"Converting to training format and saving to {output_file}...")
        adapter.convert_and_save()

        # Verify output file was created
        if os.path.exists(output_file):
            # Count lines in output file
            with open(output_file, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            logger.info(
                f"Successfully created training file with {line_count} samples")
            return True
        else:
            logger.error("Training file was not created")
            return False

    except Exception as e:
        logger.error(f"Failed to convert to training format: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_augmentation(input_file='processed_combined_data.jsonl', output_file='augmented_combined_data.jsonl', no_augmentation=False):
    """Run the complete dataset augmentation or format conversion"""

    setup_logging()
    logger = logging.getLogger(__name__)

    if no_augmentation:
        logger.info("Starting format conversion WITHOUT augmentation...")
        logger.info(f"Input: {input_file}")
        logger.info(f"Output: {output_file}")

        # Verify input file exists
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return False

        try:
            # Get consistent file names
            file_names = get_file_names(
                input_file, output_file, is_augmentation=False)
            processed_output = file_names['processed']
            final_training_file = file_names['final']

            # Convert original format to augmented format without augmentation
            if not convert_without_augmentation(input_file, processed_output):
                logger.error("Format conversion failed")
                return False

            # Convert to final training format using DataAdapter
            logger.info("Converting to training format using DataAdapter...")

            # Configure DataAdapter for no-augmentation mode (preserve all data)
            config = get_data_adapter_config(
                processed_output, final_training_file, is_augmentation=False)

            adapter = DataAdapter(config)
            adapter.load_data()
            adapter.convert_and_save()

            if os.path.exists(final_training_file):
                logger.info(f"✅ Training data ready: {final_training_file}")

                # Clean up intermediate files
                cleanup_intermediate_files(
                    [processed_output], final_training_file)

                # Report final statistics
                with open(final_training_file, 'r', encoding='utf-8') as f:
                    final_count = sum(1 for _ in f)
                logger.info(f"📊 Total training records: {final_count}")
                logger.info(
                    f"🎯 FINAL OUTPUT: {final_training_file} (converted training data)")
            else:
                logger.warning(
                    "Failed to convert to training format using DataAdapter")
                return False

            return True

        except Exception as e:
            logger.error(f"Format conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    else:
        # Original augmentation logic
        logger.info("Starting career-aware dataset augmentation...")
        logger.info(f"Input: {input_file}")
        logger.info(f"Output: {output_file}")

        # Verify input file exists
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return False

        try:
            # Load required components
            logger.info("Loading ESCO skills hierarchy...")
            esco_skills_hierarchy = load_esco_skills_hierarchy()

            logger.info("Loading career graph...")
            career_graph = load_career_graph()

            # Initialize the orchestrator
            logger.info("Initializing DatasetAugmentationOrchestrator...")
            orchestrator = DatasetAugmentationOrchestrator(
                esco_skills_hierarchy=esco_skills_hierarchy,
                career_graph=career_graph,
                lambda1=0.3,  # Weight for aspirational view
                lambda2=0.2,   # Weight for foundational view
                enable_enhanced_validation=False  # Disable strict validation
            )

            # Run the complete augmentation strategy
            logger.info(
                "Running augmentation strategy (1→4 record expansion)...")
            stats = orchestrator.augment_dataset(input_file, output_file)

            # Print comprehensive statistics
            orchestrator.print_statistics()

            logger.info("Augmentation completed successfully!")

            # Get consistent file names
            file_names = get_file_names(
                input_file, output_file, is_augmentation=True)
            processed_output = file_names['processed']
            final_training_file = file_names['final']
            intermediate_file = file_names['intermediate']

            # Post-process the augmented file to add view_type field for DataAdapter compatibility
            logger.info(
                "Post-processing augmented data for DataAdapter compatibility...")

            import json
            with open(output_file, 'r', encoding='utf-8') as infile, \
                    open(processed_output, 'w', encoding='utf-8') as outfile:

                for line in infile:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            # Add view_type field based on augmentation_type for DataAdapter filtering
                            augmentation_type = record.get(
                                'augmentation_type', 'unknown')
                            if 'Original' in augmentation_type or augmentation_type == 'original':
                                record['view_type'] = 'original'
                            else:
                                record['view_type'] = 'augmented'

                            outfile.write(json.dumps(record) + '\n')
                        except json.JSONDecodeError:
                            continue

            # Convert augmented data to training format using DataAdapter
            logger.info(
                "Converting augmented data to training format using DataAdapter...")

            # Handle potential file naming conflicts
            if final_training_file == output_file and intermediate_file:
                # Rename the current output_file to intermediate name to avoid conflicts
                if os.path.exists(output_file):
                    os.rename(output_file, intermediate_file)
                    output_file = intermediate_file

            if convert_to_training_format(processed_output, final_training_file):
                logger.info(f"✅ Training data ready: {final_training_file}")

                # Clean up intermediate files (but NOT the final training file!)
                cleanup_intermediate_files(
                    [output_file, processed_output], final_training_file)

                # Report final statistics
                with open(final_training_file, 'r', encoding='utf-8') as f:
                    final_count = sum(1 for _ in f)
                logger.info(f"📊 Total training records: {final_count}")
                logger.info(
                    f"🎯 FINAL OUTPUT: {final_training_file} (balanced training data)")
            else:
                logger.warning(
                    "Failed to convert to training format using DataAdapter")

            return True

        except Exception as e:
            logger.error(f"Augmentation failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function with argument parsing"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run career-aware dataset augmentation or format conversion')
    parser.add_argument('--input',
                        default='processed_combined_data.jsonl',
                        help='Input processed dataset file')
    parser.add_argument('--output',
                        default='augmented_combined_data.jsonl',
                        help='Output augmented dataset file')
    parser.add_argument('--no-augmentation', action='store_true',
                        help='Convert format without augmentation')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Pass the command line arguments directly to run_augmentation
    success = run_augmentation(
        input_file=args.input, output_file=args.output, no_augmentation=args.no_augmentation)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
