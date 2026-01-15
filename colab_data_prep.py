#!/usr/bin/env python3
"""
Google Colab Data Preparation Script - Career-Aware Contrastive Learning System

This script prepares the complete training data and project files for upload to Google Colab
with support for:

- Complete train/validate/test pipeline with sequential data splitting
- Two-phase training architecture (self-supervised pre-training + supervised fine-tuning)
- Research-grade model with frozen SentenceTransformer
- Career-aware augmentation with ESCO taxonomy integration
- Comprehensive evaluation (15+ metrics)
- Real-time diagnostics and monitoring
- Professional reporting with visualizations

The package includes all necessary components for running the complete ML pipeline in Colab.
"""

import os
import json
import shutil
import tarfile
from pathlib import Path
from typing import Dict


def verify_required_files() -> Dict[str, bool]:
    """
    Verify that all required files for Colab pipeline training are present.

    Returns:
        Dictionary mapping file paths to their existence status
    """
    required_files = {
        # Data files
        "augmented_combined_data_training_with_uri.jsonl": False,
        "augmented_combined_data_training.jsonl": False,
        "unaugmented_combined_data_training.jsonl": False,

        # ESCO and graph files
        "training_output/career_graph.gexf": False,
        "training_output/career_graph_data_driven.gexf": False,
        "dataset/esco_skills.json": False,
        "esco_it_career_domains_refined.json": False,

        # Configuration files
        "config/colab_pipeline_config.json": False,
        "config/colab_enhanced_training_config.json": False,
        "config/phase1_pretraining_config.json": False,
        "config/phase2_finetuning_config.json": False,

        # Core pipeline modules
        "contrastive_learning/__init__.py": False,
        "contrastive_learning/data_splitter.py": False,
        "contrastive_learning/evaluator.py": False,
        "contrastive_learning/pipeline.py": False,
        "contrastive_learning/pipeline_config.py": False,
        "contrastive_learning/trainer.py": False,
        "contrastive_learning/cli.py": False,
        
        # Two-phase training components
        "contrastive_learning/two_phase_trainer.py": False,
        "contrastive_learning/fine_tuning_trainer.py": False,
        "contrastive_learning/contrastive_classification_model.py": False,
        "contrastive_learning/training_mode_detector.py": False,
        "contrastive_learning/training_strategy.py": False,
        "contrastive_learning/two_phase_metrics.py": False,
        
        # Data processing
        "contrastive_learning/data_loader.py": False,
        "contrastive_learning/data_adapter.py": False,
        "contrastive_learning/batch_processor.py": False,
        "contrastive_learning/data_structures.py": False,
        
        # Model and loss components
        "contrastive_learning/loss_engine.py": False,
        "contrastive_learning/embedding_cache.py": False,
        "contrastive_learning/career_graph.py": False,
        
        # Augmentation system
        "augmentation/__init__.py": False,
        "augmentation/career_aware_augmenter.py": False,
        "augmentation/upward_transformer.py": False,
        "augmentation/downward_transformer.py": False,
        "augmentation/job_pool_manager.py": False,
        "augmentation/progression_constraints.py": False,
        "augmentation/semantic_validator.py": False,

        # Supporting files
        "requirements.txt": False,
    }

    for file_path in required_files:
        required_files[file_path] = Path(file_path).exists()

    return required_files


def create_pipeline_configs():
    """Create pipeline-specific configuration files for Colab."""

    # Create config directory if it doesn't exist
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    # Pipeline configuration for Colab
    colab_pipeline_config = {
        "data_splitting": {
            "strategy": "sequential",
            "ratios": {
                "train": 0.7,
                "validation": 0.15,
                "test": 0.15
            },
            "seed": 42,
            "min_samples_per_split": 1,
            "validate_splits": True,
            "output_dir": "data_splits"
        },
        "training_config_path": "config/colab_enhanced_training_config.json",
        "output_base_dir": "pipeline_experiments",
        "experiment_name": "colab_career_progression",
        "device": "cuda",
        "early_stopping": {
            "enabled": True,
            "patience": 5,
            "metric": "validation_loss",
            "mode": "min",
            "min_delta": 0.001,
            "restore_best_weights": True
        },
        "checkpointing": {
            "save_best": True,
            "save_last": True,
            "save_frequency": 1,
            "metric": "validation_contrastive_accuracy",
            "mode": "max"
        },
        "validation": {
            "frequency": "every_epoch",
            "frequency_value": 1,
            "metrics": [
                "contrastive_accuracy",
                "precision",
                "recall",
                "f1_score",
                "auc_roc",
                "precision_at_k",
                "recall_at_k",
                "map_score",
                "ndcg_score",
                "embedding_quality"
            ],
            "save_predictions": True,
            "generate_plots": True,
            "batch_size": None
        },
        "testing": {
            "load_best_model": True,
            "comprehensive_report": True,
            "save_embeddings": False,
            "save_predictions": True,
            "error_analysis": True,
            "generate_visualizations": True
        },
        "reporting": {
            "formats": ["json", "html"],
            "include_visualizations": True,
            "include_model_analysis": True,
            "include_training_history": True,
            "include_hyperparameters": True,
            "output_dir": "pipeline_reports"
        },
        "hyperparameter_optimization": {
            "enabled": False,
            "n_trials": 50,
            "optimization_metric": "validation_contrastive_accuracy",
            "optimization_direction": "maximize",
            "search_space": {},
            "pruning": True,
            "study_name": None
        }
    }

    # Save pipeline config
    pipeline_config_path = config_dir / "colab_pipeline_config.json"
    with open(pipeline_config_path, 'w') as f:
        json.dump(colab_pipeline_config, f, indent=2)

    print(f"✓ Created pipeline config: {pipeline_config_path}")

    # Enhanced training config with research-grade settings
    # CRITICAL: These hyperparameters enable proper gradient flow and prevent catastrophic forgetting
    enhanced_training_config = {
        # Core training parameters
        "batch_size": 32,
        "learning_rate": 0.00005,  # Conservative for stability
        "num_epochs": 20,
        "temperature": 0.2,  # Optimal for contrastive learning
        "negative_sampling_ratio": 0.7,  # Deprecated: mixed sampling disabled
        "pathway_weight": 0.8,  # Weight for career-aware negatives
        
        # Feature flags
        "use_pathway_negatives": True,
        "use_view_augmentation": False,  # Disable for single-phase
        "shuffle_data": True,
        
        # Logging and checkpointing
        "checkpoint_frequency": 500,
        "log_frequency": 10,
        
        # View augmentation parameters
        "max_resume_views": 5,
        "max_job_views": 5,
        "fallback_on_augmentation_failure": True,
        
        # Career distance thresholds
        "hard_negative_max_distance": 2.0,
        "medium_negative_max_distance": 4.0,
        
        # ESCO graph path
        "esco_graph_path": "training_output/career_graph_data_driven.gexf",
        
        # Text encoder configuration
        "text_encoder_model": "sentence-transformers/all-MiniLM-L6-v2",
        "text_encoder_device": None,  # Auto-detect
        
        # Research-grade model settings (CRITICAL)
        "freeze_text_encoder": True,  # Prevent catastrophic forgetting
        "projection_dim": 128,  # Smaller projection reduces overfitting
        "projection_dropout": 0.1,
        
        # Embedding cache configuration
        "embedding_cache_size": 10000,
        "enable_embedding_preload": True,
        "clear_cache_between_epochs": True,
        
        # Global negative sampling
        "global_negative_sampling": False,
        "global_negative_pool_size": 1000,
        
        # Training phase (for backward compatibility)
        "training_phase": "supervised",  # Options: "supervised", "self_supervised", "fine_tuning"
        
        # Self-supervised training settings (for two-phase training)
        "use_augmentation_labels_only": False,
        "augmentation_positive_ratio": 1.0,
        
        # Fine-tuning settings (for two-phase training)
        "pretrained_model_path": None,
        "freeze_contrastive_layers": True,
        "classification_dropout": 0.1
    }

    # Save enhanced training config
    enhanced_config_path = config_dir / "colab_enhanced_training_config.json"
    with open(enhanced_config_path, 'w') as f:
        json.dump(enhanced_training_config, f, indent=2)

    print(f"✓ Created enhanced training config: {enhanced_config_path}")
    
    # Create two-phase training configurations
    create_two_phase_configs(config_dir)


def create_two_phase_configs(config_dir: Path):
    """Create configurations for two-phase training (pre-training + fine-tuning)."""
    
    # Phase 1: Self-supervised pre-training configuration
    phase1_config = {
        "batch_size": 32,
        "learning_rate": 0.0001,
        "num_epochs": 10,
        "temperature": 0.2,
        "negative_sampling_ratio": 0.7,  # Deprecated: mixed sampling disabled
        "pathway_weight": 0.8,
        "use_pathway_negatives": True,
        "use_view_augmentation": True,
        "checkpoint_frequency": 500,
        "log_frequency": 10,
        "shuffle_data": True,
        "max_resume_views": 5,
        "max_job_views": 5,
        "fallback_on_augmentation_failure": True,
        "hard_negative_max_distance": 2.0,
        "medium_negative_max_distance": 4.0,
        "esco_graph_path": "training_output/career_graph_data_driven.gexf",
        "text_encoder_model": "sentence-transformers/all-MiniLM-L6-v2",
        "text_encoder_device": None,
        "freeze_text_encoder": True,
        "projection_dim": 128,
        "projection_dropout": 0.1,
        "embedding_cache_size": 10000,
        "enable_embedding_preload": True,
        "clear_cache_between_epochs": True,
        "global_negative_sampling": False,
        "global_negative_pool_size": 1000,
        
        # Phase 1 specific: Self-supervised training
        "training_phase": "self_supervised",
        "use_augmentation_labels_only": True,  # Only use augmented pairs
        "augmentation_positive_ratio": 0.5,  # Use 50% of augmented pairs
        
        # Not applicable for phase 1
        "pretrained_model_path": None,
        "freeze_contrastive_layers": False,
        "classification_dropout": 0.0
    }
    
    # Phase 2: Supervised fine-tuning configuration
    phase2_config = {
        "batch_size": 32,
        "learning_rate": 0.00005,  # Lower learning rate for fine-tuning
        "num_epochs": 10,
        "temperature": 0.2,
        "negative_sampling_ratio": 0.7,  # Deprecated: mixed sampling disabled
        "pathway_weight": 0.8,
        "use_pathway_negatives": True,
        "use_view_augmentation": False,  # Disable augmentation in fine-tuning
        "checkpoint_frequency": 500,
        "log_frequency": 10,
        "shuffle_data": True,
        "max_resume_views": 5,
        "max_job_views": 5,
        "fallback_on_augmentation_failure": True,
        "hard_negative_max_distance": 2.0,
        "medium_negative_max_distance": 4.0,
        "esco_graph_path": "training_output/career_graph_data_driven.gexf",
        "text_encoder_model": "sentence-transformers/all-MiniLM-L6-v2",
        "text_encoder_device": None,
        "freeze_text_encoder": True,
        "projection_dim": 128,
        "projection_dropout": 0.1,
        "embedding_cache_size": 10000,
        "enable_embedding_preload": False,  # Don't preload in fine-tuning
        "clear_cache_between_epochs": True,
        "global_negative_sampling": False,
        "global_negative_pool_size": 1000,
        
        # Phase 2 specific: Fine-tuning with classification
        "training_phase": "fine_tuning",
        "use_augmentation_labels_only": False,
        "augmentation_positive_ratio": 1.0,
        
        # Fine-tuning specific
        "pretrained_model_path": "phase1_pretraining/checkpoint_best.pt",  # Will be set dynamically
        "freeze_contrastive_layers": True,  # Freeze encoder, only train classifier
        "classification_dropout": 0.1
    }
    
    # Save phase 1 config
    phase1_config_path = config_dir / "phase1_pretraining_config.json"
    with open(phase1_config_path, 'w') as f:
        json.dump(phase1_config, f, indent=2)
    print(f"✓ Created Phase 1 (pre-training) config: {phase1_config_path}")
    
    # Save phase 2 config
    phase2_config_path = config_dir / "phase2_finetuning_config.json"
    with open(phase2_config_path, 'w') as f:
        json.dump(phase2_config, f, indent=2)
    print(f"✓ Created Phase 2 (fine-tuning) config: {phase2_config_path}")


def create_colab_package(output_dir: str = "colab_package") -> str:
    """
    Create a compressed package containing all files needed for Colab pipeline training.

    Args:
        output_dir: Directory to create the package in

    Returns:
        Path to the created tar.gz file
    """
    package_dir = Path(output_dir)
    package_dir.mkdir(exist_ok=True)

    # Create pipeline configs first
    create_pipeline_configs()

    # Files to include in the package
    files_to_package = [
        # Data files
        "augmented_combined_data_training_with_uri.jsonl",
        "augmented_combined_data_training.jsonl",
        "unaugmented_combined_data_training.jsonl",

        # ESCO and graph files
        "training_output/career_graph.gexf",
        "training_output/career_graph_data_driven.gexf",
        "dataset/esco_skills.json",
        "esco_it_career_domains_refined.json",

        # Configuration files
        "config/colab_pipeline_config.json",
        "config/colab_enhanced_training_config.json",
        "config/phase1_pretraining_config.json",
        "config/phase2_finetuning_config.json",

        # Supporting files
        "requirements.txt",
    ]

    # Directories to include (entire contents)
    dirs_to_package = [
        "contrastive_learning",  # Complete ML pipeline
        "augmentation",          # Career-aware augmentation system
        "matching",              # Job-resume matching system
        "diagnostic",            # Training diagnostics
        "scripts",               # Utility scripts
        "dataset/esco"           # ESCO taxonomy data
    ]

    print("Creating Colab pipeline training package...")

    # Copy files
    for file_path in files_to_package:
        if Path(file_path).exists():
            dest_path = package_dir / file_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest_path)
            print(f"✓ Added: {file_path}")
        else:
            print(f"✗ Missing: {file_path}")

    # Copy directories
    for dir_path in dirs_to_package:
        if Path(dir_path).exists():
            dest_path = package_dir / dir_path
            shutil.copytree(dir_path, dest_path, dirs_exist_ok=True)
            print(f"✓ Added directory: {dir_path}")
        else:
            print(f"✗ Missing directory: {dir_path}")

    # Create compressed archive
    archive_path = f"{output_dir}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(package_dir, arcname=".")

    print(f"\n📦 Package created: {archive_path}")
    print(
        f"📊 Package size: {Path(archive_path).stat().st_size / (1024*1024):.1f} MB")

    return archive_path


def create_colab_notebook() -> None:
    """
    Create a comprehensive Jupyter notebook for Google Colab pipeline training.
    """

    # Generate the setup script content as individual lines
    setup_script_lines = [
        "# Google Colab Environment Setup Script - COMPLETE ML PIPELINE",
        "# Run this cell first in your Colab notebook",
        "",
        "import os",
        "import sys",
        "import subprocess",
        "import torch",
        "",
        "def setup_colab_environment():",
        '    """Complete setup for Colab pipeline training environment."""',
        "    ",
        '    print("🚀 Setting up Google Colab ML Pipeline environment...")',
        "    ",
        "    # 1. Verify GPU availability",
        '    print("\\n1. GPU Configuration:")',
        "    if torch.cuda.is_available():",
        '        print(f"✓ CUDA available: {torch.cuda.get_device_name()}")',
        '        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")',
        "    else:",
        '        print("✗ CUDA not available - please enable GPU runtime")',
        "        return False",
        "    ",
        "    # 2. Install optimized packages for ML pipeline",
        '    print("\\n2. Installing ML pipeline packages...")',
        "    packages = [",
        '        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",',
        '        "sentence-transformers",',
        '        "transformers",',
        '        "accelerate",',
        '        "networkx",',
        '        "pandas",',
        '        "numpy",',
        '        "scikit-learn",',
        '        "nltk",',
        '        "matplotlib",',
        '        "seaborn",',
        '        "plotly",',
        '        "tqdm"',
        "    ]",
        "    ",
        "    for package in packages:",
        "        try:",
        '            subprocess.run(f"pip install {package}", shell=True, check=True, capture_output=True)',
        '            print(f"✓ Installed: {package.split()[0]}")',
        "        except subprocess.CalledProcessError as e:",
        '            print(f"✗ Failed to install: {package}")',
        "            return False",
        "    ",
        "    # 3. Setup NLTK data",
        '    print("\\n3. Setting up NLTK data...")',
        "    try:",
        "        import nltk",
        "        nltk.download('punkt', quiet=True)",
        "        nltk.download('stopwords', quiet=True)",
        '        print("✓ NLTK data downloaded")',
        "    except Exception as e:",
        '        print(f"✗ NLTK setup failed: {e}")',
        "    ",
        "    # 4. Memory optimization for pipeline",
        '    print("\\n4. Configuring pipeline optimization...")',
        "    os.environ['TOKENIZERS_PARALLELISM'] = 'false'",
        "    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'",
        "    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'",
        '    print("✓ Environment variables set for pipeline")',
        "    ",
        '    print("\\n🎯 Pipeline environment setup complete!")',
        "    return True",
        "",
        "def mount_drive_and_extract_data():",
        '    """Mount Google Drive and extract pipeline training data."""',
        "    ",
        '    print("\\n📁 Setting up pipeline data access...")',
        "    ",
        "    # Mount Google Drive",
        "    try:",
        "        from google.colab import drive",
        "        drive.mount('/content/drive')",
        '        print("✓ Google Drive mounted")',
        "    except Exception as e:",
        '        print(f"✗ Drive mount failed: {e}")',
        "        return False",
        "    ",
        "    # Extract training data (adjust path as needed)",
        '    data_path = "/content/drive/MyDrive/colab_package.tar.gz"',
        "    if os.path.exists(data_path):",
        "        try:",
        "            import tarfile",
        '            with tarfile.open(data_path, "r:gz") as tar:',
        '                tar.extractall("/content/")',
        '            print("✓ Pipeline training data extracted")',
        "            ",
        "            # Verify pipeline components",
        "            pipeline_files = [",
        '                "contrastive_learning/data_splitter.py",',
        '                "contrastive_learning/evaluator.py",',
        '                "contrastive_learning/pipeline.py",',
        '                "config/colab_pipeline_config.json"',
        "            ]",
        "            ",
        "            missing = [f for f in pipeline_files if not os.path.exists(f)]",
        "            if missing:",
        '                print(f"✗ Missing pipeline files: {missing}")',
        "                return False",
        "            else:",
        '                print("✓ All pipeline components present")',
        "            ",
        "            return True",
        "        except Exception as e:",
        '            print(f"✗ Data extraction failed: {e}")',
        "            return False",
        "    else:",
        '        print(f"✗ Data file not found at: {data_path}")',
        '        print("Please upload colab_package.tar.gz to your Google Drive")',
        "        return False",
        "",
        "def verify_pipeline_installation():",
        '    """Verify that all pipeline components are working correctly."""',
        "    ",
        '    print("\\n🔍 Verifying pipeline installation...")',
        "    ",
        "    # Test imports",
        "    try:",
        "        import torch",
        "        import sentence_transformers",
        "        import transformers",
        "        import networkx",
        "        import matplotlib.pyplot as plt",
        "        import seaborn as sns",
        '        print("✓ All packages imported successfully")',
        "    except ImportError as e:",
        '        print(f"✗ Import error: {e}")',
        "        return False",
        "    ",
        "    # Test CUDA",
        "    if torch.cuda.is_available():",
        '        print(f"✓ CUDA working: {torch.cuda.current_device()}")',
        "    else:",
        '        print("✗ CUDA not available")',
        "        return False",
        "    ",
        "    # Test pipeline components",
        "    try:",
        "        sys.path.append('/content')",
        "        from contrastive_learning.data_splitter import DataSplitter",
        "        from contrastive_learning.evaluator import ContrastiveEvaluator",
        "        from contrastive_learning.pipeline import MLPipeline",
        '        print("✓ Pipeline components imported successfully")',
        "    except ImportError as e:",
        '        print(f"✗ Pipeline import error: {e}")',
        "        return False",
        "    ",
        "    # Test data files",
        "    required_files = [",
        '        "augmented_combined_data_training.jsonl",',
        '        "training_output/career_graph.gexf",',
        '        "training_output/career_graph_data_driven.gexf",',
        '        "dataset/esco_skills.json",',
        '        "config/colab_pipeline_config.json"',
        "    ]",
        "    ",
        "    missing_files = [f for f in required_files if not os.path.exists(f)]",
        "    if missing_files:",
        '        print(f"✗ Missing files: {missing_files}")',
        "        return False",
        "    else:",
        '        print("✓ All required files present")',
        "    ",
        '    print("\\n✅ Pipeline installation verification complete!")',
        "    return True",
        "",
        "# Execute setup sequence",
        'print("🚀 Starting Colab Setup...")',
        "",
        "success = setup_colab_environment()",
        "if success:",
        "    success = mount_drive_and_extract_data()",
        "if success:",
        "    success = verify_pipeline_installation()",
        "",
        "if success:",
        '    print("\\n🎉 Ready for complete ML pipeline training!")',
        '    print("Next step: Run the pipeline training cells")',
        "else:",
        '    print("\\n❌ Setup failed. Please check the errors above.")'
    ]

    # Environment verification cell
    env_verification_lines = [
        "# Verify setup completed and show working environment",
        "import os",
        "import sys",
        "",
        'print("📍 Current Environment Status:")',
        'print(f"Working directory: {os.getcwd()}")',
        'print(f"Python executable: {sys.executable}")',
        'print(f"Python version: {sys.version}")',
        "",
        "# List extracted files",
        'if os.path.exists("contrastive_learning"):',
        '    print("✓ Pipeline code extracted successfully")',
        '    print("📁 Available directories:")',
        '    for item in sorted(os.listdir(".")):',
        "        if os.path.isdir(item) and not item.startswith('.'):",
        '            print(f"   📂 {item}/")',
        "    ",
        '    print("\\n📄 Key data files:")',
        "    data_files = [",
        '        "augmented_combined_data_training.jsonl",',
        '        "training_output/career_graph.gexf",',
        '        "training_output/career_graph_data_driven.gexf"',
        '        "dataset/esco_skills.json"',
        "    ]",
        "    ",
        "    for file_path in data_files:",
        "        if os.path.exists(file_path):",
        "            size_mb = os.path.getsize(file_path) / (1024*1024)",
        '            print(f"   ✓ {file_path} ({size_mb:.1f} MB)")',
        "        else:",
        '            print(f"   ✗ {file_path} (missing)")',
        "            ",
        '    print("\\n🎯 Environment ready for pipeline training!")',
        "else:",
        '    print("❌ Setup incomplete - pipeline code not found")',
        '    print("Please re-run the setup cell above")'
    ]

    # Data verification cell
    data_verification_lines = [
        "# Data already extracted during setup",
        "# Verify data availability and show statistics",
        "import os",
        "",
        "data_file = 'augmented_combined_data_training.jsonl'",
        "if os.path.exists(data_file):",
        "    with open(data_file, 'r') as f:",
        "        sample_count = sum(1 for line in f)",
        "    print(f'✅ Training data available: {sample_count:,} samples')",
        "    print(f'📈 Sequential splits will be:')",
        "    print(f'   Train: {int(sample_count * 0.7):,} samples (70%)')",
        "    print(f'   Validation: {int(sample_count * 0.15):,} samples (15%)')",
        "    print(f'   Test: {int(sample_count * 0.15):,} samples (15%)')",
        "else:",
        "    print('❌ Training data not found - check setup step')"
    ]

    # Pipeline training cell
    pipeline_training_lines = [
        "# Complete ML Pipeline Training Script - SEQUENTIAL DATA SPLITTING",
        "# Run this cell to execute the complete train/validate/test pipeline",
        "",
        "import os",
        "import json",
        "import torch",
        "import subprocess",
        "import sys",
        "from pathlib import Path",
        "",
        "def run_complete_pipeline():",
        '    """Execute the complete ML pipeline with sequential splitting."""',
        "    ",
        '    print("🚀 Starting Complete ML Pipeline with Sequential Data Splitting...")',
        "    ",
        "    # Check data availability",
        '    data_file = "augmented_combined_data_training.jsonl"',
        "    if not os.path.exists(data_file):",
        '        print(f"❌ Data file not found: {data_file}")',
        "        return False",
        "    ",
        '    print(f"✓ Found training data: {data_file}")',
        "    ",
        "    # Count samples",
        "    with open(data_file, 'r') as f:",
        "        sample_count = sum(1 for line in f)",
        '    print(f"📊 Total samples: {sample_count:,}")',
        "    ",
        "    # Calculate expected splits",
        "    train_samples = int(sample_count * 0.7)",
        "    val_samples = int(sample_count * 0.15)",
        "    test_samples = sample_count - train_samples - val_samples",
        "    ",
        '    print(f"📈 Expected sequential splits:")',
        '    print(f"   Train: {train_samples:,} samples (70%)")',
        '    print(f"   Validation: {val_samples:,} samples (15%)")',
        '    print(f"   Test: {test_samples:,} samples (15%)")',
        "    ",
        "    # Run the complete pipeline",
        '    print("\\n🎯 Executing Complete Pipeline...")',
        "    ",
        "    # Pipeline command with configuration file for sequential splitting",
        "    pipeline_cmd = [",
        '        sys.executable, "-m", "contrastive_learning", "pipeline",',
        "        data_file,",
        '        "--config", "config/colab_pipeline_config.json",',
        '        "--experiment-name", "colab_career_progression_sequential",',
        '        "--output-dir", "pipeline_experiments"',
        "    ]",
        "    ",
        "    try:",
        "        # Run the pipeline with real-time output streaming",
        "        print('Running command:', ' '.join(pipeline_cmd))",
        "        print('\\n' + '='*60)",
        "        print('🚀 TRAINING STARTED - Live Output Below:')",
        "        print('='*60 + '\\n')",
        "        ",
        "        # Use Popen for real-time streaming",
        "        import subprocess",
        "        process = subprocess.Popen(",
        "            pipeline_cmd,",
        "            stdout=subprocess.PIPE,",
        "            stderr=subprocess.STDOUT,",
        "            text=True,",
        "            bufsize=1,  # Line buffered",
        "            universal_newlines=True",
        "        )",
        "        ",
        "        # Stream output in real-time",
        "        output_lines = []",
        "        while True:",
        "            line = process.stdout.readline()",
        "            if not line and process.poll() is not None:",
        "                break",
        "            if line:",
        "                print(line.rstrip())  # Print without extra newlines",
        "                output_lines.append(line)",
        "        ",
        "        # Wait for process to complete",
        "        return_code = process.wait()",
        "        ",
        "        if return_code == 0:",
        '            print("\\n" + "="*60)',
        '            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")',
        '            print("="*60)',
        "            ",
        "            # Display results summary",
        '            results_file = "pipeline_experiments/colab_career_progression_sequential/pipeline_results.json"',
        "            if os.path.exists(results_file):",
        "                with open(results_file, 'r') as f:",
        "                    results = json.load(f)",
        "                ",
        '                print("\\n📈 Training Results Summary:")',
        "                if 'validation_metrics' in results:",
        "                    for metric, value in results['validation_metrics'].items():",
        "                        if isinstance(value, (int, float)):",
        '                            print(f"   {metric}: {value:.4f}")',
        "                ",
        "                if 'test_metrics' in results:",
        '                    print("\\n🎯 Test Results:")',
        "                    for metric, value in results['test_metrics'].items():",
        "                        if isinstance(value, (int, float)):",
        '                            print(f"   {metric}: {value:.4f}")',
        "            ",
        "            return True",
        "        else:",
        '            print("\\n" + "="*60)',
        '            print("❌ PIPELINE FAILED!")',
        '            print(f"Exit code: {return_code}")',
        '            print("="*60)',
        '            print("\\nCheck the output above for error details")',
        "            return False",
        "            ",
        "    except KeyboardInterrupt:",
        '        print("\\n⏸️ Training interrupted by user")',
        "        process.terminate()",
        "        return False",
        "    except Exception as e:",
        '        print(f"\\n❌ Pipeline execution failed: {e}")',
        "        return False",
        "",
        "def display_pipeline_results():",
        '    """Display comprehensive pipeline results."""',
        "    ",
        '    results_dir = Path("pipeline_experiments/colab_career_progression_sequential")',
        "    ",
        "    if not results_dir.exists():",
        '        print("❌ Results directory not found")',
        "        return",
        "    ",
        '    print("\\n📊 Pipeline Results Summary:")',
        '    print("=" * 50)',
        "    ",
        "    # Training history",
        '    history_file = results_dir / "training_history.json"',
        "    if history_file.exists():",
        "        with open(history_file, 'r') as f:",
        "            history = json.load(f)",
        "        ",
        '        print(f"\\n📈 Training completed in {len(history)} epochs")',
        "        if history:",
        "            final_loss = history[-1].get('train_loss', 'N/A')",
        "            final_val_loss = history[-1].get('val_loss', 'N/A')",
        '            print(f"   Final training loss: {final_loss}")',
        '            print(f"   Final validation loss: {final_val_loss}")',
        "    ",
        "    # Model files",
        '    model_files = list(results_dir.glob("*.pth"))',
        "    if model_files:",
        '        print(f"\\n💾 Saved models: {len(model_files)}")',
        "        for model_file in model_files:",
        "            size_mb = model_file.stat().st_size / (1024 * 1024)",
        '            print(f"   {model_file.name}: {size_mb:.1f} MB")',
        "    ",
        "    # Reports",
        '    report_files = list(results_dir.glob("*.html")) + list(results_dir.glob("*.json"))',
        "    if report_files:",
        '        print(f"\\n📄 Generated reports: {len(report_files)}")',
        "        for report_file in report_files:",
        '            print(f"   {report_file.name}")',
        "    ",
        '    print("\\n✅ Complete pipeline execution finished!")',
        "",
        "# Execute the pipeline",
        'print("🎯 Starting Pipeline Execution...")',
        "success = run_complete_pipeline()",
        "",
        "if success:",
        "    display_pipeline_results()",
        "    ",
        '    print("\\n🎯 Next Steps:")',
        '    print("1. Review the generated reports in pipeline_experiments/")',
        '    print("2. Download the trained models for deployment")',
        '    print("3. Use the test metrics to evaluate model performance")',
        "else:",
        '    print("\\n🔧 Troubleshooting:")',
        '    print("1. Check GPU memory availability")',
        '    print("2. Verify all required files are present")',
        '    print("3. Review error messages above")'
    ]

    # Results display cell with comprehensive visualizations
    results_display_lines = [
        "# Display results with comprehensive visualizations",
        "import json",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "import numpy as np",
        "from pathlib import Path",
        "",
        "# Set style for better-looking plots",
        "plt.style.use('seaborn-v0_8-darkgrid')",
        "sns.set_palette('husl')",
        "",
        "results_dir = Path('pipeline_experiments/colab_career_progression_sequential')",
        "results_file = results_dir / 'pipeline_results.json'",
        "history_file = results_dir / 'training_history.json'",
        "",
        "if results_file.exists():",
        "    with open(results_file, 'r') as f:",
        "        results = json.load(f)",
        "    ",
        "    # 1. Display Test Metrics",
        "    print('='*60)",
        "    print('🎯 TEST RESULTS')",
        "    print('='*60)",
        "    if 'test_metrics' in results:",
        "        for metric, value in results['test_metrics'].items():",
        "            if isinstance(value, (int, float)):",
        "                print(f'  {metric}: {value:.4f}')",
        "    ",
        "    # 2. Create visualizations",
        "    fig = plt.figure(figsize=(20, 12))",
        "    ",
        "    # Plot 1: Training History (Loss over epochs)",
        "    if history_file.exists():",
        "        with open(history_file, 'r') as f:",
        "            history = json.load(f)",
        "        ",
        "        if history:",
        "            ax1 = plt.subplot(2, 3, 1)",
        "            epochs = [h.get('epoch', i+1) for i, h in enumerate(history)]",
        "            train_losses = [h.get('train_loss', 0) for h in history]",
        "            val_losses = [h.get('val_loss', 0) for h in history]",
        "            ",
        "            ax1.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)",
        "            ax1.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=6)",
        "            ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')",
        "            ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')",
        "            ax1.set_title('Training & Validation Loss Over Time', fontsize=14, fontweight='bold')",
        "            ax1.legend(fontsize=10)",
        "            ax1.grid(True, alpha=0.3)",
        "    ",
        "    # Plot 2: Validation Metrics Bar Chart",
        "    if 'validation_metrics' in results:",
        "        ax2 = plt.subplot(2, 3, 2)",
        "        val_metrics = {k: v for k, v in results['validation_metrics'].items() ",
        "                      if isinstance(v, (int, float)) and k not in ['loss']}",
        "        ",
        "        if val_metrics:",
        "            metric_names = list(val_metrics.keys())",
        "            metric_values = list(val_metrics.values())",
        "            ",
        "            bars = ax2.barh(metric_names, metric_values, color=sns.color_palette('viridis', len(metric_names)))",
        "            ax2.set_xlabel('Score', fontsize=12, fontweight='bold')",
        "            ax2.set_title('Validation Metrics', fontsize=14, fontweight='bold')",
        "            ax2.set_xlim(0, 1)",
        "            ",
        "            # Add value labels on bars",
        "            for i, (bar, value) in enumerate(zip(bars, metric_values)):",
        "                ax2.text(value + 0.02, i, f'{value:.3f}', va='center', fontsize=9)",
        "    ",
        "    # Plot 3: Test Metrics Bar Chart",
        "    if 'test_metrics' in results:",
        "        ax3 = plt.subplot(2, 3, 3)",
        "        test_metrics = {k: v for k, v in results['test_metrics'].items() ",
        "                       if isinstance(v, (int, float)) and k not in ['loss']}",
        "        ",
        "        if test_metrics:",
        "            metric_names = list(test_metrics.keys())",
        "            metric_values = list(test_metrics.values())",
        "            ",
        "            bars = ax3.barh(metric_names, metric_values, color=sns.color_palette('rocket', len(metric_names)))",
        "            ax3.set_xlabel('Score', fontsize=12, fontweight='bold')",
        "            ax3.set_title('Test Metrics', fontsize=14, fontweight='bold')",
        "            ax3.set_xlim(0, 1)",
        "            ",
        "            # Add value labels on bars",
        "            for i, (bar, value) in enumerate(zip(bars, metric_values)):",
        "                ax3.text(value + 0.02, i, f'{value:.3f}', va='center', fontsize=9)",
        "    ",
        "    # Plot 4: Confusion Matrix (if available in detailed metrics)",
        "    if 'test_metrics' in results and 'detailed_metrics' in results['test_metrics']:",
        "        detailed = results['test_metrics']['detailed_metrics']",
        "        if 'confusion_matrix' in detailed:",
        "            ax4 = plt.subplot(2, 3, 4)",
        "            cm = np.array(detailed['confusion_matrix'])",
        "            ",
        "            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4, ",
        "                       cbar_kws={'label': 'Count'}, annot_kws={'size': 12, 'weight': 'bold'})",
        "            ax4.set_xlabel('Predicted', fontsize=12, fontweight='bold')",
        "            ax4.set_ylabel('Actual', fontsize=12, fontweight='bold')",
        "            ax4.set_title('Confusion Matrix', fontsize=14, fontweight='bold')",
        "            ax4.set_xticklabels(['Negative', 'Positive'])",
        "            ax4.set_yticklabels(['Negative', 'Positive'])",
        "    ",
        "    # Plot 5: Metric Comparison (Val vs Test)",
        "    ax5 = plt.subplot(2, 3, 5)",
        "    if 'validation_metrics' in results and 'test_metrics' in results:",
        "        # Find common metrics",
        "        val_m = {k: v for k, v in results['validation_metrics'].items() if isinstance(v, (int, float))}",
        "        test_m = {k: v for k, v in results['test_metrics'].items() if isinstance(v, (int, float))}",
        "        common_metrics = set(val_m.keys()) & set(test_m.keys())",
        "        common_metrics = [m for m in common_metrics if m not in ['loss']]",
        "        ",
        "        if common_metrics:",
        "            x = np.arange(len(common_metrics))",
        "            width = 0.35",
        "            ",
        "            val_values = [val_m[m] for m in common_metrics]",
        "            test_values = [test_m[m] for m in common_metrics]",
        "            ",
        "            ax5.bar(x - width/2, val_values, width, label='Validation', alpha=0.8)",
        "            ax5.bar(x + width/2, test_values, width, label='Test', alpha=0.8)",
        "            ",
        "            ax5.set_xlabel('Metrics', fontsize=12, fontweight='bold')",
        "            ax5.set_ylabel('Score', fontsize=12, fontweight='bold')",
        "            ax5.set_title('Validation vs Test Performance', fontsize=14, fontweight='bold')",
        "            ax5.set_xticks(x)",
        "            ax5.set_xticklabels(common_metrics, rotation=45, ha='right')",
        "            ax5.legend(fontsize=10)",
        "            ax5.grid(True, alpha=0.3, axis='y')",
        "            ax5.set_ylim(0, 1)",
        "    ",
        "    # Plot 6: Training Progress Summary",
        "    ax6 = plt.subplot(2, 3, 6)",
        "    ax6.axis('off')",
        "    ",
        "    # Create summary text",
        "    summary_text = '📊 TRAINING SUMMARY\\n' + '='*40 + '\\n\\n'",
        "    ",
        "    if history_file.exists() and history:",
        "        summary_text += f'✓ Epochs completed: {len(history)}\\n'",
        "        summary_text += f'✓ Final train loss: {train_losses[-1]:.4f}\\n'",
        "        summary_text += f'✓ Final val loss: {val_losses[-1]:.4f}\\n'",
        "        loss_improvement = ((train_losses[0] - train_losses[-1]) / train_losses[0]) * 100",
        "        summary_text += f'✓ Loss improvement: {loss_improvement:.1f}%\\n\\n'",
        "    ",
        "    if 'test_metrics' in results:",
        "        test_m = results['test_metrics']",
        "        summary_text += '🎯 KEY TEST METRICS:\\n'",
        "        key_metrics = ['contrastive_accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']",
        "        for metric in key_metrics:",
        "            if metric in test_m and isinstance(test_m[metric], (int, float)):",
        "                summary_text += f'  • {metric}: {test_m[metric]:.4f}\\n'",
        "    ",
        "    # Model info",
        "    model_files = list(results_dir.glob('*.pth'))",
        "    if model_files:",
        "        summary_text += f'\\n� Saved models: {len(model_files)}\\n'",
        "    ",
        "    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,",
        "            fontsize=11, verticalalignment='center', family='monospace',",
        "            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))",
        "    ",
        "    plt.tight_layout()",
        "    plt.savefig(results_dir / 'training_visualizations.png', dpi=150, bbox_inches='tight')",
        "    print('\\n' + '='*60)",
        "    print('📊 VISUALIZATIONS CREATED')",
        "    print('='*60)",
        "    print(f'✓ Saved to: {results_dir / \"training_visualizations.png\"}')",
        "    plt.show()",
        "    ",
        "    # 3. File Summary",
        "    print('\\n' + '='*60)",
        "    print('📁 GENERATED FILES')",
        "    print('='*60)",
        "    for file_path in sorted(results_dir.glob('*')):",
        "        if file_path.is_file():",
        "            size_mb = file_path.stat().st_size / (1024*1024)",
        "            print(f'  ✓ {file_path.name}: {size_mb:.1f} MB')",
        "else:",
        "    print('❌ Results not found - please run the training pipeline first')"
    ]

    # Download results cell
    download_results_lines = [
        "# Package and download results",
        "import tarfile",
        "from pathlib import Path",
        "",
        "results_dir = Path('pipeline_experiments/colab_career_progression_sequential')",
        "if results_dir.exists():",
        "    # Create archive",
        "    archive_name = 'career_progression_results.tar.gz'",
        "    with tarfile.open(archive_name, 'w:gz') as tar:",
        "        tar.add(results_dir, arcname='results')",
        "    ",
        "    print(f\"✅ Results packaged: {archive_name}\")",
        "    archive_size = Path(archive_name).stat().st_size / (1024*1024)",
        "    print(f\"📦 Archive size: {archive_size:.1f} MB\")",
        "    ",
        "    # Try to download if in Colab environment",
        "    try:",
        "        from google.colab import files",
        "        files.download(archive_name)",
        "        print('✅ Results downloaded!')",
        "    except ImportError:",
        "        print('ℹ️  Not in Colab environment - archive created locally')",
        "        print(f'📁 Archive available at: {Path(archive_name).absolute()}')",
        "    except Exception as e:",
        "        print(f'⚠️  Download failed: {e}')",
        "        print(f'📁 Archive available at: {Path(archive_name).absolute()}')",
        "        ",
        "else:",
        "    print('❌ No results to download - run the pipeline training first')"
    ]

    notebook_content = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "accelerator": "GPU"
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"id": "title"},
                "source": [
                    "# Career-Aware Contrastive Learning System (CDCL)\n",
                    "\n",
                    "## 🚀 Complete ML Pipeline with Two-Phase Training\n",
                    "\n",
                    "This notebook implements a sophisticated contrastive learning pipeline for career-aware job-resume matching:\n",
                    "\n",
                    "### **Key Features**:\n",
                    "- **Sequential Data Splitting**: Prevents temporal data leakage (70% train / 15% val / 15% test)\n",
                    "- **Two-Phase Training**: Self-supervised pre-training + supervised fine-tuning\n",
                    "- **Career Progression Awareness**: ESCO taxonomy integration\n",
                    "- **Research-Grade Architecture**: Frozen SentenceTransformer prevents catastrophic forgetting\n",
                    "- **Comprehensive Evaluation**: 15+ metrics (Precision@K, MAP, NDCG, etc.)\n",
                    "- **Professional Reporting**: HTML + JSON reports with visualizations\n",
                    "\n",
                    "### **Training Modes**:\n",
                    "1. **Single-Phase**: Traditional supervised contrastive learning\n",
                    "2. **Two-Phase**: Pre-training on augmented data + fine-tuning on labeled data\n",
                    "\n",
                    "**Run each cell in order! 🎯**"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "setup-header"},
                "source": ["## 1. Environment Setup"]
            },
            {
                "cell_type": "code",
                "metadata": {"id": "setup-cell"},
                "execution_count": None,
                "outputs": [],
                "source": "\n".join(setup_script_lines)
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "env-verification-header"},
                "source": ["## 1.5. Environment Verification"]
            },
            {
                "cell_type": "code",
                "metadata": {"id": "env-verification-cell"},
                "execution_count": None,
                "outputs": [],
                "source": "\n".join(env_verification_lines)
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "data-header"},
                "source": ["## 2. Data Verification"]
            },
            {
                "cell_type": "code",
                "metadata": {"id": "data-cell"},
                "execution_count": None,
                "outputs": [],
                "source": "\n".join(data_verification_lines)
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "pipeline-header"},
                "source": ["## 3. Complete Pipeline Training"]
            },
            {
                "cell_type": "code",
                "metadata": {"id": "pipeline-cell"},
                "execution_count": None,
                "outputs": [],
                "source": "\n".join(pipeline_training_lines)
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "results-header"},
                "source": ["## 4. Results & Visualization"]
            },
            {
                "cell_type": "code",
                "metadata": {"id": "results-cell"},
                "execution_count": None,
                "outputs": [],
                "source": "\n".join(results_display_lines)
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "download-header"},
                "source": ["## 5. Download Results"]
            },
            {
                "cell_type": "code",
                "metadata": {"id": "download-cell"},
                "execution_count": None,
                "outputs": [],
                "source": "\n".join(download_results_lines)
            }
        ]
    }

    # Save notebook
    notebook_path = "colab_training_notebook.ipynb"
    with open(notebook_path, 'w') as f:
        json.dump(notebook_content, f, indent=2)

    print(f"✓ Created Colab notebook: {notebook_path}")


def main():
    """Main function to prepare all Colab materials."""
    print("🚀 Career-Aware Contrastive Learning System (CDCL) - Colab Preparation")
    print("=" * 70)
    print("Preparing complete ML pipeline with two-phase training for Google Colab")
    print("=" * 70)

    # 1. Verify required files
    print("\n1. Verifying required files...")
    file_status = verify_required_files()

    missing_files = [f for f, exists in file_status.items() if not exists]
    available_files = [f for f, exists in file_status.items() if exists]

    print(f"✓ Available files: {len(available_files)}")
    print(f"✗ Missing files: {len(missing_files)}")

    if missing_files:
        print("\nMissing files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\n⚠️  Some files are missing. Package will be created with available files.")

    # 2. Create training package
    print("\n2. Creating Colab training package...")
    archive_path = create_colab_package()

    # 3. Create Colab notebook
    print("\n3. Creating Colab notebook...")
    create_colab_notebook()

    # 4. Package summary
    print("\n4. Package summary...")
    print("✓ Notebook-only approach - cleaner and simpler")

    # 5. Summary
    print("\n" + "=" * 60)
    print("📦 COLAB PACKAGE READY!")
    print("=" * 60)

    print(f"\n📁 Files created:")
    print(f"   📦 {archive_path} ({Path(archive_path).stat().st_size / (1024*1024):.1f} MB)")
    print(f"   📓 colab_training_notebook.ipynb")

    print(f"\n🎯 Next steps:")
    print(f"   1. Upload {archive_path} to Google Drive (root directory)")
    print(f"   2. Open colab_training_notebook.ipynb in Google Colab")
    print(f"   3. Enable GPU runtime (Runtime → Change runtime type → GPU)")
    print(f"   4. Run all cells for complete ML pipeline!")

    print(f"\n✨ Features included:")
    print(f"   ✅ Sequential data splitting (70%/15%/15%) - prevents data leakage")
    print(f"   ✅ Two-phase training architecture (pre-training + fine-tuning)")
    print(f"   ✅ Research-grade model (frozen SentenceTransformer)")
    print(f"   ✅ Career-aware augmentation system")
    print(f"   ✅ ESCO taxonomy integration")
    print(f"   ✅ Comprehensive evaluation (Precision@K, MAP, NDCG, AUC)")
    print(f"   ✅ Real-time diagnostics and monitoring")
    print(f"   ✅ Professional reports and visualizations")
    print(f"   ✅ Embedding cache for efficiency")

    print(f"\n📊 Training Configurations:")
    print(f"   • Single-phase: Standard supervised contrastive learning")
    print(f"   • Two-phase: Self-supervised pre-training → supervised fine-tuning")
    print(f"   • Batch size: 32 (optimized for T4 GPU)")
    print(f"   • Learning rate: 0.00005 (conservative for stability)")
    print(f"   • Temperature: 0.2 (optimal for contrastive learning)")

    print(f"\n� Project Components:")
    print(f"   🧠 Contrastive Learning: Complete ML pipeline with strategy pattern")
    print(f"   📈 Two-Phase Training: Pre-training → Fine-tuning")
    print(f"   🔄 Career Augmentation: Aspirational + foundational views")
    print(f"   🎯 Matching System: Explainable career-aware matching")
    print(f"   🔍 Diagnostics: Real-time training monitoring")
    print(f"   📊 Evaluation: 15+ metrics with visualizations")

    print(f"\n�🚀 Ready for Google Colab training!")


if __name__ == "__main__":
    main()
