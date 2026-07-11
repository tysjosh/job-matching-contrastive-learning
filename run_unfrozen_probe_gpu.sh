#!/usr/bin/env bash
set -e

echo "=== UF-Ordinal__cnamuangtoun__s13 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/UF-Ordinal__cnamuangtoun__s13/training_config.json --output-dir results/research_runs/UF-Ordinal__cnamuangtoun__s13/phase1_pretraining --seed 13
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/UF-Ordinal__cnamuangtoun__s13/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/UF-Ordinal__cnamuangtoun__s13/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/UF-Ordinal__cnamuangtoun__s13/phase1_evaluation

echo "=== UF-Ordinal__cnamuangtoun__s21 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/UF-Ordinal__cnamuangtoun__s21/training_config.json --output-dir results/research_runs/UF-Ordinal__cnamuangtoun__s21/phase1_pretraining --seed 21
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/UF-Ordinal__cnamuangtoun__s21/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/UF-Ordinal__cnamuangtoun__s21/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/UF-Ordinal__cnamuangtoun__s21/phase1_evaluation

echo "=== UF-Ordinal__cnamuangtoun__s42 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/UF-Ordinal__cnamuangtoun__s42/training_config.json --output-dir results/research_runs/UF-Ordinal__cnamuangtoun__s42/phase1_pretraining --seed 42
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/UF-Ordinal__cnamuangtoun__s42/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/UF-Ordinal__cnamuangtoun__s42/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/UF-Ordinal__cnamuangtoun__s42/phase1_evaluation

echo "=== UF-Ordinal__cnamuangtoun__s87 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/UF-Ordinal__cnamuangtoun__s87/training_config.json --output-dir results/research_runs/UF-Ordinal__cnamuangtoun__s87/phase1_pretraining --seed 87
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/UF-Ordinal__cnamuangtoun__s87/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/UF-Ordinal__cnamuangtoun__s87/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/UF-Ordinal__cnamuangtoun__s87/phase1_evaluation

echo "=== UF-Ordinal__cnamuangtoun__s123 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/UF-Ordinal__cnamuangtoun__s123/training_config.json --output-dir results/research_runs/UF-Ordinal__cnamuangtoun__s123/phase1_pretraining --seed 123
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/UF-Ordinal__cnamuangtoun__s123/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/UF-Ordinal__cnamuangtoun__s123/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/UF-Ordinal__cnamuangtoun__s123/phase1_evaluation

echo "=== UF-InfoNCE__cnamuangtoun__s13 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/UF-InfoNCE__cnamuangtoun__s13/training_config.json --output-dir results/research_runs/UF-InfoNCE__cnamuangtoun__s13/phase1_pretraining --seed 13
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/UF-InfoNCE__cnamuangtoun__s13/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/UF-InfoNCE__cnamuangtoun__s13/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/UF-InfoNCE__cnamuangtoun__s13/phase1_evaluation

echo "=== UF-InfoNCE__cnamuangtoun__s21 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/UF-InfoNCE__cnamuangtoun__s21/training_config.json --output-dir results/research_runs/UF-InfoNCE__cnamuangtoun__s21/phase1_pretraining --seed 21
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/UF-InfoNCE__cnamuangtoun__s21/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/UF-InfoNCE__cnamuangtoun__s21/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/UF-InfoNCE__cnamuangtoun__s21/phase1_evaluation

echo "=== UF-InfoNCE__cnamuangtoun__s42 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/UF-InfoNCE__cnamuangtoun__s42/training_config.json --output-dir results/research_runs/UF-InfoNCE__cnamuangtoun__s42/phase1_pretraining --seed 42
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/UF-InfoNCE__cnamuangtoun__s42/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/UF-InfoNCE__cnamuangtoun__s42/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/UF-InfoNCE__cnamuangtoun__s42/phase1_evaluation

echo "=== UF-InfoNCE__cnamuangtoun__s87 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/UF-InfoNCE__cnamuangtoun__s87/training_config.json --output-dir results/research_runs/UF-InfoNCE__cnamuangtoun__s87/phase1_pretraining --seed 87
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/UF-InfoNCE__cnamuangtoun__s87/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/UF-InfoNCE__cnamuangtoun__s87/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/UF-InfoNCE__cnamuangtoun__s87/phase1_evaluation

echo "=== UF-InfoNCE__cnamuangtoun__s123 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/UF-InfoNCE__cnamuangtoun__s123/training_config.json --output-dir results/research_runs/UF-InfoNCE__cnamuangtoun__s123/phase1_pretraining --seed 123
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/UF-InfoNCE__cnamuangtoun__s123/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/UF-InfoNCE__cnamuangtoun__s123/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/UF-InfoNCE__cnamuangtoun__s123/phase1_evaluation
