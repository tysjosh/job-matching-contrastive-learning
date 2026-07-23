#!/usr/bin/env bash
set -e

echo "=== EO-A__cnamuangtoun__s13 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-A__cnamuangtoun__s13/training_config.json --output-dir results/research_runs/EO-A__cnamuangtoun__s13/phase1_pretraining --seed 13
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-A__cnamuangtoun__s13/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-A__cnamuangtoun__s13/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-A__cnamuangtoun__s13/phase1_evaluation

echo "=== EO-A__cnamuangtoun__s21 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-A__cnamuangtoun__s21/training_config.json --output-dir results/research_runs/EO-A__cnamuangtoun__s21/phase1_pretraining --seed 21
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-A__cnamuangtoun__s21/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-A__cnamuangtoun__s21/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-A__cnamuangtoun__s21/phase1_evaluation

echo "=== EO-A__cnamuangtoun__s42 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-A__cnamuangtoun__s42/training_config.json --output-dir results/research_runs/EO-A__cnamuangtoun__s42/phase1_pretraining --seed 42
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-A__cnamuangtoun__s42/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-A__cnamuangtoun__s42/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-A__cnamuangtoun__s42/phase1_evaluation

echo "=== EO-A__cnamuangtoun__s87 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-A__cnamuangtoun__s87/training_config.json --output-dir results/research_runs/EO-A__cnamuangtoun__s87/phase1_pretraining --seed 87
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-A__cnamuangtoun__s87/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-A__cnamuangtoun__s87/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-A__cnamuangtoun__s87/phase1_evaluation

echo "=== EO-A__cnamuangtoun__s123 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-A__cnamuangtoun__s123/training_config.json --output-dir results/research_runs/EO-A__cnamuangtoun__s123/phase1_pretraining --seed 123
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-A__cnamuangtoun__s123/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-A__cnamuangtoun__s123/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-A__cnamuangtoun__s123/phase1_evaluation

echo "=== EO-B__cnamuangtoun__s13 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-B__cnamuangtoun__s13/training_config.json --output-dir results/research_runs/EO-B__cnamuangtoun__s13/phase1_pretraining --seed 13
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-B__cnamuangtoun__s13/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-B__cnamuangtoun__s13/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-B__cnamuangtoun__s13/phase1_evaluation

echo "=== EO-B__cnamuangtoun__s21 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-B__cnamuangtoun__s21/training_config.json --output-dir results/research_runs/EO-B__cnamuangtoun__s21/phase1_pretraining --seed 21
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-B__cnamuangtoun__s21/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-B__cnamuangtoun__s21/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-B__cnamuangtoun__s21/phase1_evaluation

echo "=== EO-B__cnamuangtoun__s42 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-B__cnamuangtoun__s42/training_config.json --output-dir results/research_runs/EO-B__cnamuangtoun__s42/phase1_pretraining --seed 42
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-B__cnamuangtoun__s42/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-B__cnamuangtoun__s42/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-B__cnamuangtoun__s42/phase1_evaluation

echo "=== EO-B__cnamuangtoun__s87 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-B__cnamuangtoun__s87/training_config.json --output-dir results/research_runs/EO-B__cnamuangtoun__s87/phase1_pretraining --seed 87
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-B__cnamuangtoun__s87/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-B__cnamuangtoun__s87/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-B__cnamuangtoun__s87/phase1_evaluation

echo "=== EO-B__cnamuangtoun__s123 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-B__cnamuangtoun__s123/training_config.json --output-dir results/research_runs/EO-B__cnamuangtoun__s123/phase1_pretraining --seed 123
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-B__cnamuangtoun__s123/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-B__cnamuangtoun__s123/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-B__cnamuangtoun__s123/phase1_evaluation

echo "=== EO-C__cnamuangtoun__s13 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-C__cnamuangtoun__s13/training_config.json --output-dir results/research_runs/EO-C__cnamuangtoun__s13/phase1_pretraining --seed 13
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-C__cnamuangtoun__s13/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-C__cnamuangtoun__s13/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-C__cnamuangtoun__s13/phase1_evaluation

echo "=== EO-C__cnamuangtoun__s21 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-C__cnamuangtoun__s21/training_config.json --output-dir results/research_runs/EO-C__cnamuangtoun__s21/phase1_pretraining --seed 21
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-C__cnamuangtoun__s21/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-C__cnamuangtoun__s21/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-C__cnamuangtoun__s21/phase1_evaluation

echo "=== EO-C__cnamuangtoun__s42 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-C__cnamuangtoun__s42/training_config.json --output-dir results/research_runs/EO-C__cnamuangtoun__s42/phase1_pretraining --seed 42
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-C__cnamuangtoun__s42/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-C__cnamuangtoun__s42/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-C__cnamuangtoun__s42/phase1_evaluation

echo "=== EO-C__cnamuangtoun__s87 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-C__cnamuangtoun__s87/training_config.json --output-dir results/research_runs/EO-C__cnamuangtoun__s87/phase1_pretraining --seed 87
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-C__cnamuangtoun__s87/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-C__cnamuangtoun__s87/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-C__cnamuangtoun__s87/phase1_evaluation

echo "=== EO-C__cnamuangtoun__s123 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-C__cnamuangtoun__s123/training_config.json --output-dir results/research_runs/EO-C__cnamuangtoun__s123/phase1_pretraining --seed 123
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-C__cnamuangtoun__s123/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-C__cnamuangtoun__s123/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-C__cnamuangtoun__s123/phase1_evaluation

echo "=== EO-D__cnamuangtoun__s13 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-D__cnamuangtoun__s13/training_config.json --output-dir results/research_runs/EO-D__cnamuangtoun__s13/phase1_pretraining --seed 13
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-D__cnamuangtoun__s13/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-D__cnamuangtoun__s13/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-D__cnamuangtoun__s13/phase1_evaluation

echo "=== EO-D__cnamuangtoun__s21 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-D__cnamuangtoun__s21/training_config.json --output-dir results/research_runs/EO-D__cnamuangtoun__s21/phase1_pretraining --seed 21
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-D__cnamuangtoun__s21/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-D__cnamuangtoun__s21/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-D__cnamuangtoun__s21/phase1_evaluation

echo "=== EO-D__cnamuangtoun__s42 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-D__cnamuangtoun__s42/training_config.json --output-dir results/research_runs/EO-D__cnamuangtoun__s42/phase1_pretraining --seed 42
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-D__cnamuangtoun__s42/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-D__cnamuangtoun__s42/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-D__cnamuangtoun__s42/phase1_evaluation

echo "=== EO-D__cnamuangtoun__s87 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-D__cnamuangtoun__s87/training_config.json --output-dir results/research_runs/EO-D__cnamuangtoun__s87/phase1_pretraining --seed 87
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-D__cnamuangtoun__s87/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-D__cnamuangtoun__s87/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-D__cnamuangtoun__s87/phase1_evaluation

echo "=== EO-D__cnamuangtoun__s123 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-D__cnamuangtoun__s123/training_config.json --output-dir results/research_runs/EO-D__cnamuangtoun__s123/phase1_pretraining --seed 123
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-D__cnamuangtoun__s123/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-D__cnamuangtoun__s123/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-D__cnamuangtoun__s123/phase1_evaluation

echo "=== EO-E__cnamuangtoun__s13 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-E__cnamuangtoun__s13/training_config.json --output-dir results/research_runs/EO-E__cnamuangtoun__s13/phase1_pretraining --seed 13
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-E__cnamuangtoun__s13/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-E__cnamuangtoun__s13/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-E__cnamuangtoun__s13/phase1_evaluation

echo "=== EO-E__cnamuangtoun__s21 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-E__cnamuangtoun__s21/training_config.json --output-dir results/research_runs/EO-E__cnamuangtoun__s21/phase1_pretraining --seed 21
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-E__cnamuangtoun__s21/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-E__cnamuangtoun__s21/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-E__cnamuangtoun__s21/phase1_evaluation

echo "=== EO-E__cnamuangtoun__s42 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-E__cnamuangtoun__s42/training_config.json --output-dir results/research_runs/EO-E__cnamuangtoun__s42/phase1_pretraining --seed 42
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-E__cnamuangtoun__s42/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-E__cnamuangtoun__s42/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-E__cnamuangtoun__s42/phase1_evaluation

echo "=== EO-E__cnamuangtoun__s87 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-E__cnamuangtoun__s87/training_config.json --output-dir results/research_runs/EO-E__cnamuangtoun__s87/phase1_pretraining --seed 87
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-E__cnamuangtoun__s87/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-E__cnamuangtoun__s87/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-E__cnamuangtoun__s87/phase1_evaluation

echo "=== EO-E__cnamuangtoun__s123 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-E__cnamuangtoun__s123/training_config.json --output-dir results/research_runs/EO-E__cnamuangtoun__s123/phase1_pretraining --seed 123
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-E__cnamuangtoun__s123/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-E__cnamuangtoun__s123/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-E__cnamuangtoun__s123/phase1_evaluation

echo "=== EO-OntNeg__cnamuangtoun__s13 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-OntNeg__cnamuangtoun__s13/training_config.json --output-dir results/research_runs/EO-OntNeg__cnamuangtoun__s13/phase1_pretraining --seed 13
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-OntNeg__cnamuangtoun__s13/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-OntNeg__cnamuangtoun__s13/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-OntNeg__cnamuangtoun__s13/phase1_evaluation

echo "=== EO-OntNeg__cnamuangtoun__s21 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-OntNeg__cnamuangtoun__s21/training_config.json --output-dir results/research_runs/EO-OntNeg__cnamuangtoun__s21/phase1_pretraining --seed 21
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-OntNeg__cnamuangtoun__s21/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-OntNeg__cnamuangtoun__s21/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-OntNeg__cnamuangtoun__s21/phase1_evaluation

echo "=== EO-OntNeg__cnamuangtoun__s42 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-OntNeg__cnamuangtoun__s42/training_config.json --output-dir results/research_runs/EO-OntNeg__cnamuangtoun__s42/phase1_pretraining --seed 42
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-OntNeg__cnamuangtoun__s42/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-OntNeg__cnamuangtoun__s42/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-OntNeg__cnamuangtoun__s42/phase1_evaluation

echo "=== EO-OntNeg__cnamuangtoun__s87 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-OntNeg__cnamuangtoun__s87/training_config.json --output-dir results/research_runs/EO-OntNeg__cnamuangtoun__s87/phase1_pretraining --seed 87
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-OntNeg__cnamuangtoun__s87/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-OntNeg__cnamuangtoun__s87/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-OntNeg__cnamuangtoun__s87/phase1_evaluation

echo "=== EO-OntNeg__cnamuangtoun__s123 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-OntNeg__cnamuangtoun__s123/training_config.json --output-dir results/research_runs/EO-OntNeg__cnamuangtoun__s123/phase1_pretraining --seed 123
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-OntNeg__cnamuangtoun__s123/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-OntNeg__cnamuangtoun__s123/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-OntNeg__cnamuangtoun__s123/phase1_evaluation

echo "=== EO-ISCONeg__cnamuangtoun__s13 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-ISCONeg__cnamuangtoun__s13/training_config.json --output-dir results/research_runs/EO-ISCONeg__cnamuangtoun__s13/phase1_pretraining --seed 13
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-ISCONeg__cnamuangtoun__s13/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-ISCONeg__cnamuangtoun__s13/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-ISCONeg__cnamuangtoun__s13/phase1_evaluation

echo "=== EO-ISCONeg__cnamuangtoun__s21 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-ISCONeg__cnamuangtoun__s21/training_config.json --output-dir results/research_runs/EO-ISCONeg__cnamuangtoun__s21/phase1_pretraining --seed 21
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-ISCONeg__cnamuangtoun__s21/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-ISCONeg__cnamuangtoun__s21/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-ISCONeg__cnamuangtoun__s21/phase1_evaluation

echo "=== EO-ISCONeg__cnamuangtoun__s42 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-ISCONeg__cnamuangtoun__s42/training_config.json --output-dir results/research_runs/EO-ISCONeg__cnamuangtoun__s42/phase1_pretraining --seed 42
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-ISCONeg__cnamuangtoun__s42/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-ISCONeg__cnamuangtoun__s42/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-ISCONeg__cnamuangtoun__s42/phase1_evaluation

echo "=== EO-ISCONeg__cnamuangtoun__s87 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-ISCONeg__cnamuangtoun__s87/training_config.json --output-dir results/research_runs/EO-ISCONeg__cnamuangtoun__s87/phase1_pretraining --seed 87
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-ISCONeg__cnamuangtoun__s87/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-ISCONeg__cnamuangtoun__s87/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-ISCONeg__cnamuangtoun__s87/phase1_evaluation

echo "=== EO-ISCONeg__cnamuangtoun__s123 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-ISCONeg__cnamuangtoun__s123/training_config.json --output-dir results/research_runs/EO-ISCONeg__cnamuangtoun__s123/phase1_pretraining --seed 123
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-ISCONeg__cnamuangtoun__s123/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-ISCONeg__cnamuangtoun__s123/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-ISCONeg__cnamuangtoun__s123/phase1_evaluation

echo "=== EO-RandNeg__cnamuangtoun__s13 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-RandNeg__cnamuangtoun__s13/training_config.json --output-dir results/research_runs/EO-RandNeg__cnamuangtoun__s13/phase1_pretraining --seed 13
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-RandNeg__cnamuangtoun__s13/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-RandNeg__cnamuangtoun__s13/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-RandNeg__cnamuangtoun__s13/phase1_evaluation

echo "=== EO-RandNeg__cnamuangtoun__s21 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-RandNeg__cnamuangtoun__s21/training_config.json --output-dir results/research_runs/EO-RandNeg__cnamuangtoun__s21/phase1_pretraining --seed 21
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-RandNeg__cnamuangtoun__s21/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-RandNeg__cnamuangtoun__s21/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-RandNeg__cnamuangtoun__s21/phase1_evaluation

echo "=== EO-RandNeg__cnamuangtoun__s42 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-RandNeg__cnamuangtoun__s42/training_config.json --output-dir results/research_runs/EO-RandNeg__cnamuangtoun__s42/phase1_pretraining --seed 42
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-RandNeg__cnamuangtoun__s42/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-RandNeg__cnamuangtoun__s42/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-RandNeg__cnamuangtoun__s42/phase1_evaluation

echo "=== EO-RandNeg__cnamuangtoun__s87 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-RandNeg__cnamuangtoun__s87/training_config.json --output-dir results/research_runs/EO-RandNeg__cnamuangtoun__s87/phase1_pretraining --seed 87
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-RandNeg__cnamuangtoun__s87/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-RandNeg__cnamuangtoun__s87/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-RandNeg__cnamuangtoun__s87/phase1_evaluation

echo "=== EO-RandNeg__cnamuangtoun__s123 ==="
python -m contrastive_learning train preprocess/data_splits_v7/train.jsonl --config results/research_runs/EO-RandNeg__cnamuangtoun__s123/training_config.json --output-dir results/research_runs/EO-RandNeg__cnamuangtoun__s123/phase1_pretraining --seed 123
python run_ordinal_evaluation.py --ordinal-checkpoint results/research_runs/EO-RandNeg__cnamuangtoun__s123/phase1_pretraining/best_checkpoint.pt --ordinal-config results/research_runs/EO-RandNeg__cnamuangtoun__s123/training_config.json --dataset preprocess/data_splits_v7/test.jsonl --output-dir results/research_runs/EO-RandNeg__cnamuangtoun__s123/phase1_evaluation
