#!/bin/bash

#SBATCH --partition=GPU-a100s
#SBATCH --gres=gpu:a100s:1
#SBATCH --nodes=1
#SBATCH --job-name=test_eval
#SBATCH --output=%x-%j.out

# Activate environment
source .venv/bin/activate

MODEL_PATHS=(
    "logs/train_test_neb_all_323916/runs/2025-12-01_15-22-36/checkpoints/epoch_168.ckpt"
)


CSV=$(IFS=,; echo "${MODEL_PATHS[*]}")

uv run flow_train.py -m \
    experiment=flow2_neb \
    model=flow_neb \
    data=t1x \
    logger=[wandb,csv] \
    model.num_steps=25 \
    model.num_samples=20 \
    trainer=gpu \
    model.neb_img_subset=all \
    model.save_all_samples=True \
    model.preserve_path_coherence=false \
    model.rmse_extended_metrics=false \
    task_name=test_neb_20sam_${SLURM_JOB_ID}\
    project=goflow-neb-t1x \
    train=false \
    custom_model_weight_path="$CSV"
