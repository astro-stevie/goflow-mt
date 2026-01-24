#!/bin/bash

#SBATCH --partition=GPU-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --job-name=test_eval
#SBATCH --output=%x-%j.out

# Activate environment
source .venv/bin/activate

MODEL_PATHS=(
    "logs/train_neb_361841_0lam/runs/2026-01-22_15-24-47/checkpoints/epoch_168.ckpt"
)


CSV=$(IFS=,; echo "${MODEL_PATHS[*]}")

uv run flow_train.py -m \
    experiment=flow2_neb \
    model=flow_neb \
    data=t1x \
    logger=[wandb,csv] \
    model.num_steps=25 \
    model.num_samples=1 \
    trainer=gpu \
    model.neb_img_subset=all \
    model.save_all_samples=True \
    model.preserve_path_coherence=true \
    model.rmse_extended_metrics=true \
    task_name=test_neb_${SLURM_JOB_ID}_pc \
    project=goflow-neb-t1x-rmse \
    train=false \
    custom_model_weight_path="$CSV"
