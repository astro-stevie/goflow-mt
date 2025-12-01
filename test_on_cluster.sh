#!/bin/bash

#SBATCH --partition=GPU-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --job-name=test_eval
#SBATCH --output=%x-%j.out

# Activate environment
source .venv/bin/activate

MODEL_PATHS=(
    "logs/train_test_neb_all_323433/runs/2025-11-28_15-39-16/checkpoints/epoch_172.ckpt"
)

CSV=$(IFS=,; echo "${MODEL_PATHS[*]}")

uv run flow_train.py -m \
    experiment=flow2_neb \
    model=flow_neb \
    data=t1x \
    model.neb_img_subset=all \
    logger=[wandb,csv] \
    model.num_samples=25 \
    model.num_steps=25 \
    trainer=gpu \
    model.save_all_samples=True \
    task_name=test_neb_all_${SLURM_JOB_ID} \
    project=goflow-neb-t1x \
    train=false \
    custom_model_weight_path="$CSV"
