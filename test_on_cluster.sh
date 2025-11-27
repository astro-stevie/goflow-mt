#!/bin/bash

#SBATCH --partition=GPU-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --job-name=test_eval
#SBATCH --output=%x-%j.out

# Activate environment
source .venv/bin/activate

MODEL_PATHS=(
    "logs/train_neb_TS_322093/runs/2025-11-24_14-08-56/checkpoints/epoch_195.ckpt"
)

CSV=$(IFS=,; echo "${MODEL_PATHS[*]}")

uv run flow_train.py -m \
    experiment=flow2 \
    model=flow_neb \
    data=rdb7 \
    model.neb_img_subset=[1] \
    logger=[wandb,csv] \
    model.num_samples=25 \
    model.num_steps=25 \
    trainer=gpu \
    model.save_all_samples=True \
    task_name=test_neb_TS_${SLURM_JOB_ID} \
    project=goflow-neb-rdb7 \
    train=false \
    custom_model_weight_path="$CSV"
