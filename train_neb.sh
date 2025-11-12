#!/bin/bash

#SBATCH --partition=GPU-a100s
#SBATCH --gres=gpu:a100s:1
#SBATCH --nodes=1
#SBATCH --job-name=train_neb
#SBATCH --output=%x-%j.out

# Submit with: sbatch train_neb.sh

# Activate environment
source .venv/bin/activate

echo "Starting NEB full training experiment..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"

# Run full NEB training with proper configs
uv run flow_train.py \
    experiment=flow2 \
    model=flow_neb \
    data=rdb7 \
    model.neb_img_subset=all \
    logger=[wandb,csv] \
    model.num_steps=25 \
    model.num_samples=50 \
    task_name=train_neb_all_${SLURM_JOB_ID} \
    project=goflow-neb-rdb7 \
    test=false

echo "NEB training experiment complete!"
echo "Check logs at: /home/stephg/goflow/logs/train_neb_all_${SLURM_JOB_ID}.out"