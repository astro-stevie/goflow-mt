#!/bin/bash

#SBATCH --partition=GPU-a100s
#SBATCH --gres=gpu:a100s:1
#SBATCH --nodes=1
#SBATCH --job-name=train_test_all_model_sizes
#SBATCH --output=%x-%j.out

# Submit with: sbatch train_test_flow_ref.sh

echo "Starting experiment..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"

# Activate environment
source .venv/bin/activate

uv run flow_train.py -m \
    experiment=flow2 \
    model=flow \
    data=rdb7 \
    trainer=gpu \
    model.num_steps=25 \
    model.num_samples=50 \
    task_name=train_test_flow_ref_${SLURM_JOB_ID} \
    project=goflow-neb \
    logger=[wandb,csv]
    