#!/bin/bash

#SBATCH --partition=GPU-a100s
#SBATCH --gres=gpu:a100s:1
#SBATCH --nodes=1
#SBATCH --job-name=test_neb_ablation
#SBATCH --output=%x-%j.out

# NEB ablation study script for cluster
# Submit with: sbatch test_neb_ablation.sh

echo "Starting NEB ablation study on cluster..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Test different NEB configurations
# This will test various NEB image counts and sampling parameters
uv run flow_train.py -m \
    model.neb_images=5,8,10 \
    model.num_samples=10,25 \
    trainer=gpu \
    trainer.max_epochs=10 \
    +trainer.limit_train_batches=200 \
    +trainer.limit_val_batches=50 \
    data=t1x \
    experiment=flow_neb_test \
    task_name=test_neb_ablation \
    logger=[wandb,csv] \
    train=True \
    test=False

echo "NEB ablation study complete!"
echo "Results saved to: logs/test_neb_ablation/"