#!/bin/bash

#SBATCH --partition=GPU-a100s
#SBATCH --gres=gpu:a100s:1
#SBATCH --nodes=1
#SBATCH --job-name=test_cluster_neb
#SBATCH --output=%x-%j.out

# Cluster testing script for NEB multi-image implementation
# Submit with: sbatch test_cluster_neb.sh

echo "Starting NEB multi-image cluster test..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Test NEB training with GPU config - trains for a few epochs
uv run flow_train.py \
    debug=neb_test \
    experiment=flow_neb_test \
    data=t1x \
    trainer=gpu \
    trainer.max_epochs=5 \
    +trainer.limit_train_batches=100 \
    +trainer.limit_val_batches=20 \
    logger=[wandb,csv] \
    train=True \
    test=False

echo "Cluster NEB test complete!"
echo "Check logs in: logs/experiment_runs/"