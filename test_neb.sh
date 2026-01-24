#!/bin/bash

#SBATCH --partition=GPU-a100s
#SBATCH --gres=gpu:a100s:1
#SBATCH --nodes=1
#SBATCH --job-name=test_neb
#SBATCH --output=%x-%j.out

# Submit with: sbatch test_neb.sh

# Activate environment
source .venv/bin/activate

echo "Starting NEB testing experiment..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"

# Use the trained checkpoint for testing only
CHECKPOINT_PATH="goflow/logs/train_neb_springF_1l_351054/runs/2026-01-08_18-50-51/checkpoints/epoch_168.ckpt"

if [ -f "$CHECKPOINT_PATH" ]; then
    echo "Found checkpoint: $CHECKPOINT_PATH"
    
    # Run test only with the trained NEB model
    uv run flow_train.py \
        experiment=flow2_neb \
        model=flow_neb \
        data=t1x \
        model.neb_img_subset=all \
        trainer=gpu \
        logger=[wandb,csv] \
        model.num_steps=25 \
        model.num_samples=50 \
        task_name=test_neb_${SLURM_JOB_ID}_1lam \
        project=goflow-neb-t1x-rmse \
        train=false \
        test=true \
        ckpt_path="$CHECKPOINT_PATH" \
        custom_model_weight_path=null
        
    echo "NEB testing experiment complete!"
    echo "Check logs at: /home/stephg/goflow/logs/test_neb_all_${SLURM_JOB_ID}.out"
else
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Available checkpoints in directory:"
    ls -la logs/train_test_neb_all_323916/runs/2025-12-01_15-22-36/checkpoints/ || echo "Directory not found"
fi