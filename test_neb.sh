#!/bin/bash

#SBATCH --partition=GPU-a100s
#SBATCH --gres=gpu:a100s:1
#SBATCH --nodes=1
#SBATCH --job-name=test_neb
#SBATCH --output=%x-%j.out

# Submit with: sbatch test_neb.sh

echo "Starting NEB testing experiment..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"

# Use the trained checkpoint for testing only
CHECKPOINT_PATH="logs/train_neb_full_298844/runs/2025-10-07_18-50-32/checkpoints/last.ckpt"

# Optional performance overrides (can be set in environment before sbatch)
# Example: export NUM_WORKERS=6 PIN_MEMORY=true MP_CTX=spawn
NUM_WORKERS=${NUM_WORKERS:-2}
PIN_MEMORY=${PIN_MEMORY:-true}
MP_CTX=${MP_CTX:-spawn}

if [ -f "$CHECKPOINT_PATH" ]; then
    echo "Found checkpoint: $CHECKPOINT_PATH"
    
    # Run test only with the trained NEB model
    uv run flow_train.py \
        experiment=flow_neb_test \
        model=flow_neb \
        data=t1x \
        trainer=gpu \
        logger=wandb \
        model.num_steps=25 \
        model.num_samples=50 \
        task_name=test_neb_full_${SLURM_JOB_ID} \
        project=goflow-neb \
        train=false \
        test=true \
        ckpt_path="$CHECKPOINT_PATH" \
        custom_model_weight_path=null \
        data.num_workers=${NUM_WORKERS} \
        data.pin_memory=${PIN_MEMORY} \
        data.multiprocessing_context=${MP_CTX}
        
    echo "NEB testing experiment complete!"
    echo "Check logs at: /home/stephg/goflow/logs/test_neb_full_${SLURM_JOB_ID}.out"
else
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Available checkpoints in directory:"
    ls -la logs/train_neb_full_298844/runs/2025-10-07_18-50-32/checkpoints/ || echo "Directory not found"
fi