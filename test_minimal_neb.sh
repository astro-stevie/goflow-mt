#!/bin/bash

# Test-only script for NEB using a minimal configuration
# This avoids config conflicts with hardcoded checkpoint paths

echo "Starting NEB test-only run using checkpoint (minimal config)..."

# Use the latest checkpoint for testing only
CHECKPOINT_PATH="logs/train_neb_full_298844/runs/2025-10-07_18-50-32/checkpoints/last.ckpt"

if [ -f "$CHECKPOINT_PATH" ]; then
    echo "Found checkpoint: $CHECKPOINT_PATH"
    
    # Run test only with minimal config to avoid checkpoint conflicts
    python flow_train.py \
        model=flow_neb \
        data=t1x \
        trainer=cpu \
        logger=csv \
        train=False \
        test=True \
        ckpt_path="$CHECKPOINT_PATH" \
        trainer.max_epochs=1 \
        data.batch_size=2 \
        model.num_steps=5 \
        model.num_samples=1 \
        custom_model_weight_path=null
        
    echo "Test-only run complete!"
else
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Available checkpoints:"
    ls -la logs/train_neb_full_298844/runs/2025-10-07_18-50-32/checkpoints/
fi