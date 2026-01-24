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

# Run experiment 1: flow1_neb
echo "Starting experiment flow2_neb..."
uv run flow_train.py \
    experiment=flow2_neb \
    model=flow_neb \
    data=t1x \
    logger=[wandb,csv] \
    model.num_steps=25 \
    model.num_samples=50 \
    model.neb_img_subset=[0,1,2,3]\
    task_name=train_test_neb_TS_0-3i_${SLURM_JOB_ID}\
    project=goflow-neb-t1x \
    test=true \
    trainer.gradient_clip_val=1.0

# Run experiment 1: flow1_neb
echo "Starting experiment flow2_neb..."
uv run flow_train.py \
    experiment=flow2_neb \
    model=flow_neb \
    data=t1x \
    logger=[wandb,csv] \
    model.num_steps=25 \
    model.num_samples=50 \
    model.neb_img_subset=[3,4,5,6,7]\
    task_name=train_test_neb_TS_3-7i_${SLURM_JOB_ID}\
    project=goflow-neb-t1x \
    test=true \
    trainer.gradient_clip_val=1.0

# Run experiment 1: flow1_neb
echo "Starting experiment flow2_neb..."
uv run flow_train.py \
    experiment=flow2_neb \
    model=flow_neb \
    data=t1x \
    logger=[wandb,csv] \
    model.num_steps=25 \
    model.num_samples=50 \
    model.neb_img_subset=[6,7,8,9]\
    task_name=train_test_neb_TS_6-9i_${SLURM_JOB_ID}\
    project=goflow-neb-t1x \
    test=true \
    trainer.gradient_clip_val=1.0

# Run experiment 1: flow1_neb
echo "Starting experiment flow2_neb..."
uv run flow_train.py \
    experiment=flow2_neb \
    model=flow_neb \
    data=t1x \
    logger=[wandb,csv] \
    model.num_steps=25 \
    model.num_samples=50 \
    model.neb_img_subset=[0,2,3,6,8]\
    task_name=train_test_neb_TS_eveni_${SLURM_JOB_ID}\
    project=goflow-neb-t1x \
    test=true \
    trainer.gradient_clip_val=1.0

# Run experiment 1: flow1_neb
echo "Starting experiment flow2_neb..."
uv run flow_train.py \
    experiment=flow2_neb \
    model=flow_neb \
    data=t1x \
    logger=[wandb,csv] \
    model.num_steps=25 \
    model.num_samples=50 \
    model.neb_img_subset=[0,5,9]\
    task_name=train_test_neb_TS_rtsp_${SLURM_JOB_ID}\
    project=goflow-neb-t1x \
    test=true \
    trainer.gradient_clip_val=1.0

echo "All NEB training experiments complete!"
echo "Check logs at: /home/stephg/goflow/logs/train_test_neb_TS_${SLURM_JOB_ID}.out"