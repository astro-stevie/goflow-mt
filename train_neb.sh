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
    experiment=flow2_neb \
    model=flow_neb \
    data=t1x \
    logger=[wandb,csv] \
<<<<<<< HEAD
<<<<<<< HEAD
    model.num_steps=25 \
=======
    model.neb_img_subset=all \
    model.num_steps=50 \
>>>>>>> fbab8f8 (added linear interpolation of energy-based spring froces and alighment to reference rotation framework)
    model.num_samples=50 \
<<<<<<< HEAD
    task_name=train_test_neb_TS_${SLURM_JOB_ID} \
    project=goflow-neb-t1x \
=======
    model.lambda_spring=0.01 \
    model.spring_loss_schedule=cosine \
    task_name=train_neb_cos_springF_${SLURM_JOB_ID} \
=======
    model.neb_img_subset=[0,1,2,3] \
    model.num_steps=25 \
    model.num_samples=50 \
    model.lambda_spring=1.0 \
    model.spring_loss_schedule=linear \
    task_name=train_neb_0-3i_springF_${SLURM_JOB_ID} \
>>>>>>> ec54497 (added a notebook for finding reactions)
    project=goflow-neb-springF \
    test=true \
    trainer.gradient_clip_val=1.0 \

uv run flow_train.py \
    experiment=flow2_neb \
    model=flow_neb \
    data=t1x \
    logger=[wandb,csv] \
    model.neb_img_subset=[2,3,4,5,6] \
    model.num_steps=25 \
    model.num_samples=50 \
    model.lambda_spring=1.0 \
    model.spring_loss_schedule=linear \
    task_name=train_neb_2-6i_springF_${SLURM_JOB_ID} \
    project=goflow-neb-springF \
    test=true \
    trainer.gradient_clip_val=1.0 \

uv run flow_train.py \
    experiment=flow2_neb \
    model=flow_neb \
    data=t1x \
    logger=[wandb,csv] \
    model.neb_img_subset=[6,7,8,9] \
    model.num_steps=25 \
    model.num_samples=50 \
    model.lambda_spring=1.0 \
    model.spring_loss_schedule=linear \
    task_name=train_neb_6-9i_springF_${SLURM_JOB_ID} \
    project=goflow-neb-springF \
    test=true \
    trainer.gradient_clip_val=1.0 \

uv run flow_train.py \
    experiment=flow2_neb \
    model=flow_neb \
    data=t1x \
    logger=[wandb,csv] \
    model.neb_img_subset=[0,2,3,6,8] \
    model.num_steps=25 \
    model.num_samples=50 \
    model.lambda_spring=1.0 \
    model.spring_loss_schedule=linear \
    task_name=train_neb_eveni_springF_${SLURM_JOB_ID} \
    project=goflow-neb-springF \
    test=true \
    trainer.gradient_clip_val=1.0 \

uv run flow_train.py \
    experiment=flow2_neb \
    model=flow_neb \
    data=t1x \
    logger=[wandb,csv] \
    model.neb_img_subset=[0,5,9] \
    model.num_steps=25 \
    model.num_samples=50 \
    model.lambda_spring=1.0 \
    model.spring_loss_schedule=linear \
    task_name=train_neb_rstp_springF_${SLURM_JOB_ID} \
    project=goflow-neb-springF \
>>>>>>> 1a8a47f (original neb version + logging of GT distances for spring forces)
    test=true \
    trainer.gradient_clip_val=1.0 \

echo "NEB training experiment complete!"
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
echo "Check logs at: /home/stephg/goflow/logs/train_test_neb_all_${SLURM_JOB_ID}.out"
=======
echo "Check logs at: /home/stephg/goflow/logs/train_neb_GT_springF_${SLURM_JOB_ID}.out"
>>>>>>> 1a8a47f (original neb version + logging of GT distances for spring forces)
=======
echo "Check logs at: /home/stephg/goflow/logs/train_neb_cos_springF_${SLURM_JOB_ID}.out"
>>>>>>> fbab8f8 (added linear interpolation of energy-based spring froces and alighment to reference rotation framework)
=======
echo "Check logs at: /home/stephg/goflow/logs/train_neb_springF_${SLURM_JOB_ID}.out"
>>>>>>> ec54497 (added a notebook for finding reactions)
