#!/bin/bash

# NCCL configuration (uncomment as needed)
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_NET_GDR_LEVEL=3
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_TOPO_FILE=/tmp/topo.txt

# Training configuration
# name="single_arm_franka_simulation_all"
# config_file=configs/train/config_single_arm_franka_simulation_all.yaml

name="dual_arm_franka_simulation_all"
config_file=configs/train/config_dual_arm_franka_simulation_all.yaml

save_root="/mnt/data/output_unifolm-wma"

mkdir -p $save_root/$name

# Determine if running under Slurm
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Running under Slurm (Job ID: $SLURM_JOB_ID)"
    
    # Slurm sets these automatically
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    export MASTER_PORT=${MASTER_PORT:-12366}
    NODE_RANK=$SLURM_NODEID
    NNODES=$SLURM_NNODES
    
    # Robust GPU detection: SLURM_GPUS_PER_NODE is not always set by Slurm
    if [ -z "$SLURM_GPUS_PER_NODE" ]; then
        echo "SLURM_GPUS_PER_NODE is not set. Attempting to detect via nvidia-smi..."
        num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        export SLURM_GPUS_PER_NODE=$num_gpus
    else
        num_gpus=$SLURM_GPUS_PER_NODE
    fi
    
    echo "Node: $SLURM_NODEID/$SLURM_NNODES"
    echo "Node name: $(hostname)"
    echo "Master: $MASTER_ADDR:$MASTER_PORT"
    echo "GPUs per node: $num_gpus"
    
else
    # Manual configuration (non-Slurm)
    export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    export MASTER_PORT=${MASTER_PORT:-12366}
    NODE_RANK=${NODE_RANK:-0}
    NNODES=${NNODES:-1}
    
    if [ $NNODES -eq 1 ]; then
        # Single node: use specific GPUs
        gpus="0,1,2,3,4,5,6,7"
        export CUDA_VISIBLE_DEVICES=$gpus
        num_gpus=$(echo $gpus | awk -F',' '{print NF}')
        echo "Single-node training"
        echo "Using GPUs: $gpus (count: $num_gpus)"
    else
        # Multi-node: use all available GPUs
        num_gpus=${NPROC_PER_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
        echo "Multi-node training"
        echo "Node rank: $NODE_RANK/$NNODES"
        echo "Master: $MASTER_ADDR:$MASTER_PORT"
        echo "GPUs per node: $num_gpus"
    fi
fi

# Calculate total GPUs
total_gpus=$((num_gpus * NNODES))
echo "Total GPUs across all nodes: $total_gpus"

# Run training with torchrun
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$num_gpus \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    ./scripts/trainer.py \
    --base $config_file \
    --train \
    --name $name \
    --logdir $save_root \
    --devices $num_gpus \
    --total_gpus=$total_gpus \
    lightning.trainer.num_nodes=$NNODES