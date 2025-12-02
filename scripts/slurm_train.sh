#!/bin/bash
#SBATCH --job-name=unifolm-wma_simulation_single_arm_franka_all        # Job name
#SBATCH --nodes=2                          # Number of nodes
#SBATCH --ntasks-per-node=1                # One task per node (torchrun handles multi-GPU)
#SBATCH --gpus-per-node=8                  # 8 GPUs per node (B200), #SBATCH --gres=gpu:8  for older clusters
#SBATCH --cpus-per-task=160                # CPUs per task (adjusted for worker-0)
#SBATCH --mem=0                            # Request all memory on node (or specify like 500G)
#SBATCH --time=168:00:00                    # Time limit (168 hours, 7 days)
#SBATCH --partition=main                     # Partition name (main)
#SBATCH --output=logs/slurm_%j.out         # Standard output log (%j = job ID)
#SBATCH --error=logs/slurm_%j.err          # Standard error log
#SBATCH --exclusive                        # Exclusive node access (recommended for multi-node)

# Change to project directory
cd /root/chen/repo/unifolm-world-model-action || exit 1

# Create log directory
mkdir -p logs

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Working directory: $(pwd)"
echo "=========================================="

# Load required modules (adjust based on your cluster)
# Note: 'module' command not found on this cluster. Relying on system paths and conda.
# module purge
# module load cuda/12.1
# module load cudnn/8.9
# module load nccl/2.18
# module load python/3.10

# Activate your Python environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate unifolm-wma

# NCCL configuration for InfiniBand (uncomment if using IB)
export NCCL_DEBUG=INFO                     # Set to INFO for debugging, WARN for production
# export NCCL_IB_DISABLE=0                 # Enable InfiniBand
# export NCCL_IB_GID_INDEX=3               # Adjust based on your IB setup
# export NCCL_NET_GDR_LEVEL=3              # Enable GPU Direct RDMA
# export NCCL_SOCKET_IFNAME=ib0            # InfiniBand interface name

# NCCL configuration for Ethernet (uncomment if using Ethernet)
# export NCCL_SOCKET_IFNAME=eth0           # Ethernet interface name
# export NCCL_IB_DISABLE=1                 # Disable InfiniBand

# Optional: Set master port (default is handled in script)
export MASTER_PORT=12366

# Print node information
echo "=========================================="
echo "Node Information:"
srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES bash -c '
    echo "Node $SLURM_NODEID ($(hostname)):"
    echo "  - GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
    echo "  - GPU Types: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "  - Memory: $(free -h | grep Mem | awk "{print \$2}")"
'
echo "=========================================="

# Launch training with srun
# srun ensures the script runs on all allocated nodes
srun bash scripts/train_torchrun.sh