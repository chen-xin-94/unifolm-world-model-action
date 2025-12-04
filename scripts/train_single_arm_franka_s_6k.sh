# NCCL configuration
# export NCCL_DEBUG=debug
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_NET_GDR_LEVEL=3
# export CUDA_LAUNCH_BLOCKING=1

# export NCCL_TOPO_FILE=/tmp/topo.txt
# export MASTER_ADDR="master.ip."
# export MASTER_PROT=12366


# args
name="single_arm_franka_simulation_6k"
config_file=configs/train/config_single_arm_franka_simulation_6k.yaml

# save root dir for logs, checkpoints, tensorboard record, etc.
save_root="/raid/chen.xin/repo/unifolm-world-model-action/output/train"

mkdir -p $save_root/$name

## run
# GPU configuration for one node
gpus="1,2,3,7"
num_gpus=$(echo $gpus | awk -F',' '{print NF}')
echo "Using GPUs: $gpus (count: $num_gpus)"

CUDA_VISIBLE_DEVICES=$gpus python3 -m torch.distributed.launch \
--nproc_per_node=$num_gpus --nnodes=1 --master_addr=127.0.0.1 --master_port=12366 --node_rank=0 \
./scripts/trainer.py \
--base $config_file \
--train \
--name $name \
--logdir $save_root \
--devices $num_gpus \
--total_gpus=$num_gpus \
lightning.trainer.num_nodes=1
