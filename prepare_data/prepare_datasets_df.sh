#!/bin/bash

# ===============
# first batch
# ===============

# FR3 dual-arm with Franka Hand
python /raid/chen/repo/unifolm-world-model-action/prepare_data/prepare_training_data.py \
    --source_dir /mnt/central_storage/data_pool/data_foundry/avla_nov_8_merged_per_embodiment_2025-11-12\
    --target_dir datasets_converted\
    --dataset_name "fr3_dual_arm_franka_hand" \
    --robot_name "Franka FR3 Dual Arm with Franka Hand" \
    --convert_av1

# FR3 dual-arm with Robotiq 2F
python /raid/chen/repo/unifolm-world-model-action/prepare_data/prepare_training_data.py \
    --source_dir /mnt/central_storage/data_pool/data_foundry/avla_nov_8_merged_per_embodiment_2025-11-12\
    --target_dir datasets_converted\
    --dataset_name "fr3_dual_arm_robotiq_2f" \
    --robot_name "Franka FR3 Dual Arm with Robotiq 2F Gripper" \
    --convert_av1

# Thor3 dual-arm with Robotiq 2F
python /raid/chen/repo/unifolm-world-model-action/prepare_data/prepare_training_data.py \
    --source_dir /mnt/central_storage/data_pool/data_foundry/avla_nov_8_merged_per_embodiment_2025-11-12\
    --target_dir datasets_converted\
    --dataset_name "thor3_dual_arm_robotiq_2f" \
    --robot_name "Thor3 Dual Arm with Robotiq 2F Gripper" \
    --convert_av1

# FR3 single-arm with Franka Hand
python /raid/chen/repo/unifolm-world-model-action/prepare_data/prepare_training_data.py \
    --source_dir /mnt/central_storage/data_pool/data_foundry/avla_nov_8_merged_per_embodiment_2025-11-12\
    --target_dir datasets_converted\
    --dataset_name "fr3_single_arm_franka_hand" \
    --robot_name "Franka FR3 Single Arm with Franka Hand" \
    --convert_av1

# FR3 single-arm with Robotiq 2F
python /raid/chen/repo/unifolm-world-model-action/prepare_data/prepare_training_data.py \
    --source_dir /mnt/central_storage/data_pool/data_foundry/avla_nov_8_merged_per_embodiment_2025-11-12\
    --target_dir datasets_converted\
    --dataset_name "fr3_single_arm_robotiq_2f" \
    --robot_name "Franka FR3 Single Arm with Robotiq 2F Gripper" \
    --convert_av1

# Thor3 single-arm with Robotiq 2F
python /raid/chen/repo/unifolm-world-model-action/prepare_data/prepare_training_data.py \
    --source_dir /mnt/central_storage/data_pool/data_foundry/avla_nov_8_merged_per_embodiment_2025-11-12\
    --target_dir datasets_converted\
    --dataset_name "thor3_single_robotiq_2f" \
    --robot_name "Thor3 Single Arm with Robotiq 2F Gripper" \
    --convert_av1

# ===============
# second batch
# ===============

# FR3 dual-arm with Franka Hand
python /raid/chen/repo/unifolm-world-model-action/prepare_data/prepare_training_data.py \
    --source_dir /mnt/central_storage/data_pool/data_foundry/avla-dataset-nov-24_merged_per_embodiment\
    --target_dir /mnt/central_storage/unifolm_world_model_action/datasets_converted\
    --dataset_name "fr3_dual_arm_franka_hand" \
    --robot_name "Franka FR3 Dual Arm with Franka Hand" \
    --convert_av1

# # FR3 dual-arm with Robotiq 2F
# python /raid/chen/repo/unifolm-world-model-action/prepare_data/prepare_training_data.py \
#     --source_dir /mnt/central_storage/data_pool/data_foundry/avla-dataset-nov-24_merged_per_embodiment\
#     --target_dir /mnt/central_storage/unifolm_world_model_action/datasets_converted\
#     --dataset_name "fr3_dual_arm_robotiq_2f" \
#     --robot_name "Franka FR3 Dual Arm with Robotiq 2F Gripper" \
#     --convert_av1

# # Thor3 dual-arm with Robotiq 2F
# python /raid/chen/repo/unifolm-world-model-action/prepare_data/prepare_training_data.py \
#     --source_dir /mnt/central_storage/data_pool/data_foundry/avla-dataset-nov-24_merged_per_embodiment\
#     --target_dir /mnt/central_storage/unifolm_world_model_action/datasets_converted\
#     --dataset_name "thor3_dual_arm_robotiq_2f" \
#     --robot_name "Thor3 Dual Arm with Robotiq 2F Gripper" \
#     --convert_av1

# FR3 single-arm with Franka Hand
python /raid/chen/repo/unifolm-world-model-action/prepare_data/prepare_training_data.py \
    --source_dir /mnt/central_storage/data_pool/data_foundry/avla-dataset-nov-24_merged_per_embodiment\
    --target_dir /mnt/central_storage/unifolm_world_model_action/datasets_converted\
    --dataset_name "fr3_single_arm_franka_hand" \
    --robot_name "Franka FR3 Single Arm with Franka Hand" \
    --convert_av1

# # FR3 single-arm with Robotiq 2F
# python /raid/chen/repo/unifolm-world-model-action/prepare_data/prepare_training_data.py \
#     --source_dir /mnt/central_storage/data_pool/data_foundry/avla-dataset-nov-24_merged_per_embodiment\
#     --target_dir /mnt/central_storage/unifolm_world_model_action/datasets_converted\
#     --dataset_name "fr3_single_arm_robotiq_2f" \
#     --robot_name "Franka FR3 Single Arm with Robotiq 2F Gripper" \
#     --convert_av1

# Thor3 single-arm with Robotiq 2F
python /raid/chen/repo/unifolm-world-model-action/prepare_data/prepare_training_data.py \
    --source_dir /mnt/central_storage/data_pool/data_foundry/avla-dataset-nov-24_merged_per_embodiment\
    --target_dir /mnt/central_storage/unifolm_world_model_action/datasets_converted\
    --dataset_name "thor3_single_robotiq_2f" \
    --robot_name "Thor3 Single Arm with Robotiq 2F Gripper" \
    --convert_av1
