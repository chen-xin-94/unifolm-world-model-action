#!/bin/bash

# FR3 dual-arm with Franka Hand
python prepare_data/prepare_training_data.py \
    --source_dir /mnt/central_storage/data_pool/data_foundry/avla_nov_8_merged_per_embodiment_2025-11-12\
    --target_dir datasets_converted\
    --dataset_name "fr3_dual_arm_franka_hand" \
    --robot_name "Franka FR3 Dual Arm with Franka Hand"

# FR3 dual-arm with Robotiq 2F
python prepare_data/prepare_training_data.py \
    --source_dir /mnt/central_storage/data_pool/data_foundry/avla_nov_8_merged_per_embodiment_2025-11-12\
    --target_dir datasets_converted\
    --dataset_name "fr3_dual_arm_robotiq_2f" \
    --robot_name "Franka FR3 Dual Arm with Robotiq 2F Gripper"

# Thor3 dual-arm with Robotiq 2F
python prepare_data/prepare_training_data.py \
    --source_dir /mnt/central_storage/data_pool/data_foundry/avla_nov_8_merged_per_embodiment_2025-11-12\
    --target_dir datasets_converted\
    --dataset_name "thor3_dual_arm_robotiq_2f" \
    --robot_name "Thor3 Dual Arm with Robotiq 2F Gripper"

# # FR3 single-arm with Franka Hand
# python prepare_data/prepare_training_data.py \
#     --source_dir /mnt/central_storage/data_pool/data_foundry/avla_nov_8_merged_per_embodiment_2025-11-12\
#     --target_dir datasets_converted\
#     --dataset_name "fr3_single_arm_franka_hand" \
#     --robot_name "Franka FR3 Single Arm with Franka Hand" \
#     --skip_videos

# # FR3 single-arm with Robotiq 2F
# python prepare_data/prepare_training_data.py \
#     --source_dir /mnt/central_storage/data_pool/data_foundry/avla_nov_8_merged_per_embodiment_2025-11-12\
#     --target_dir datasets_converted\
#     --dataset_name "fr3_single_arm_robotiq_2f" \
#     --robot_name "Franka FR3 Single Arm with Robotiq 2F Gripper" \
#     --skip_videos

# # Thor3 single-arm with Robotiq 2F
# python prepare_data/prepare_training_data.py \
#     --source_dir /mnt/central_storage/data_pool/data_foundry/avla_nov_8_merged_per_embodiment_2025-11-12\
#     --target_dir datasets_converted\
#     --dataset_name "thor3_single_robotiq_2f" \
#     --robot_name "Thor3 Single Arm with Robotiq 2F Gripper" \
#     --skip_videos

