#!/bin/bash

# # Rename to final dataset name (run only once)
# mv datasets/Z1_StackBox_Dataset datasets/unitree_z1_stackbox
# mv datasets/Z1_Dual_Dex1_StackBox_Dataset datasets/unitree_z1_dual_arm_stackbox
# mv datasets/Z1_Dual_Dex1_StackBox_Dataset_V2 datasets/unitree_z1_dual_arm_stackbox_v2
# mv datasets/Z1_Dual_Dex1_CleanupPencils_Dataset datasets/unitree_z1_dual_arm_cleanup_pencils
# mv datasets/G1_Dex1_MountCameraRedGripper_Dataset datasets/unitree_g1_pack_camera


# Z1_StackBox Dataset
python prepare_data/prepare_training_data.py \
    --source_dir datasets\
    --target_dir datasets_converted\
    --dataset_name "unitree_z1_stackbox" \
    --robot_name "Unitree Z1 Robot Arm"

# Z1_DualArm_StackBox Dataset
python prepare_data/prepare_training_data.py \
    --source_dir datasets\
    --target_dir datasets_converted\
    --dataset_name "unitree_z1_dual_arm_stackbox" \
    --robot_name "Unitree Z1 Robot Arm"

# Z1_DualArm_StackBox_V2 Dataset
python prepare_data/prepare_training_data.py \
    --source_dir datasets\
    --target_dir datasets_converted\
    --dataset_name "unitree_z1_dual_arm_stackbox_v2" \
    --robot_name "Unitree Z1 Robot Arm"

# Z1_DualArm_Cleanup_Pencils Dataset
python prepare_data/prepare_training_data.py \
    --source_dir datasets\
    --target_dir datasets_converted\
    --dataset_name "unitree_z1_dual_arm_cleanup_pencils" \
    --robot_name "Unitree Z1 Robot Arm"

# G1_Pack_Camera Dataset
python prepare_data/prepare_training_data.py \
    --source_dir datasets\
    --target_dir datasets_converted\
    --dataset_name "unitree_g1_pack_camera" \
    --robot_name "Unitree G1 Robot with Gripper"

