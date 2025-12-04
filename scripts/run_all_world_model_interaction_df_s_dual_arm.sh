#!/bin/bash

# Pipeline script to run the dual-arm world model interaction scripts (standard + unseen)
# Usage: ./run_all_world_model_interaction_df_dual_arm.sh <model_name> <CUDA_VISIBLE_DEVICES>
# Example: ./run_all_world_model_interaction_df_dual_arm.sh "dual_arm_franka_simulation/epoch=85-step=3000" "7"

# Parse command line arguments
model_name=${1:-'dual_arm_franka_simulation/epoch=57-step=2000'}
CUDA_DEVICES=${2:-'2'}

echo "=========================================="
echo "Running dual-arm world model interaction scripts"
echo "Model: ${model_name}"
echo "CUDA Devices: ${CUDA_DEVICES}"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo ""
echo "[1/2] Running dual-arm world model interaction..."
bash "${SCRIPT_DIR}/run_world_model_interaction_df_s_dual_arm.sh" "${model_name}" "${CUDA_DEVICES}"
if [ $? -ne 0 ]; then
    echo "Error: Dual-arm script failed!"
    exit 1
fi

echo ""
echo "[2/2] Running dual-arm unseen world model interaction..."
bash "${SCRIPT_DIR}/run_world_model_interaction_df_s_dual_arm_unseen.sh" "${model_name}" "${CUDA_DEVICES}"
if [ $? -ne 0 ]; then
    echo "Error: Unseen script failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "All scripts completed successfully!"
echo "=========================================="
