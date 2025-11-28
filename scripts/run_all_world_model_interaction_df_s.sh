#!/bin/bash

# Pipeline script to run all 4 world model interaction scripts
# Usage: ./run_all_world_model_interaction_df_s.sh <model_name> <CUDA_VISIBLE_DEVICES>
# Example: ./run_all_world_model_interaction_df_s.sh "single_arm_franka_simulation/epoch=45-step=3000" "0"

# Parse command line arguments
model_name=${1:-'single_arm_franka_simulation/epoch=75-step=5000'}
CUDA_DEVICES=${2:-'7'}

echo "=========================================="
echo "Running all world model interaction scripts"
echo "Model: ${model_name}"
echo "CUDA Devices: ${CUDA_DEVICES}"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the base script
echo ""
echo "[1/4] Running base world model interaction..."
bash "${SCRIPT_DIR}/run_world_model_interaction_df_s.sh" "${model_name}" "${CUDA_DEVICES}"
if [ $? -ne 0 ]; then
    echo "Error: Base script failed!"
    exit 1
fi

# Run the multiview script
echo ""
echo "[2/4] Running multiview world model interaction..."
bash "${SCRIPT_DIR}/run_world_model_interaction_df_s_multiview.sh" "${model_name}" "${CUDA_DEVICES}"
if [ $? -ne 0 ]; then
    echo "Error: Multiview script failed!"
    exit 1
fi

# Run the OOD script
echo ""
echo "[3/4] Running OOD world model interaction..."
bash "${SCRIPT_DIR}/run_world_model_interaction_df_s_OOD.sh" "${model_name}" "${CUDA_DEVICES}"
if [ $? -ne 0 ]; then
    echo "Error: OOD script failed!"
    exit 1
fi

# Run the unseen script
echo ""
echo "[4/4] Running unseen world model interaction..."
bash "${SCRIPT_DIR}/run_world_model_interaction_df_s_unseen.sh" "${model_name}" "${CUDA_DEVICES}"
if [ $? -ne 0 ]; then
    echo "Error: Unseen script failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "All scripts completed successfully!"
echo "=========================================="
