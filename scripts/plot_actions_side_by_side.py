#!/usr/bin/env python3
"""
Plot actions side-by-side from CSV (model predictions) and Parquet (ground truth) files.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import argparse
from pathlib import Path


def parse_action_string(action_str):
    """Parse action string to numpy array."""
    if isinstance(action_str, str):
        return np.array(ast.literal_eval(action_str))
    elif isinstance(action_str, list):
        return np.array(action_str)
    else:
        return action_str


def load_csv_data(csv_path):
    """Load and parse CSV data."""
    df = pd.read_csv(csv_path)
    
    # Parse action strings to arrays
    actions = []
    for action_str in df['action']:
        actions.append(parse_action_string(action_str))
    
    actions = np.array(actions)
    timestamps = df['timestamp_sec'].values
    
    return actions, timestamps


def load_parquet_data(parquet_path):
    """Load and parse Parquet data."""
    df = pd.read_parquet(parquet_path)
    
    # Parse action arrays
    actions = []
    for action in df['action']:
        actions.append(parse_action_string(action))
    
    actions = np.array(actions)
    timestamps = df['timestamp'].values
    
    return actions, timestamps


def plot_actions_side_by_side(csv_path, parquet_path, output_path=None):
    """Plot actions from CSV and Parquet files side-by-side."""
    
    # Load data
    print(f"Loading CSV data from: {csv_path}")
    csv_actions, csv_timestamps = load_csv_data(csv_path)
    
    print(f"Loading Parquet data from: {parquet_path}")
    parquet_actions, parquet_timestamps = load_parquet_data(parquet_path)
    
    # Get action dimensions
    action_dim = csv_actions.shape[1]
    print(f"Action dimension: {action_dim}")
    print(f"CSV actions shape: {csv_actions.shape}")
    print(f"Parquet actions shape: {parquet_actions.shape}")
    
    # Create subplots - one for each action dimension
    fig, axes = plt.subplots(action_dim, 1, figsize=(12, 2.5 * action_dim), sharex=True)
    
    if action_dim == 1:
        axes = [axes]
    
    action_labels = [
        'X Position', 'Y Position', 'Z Position',
        'Roll', 'Pitch', 'Yaw', 'Gripper'
    ]
    
    for i in range(action_dim):
        ax = axes[i]
        
        # Plot ground truth (parquet)
        ax.plot(parquet_timestamps, parquet_actions[:, i], 
                label='Ground Truth (Parquet)', 
                color='blue', linewidth=2, alpha=0.7)
        
        # Plot model predictions (csv)
        ax.plot(csv_timestamps, csv_actions[:, i], 
                label='Model Prediction (CSV)', 
                color='red', linewidth=2, alpha=0.7, linestyle='--')
        
        # Formatting
        label = action_labels[i] if i < len(action_labels) else f'Action {i}'
        ax.set_ylabel(label, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        
        # Add value range in the corner
        gt_min, gt_max = parquet_actions[:, i].min(), parquet_actions[:, i].max()
        pred_min, pred_max = csv_actions[:, i].min(), csv_actions[:, i].max()
        ax.text(0.02, 0.98, f'GT: [{gt_min:.3f}, {gt_max:.3f}]\nPred: [{pred_min:.3f}, {pred_max:.3f}]',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    axes[-1].set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    
    # Add title
    csv_name = Path(csv_path).stem
    parquet_name = Path(parquet_path).stem
    fig.suptitle(f'Action Comparison: {csv_name} vs {parquet_name}', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot actions side-by-side from CSV and Parquet files'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='output/single_arm_franka_simulation/epoch=121-step=8000/fr3_single_arm_franka_hand/inference/3000_actions_states_fs4.csv',
        help='Path to CSV file with model predictions'
    )
    parser.add_argument(
        '--parquet',
        type=str,
        default='datasets/avla-dataset-nov-24_merged_per_embodiment/fr3_single_arm_franka_hand/data/chunk-003/episode_003000.parquet',
        help='Path to Parquet file with ground truth'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for the plot (if not specified, will display)'
    )
    
    args = parser.parse_args()
    
    # Generate default output path if not specified
    if args.output is None:
        csv_path = Path(args.csv)
        output_dir = csv_path.parent
        args.output = output_dir / f"{csv_path.stem}_comparison.png"
    
    plot_actions_side_by_side(args.csv, args.parquet, args.output)


if __name__ == '__main__':
    main()
