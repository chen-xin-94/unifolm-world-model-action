import h5py
import os
import numpy as np

# Variables for folder and file name
folder_path = 'datasets_converted/transitions/fr3_single_arm_franka_hand'
file_name = '0.h5'

file_path = os.path.join(folder_path, file_name)

def print_structure(name, obj):
    print(f"Name: {name}")
    if isinstance(obj, h5py.Group):
        print(f"Type: Group")
    elif isinstance(obj, h5py.Dataset):
        print(f"Type: Dataset")
        print(f"Shape: {obj.shape}")
        print(f"Dtype: {obj.dtype}")
        # Print first few elements for preview
        try:
            if obj.ndim == 0:
                print(f"Value: {obj[()]}")
            elif obj.size > 0:
                # Flatten for easier printing of first few items
                flat_data = obj[...].flatten()
                print(f"Value (first few): {flat_data[:5]}")
        except Exception as e:
            print(f"Could not print value: {e}")
            
    # Print attributes
    if len(obj.attrs) > 0:
        print("Attributes:")
        for k, v in obj.attrs.items():
            print(f"  {k}: {v}")
    print("-" * 40)

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    with h5py.File(file_path, 'r') as f:
        print(f"Inspecting file: {file_path}")
        print("=" * 40)
        # Print file level attributes
        if len(f.attrs) > 0:
            print("File Attributes:")
            for k, v in f.attrs.items():
                print(f"  {k}: {v}")
            print("=" * 40)
            
        f.visititems(print_structure)
