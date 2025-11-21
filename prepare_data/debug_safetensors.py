import os
from safetensors import safe_open
import argparse

def inspect_safetensors(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    print(f"Inspecting file: {file_path}")
    print("=" * 40)

    try:
        with safe_open(file_path, framework="np", device="cpu") as f:
            keys = f.keys()
            print(f"Found {len(keys)} tensors:")
            for key in keys:
                tensor = f.get_tensor(key)
                print(f"Name: {key}")
                print(f"  Shape: {tensor.shape}")
                print(f"  Dtype: {tensor.dtype}")
                # Print first few elements
                if tensor.size > 0:
                    flat_data = tensor.flatten()
                    print(f"  Value (first few): {flat_data[:5]}")
                print("-" * 40)
            
            # Check for metadata if possible (safetensors usually stores tensors, metadata might be separate or in header)
            # safe_open doesn't directly expose metadata in the context manager in the same way as h5py attrs
            # but we can check if there are any specific keys that look like metadata
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect safetensors file.")
    parser.add_argument("file_path", type=str, help="Path to the safetensors file")
    args = parser.parse_args()

    inspect_safetensors(args.file_path)
