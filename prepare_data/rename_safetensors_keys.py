import os
from safetensors.numpy import save_file, load_file
import argparse

def rename_safetensors_keys(file_path, old_key_prefix, new_key_prefix):
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    print(f"Processing file: {file_path}")
    
    try:
        tensors = load_file(file_path)
        new_tensors = {}
        renamed_count = 0
        
        for key, tensor in tensors.items():
            if key.startswith(old_key_prefix):
                new_key = key.replace(old_key_prefix, new_key_prefix, 1)
                new_tensors[new_key] = tensor
                renamed_count += 1
                print(f"Renaming {key} -> {new_key}")
            else:
                new_tensors[key] = tensor
        
        if renamed_count > 0:
            save_file(new_tensors, file_path)
            print(f"Successfully renamed {renamed_count} keys in {file_path}")
        else:
            print("No keys found to rename.")
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename keys in a safetensors file.")
    parser.add_argument("file_path", type=str, help="Path to the safetensors file")
    parser.add_argument("--old_prefix", type=str, default="observation.state.franka_robot_ee", help="The prefix of keys to be renamed")
    parser.add_argument("--new_prefix", type=str, default="observation.state", help="The new prefix for the keys")
    args = parser.parse_args()

    rename_safetensors_keys(args.file_path, args.old_prefix, args.new_prefix)
