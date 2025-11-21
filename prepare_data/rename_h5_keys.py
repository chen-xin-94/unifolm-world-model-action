import h5py
import os
import argparse
from tqdm import tqdm

def rename_keys(folder_path, old_key):
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        return

    files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
    print(f"Found {len(files)} HDF5 files in {folder_path}")

    count = 0
    for filename in tqdm(files, desc="Processing files"):
        file_path = os.path.join(folder_path, filename)
        try:
            with h5py.File(file_path, 'r+') as f:
                if old_key in f:
                    f.move(old_key, 'observation.state')
                    count += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Renamed keys in {count} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename HDF5 keys in a folder.")
    parser.add_argument("folder", type=str, help="Path to the folder containing HDF5 files")
    parser.add_argument("--old_key", type=str, default="observation.state.franka_robot_ee", help="The key to be renamed")
    args = parser.parse_args()

    rename_keys(args.folder, args.old_key)
