import os
import h5py
import numpy as np
import glob

# Try finding an h5 file
arrays_dir = r"D:\RemoteSensing-Project\Dataset\CropHarvest\features\arrays"
h5_files = glob.glob(os.path.join(arrays_dir, "*.h5"))

if not h5_files:
    print("No .h5 files found in arrays directory.")
else:
    sample_file = h5_files[0]
    print(f"Inspecting file: {sample_file}")
    with h5py.File(sample_file, 'r') as f:
        print("Keys:", list(f.keys()))
        if 'array' in f:
            data = f['array'][:]
            print(f"Array shape: {data.shape}")
        else:
            print("No 'array' dataset found.")
