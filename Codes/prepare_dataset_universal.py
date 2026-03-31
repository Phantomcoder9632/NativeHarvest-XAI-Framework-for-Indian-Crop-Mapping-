import os
import json
import glob
import h5py
import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split

labels_file = r"D:\RemoteSensing-Project\Dataset\CropHarvest\labels.geojson"
arrays_dir = r"D:\RemoteSensing-Project\Dataset\CropHarvest\features\arrays"
out_dir = r"D:\RemoteSensing-Project\Dataset\processed_universal"
os.makedirs(out_dir, exist_ok=True)

print("Loading labels...")
with open(labels_file, "r") as f:
    geojson = json.load(f)

BAND_INDICES = [17, 0, 1]  # NDVI, VV, VH

# Define the master list of crops cultivated in Indian Terrain
indian_crops = [
    'maize', 'wheat', 'rice', 'paddy', 'sorghum', 'millet', 
    'sugarcane', 'cotton', 'groundnut', 'soybean', 'cassava',
    'chickpea', 'mustard', 'fallowland', 'non-crop'
]

# We will balance the dataset by capping max samples per crop
MAX_SAMPLES_PER_CROP = 500

crop_counts = {c: 0 for c in indian_crops}
X_list = []
y_list = []

# Dynamic label mapping for whatever we actually find
label_to_int = {}
current_label_idx = 0

print("Extracting Universal Dataset focused on Indian Terrain crops...")
for feat in geojson["features"]:
    props = feat["properties"]
    label_str = str(props.get("label", "")).lower()
    
    # Check if the label matches our target list
    matched_crop = None
    for target in indian_crops:
        if target in label_str:
            matched_crop = target
            break
            
    if not matched_crop:
        continue
        
    if crop_counts[matched_crop] >= MAX_SAMPLES_PER_CROP:
        continue
        
    idx = props.get("index")
    raw_dataset = props.get("dataset")
    h5_filename = f"{idx}_{raw_dataset}.h5"
    h5_path = os.path.join(arrays_dir, h5_filename)
    
    if not os.path.exists(h5_path):
        continue
        
    # Standardize label mapping
    if matched_crop not in label_to_int:
        label_to_int[matched_crop] = current_label_idx
        current_label_idx += 1
    
    label_int = label_to_int[matched_crop]
    
    try:
        with h5py.File(h5_path, 'r') as f:
            data = f['array'][:]
        selected = data[:, BAND_INDICES]
        # Skip if array is malformed
        if selected.shape[0] != 12:
            continue
            
        orig_days = np.arange(15, 360, 30)
        target_days = np.arange(1, 361)
        f_interp = interp1d(orig_days, selected, axis=0, kind='linear', fill_value="extrapolate")
        interpolated = f_interp(target_days)
        
        X_list.append(interpolated)
        y_list.append(label_int)
        crop_counts[matched_crop] += 1
        
    except Exception as e:
        continue
        
    if len(X_list) % 1000 == 0:
        print(f"Processed {len(X_list)} valid samples so far...")

X = np.array(X_list)
y = np.array(y_list)
print(f"Total samples extracted universally: {X.shape[0]}")
print(f"Label mapping: {label_to_int}")

with open(os.path.join(out_dir, "label_mapping.json"), "w") as f:
    json.dump(label_to_int, f, indent=4)

# Without stratify to prevent dimension errors
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Data shapes:\nTrain: {X_train.shape}\nVal:   {X_val.shape}\nTest:  {X_test.shape}")

np.save(os.path.join(out_dir, "X_train.npy"), X_train)
np.save(os.path.join(out_dir, "y_train.npy"), y_train)
np.save(os.path.join(out_dir, "X_val.npy"), X_val)
np.save(os.path.join(out_dir, "y_val.npy"), y_val)
np.save(os.path.join(out_dir, "X_test.npy"), X_test)
np.save(os.path.join(out_dir, "y_test.npy"), y_test)

print("Universal Dataset successfully prepared and balanced.")
