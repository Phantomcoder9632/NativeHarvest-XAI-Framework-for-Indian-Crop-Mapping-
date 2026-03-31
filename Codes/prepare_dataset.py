import os
import json
import glob
import h5py
import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split

labels_file = r"D:\RemoteSensing-Project\Dataset\CropHarvest\labels.geojson"
arrays_dir = r"D:\RemoteSensing-Project\Dataset\CropHarvest\features\arrays"
out_dir = r"D:\RemoteSensing-Project\Dataset\processed"

os.makedirs(out_dir, exist_ok=True)

print("Loading labels...")
with open(labels_file, "r") as f:
    geojson = json.load(f)

# BAND_INDICES = [17, 0, 1]  -> NDVI, VV, VH
BAND_INDICES = [17, 0, 1] 

X_list = []
y_list = []

label_to_int = {}
current_label_idx = 0

print("Extracting Kenya dataset samples...")
for feat in geojson["features"]:
    props = feat["properties"]
    dataset = str(props.get("dataset", "")).lower()
    
    if "kenya" not in dataset:
        continue
        
    label_str = str(props.get("label", "unknown")).lower()
    idx = props.get("index")
    raw_dataset = props.get("dataset")
    
    h5_filename = f"{idx}_{raw_dataset}.h5"
    h5_path = os.path.join(arrays_dir, h5_filename)
    
    if not os.path.exists(h5_path):
        continue
        
    if label_str not in label_to_int:
        label_to_int[label_str] = current_label_idx
        current_label_idx += 1
        
    label_int = label_to_int[label_str]
    
    # Process features
    with h5py.File(h5_path, 'r') as f:
        data = f['array'][:]  # (12, 18)
    
    selected = data[:, BAND_INDICES]
    
    orig_days = np.arange(15, 360, 30)
    target_days = np.arange(1, 361)
    f_interp = interp1d(orig_days, selected, axis=0, kind='linear', fill_value="extrapolate")
    interpolated = f_interp(target_days) # (360, 3)
    
    X_list.append(interpolated)
    y_list.append(label_int)

if len(X_list) == 0:
    print("No matching samples found for Kenya!")
else:
    X = np.array(X_list)
    y = np.array(y_list)
    print(f"Total samples extracted: {X.shape[0]}")
    print(f"Label mapping: {label_to_int}")
    
    # Save mapping
    with open(os.path.join(out_dir, "label_mapping.json"), "w") as f:
        json.dump(label_to_int, f, indent=4)
    
    # Split 80:10:10 (Train:Val:Test) Without stratify to prevent class count < 2 errors
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Then split the 20% evenly into 10% val and 10% test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Data shapes:\nTrain: {X_train.shape}, {y_train.shape}\nVal:   {X_val.shape}, {y_val.shape}\nTest:  {X_test.shape}, {y_test.shape}")
    
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "X_val.npy"), X_val)
    np.save(os.path.join(out_dir, "y_val.npy"), y_val)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)
    print("All sets saved successfully to D:\\RemoteSensing-Project\\Dataset\\processed")
