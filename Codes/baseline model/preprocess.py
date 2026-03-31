import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import h5py

labels_file = r"D:\RemoteSensing-Project\Dataset\CropHarvest\labels.geojson"
arrays_dir = r"D:\RemoteSensing-Project\Dataset\CropHarvest\features\arrays"
plots_dir = r"D:\RemoteSensing-Project\Results-plots"
os.makedirs(plots_dir, exist_ok=True)

print("Loading labels...")
with open(labels_file, "r") as f:
    geojson = json.load(f)

maize_ids = []
non_maize_ids = []

for feat in geojson["features"]:
    props = feat["properties"]
    label = str(props.get("label", "")).lower()
    idx = props.get("index")
    dataset = props.get("dataset")
    
    # Try variations of the filename since we aren't 100% sure how they map
    h5_filename = f"{idx}_{dataset}.h5"
    h5_path = os.path.join(arrays_dir, h5_filename)
    
    if not os.path.exists(h5_path):
        continue
        
    if "maize" in label and len(maize_ids) < 10:
        maize_ids.append(h5_path)
    elif "maize" not in label and len(non_maize_ids) < 10:
        non_maize_ids.append(h5_path)
        
    if len(maize_ids) >= 10 and len(non_maize_ids) >= 10:
        break

print(f"Found {len(maize_ids)} Maize files, {len(non_maize_ids)} Non-Maize files.")

# Indicators for NDVI, VV, VH
BAND_INDICES = [17, 0, 1] 

def process_file(h5_path):
    with h5py.File(h5_path, 'r') as f:
        data = f['array'][:]
    selected = data[:, BAND_INDICES]
    orig_days = np.arange(15, 360, 30)
    target_days = np.arange(1, 361)
    
    f_interp = interp1d(orig_days, selected, axis=0, kind='linear', fill_value="extrapolate")
    interpolated = f_interp(target_days)
    return interpolated

if maize_ids or non_maize_ids:
    plt.figure(figsize=(10, 6))
    for item in maize_ids:
        res = process_file(item)
        plt.plot(res[:, 0], color='blue', alpha=0.5, label='Maize' if item == maize_ids[0] else "")

    for item in non_maize_ids:
        res = process_file(item)
        plt.plot(res[:, 0], color='red', alpha=0.5, label='Non-Maize' if item == non_maize_ids[0] else "")

    plt.title("NDVI Time-Series for 10 Maize and 10 Non-Maize Samples")
    plt.xlabel("Day of Year")
    plt.ylabel("NDVI")
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(plots_dir, "NDVI_Time_Series.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
else:
    print("Cannot plot, not enough data found.")
    # let's print unique labels to see what we have
    labels = set([str(f["properties"].get("label")).lower() for f in geojson["features"][:1000]])
    print("Some available labels:", labels)
