"""
Split native_india_arrays.h5 -> Train (70%) / Val (15%) / Test (15%)
Optimizations applied:
  - Rare classes (< 5 samples) removed
  - Label remapped to contiguous 0-indexed integers
  - Minority classes oversampled in TRAINING split only (via RandomOverSampler)
  - Val / Test are kept as-is (no data leakage)
"""
import os, json
import h5py
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

H5_PATH = r"D:\RemoteSensing-Project\Dataset\native_india_arrays.h5"
OUT_DIR = r"D:\RemoteSensing-Project\Dataset\native_processed"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading native_india_arrays.h5 ...")
with h5py.File(H5_PATH, 'r') as hf:
    X   = hf['features'][:]  # (N, seq_len, 3)
    y_raw = hf['labels'][:]

N, SEQ, CH = X.shape
print(f"  Raw samples  : {N}  |  Sequence: ({SEQ},{CH})")
print(f"  Raw classes  : {sorted(np.unique(y_raw).tolist())}")

# ── 1. Drop rare classes (< 5 samples) ─────────────────────────────────────
counts = Counter(y_raw.tolist())
keep   = {c for c, n in counts.items() if n >= 5}
mask   = np.array([v in keep for v in y_raw])
X, y_raw = X[mask], y_raw[mask]
print(f"  After filtering rare classes : {len(X)} samples, {len(keep)} classes")

# ── 2. Remap labels to 0-indexed ────────────────────────────────────────────
unique_crops = sorted(np.unique(y_raw).tolist())
crop_to_idx  = {int(c): i for i, c in enumerate(unique_crops)}
y = np.array([crop_to_idx[int(v)] for v in y_raw])
print(f"  Label map: {crop_to_idx}")
with open(os.path.join(OUT_DIR, "crop_label_map.json"), "w") as f:
    json.dump(crop_to_idx, f, indent=2)

# ── 3. Stratified 70/15/15 split ────────────────────────────────────────────
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

print(f"\nBefore oversampling — Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
print(f"  Train class distribution: {sorted(Counter(y_train.tolist()).items())}")

# ── 4. Oversample ONLY the training split ───────────────────────────────────
# Flatten time dimension for oversampler, then reshape back
X_flat  = X_train.reshape(len(X_train), -1)          # (N, seq*ch)
ros     = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_flat, y_train)
X_train = X_res.reshape(len(X_res), SEQ, CH)
y_train = y_res

print(f"After  oversampling — Train: {len(X_train)}")
print(f"  Train class distribution: {sorted(Counter(y_train.tolist()).items())}")

# ── 5. Save ─────────────────────────────────────────────────────────────────
np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUT_DIR, "X_val.npy"),   X_val)
np.save(os.path.join(OUT_DIR, "y_val.npy"),   y_val)
np.save(os.path.join(OUT_DIR, "X_test.npy"),  X_test)
np.save(os.path.join(OUT_DIR, "y_test.npy"),  y_test)
print(f"\nAll splits saved to: {OUT_DIR}")
