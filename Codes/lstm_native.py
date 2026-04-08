"""
Train the Crop-Classifier LSTM on the Native Indian AgriFieldNet dataset.
Uses VV + VH Sentinel-1 radar tensors harvested via Google Earth Engine.
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ─── Paths ──────────────────────────────────────────────────────────────────
data_dir   = r"D:\RemoteSensing-Project\Dataset\processed"
models_dir = r"D:\RemoteSensing-Project\Models"
plots_dir  = r"D:\RemoteSensing-Project\Results-plots"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir,  exist_ok=True)

# ─── Hyperparameters ─────────────────────────────────────────────────────────
BATCH_SIZE  = 32
EPOCHS      = 40
LR          = 0.001
INPUT_SIZE  = 3       # NDVI (zeros), VV, VH
HIDDEN_SIZE = 128     # Larger hidden size for richer Indian dataset
DROPOUT     = 0.3

# ─── Load data ───────────────────────────────────────────────────────────────
print("Loading Native Indian dataset splits ...")
X_train = np.load(os.path.join(data_dir, "X_train.npy"))
y_train = np.load(os.path.join(data_dir, "y_train.npy"))
X_val   = np.load(os.path.join(data_dir, "X_val.npy"))
y_val   = np.load(os.path.join(data_dir, "y_val.npy"))
X_test  = np.load(os.path.join(data_dir, "X_test.npy"))
y_test  = np.load(os.path.join(data_dir, "y_test.npy"))

NUM_CLASSES = int(max(y_train.max(), y_val.max(), y_test.max())) + 1
print(f"  Train: {len(X_train)}   Val: {len(X_val)}   Test: {len(X_test)}")
print(f"  Number of crop classes  : {NUM_CLASSES}")
print(f"  Tensor shape per sample : {X_train.shape[1:]}")

# Load label-to-name mapping if available
label_map_path = os.path.join(data_dir, "crop_label_map.json")
if os.path.exists(label_map_path):
    with open(label_map_path) as f:
        crop_map = json.load(f)
    # AgriFieldNet official codes -> crop names
    AGRI_CODES = {
        1: "Wheat", 2: "Mustard", 3: "Lentil", 4: "No Crop/Fallow",
        5: "Green pea", 6: "Sugarcane", 8: "Garlic", 9: "Maize",
        13: "Gram", 14: "Watermelon", 15: "Unsown"
    }
    idx_to_name = {v: AGRI_CODES.get(int(k), f"Crop-{k}") for k, v in crop_map.items()}
    print(f"  Crops in dataset: {list(idx_to_name.values())}")

# ─── Preprocessing: NaN fill + standardize on training stats ─────────────────
X_train = np.nan_to_num(X_train)
X_val   = np.nan_to_num(X_val)
X_test  = np.nan_to_num(X_test)

# Only standardize channels 1 and 2 (VV/VH); channel 0 is all-zeros (NDVI placeholder)
for ch in [1, 2]:
    mu  = X_train[:, :, ch].mean()
    std = X_train[:, :, ch].std() + 1e-8
    X_train[:, :, ch] = (X_train[:, :, ch] - mu) / std
    X_val[:, :, ch]   = (X_val[:, :, ch]   - mu) / std
    X_test[:, :, ch]  = (X_test[:, :, ch]  - mu) / std

# ─── PyTorch Tensors & DataLoaders ───────────────────────────────────────────
def to_loader(X, y, shuffle=True):
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = to_loader(X_train, y_train, shuffle=True)
val_loader   = to_loader(X_val,   y_val,   shuffle=False)
test_loader  = to_loader(X_test,  y_test,  shuffle=False)

# ─── Model ───────────────────────────────────────────────────────────────────
class CropClassifierLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=2, batch_first=True,
                            dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# --- Compute MILD class weights: sqrt(1/freq), to balance without collapse --
counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(float)
counts = np.where(counts == 0, 1, counts)
class_weights = np.sqrt(1.0 / counts)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"Sqrt class weights: {np.round(class_weights, 2).tolist()}")

model     = CropClassifierLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, DROPOUT).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_t)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# ─── Training Loop ────────────────────────────────────────────────────────────
train_losses, val_losses, train_accs, val_accs = [], [], [], []
best_val_acc = 0.0

print("Starting training ...")
for epoch in range(EPOCHS):
    # -- Train --
    model.train()
    run_loss, correct, total = 0.0, 0, 0
    for bX, by in train_loader:
        bX, by = bX.to(device), by.to(device)
        optimizer.zero_grad()
        out  = model(bX)
        loss = criterion(out, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        run_loss += loss.item() * bX.size(0)
        correct  += (out.argmax(1) == by).sum().item()
        total    += by.size(0)
    ep_train_loss = run_loss / total
    ep_train_acc  = correct  / total * 100

    # -- Validate --
    model.eval()
    vl, vc, vt = 0.0, 0, 0
    with torch.no_grad():
        for bX, by in val_loader:
            bX, by = bX.to(device), by.to(device)
            out  = model(bX)
            vl  += criterion(out, by).item() * bX.size(0)
            vc  += (out.argmax(1) == by).sum().item()
            vt  += by.size(0)
    ep_val_loss = vl / vt
    ep_val_acc  = vc / vt * 100

    scheduler.step(ep_val_loss)
    train_losses.append(ep_train_loss); val_losses.append(ep_val_loss)
    train_accs.append(ep_train_acc);    val_accs.append(ep_val_acc)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}]  "
              f"Train: Loss={ep_train_loss:.4f} Acc={ep_train_acc:.2f}%  |  "
              f"Val:   Loss={ep_val_loss:.4f} Acc={ep_val_acc:.2f}%")

    if ep_val_acc > best_val_acc:
        best_val_acc = ep_val_acc
        torch.save(model.state_dict(), os.path.join(models_dir, "lstm_native_best.pth"))

print(f"\nTraining complete. Best Val Accuracy: {best_val_acc:.2f}%")

# ─── Test Evaluation ─────────────────────────────────────────────────────────
model.load_state_dict(torch.load(os.path.join(models_dir, "lstm_native_best.pth")))
model.eval()
tc, tt = 0, 0
with torch.no_grad():
    for bX, by in test_loader:
        bX, by = bX.to(device), by.to(device)
        tc += (model(bX).argmax(1) == by).sum().item()
        tt += by.size(0)
print(f"Final Test Accuracy (Native India Model): {tc/tt*100:.2f}%")

# ─── Learning Curves ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Native Indian LSTM – Training Curves", fontsize=14)

axes[0].plot(train_losses, label="Train Loss", color="#e07b39")
axes[0].plot(val_losses,   label="Val Loss",   color="#3978e0")
axes[0].set(title="Loss Curve", xlabel="Epoch", ylabel="Loss")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(train_accs, label="Train Accuracy", color="#e07b39")
axes[1].plot(val_accs,   label="Val Accuracy",   color="#3978e0")
axes[1].set(title="Accuracy Curve", xlabel="Epoch", ylabel="Accuracy (%)")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
out_path = os.path.join(plots_dir, "Native_LSTM_Training_Curves.png")
plt.savefig(out_path, dpi=150)
print(f"Training curves saved → {out_path}")
