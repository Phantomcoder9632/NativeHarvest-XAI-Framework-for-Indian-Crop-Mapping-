import h5py
import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

class AttentionLSTM(nn.Module):
    def __init__(self, i, h, c, d):
        super().__init__()
        self.lstm = nn.LSTM(i, h, 2, batch_first=True, dropout=d)
        self.attn_fc = nn.Linear(h, 1)
        self.drop = nn.Dropout(d)
        self.fc = nn.Sequential(nn.Linear(h, 128), nn.ReLU(), nn.Dropout(d*0.5), nn.Linear(128, c))
    def forward(self, x):
        o, _ = self.lstm(x)
        w = torch.softmax(self.attn_fc(o), dim=1)
        ctx = (w * o).sum(dim=1)
        return self.fc(self.drop(ctx))

h5_path = r"D:\RemoteSensing-Project\Dataset\native_india_arrays.h5"
model_path = r"D:\RemoteSensing-Project\Models\lstm_native_optimized.pth"

with h5py.File(h5_path, 'r') as hf:
    X_all = hf['features'][:]
    y_raw = hf['labels'][:]

unique_crops = sorted(np.unique(y_raw[np.array([sum(y_raw==c)>=5 for c in y_raw])]).tolist())
c2i = {int(c): i for i, c in enumerate(unique_crops)}
mask = np.array([v in unique_crops for v in y_raw])
X_all, y_all = X_all[mask], y_raw[mask]
y_mapped = np.array([c2i[int(v)] for v in y_all])

X_train, X_tmp, y_train, y_tmp = train_test_split(X_all, y_mapped, test_size=0.30, random_state=42, stratify=y_mapped)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

for ch in [1, 2]:
    mu = X_train[:,:,ch].mean()
    std = X_train[:,:,ch].std() + 1e-8
    X_test[:,:,ch] = (np.nan_to_num(X_test[:,:,ch]) - mu) / std

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AttentionLSTM(3, 256, len(unique_crops), 0.5).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

Xt = torch.tensor(X_test, dtype=torch.float32).to(device)
yt = torch.tensor(y_test, dtype=torch.long).to(device)

with torch.no_grad():
    logits = model(Xt)
    preds = logits.argmax(1).cpu().numpy()
    targets = yt.cpu().numpy()

acc = (preds == targets).mean() * 100
print(f"Overall Accuracy: {acc:.2f}%")

AGRI_CODES = {1:'Wheat', 2:'Mustard', 3:'Lentil', 4:'No Crop', 6:'Sugarcane', 8:'Garlic', 9:'Maize', 13:'Gram', 15:'Unsown', 36:'Other'}
print("\nPer-Class Breakdown:")
for i, code in enumerate(unique_crops):
    m = targets == i
    if m.any():
        c_acc = (preds[m] == i).mean() * 100
        print(f"  {AGRI_CODES.get(code, f'Crop-{code}'):<16}: {c_acc:>6.1f}% ({sum(preds[m]==i)}/{sum(m)})")
