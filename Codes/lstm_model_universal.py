import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

data_dir = r"D:\RemoteSensing-Project\Dataset\processed_universal"
models_dir = r"D:\RemoteSensing-Project\Models"
plots_dir  = r"D:\RemoteSensing-Project\Results-plots"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Load class mapping to dynamically determine output nodes
with open(os.path.join(data_dir, "label_mapping.json"), "r") as f:
    label_mapping = json.load(f)

BATCH_SIZE = 128
EPOCHS = 20
LR = 0.001
INPUT_SIZE = 3     # NDVI, VV, VH
HIDDEN_SIZE = 64
NUM_CLASSES = len(label_mapping)
print(f"Universal LSTM configured for {NUM_CLASSES} agricultural classes detected globally.")

print("Loading universal processed data...")
X_train = np.load(os.path.join(data_dir, "X_train.npy"))
y_train = np.load(os.path.join(data_dir, "y_train.npy"))
X_val   = np.load(os.path.join(data_dir, "X_val.npy"))
y_val   = np.load(os.path.join(data_dir, "y_val.npy"))
X_test  = np.load(os.path.join(data_dir, "X_test.npy"))
y_test  = np.load(os.path.join(data_dir, "y_test.npy"))

X_train = np.nan_to_num(X_train)
X_val = np.nan_to_num(X_val)
X_test = np.nan_to_num(X_test)

train_mean = np.mean(X_train, axis=(0, 1), keepdims=True)
train_std  = np.std(X_train, axis=(0, 1), keepdims=True) + 1e-8
X_train = (X_train - train_mean) / train_std
X_val   = (X_val - train_mean) / train_std
X_test  = (X_test - train_mean) / train_std

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t   = torch.tensor(X_val, dtype=torch.float32)
y_val_t   = torch.tensor(y_val, dtype=torch.long)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_test_t  = torch.tensor(y_test, dtype=torch.long)

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds   = TensorDataset(X_val_t, y_val_t)
test_ds  = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

class UniversalCropClassifierLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(UniversalCropClassifierLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using compute device: {device}")

model = UniversalCropClassifierLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val_acc = 0.0

print("Initiating Universal Matrix Training Loop...")
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_X.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)
        
    epoch_train_loss = running_loss / total
    epoch_train_acc = correct / total * 100.0
    
    model.eval()
    val_loss, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            val_loss += criterion(outputs, batch_y).item() * batch_X.size(0)
            _, preds = torch.max(outputs, 1)
            v_correct += (preds == batch_y).sum().item()
            v_total += batch_y.size(0)
            
    epoch_val_loss = val_loss / v_total
    epoch_val_acc = v_correct / v_total * 100.0
    
    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    train_accs.append(epoch_train_acc)
    val_accs.append(epoch_val_acc)
    
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")
        
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), os.path.join(models_dir, "lstm_universal_best.pth"))

print("Universal Master Training finished.")
model.load_state_dict(torch.load(os.path.join(models_dir, "lstm_universal_best.pth")))
model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        _, preds = torch.max(outputs, 1)
        test_correct += (preds == batch_y).sum().item()
        test_total += batch_y.size(0)

test_acc = test_correct / test_total * 100.0
print(f"Universal Master Model - Final Test Accuracy: {test_acc:.2f}%")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.title('Universal Learning Curve (Loss)')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train')
plt.plot(val_accs, label='Val')
plt.title('Universal Accuracy Curve')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "LSTM_Universal_Training_Curves.png"))
