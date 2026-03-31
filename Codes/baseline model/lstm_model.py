import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Directories
data_dir = r"D:\RemoteSensing-Project\Dataset\processed"
models_dir = r"D:\RemoteSensing-Project\Models"
plots_dir  = r"D:\RemoteSensing-Project\Results-plots"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 30
LR = 0.001
INPUT_SIZE = 3     # NDVI, VV, VH
HIDDEN_SIZE = 64
NUM_CLASSES = 14   # Based on the json mapping

# Load data
print("Loading processed data...")
X_train = np.load(os.path.join(data_dir, "X_train.npy"))
y_train = np.load(os.path.join(data_dir, "y_train.npy"))
X_val   = np.load(os.path.join(data_dir, "X_val.npy"))
y_val   = np.load(os.path.join(data_dir, "y_val.npy"))
X_test  = np.load(os.path.join(data_dir, "X_test.npy"))
y_test  = np.load(os.path.join(data_dir, "y_test.npy"))

# NaN filling in case interpolation created NaNs (extrapolation)
X_train = np.nan_to_num(X_train)
X_val = np.nan_to_num(X_val)
X_test = np.nan_to_num(X_test)

# Standardization -> subtract mean, divide by std across training set
train_mean = np.mean(X_train, axis=(0, 1), keepdims=True)
train_std  = np.std(X_train, axis=(0, 1), keepdims=True) + 1e-8

X_train = (X_train - train_mean) / train_std
X_val   = (X_val - train_mean) / train_std
X_test  = (X_test - train_mean) / train_std

# Convert to PyTorch Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t   = torch.tensor(X_val, dtype=torch.float32)
y_val_t   = torch.tensor(y_val, dtype=torch.long)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_test_t  = torch.tensor(y_test, dtype=torch.long)

# Datasets & Loaders
train_ds = TensorDataset(X_train_t, y_train_t)
val_ds   = TensorDataset(X_val_t, y_val_t)
test_ds  = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# Model Definition
class CropClassifierLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CropClassifierLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        # We take the output of the last time step for classification
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)
        # out shape: (batch, seq_len, hidden_size)
        # Get the representation from the last time step
        last_out = out[:, -1, :]
        logits = self.fc(last_out)
        return logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = CropClassifierLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training Loop
train_losses = []
val_losses = []
train_accs = []
val_accs = []

best_val_acc = 0.0

print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
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
    
    # Validation
    model.eval()
    val_loss = 0.0
    v_correct = 0
    v_total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            val_loss += loss.item() * batch_X.size(0)
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
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")
        
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), os.path.join(models_dir, "lstm_best.pth"))

print("Training finished.")

# Test Evaluation
model.load_state_dict(torch.load(os.path.join(models_dir, "lstm_best.pth")))
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        _, preds = torch.max(outputs, 1)
        test_correct += (preds == batch_y).sum().item()
        test_total += batch_y.size(0)

test_acc = test_correct / test_total * 100.0
print(f"Final Test Accuracy (Best Model): {test_acc:.2f}%")

# Plot Learning Curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Val Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "LSTM_Training_Curves.png"))
print("Plots saved.")
