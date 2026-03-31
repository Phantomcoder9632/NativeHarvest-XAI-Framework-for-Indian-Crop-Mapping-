import os
import torch
import torch.nn as nn
import numpy as np
import shap

data_dir = r"D:\RemoteSensing-Project\Dataset\processed"
models_dir = r"D:\RemoteSensing-Project\Models"

INPUT_SIZE = 3
HIDDEN_SIZE = 64
NUM_CLASSES = 14

class CropClassifierLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CropClassifierLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

device = torch.device('cpu')
model = CropClassifierLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
model.load_state_dict(torch.load(os.path.join(models_dir, "lstm_best.pth"), map_location=device, weights_only=True))
model.eval()

X_train = np.load(os.path.join(data_dir, "X_train.npy"))
X_test = np.load(os.path.join(data_dir, "X_test.npy"))

np.random.seed(42)
bg_idx = np.random.choice(X_train.shape[0], 5, replace=False)
test_idx = np.random.choice(X_test.shape[0], 2, replace=False)

background = torch.tensor(X_train[bg_idx], dtype=torch.float32).to(device)
test_samples = torch.tensor(X_test[test_idx], dtype=torch.float32).to(device)

explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(test_samples)

print(type(shap_values))
if isinstance(shap_values, list):
    print("List length:", len(shap_values))
    print("Element shape:", np.array(shap_values[0]).shape)
else:
    print("Shape:", np.array(shap_values).shape)
