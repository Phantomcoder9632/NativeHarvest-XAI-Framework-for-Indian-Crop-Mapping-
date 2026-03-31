import os
import torch
import torch.nn as nn
import numpy as np
import shap
import matplotlib.pyplot as plt

data_dir = r"D:\RemoteSensing-Project\Dataset\processed"
models_dir = r"D:\RemoteSensing-Project\Models"
plots_dir  = r"D:\RemoteSensing-Project\Results-plots"

INPUT_SIZE = 3
HIDDEN_SIZE = 64
NUM_CLASSES = 14
# Our bands were extracted in order: [17, 0, 1] which is NDVI, VV, VH
FEATURE_NAMES = ["NDVI", "VV", "VH"]

class CropClassifierLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CropClassifierLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

device = torch.device('cpu') # Avoid cuda runtime RNN errors with SHAP
model = CropClassifierLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
model.load_state_dict(torch.load(os.path.join(models_dir, "lstm_best.pth"), map_location=device, weights_only=True))
model.eval()

print("Loading data for SHAP interpretation...")
X_train = np.load(os.path.join(data_dir, "X_train.npy"))
X_test = np.load(os.path.join(data_dir, "X_test.npy"))

np.random.seed(42)
bg_idx = np.random.choice(X_train.shape[0], 15, replace=False)
test_idx = np.random.choice(X_test.shape[0], 15, replace=False)

background = torch.tensor(X_train[bg_idx], dtype=torch.float32).to(device)
test_samples = torch.tensor(X_test[test_idx], dtype=torch.float32).to(device)

print("Calculating SHAP values (this might take a minute)...")
explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(test_samples)

print("Processing SHAP values for summary plot...")
# shap_values has shape (samples, timesteps, features, classes)
overall_shap = np.sum(np.abs(shap_values), axis=(1, 3))

features_2d = np.mean(test_samples.numpy(), axis=1)

# Generate Bar Plot
plt.figure(figsize=(8, 6))
shap.summary_plot(overall_shap, features=features_2d, feature_names=FEATURE_NAMES, plot_type="bar", show=False)
plt.title("Feature Importance (SHAP Bar Plot)")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "SHAP_Summary_Bar.png"))
plt.close()

# Generate Beeswarm Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(overall_shap, features=features_2d, feature_names=FEATURE_NAMES, show=False)
plt.title("Feature Impact (SHAP Beeswarm Plot)")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "SHAP_Summary_Beeswarm.png"))
plt.close()

print("SHAP explanation plots saved successfully.")
