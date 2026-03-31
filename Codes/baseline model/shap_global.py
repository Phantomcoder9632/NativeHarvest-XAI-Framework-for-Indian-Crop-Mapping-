"""
SHAP Explainability for Global Model (Module A)
-----------------------------------------------
Visualizes temporal feature importance for the original CropHarvest 
Global model. This provides the XAI comparison requested by the user.
"""
import os, json
import numpy as np
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR   = r"D:\RemoteSensing-Project\Dataset\processed"
MODEL_PATH = r"D:\RemoteSensing-Project\Models\lstm_best.pth"
OUT_DIR    = r"D:\RemoteSensing-Project\Results-plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Model Definition (Module A) ─────────────────────────────────────────────
class CropClassifierLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_classes=14):
        super(CropClassifierLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def run_shap_global():
    print("Loading Global Model and Data ...")
    device = torch.device('cpu') # Use CPU for SHAP stability
    model  = CropClassifierLSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    # Load mapping to find 'Maize' (Index 1)
    with open(os.path.join(DATA_DIR, "label_mapping.json")) as f:
        cmap = json.load(f)
    print(f"  Label map: {cmap}")

    # Standardize if we have the stats, otherwise use raw for SHAP attribution
    X_t = torch.tensor(X_test[:50], dtype=torch.float32).to(device) # bg + samples
    
    print("Calculating SHAP values (GradientExplainer) ...")
    explainer = shap.GradientExplainer(model, X_t[:20]) # Background
    # Predict for Maize (Index 1)
    test_sample = X_t[1:2] 
    shap_vals   = explainer.shap_values(test_sample)

    # GradientExplainer returns (samples, steps, features, classes)
    print(f"  SHAP Result Type: {type(shap_vals)}")
    if isinstance(shap_vals, list):
        # Case where it returns a list of classes
        maize_importance = np.abs(shap_vals[1][0]) # (360, 3)
    else:
        # Case where it returns a 4D array (1, 360, 3, 14)
        print(f"  Array shape: {shap_vals.shape}")
        maize_importance = np.abs(shap_vals[0, :, :, 1]) # (360, 3)
    
    # ── Plotting ─────────────────────────────────────────────────────────────
    print("Generating Temporal Importance Plot ...")
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # temporal_imp should be (360,)
    temporal_imp = (maize_importance[:, 0] + maize_importance[:, 1] + maize_importance[:, 2]).flatten()
    print(f"  Plotting importance shape: {temporal_imp.shape}")
    
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x_ticks = np.linspace(0, 360, 12)
    
    ax.fill_between(range(360), temporal_imp, color="#3978e0", alpha=0.4, label="Global Model Attention")
    ax.plot(temporal_imp, color="#3978e0", linewidth=1.5)
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(months)
    ax.set_title("SHAP: Global Model (Module A) Crop Detection Logic\nTemporal Feature Importance (Maize Analysis)", fontsize=14)
    ax.set_xlabel("Month of the Year")
    ax.set_ylabel("AI Probability Contribution (SHAP Value)")
    ax.legend(); ax.grid(alpha=0.3)

    out_path = os.path.join(OUT_DIR, "SHAP_Global_Maize_Importance.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"SUCCESS: SHAP plot saved to {out_path}")

if __name__ == "__main__":
    run_shap_global()
