"""
SHAP Explainability for Native Indian LSTM (v4)
-----------------------------------------------
This script uses SHAP (GradientExplainer) to visualize why the 
Attention-LSTM model is so accurate for Wheat (80.8%).
It identifies the specific months (phenological peaks) in 
the 365-day SAR cycle that drive the prediction.
"""
import os, json
import numpy as np
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
import h5py

# ── Paths ───────────────────────────────────────────────────────────────────
H5_PATH    = r"D:\RemoteSensing-Project\Dataset\native_india_arrays.h5"
MODEL_PATH = r"D:\RemoteSensing-Project\Models\lstm_native_optimized.pth"
OUT_DIR    = r"D:\RemoteSensing-Project\Results-plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Model Definition (v4) ────────────────────────────────────────────────────
class AttentionLSTM(nn.Module):
    def __init__(self, i=3, h=256, c=10, d=0.5):
        super().__init__()
        self.lstm = nn.LSTM(i, h, 2, batch_first=True, dropout=d)
        self.attn_fc = nn.Linear(h, 1)
        self.drop = nn.Dropout(d)
        self.fc = nn.Sequential(
            nn.Linear(h, 128), 
            nn.ReLU(), 
            nn.Dropout(d * 0.5), 
            nn.Linear(128, c)
        )
    def forward(self, x):
        o, _ = self.lstm(x)
        w = torch.softmax(self.attn_fc(o), dim=1)
        ctx = (w * o).sum(dim=1)
        return self.fc(self.drop(ctx))

# ── SHAP Logic ───────────────────────────────────────────────────────────────
def run_shap_analysis():
    print("Loading Native Model and Data ...")
    # Using CPU for SHAP to avoid CuDNN RNN backward-pass restrictions
    device = torch.device('cpu') 
    model = AttentionLSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    with h5py.File(H5_PATH, 'r') as hf:
        X = hf['features'][:100] # Use more samples for better background
        y = hf['labels'][:100]

    X = torch.tensor(X, dtype=torch.float32).to(device)
    # Background for GradientExplainer
    background = X[:50]
    # Find a sample that is actually Wheat (Label 1 -> Index 0)
    wheat_indices = np.where(y == 1)[0]
    if len(wheat_indices) > 0:
        test_sample = X[wheat_indices[0]:wheat_indices[0]+1]
        print(f"  Using Wheat sample at index {wheat_indices[0]}")
    else:
        test_sample = X[0:1]

    print("Calculating SHAP values (GradientExplainer) ...")
    explainer = shap.GradientExplainer(model, background)
    shap_vals = explainer.shap_values(test_sample)

    # GradientExplainer returns (samples, steps, features) for EACH class
    # If test_sample is (1, 365, 3), each class array should be (1, 365, 3)
    print(f"  SHAP Result Type: {type(shap_vals)}")
    if isinstance(shap_vals, list):
        print(f"  List length: {len(shap_vals)}")
        print(f"  Class-0 shape: {getattr(shap_vals[0], 'shape', 'No Shape')}")
        # Slicing for list: [class][sample, step, feature]
        wheat_importance = np.abs(shap_vals[0][0]) # (365, 3)
    else:
        print(f"  Array shape: {shap_vals.shape}") # (1, 365, 3, 10)
        # Slicing: [sample, step, feature, class]
        wheat_importance = np.abs(shap_vals[0, :, :, 0]) # (365, 3)
    
    # ── Plotting ─────────────────────────────────────────────────────────────
    print("Generating Phenomenological Importance Plot ...")
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # temporal_imp should be (365,)
    temporal_imp = (wheat_importance[:, 1] + wheat_importance[:, 2]).flatten()
    print(f"  Plotting importance shape: {temporal_imp.shape}")
    
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x_ticks = np.linspace(0, 365, 12)
    
    ax.fill_between(range(365), temporal_imp, color="#e07b39", alpha=0.4, label="AI Attention Score")
    ax.plot(temporal_imp, color="#e07b39", linewidth=1.5)
    
    # Highlight Rabi (Winter) Season
    ax.axvspan(0, 100, color='green', alpha=0.1, label="Rabi (Winter) Growth") # Jan-Mar
    ax.axvspan(300, 365, color='green', alpha=0.1) # Oct-Dec
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(months)
    ax.set_title("SHAP: Why is Native Model 80.8% Accurate for Wheat?\nFeature Importance over 365-Day Seasonal Cycle", fontsize=14)
    ax.set_xlabel("Month of the Year (2021)")
    ax.set_ylabel("AI Probability Contribution (SHAP Value)")
    ax.legend()
    ax.grid(alpha=0.3)

    out_path = os.path.join(OUT_DIR, "SHAP_Native_Wheat_Importance.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"SUCCESS: SHAP plot saved to {out_path}")

if __name__ == "__main__":
    run_shap_analysis()
