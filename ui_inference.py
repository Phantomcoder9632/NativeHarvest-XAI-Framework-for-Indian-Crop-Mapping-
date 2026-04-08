from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None

    class _MissingTorchModule:
        pass

    class _MissingNN:
        Module = _MissingTorchModule

    nn = _MissingNN()


DEFAULT_LABELS = [
    "cassava",
    "maize",
    "sorghum",
    "bean",
    "groundnut",
    "fallowland",
    "millet",
    "tomato",
    "sugarcane",
    "sweetpotato",
    "banana",
    "soybean",
    "cabbage",
    "none",
]

DISPLAY_NAME_MAP = {
    "cassava": "Cassava",
    "maize": "Maize",
    "sorghum": "Sorghum",
    "bean": "Bean",
    "groundnut": "Groundnut",
    "fallowland": "Fallow Land",
    "millet": "Millet",
    "tomato": "Tomato",
    "sugarcane": "Sugarcane",
    "sweetpotato": "Sweet Potato",
    "banana": "Banana",
    "soybean": "Soybean",
    "cabbage": "Cabbage",
    "none": "No Crop / None",
    "mustard": "Mustard",
    "wheat": "Wheat",
    "rice": "Rice",
    "cotton": "Cotton",
    "gram": "Gram",
    "lentil": "Lentil",
    "garlic": "Garlic",
    "other": "Other",
    "unsown": "Unsown",
}

RABI_CROPS = {"wheat", "mustard", "gram", "lentil", "garlic"}
KHARIF_CROPS = {"maize", "rice", "sugarcane", "cotton"}
SUPPORTED_CROP_CHOICES = [
    "Wheat",
    "Mustard",
    "Gram",
    "Maize",
    "Rice",
    "Sugarcane",
    "Cotton",
]


@dataclass
class ModelAssets:
    project_root: Path
    model_key: str
    model_label: str
    model_path: Path | None
    dataset_path: Path | None
    processed_dir: Path | None
    crop_names: list[str]
    channel_stats: dict[int, tuple[float, float]]
    sequence_length: int
    input_size: int
    architecture: str
    season: str
    best_for: list[str]
    summary: str
    dataset_story: dict[str, str | int | float]
    benchmark_metrics: dict[str, str | int | float | None]


class AttentionLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=dropout)
        self.attn_fc = nn.Linear(hidden_size, 1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        attn_w = torch.softmax(self.attn_fc(out), dim=1)
        ctx = (attn_w * out).sum(dim=1)
        return self.fc(self.drop(ctx))


class CropClassifierLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


class UniversalCropClassifierLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _display_name(value: str) -> str:
    return DISPLAY_NAME_MAP.get(value.lower(), value.replace("_", " ").title())


def _load_label_names(processed_dir: Path | None, fallback: list[str] | None = None) -> list[str]:
    fallback = fallback or DEFAULT_LABELS
    if processed_dir is None:
        return fallback

    mapping_path = _first_existing([processed_dir / "label_mapping.json", processed_dir / "crop_label_map.json"])
    if mapping_path is None:
        return fallback

    with open(mapping_path, "r", encoding="utf-8") as fh:
        mapping = json.load(fh)

    if all(isinstance(v, int) for v in mapping.values()):
        ordered = sorted(mapping.items(), key=lambda item: item[1])
        return [_display_name(str(name)) for name, _ in ordered]

    inverse = {int(v): _display_name(str(k)) for k, v in mapping.items()}
    return [inverse[idx] for idx in sorted(inverse)]


def _checkpoint_num_classes(state: dict, model_key: str) -> int | None:
    if model_key == "native" and "fc.3.weight" in state:
        return int(state["fc.3.weight"].shape[0])
    if model_key == "universal" and "fc.weight" in state:
        return int(state["fc.weight"].shape[0])
    return None


def _aligned_crop_names(crop_names: list[str], num_classes: int, model_key: str) -> list[str]:
    if len(crop_names) == num_classes:
        return crop_names

    native_default = ["Wheat", "Mustard", "Lentil", "No Crop", "Sugarcane", "Garlic", "Maize", "Gram", "Unsown", "Other"]
    universal_default = ["Maize", "Wheat", "Rice", "Paddy", "Sorghum", "Millet", "Sugarcane", "Cotton", "Groundnut", "Soybean", "Mustard"]
    fallback = native_default if model_key == "native" else universal_default

    if len(fallback) >= num_classes:
        return fallback[:num_classes]

    if len(crop_names) >= num_classes:
        return crop_names[:num_classes]

    padded = list(crop_names)
    while len(padded) < num_classes:
        padded.append(f"Class {len(padded)}")
    return padded


def normalize_crop_name(crop_name: str) -> str:
    return crop_name.strip().lower()


def route_crop_to_model(crop_name: str) -> str:
    normalized = normalize_crop_name(crop_name)
    if normalized in RABI_CROPS:
        return "native"
    if normalized in KHARIF_CROPS:
        return "universal"
    return "native"


def _compute_channel_stats(processed_dir: Path | None) -> tuple[dict[int, tuple[float, float]], int, int]:
    channel_stats = {1: (0.0, 1.0), 2: (0.0, 1.0)}
    sequence_length = 365
    input_size = 3
    if processed_dir is None:
        return channel_stats, sequence_length, input_size

    x_train_path = processed_dir / "X_train.npy"
    if not x_train_path.exists():
        return channel_stats, sequence_length, input_size

    x_train = np.load(x_train_path, mmap_mode="r")
    sequence_length = int(x_train.shape[1])
    input_size = int(x_train.shape[2])
    for ch in [1, 2]:
        mu = float(np.nanmean(x_train[:, :, ch]))
        std = float(np.nanstd(x_train[:, :, ch]) + 1e-8)
        channel_stats[ch] = (mu, std)
    return channel_stats, sequence_length, input_size


def _external_root() -> Path:
    return Path(r"D:\RemoteSensing-Project")


def _native_assets(project_root: Path) -> ModelAssets:
    external_root = _external_root()
    processed_dir = _first_existing([
        project_root / "Dataset" / "native_processed",
        project_root / "Dataset" / "processed",
        external_root / "Dataset" / "native_processed",
        external_root / "Dataset" / "processed",
    ])
    channel_stats, sequence_length, input_size = _compute_channel_stats(processed_dir)
    return ModelAssets(
        project_root=project_root,
        model_key="native",
        model_label="Native Indian Optimized",
        model_path=_first_existing([
            project_root / "Models" / "lstm_native_optimized.pth",
            project_root / "Models" / "lstm_native_best.pth",
            external_root / "Models" / "lstm_native_optimized.pth",
            external_root / "Models" / "lstm_native_best.pth",
        ]),
        dataset_path=_first_existing([
            project_root / "Dataset" / "native_india_arrays.h5",
            external_root / "Dataset" / "native_india_arrays.h5",
        ]),
        processed_dir=processed_dir,
        crop_names=_load_label_names(processed_dir, ["Wheat", "Mustard", "Lentil", "No Crop", "Sugarcane", "Garlic", "Maize", "Gram", "Unsown", "Other"]),
        channel_stats=channel_stats,
        sequence_length=sequence_length,
        input_size=input_size,
        architecture="attention",
        season="Rabi / Winter",
        best_for=["Wheat", "Mustard", "Gram"],
        summary="Built for native Indian seasonal behaviour with a full 365-day cycle, especially to recover winter crop signatures.",
        dataset_story={
            "primary_dataset": "Native India seasonal tensor dataset",
            "geography": "Indian crop fields with district-level validation context",
            "timeline": "365-day Jan-Dec observation window",
            "field_count": 17644,
            "global_arrays": 4633,
            "signals": "NDVI, VV, VH",
        },
        benchmark_metrics={
            "accuracy": 80.8,
            "accuracy_label": "Wheat detection accuracy in Northern India",
            "precision": None,
            "precision_note": "Precision is not documented for the native deployment build yet.",
            "spatial_survey_accuracy": 93.3,
            "mean_accuracy_error": 6.70,
        },
    )


def _universal_assets(project_root: Path) -> ModelAssets:
    external_root = _external_root()
    processed_dir = _first_existing([
        project_root / "Dataset" / "processed_universal",
        external_root / "Dataset" / "processed_universal",
    ])
    channel_stats, sequence_length, input_size = _compute_channel_stats(processed_dir)
    return ModelAssets(
        project_root=project_root,
        model_key="universal",
        model_label="Global Transfer",
        model_path=_first_existing([
            project_root / "Models" / "lstm_universal_best.pth",
            project_root / "Models" / "lstm_model.pth",
            external_root / "Models" / "lstm_universal_best.pth",
            external_root / "Models" / "lstm_model.pth",
        ]),
        dataset_path=_first_existing([
            project_root / "Dataset" / "processed_universal",
            external_root / "Dataset" / "processed_universal",
        ]),
        processed_dir=processed_dir,
        crop_names=_load_label_names(processed_dir, ["Maize", "Wheat", "Rice", "Sorghum", "Millet", "Sugarcane", "Cotton", "Groundnut", "Soybean", "Mustard", "Fallowland", "Non-Crop"]),
        channel_stats=channel_stats,
        sequence_length=sequence_length,
        input_size=input_size,
        architecture="universal_lstm",
        season="Kharif / Monsoon",
        best_for=["Maize", "Rice", "Sugarcane", "Cotton"],
        summary="Trained on the NASA CropHarvest-style global transfer setup and strongest on broad tropical monsoon staples.",
        dataset_story={
            "primary_dataset": "NASA CropHarvest / global transfer dataset",
            "geography": "Global crop signatures transferred into Indian seasonal comparison",
            "timeline": "Seasonal sequence modelling for broad tropical staples",
            "field_count": 4633,
            "global_arrays": 4633,
            "signals": "NDVI, VV, VH",
        },
        benchmark_metrics={
            "accuracy": 88.04,
            "accuracy_label": "General binary crop classification accuracy",
            "precision": None,
            "precision_note": "Precision is not documented for the universal model in this repo yet.",
            "spatial_survey_accuracy": None,
            "mean_accuracy_error": None,
        },
    )


def discover_model_catalog(project_root: Path) -> dict[str, ModelAssets]:
    return {
        "native": _native_assets(project_root),
        "universal": _universal_assets(project_root),
    }


def discover_assets(project_root: Path, crop_name: str = "Wheat") -> ModelAssets:
    catalog = discover_model_catalog(project_root)
    return catalog[route_crop_to_model(crop_name)]


def build_manual_profile(
    ndvi_peak: float,
    vv_base: float,
    vv_wave: float,
    vh_base: float,
    vh_wave: float,
    peak_day: int,
    spread: int,
    length: int = 365,
) -> pd.DataFrame:
    days = np.arange(length)
    sigma = max(spread, 5)
    ndvi = 0.15 + ndvi_peak * np.exp(-0.5 * ((days - peak_day) / sigma) ** 2)
    vv = vv_base + vv_wave * np.sin(2 * np.pi * days / length)
    vh = vh_base + vh_wave * np.cos(2 * np.pi * (days - peak_day) / length)
    return pd.DataFrame({"day": days + 1, "NDVI": np.clip(ndvi, 0.0, 1.0), "VV": vv, "VH": vh})


def prepare_input_frame(frame: pd.DataFrame, length: int) -> pd.DataFrame:
    renamed = frame.rename(columns={c: c.strip().upper() for c in frame.columns})
    required = ["NDVI", "VV", "VH"]
    missing = [col for col in required if col not in renamed.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    trimmed = renamed[required].apply(pd.to_numeric, errors="coerce")
    trimmed = trimmed.interpolate(limit_direction="both").ffill().bfill()
    trimmed = trimmed.iloc[:length].copy()

    if len(trimmed) < length:
        pad = pd.DataFrame([trimmed.iloc[-1].to_dict()] * (length - len(trimmed)))
        trimmed = pd.concat([trimmed, pad], ignore_index=True)

    trimmed.insert(0, "DAY", np.arange(1, length + 1))
    return trimmed


def normalize_features(frame: pd.DataFrame, assets: ModelAssets) -> np.ndarray:
    arr = frame[["NDVI", "VV", "VH"]].to_numpy(dtype=np.float32)
    arr = np.nan_to_num(arr)
    for ch in [1, 2]:
        mu, std = assets.channel_stats[ch]
        arr[:, ch] = (arr[:, ch] - mu) / std
    return arr


def load_model(assets: ModelAssets):
    if assets.model_path is None or torch is None:
        return None

    state = torch.load(assets.model_path, map_location="cpu")
    num_classes = _checkpoint_num_classes(state, assets.model_key) or len(assets.crop_names)
    assets.crop_names = _aligned_crop_names(assets.crop_names, num_classes, assets.model_key)
    if assets.model_key == "native":
        model = AttentionLSTM(assets.input_size, 256, num_classes, 0.5)
    elif assets.model_key == "universal":
        model = UniversalCropClassifierLSTM(assets.input_size, 64, num_classes)
    elif assets.architecture == "attention":
        model = AttentionLSTM(assets.input_size, 256, num_classes, 0.5)
    else:
        model = CropClassifierLSTM(assets.input_size, 128, num_classes, 0.3)

    model.load_state_dict(state)
    model.eval()
    return model


def predict(frame: pd.DataFrame, assets: ModelAssets, model, crop_focus: str | None = None) -> dict:
    prepared = prepare_input_frame(frame, assets.sequence_length)
    features = normalize_features(prepared, assets)

    if model is None:
        mean_ndvi = float(prepared["NDVI"].mean())
        radar_signal = float(prepared["VH"].mean() - prepared["VV"].mean())
        seed_vector = np.linspace(0.2, 0.9, num=len(assets.crop_names))
        weights = seed_vector + mean_ndvi + radar_signal * 0.05
        probs = np.exp(weights) / np.exp(weights).sum()
    else:
        with torch.no_grad():
            tensor = torch.tensor(features[None, :, :], dtype=torch.float32)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    ranking = pd.DataFrame(
        {
            "crop_code": list(range(len(assets.crop_names))),
            "crop_name": assets.crop_names,
            "probability": probs,
        }
    ).sort_values("probability", ascending=False, ignore_index=True)

    crop_focus_name = crop_focus or str(ranking.loc[0, "crop_name"])
    return {
        "prepared_frame": prepared,
        "top_code": int(ranking.loc[0, "crop_code"]),
        "top_name": str(ranking.loc[0, "crop_name"]),
        "confidence": float(ranking.loc[0, "probability"]),
        "ranking": ranking,
        "ndvi_summary": float(prepared["NDVI"].mean()),
        "signal_summary": {
            "ndvi_mean": float(prepared["NDVI"].mean()),
            "vv_mean": float(prepared["VV"].mean()),
            "vh_mean": float(prepared["VH"].mean()),
            "season_length": int(len(prepared)),
        },
        "inference_story": {
            "mode": "trained_model" if model is not None else "demo_scoring",
            "architecture": assets.architecture,
            "sequence_length": assets.sequence_length,
            "data_source": assets.dataset_story["primary_dataset"],
            "model_label": assets.model_label,
            "crop_focus": crop_focus_name,
            "season": assets.season,
        },
    }
