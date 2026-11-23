"""HMS EEG classification: simple scikit-learn teaching baseline.

Run in a Kaggle notebook with the HMS competition dataset attached, or locally:
    python hms_sklearn_starter.py --data-dir /path/to/hms-harmful-brain-activity-classification

Requires: numpy, pandas, scipy, scikit-learn, pyarrow.
This is a teaching baseline, not a competitive Kaggle solution.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
FS = 200
# A small, interpretable subset of EEG electrodes (not the EKG channel).
CHANNELS = [
    "Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1",
    "Fz", "Cz", "Pz",
    "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2",
]
BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30)]
LEFT_CHANNELS = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1"]
RIGHT_CHANNELS = ["Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"]
BIPOLAR_PAIRS = [
    ("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
]
SPEC_REGIONS = ["LL", "RL", "LP", "RP"]


def extract_spectrogram_features(spec_path, offset_seconds=0.0):
    """Extract summary log-power features from the 10-min and central spectrogram."""
    spec = pd.read_parquet(spec_path)
    if "time" in spec.columns:
        t = spec["time"].to_numpy(dtype=np.float64)
        center = float(offset_seconds) + 300.0
        mask_center = (t >= center - 10.0) & (t <= center + 10.0)
        spec_center = spec.loc[mask_center] if mask_center.any() else spec
    else:
        mid = len(spec) // 2
        spec_center = spec.iloc[max(0, mid - 5) : min(len(spec), mid + 5)]

    features = []
    for win_df in (spec_center, spec):
        for region in SPEC_REGIONS:
            cols = [c for c in win_df.columns if c.startswith(f"{region}_")]
            if not cols:
                features.extend([0.0] * (2 * len(BANDS)))
                continue
            freqs = np.array([float(c.split("_")[1]) for c in cols])
            vals = np.log1p(np.clip(win_df[cols].to_numpy(dtype=np.float64), 0.0, None))
            for low, high in BANDS:
                bmask = (freqs >= low) & (freqs < high)
                if bmask.any():
                    band_vals = vals[:, bmask]
                    features.append(float(np.nanmean(band_vals)))
                    features.append(float(np.nanstd(band_vals)))
                else:
                    features.extend([0.0, 0.0])

    arr = np.asarray(features, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

def _channel_features(x):
    feats = [np.mean(x), np.std(x), np.ptp(x)]
    frequencies, psd = welch(x, fs=FS, nperseg=min(512, len(x)))
    total_mask = (frequencies >= 0.5) & (frequencies < 30.0)
    psd_total = psd[total_mask]
    total_power = np.sum(psd_total) + 1e-12
    psd_norm = psd_total / total_power
    spec_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12)) / np.log2(len(psd_norm) + 1e-12)
    feats.append(spec_entropy)

    log_bp, rel_bp = [], []
    for low, high in BANDS:
        mask = (frequencies >= low) & (frequencies < high)
        bp = np.sum(psd[mask])
        log_bp.append(np.log1p(bp))
        rel_bp.append(bp / total_power)
    feats.extend(log_bp)
    feats.extend(rel_bp)
    return feats, np.array(log_bp, dtype=np.float64)

def extract_features(eeg_path, offset_seconds=0.0):
    """Extract features from the central 10 s of a 50 s annotated segment.

    train.csv offsets refer to the START of each 50 s segment; the labels
    concern its central 10 s. Test EEG files are exactly 50 s long.
    """
    frame = pd.read_parquet(eeg_path)
    channels = [c for c in CHANNELS if c in frame.columns]
    if not channels:
        raise ValueError(f"No expected EEG channels in {eeg_path}")

    start = int(round((float(offset_seconds) + 20.0) * FS))
    end = start + 10 * FS
    if end > len(frame):
        raise ValueError(f"EEG window outside file: {eeg_path}, offset={offset_seconds}")
    segment = frame.iloc[start:end][channels].to_numpy(dtype=np.float64)

    arr = _extract_window_features(segment, channels)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

def _extract_window_features(segment, channels):
    clean_signals = {}
    features = []
    channel_log_bands = {}
    for j, ch in enumerate(channels):
        x = segment[:, j]
        if not np.isfinite(x).any():
            x = np.zeros_like(x)
        else:
            fill = np.nanmedian(x[np.isfinite(x)])
            x = np.where(np.isfinite(x), x, fill)
        x = np.clip(x, -1000, 1000)
        clean_signals[ch] = x

        feats, log_bp = _channel_features(x)
        features.extend(feats)
        channel_log_bands[ch]= np.array(log_bp, dtype=np.float64)


    for ch1, ch2 in BIPOLAR_PAIRS:
        if ch1 in clean_signals and ch2 in clean_signals:
            feats, _ = _channel_features(clean_signals[ch1] - clean_signals[ch2])
            features.extend(feats)
        else:
            features.extend([0.0] * 12)

    for l_ch, r_ch in zip(LEFT_CHANNELS, RIGHT_CHANNELS):
        if l_ch in channel_log_bands and r_ch in channel_log_bands:
            features.extend((channel_log_bands[l_ch] - channel_log_bands[r_ch]).tolist())
        else:
            features.extend([0.0] * len(BANDS))

    arr = np.asarray(features, dtype=np.float32)
    return arr


def build_features(metadata, eeg_folder, spec_folder, is_train=True):
    X = []
    for i, row in enumerate(metadata.itertuples(index=False)):
        eeg_path = eeg_folder / f"{row.eeg_id}.parquet"
        spec_path = spec_folder / f"{row.spectrogram_id}.parquet"
        eeg_offset = row.eeg_label_offset_seconds if is_train else 0.0
        spec_offset = row.spectrogram_label_offset_seconds if is_train else 0.0
        f_eeg = extract_features(eeg_path, eeg_offset)
        f_spec = extract_spectrogram_features(spec_path, spec_offset)
        X.append(np.concatenate([f_eeg, f_spec]))
        if (i + 1) % 200 == 0:
            print(f"  Extracted {i + 1}/{len(metadata)} recordings")
    return np.vstack(X)


def predict_soft_probabilities(fold_models, X):
    """Predict normalized 6-class probability distributions from per-class regressors."""
    preds = np.column_stack([m.predict(X) for m in fold_models])
    preds = np.clip(preds, 1e-6, None)
    return preds / preds.sum(axis=1, keepdims=True)

def calibrate_probabilities(probs, temp=1.0, eps=1e-4):
    """Apply temperature scaling and a minimum probability floor for KL divergence."""
    p = np.clip(probs, eps, 1.0)
    p = np.power(p, 1.0 / temp)
    p = np.clip(p, eps, None)
    return p / p.sum(axis=1, keepdims=True)

def predict_mlp_probabilities(mlp_model, X):
    """Predict normalized 6-class probability distributions from an MLPRegressor."""
    preds = np.clip(mlp_model.predict(X), 1e-6, None)
    return preds / preds.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/kaggle/input/hms-harmful-brain-activity-classification"))
    parser.add_argument("--max-train", type=int, default=0, help="Limit examples for a quick classroom run; 0 uses all rows")
    parser.add_argument("--submission", action="store_true", help="Also predict the provided test EEGs")
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    # Keep one annotated window per EEG for a fast, simple starting point.
    agg_dict = {col: "sum" for col in TARGETS}
    agg_dict["patient_id"] = "first"
    agg_dict["eeg_label_offset_seconds"] = "median"
    agg_dict["spectrogram_id"] = "first"
    agg_dict["spectrogram_label_offset_seconds"] = "median"
    train = train.groupby("eeg_id", as_index=False).agg(agg_dict)
    if args.max_train > 0 and len(train) > args.max_train:
        train = train.sample(n=args.max_train, random_state=42).reset_index(drop=True)

    # Soft targets are the normalized expert votes; the simple RF fits hard labels.
    votes = train[TARGETS].to_numpy(dtype=float)
    soft_targets = votes / votes.sum(axis=1, keepdims=True)
    hard_targets = np.argmax(soft_targets, axis=1)

    cache_file = Path(f"/tmp/features_train_g_{len(train)}.npz")
    if cache_file.exists():
        print(f"Loading cached training features from {cache_file}...")
        X = np.load(cache_file)["X"]
    else:
        print("Extracting training EEG features...")
        X = build_features(train, args.data_dir / "train_eegs", args.data_dir / "train_spectrograms", is_train=True)
        np.savez(cache_file, X=X)
        print(f"Saved cached features to {cache_file}")

    # 5-fold cross-validation grouped by patient_id to prevent data leakage.
    gkf = GroupKFold(n_splits=5)
    oof_hgb = np.zeros_like(soft_targets)
    oof_mlp = np.zeros_like(soft_targets)
    hgb_models = []
    mlp_models = []
    sample_weights = np.log1p(votes.sum(axis=1))

    for fold, (idx_train, idx_val) in enumerate(gkf.split(X, hard_targets, groups=train["patient_id"])):
        fold_models = []
        for cls_idx in range(len(TARGETS)):
            reg = HistGradientBoostingRegressor(
                max_iter=250,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=42 + fold * 10 + cls_idx,
            )
            reg.fit(
                X[idx_train],
                soft_targets[idx_train, cls_idx],
                sample_weight=sample_weights[idx_train],
            )
            fold_models.append(reg)
        oof_hgb[idx_val] = predict_soft_probabilities(fold_models, X[idx_val])
        hgb_models.append(fold_models)

        mlp = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0.0),
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(256, 128),
                alpha=1.0,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=40,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42 + fold,
            ),
        )
        mlp.fit(X[idx_train], soft_targets[idx_train])
        oof_mlp[idx_val] = predict_mlp_probabilities(mlp, X[idx_val])
        mlp_models.append(mlp)

    y_true = np.clip(soft_targets, 1e-12, 1.0)

    def kl_loss(params):
        w, temp, eps = params
        blend = w * oof_hgb + (1.0 - w) * oof_mlp
        cal = calibrate_probabilities(blend, temp=temp, eps=eps)
        return float(np.mean(np.sum(y_true * (np.log(y_true) - np.log(cal)), axis=1)))

    res = minimize(
        kl_loss,
        x0=[0.75, 0.85, 0.01],
        bounds=[(0.0, 1.0), (0.5, 2.0), (1e-5, 0.05)],
        method="Nelder-Mead",
    )
    best_w, best_temp, best_eps = res.x
    oof_prob = calibrate_probabilities(best_w * oof_hgb + (1.0 - best_w) * oof_mlp, temp=best_temp, eps=best_eps)

    oof_pred = np.argmax(oof_prob, axis=1)
    print(f"Optimal temperature: {best_temp:.3f} | Floor eps: {best_eps:.5f}")
    print(f"5-fold OOF accuracy (consensus/majority class): {accuracy_score(hard_targets, oof_pred):.4f}")
    print(f"5-fold OOF log loss (hard labels): {log_loss(hard_targets, oof_prob, labels=np.arange(6)):.4f}")
    kl = np.mean(np.sum(y_true * (np.log(y_true) - np.log(oof_prob)), axis=1))
    print(f"5-fold OOF KL divergence (soft expert votes): {kl:.4f}")

    if args.submission:
        test = pd.read_csv(args.data_dir / "test.csv")
        print("Extracting test EEG features...")
        X_test = build_features(test, args.data_dir / "test_eegs", args.data_dir / "test_spectrograms", is_train=False)
        pred_hgb = np.mean([predict_soft_probabilities(m, X_test) for m in hgb_models], axis=0)
        pred_mlp = np.mean([predict_mlp_probabilities(m, X_test) for m in mlp_models], axis=0)
        pred = calibrate_probabilities(best_w * pred_hgb + (1.0 - best_w) * pred_mlp, temp=best_temp, eps=best_eps)
        submission = pd.DataFrame(pred, columns=TARGETS)
        submission.insert(0, "eeg_id", test["eeg_id"].to_numpy())
        submission.to_csv("submission.csv", index=False)
        print("Saved submission.csv")


if __name__ == "__main__":
    main()