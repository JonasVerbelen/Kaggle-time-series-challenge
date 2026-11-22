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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold

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

    start_50 = int(round(float(offset_seconds) * FS))
    end_50 = start_50 + 50 * FS
    if end_50 > len(frame):
        raise ValueError(f"EEG window outside file: {eeg_path}, offset={offset_seconds}")
    seg_50 = frame.iloc[start_50:end_50][channels].to_numpy(dtype=np.float64)
    seg_10 = seg_50[20 * FS : 30 * FS]

    feat_10 = _extract_window_features(seg_10, channels)
    feat_50 = _extract_window_features(seg_50, channels)
    feat_diff = feat_10 - feat_50

    arr = np.concatenate([feat_10, feat_50, feat_diff])
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


def build_features(metadata, eeg_folder, is_train=True):
    X = []
    for i, row in enumerate(metadata.itertuples(index=False)):
        eeg_path = eeg_folder / f"{row.eeg_id}.parquet"
        offset = row.eeg_label_offset_seconds if is_train else 0.0
        X.append(extract_features(eeg_path, offset))
        if (i + 1) % 200 == 0:
            print(f"  Extracted {i + 1}/{len(metadata)} recordings")
    return np.vstack(X)


def align_probabilities(model, X, n_classes=6):
    """Keep all six classes even when a small training subset omits one."""
    p = np.zeros((len(X), n_classes), dtype=float)
    p[:, model.classes_.astype(int)] = model.predict_proba(X)
    return np.clip(p, 1e-8, None) / np.clip(p, 1e-8, None).sum(axis=1, keepdims=True)


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
    train = train.groupby("eeg_id", as_index=False).agg(agg_dict)
    if args.max_train > 0 and len(train) > args.max_train:
        train = train.sample(n=args.max_train, random_state=42).reset_index(drop=True)

    # Soft targets are the normalized expert votes; the simple RF fits hard labels.
    votes = train[TARGETS].to_numpy(dtype=float)
    soft_targets = votes / votes.sum(axis=1, keepdims=True)
    hard_targets = np.argmax(soft_targets, axis=1)

    cache_file = Path(f"/tmp/features_train_f_{len(train)}.npz")
    if cache_file.exists():
        print(f"Loading cached training features from {cache_file}...")
        X = np.load(cache_file)["X"]
    else:
        print("Extracting training EEG features...")
        X = build_features(train, args.data_dir / "train_eegs", is_train=True)
        np.savez(cache_file, X=X)
        print(f"Saved cached features to {cache_file}")

    # 5-fold cross-validation grouped by patient_id to prevent data leakage.
    gkf = GroupKFold(n_splits=5)
    oof_prob = np.zeros_like(soft_targets)
    models = []

    for fold, (idx_train, idx_val) in enumerate(gkf.split(X, hard_targets, groups=train["patient_id"])):
        model = RandomForestClassifier(
            n_estimators=150, min_samples_leaf=2, random_state=42 + fold, n_jobs=-1
        )
        model.fit(X[idx_train], hard_targets[idx_train])
        oof_prob[idx_val] = align_probabilities(model, X[idx_val])
        models.append(model)

    oof_pred = np.argmax(oof_prob, axis=1)
    print(f"5-fold OOF accuracy (consensus/majority class): {accuracy_score(hard_targets, oof_pred):.4f}")
    print(f"5-fold OOF log loss (hard labels): {log_loss(hard_targets, oof_prob, labels=np.arange(6)):.4f}")
    y_true = np.clip(soft_targets, 1e-12, 1.0)
    kl = np.mean(np.sum(y_true * (np.log(y_true) - np.log(oof_prob)), axis=1))
    print(f"5-fold OOF KL divergence (soft expert votes): {kl:.4f}")

    if args.submission:
        test = pd.read_csv(args.data_dir / "test.csv")
        print("Extracting test EEG features...")
        X_test = build_features(test, args.data_dir / "test_eegs", is_train=False)
        pred = np.mean([align_probabilities(m, X_test) for m in models], axis=0)
        submission = pd.DataFrame(pred, columns=TARGETS)
        submission.insert(0, "eeg_id", test["eeg_id"].to_numpy())
        submission.to_csv("submission.csv", index=False)
        print("Saved submission.csv")


if __name__ == "__main__":
    main()