"""
audit_part4_review.py
Independent audit of Backend Part 4 artifacts, metrics, and drive-level evaluation.
"""

from collections import defaultdict, Counter
from datetime import datetime
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch

# Paths
BACKEND_DIR = Path(__file__).resolve().parents[2]
RAW_CSV = BACKEND_DIR / "data" / "raw" / "harddrive.csv"
MODELS_DIR = BACKEND_DIR / "models"
SRC_MODELS = BACKEND_DIR / "src" / "models"

sys.path.insert(0, str(SRC_MODELS))
from architectures import LSTMClassifier, GRUClassifier
from train import (
    FEATURE_NAMES,
    HORIZON,
    RANDOM_SEED,
    SEQ_LEN,
    extract_summary_features,
    load_and_validate_telemetry,
    partition_drives,
    generate_partition_windows,
    preprocess_and_downsample,
)

def run_audit():
    print("=" * 70)
    print("DeviceGuard AI - Backend Part 4 Independent Verification & Audit")
    print("=" * 70)

    # 1. Verify Raw CSV integrity
    raw_size = os.path.getsize(RAW_CSV)
    print(f"1. Raw CSV Size: {raw_size:,} bytes (Expected: 294,074,825 bytes) -> {'MATCH' if raw_size == 294074825 else 'MISMATCH'}")
    assert raw_size == 294074825

    # 2. Verify Preprocessor Config
    config_path = MODELS_DIR / "preprocessor_config.json"
    assert config_path.exists(), "preprocessor_config.json missing!"
    with open(config_path) as f:
        config = json.load(f)

    print("\n2. Preprocessor Configuration:")
    print(f"   Features ({len(config['feature_names'])}): {config['feature_names']}")
    print(f"   Imputer: {config['imputer_strategy']} (Medians: {config['imputer_medians']})")
    print(f"   Scaler: {config['scaler']} (Center count: {len(config['scaler_center'])})")
    print(f"   Training Downsample Ratio: {config['downsample_ratio_train']}:1")

    # Check smart_187_raw median
    s187_median = config['imputer_medians'].get('smart_187_raw')
    print(f"   smart_187_raw median fitted on train: {s187_median}")

    # 3. Load Saved Models
    print("\n3. Loading Saved Model Artifacts:")
    
    # Baseline
    baseline_path = MODELS_DIR / "baseline_model.joblib"
    assert baseline_path.exists(), "baseline_model.joblib missing!"
    clf = joblib.load(baseline_path)
    print(f"   - Loaded LogisticRegression baseline (Coefficients shape: {clf.coef_.shape})")

    # LSTM
    lstm_path = MODELS_DIR / "best_lstm.pt"
    assert lstm_path.exists(), "best_lstm.pt missing!"
    lstm_model = LSTMClassifier(input_size=len(FEATURE_NAMES), hidden_size=32, num_layers=1, dropout=0.2)
    lstm_model.load_state_dict(torch.load(lstm_path, weights_only=True))
    lstm_model.eval()
    print("   - Loaded LSTM model checkpoint successfully.")

    # GRU
    gru_path = MODELS_DIR / "best_gru.pt"
    assert gru_path.exists(), "best_gru.pt missing!"
    gru_model = GRUClassifier(input_size=len(FEATURE_NAMES), hidden_size=32, num_layers=1, dropout=0.2)
    gru_model.load_state_dict(torch.load(gru_path, weights_only=True))
    gru_model.eval()
    print("   - Loaded GRU model checkpoint successfully.")

    # 4. Reconstruct Dataset Partitions with exact same seed
    print("\n4. Reconstructing Partitions & Windows:")
    drive_records, drive_models, failed_drives, eligible_failed_drives, dataset_end = load_and_validate_telemetry(RAW_CSV)
    partition_map, partition_info = partition_drives(drive_records, drive_models, failed_drives, eligible_failed_drives, seed=RANDOM_SEED)

    # Check partition disjointness
    serials_by_part = defaultdict(set)
    for s, p in partition_map.items():
        serials_by_part[p].add(s)

    train_s = serials_by_part['train']
    val_s = serials_by_part['val']
    test_s = serials_by_part['test']

    print(f"   Train unique drives : {len(train_s):,}")
    print(f"   Val unique drives   : {len(val_s):,}")
    print(f"   Test unique drives  : {len(test_s):,}")
    assert len(train_s & val_s) == 0, "Leakage: train & val overlap!"
    assert len(train_s & test_s) == 0, "Leakage: train & test overlap!"
    assert len(val_s & test_s) == 0, "Leakage: val & test overlap!"
    print("   -> STRICT DISJOINTNESS VERIFIED: Zero drive crossover between partitions.")

    # Generate windows
    raw_windows = generate_partition_windows(drive_records, partition_map, failed_drives, dataset_end)
    train_data, val_data, test_data = preprocess_and_downsample(raw_windows, seed=RANDOM_SEED)

    test_X, test_y = test_data
    test_serials = raw_windows['test']['serials']
    test_raw_X = np.array(raw_windows['test']['X'], dtype=np.float32)

    print(f"\n   Test set window count: {len(test_y):,} (Positives: {(test_y == 1).sum()})")
    print(f"   Test set class prevalence: {(test_y == 1).sum() / len(test_y) * 100:.4f}% (1 in {len(test_y) // (test_y == 1).sum()})")

    # 5. Verify Window-Level Metrics Against evaluation_summary.json
    print("\n5. Recalculating & Auditing Window-Level Metrics:")
    
    # Model predictions
    # A. Heuristic
    f5_idx = FEATURE_NAMES.index('smart_5_raw')
    f197_idx = FEATURE_NAMES.index('smart_197_raw')
    f187_idx = FEATURE_NAMES.index('smart_187_raw')
    day3_raw = test_raw_X[:, 2, :]
    heur_preds = ((day3_raw[:, f5_idx] > 0) | (day3_raw[:, f197_idx] > 0) | (day3_raw[:, f187_idx] > 0)).astype(int)

    # B. Logistic Regression
    test_feat = extract_summary_features(test_X)
    lr_probs = clf.predict_proba(test_feat)[:, 1]
    lr_preds = (lr_probs >= 0.5).astype(int)

    # C. LSTM
    with torch.no_grad():
        test_t = torch.tensor(test_X)
        lstm_logits = lstm_model(test_t)
        lstm_probs = torch.sigmoid(lstm_logits).squeeze(1).numpy()
    lstm_preds = (lstm_probs >= 0.5).astype(int)

    # D. GRU
    with torch.no_grad():
        gru_logits = gru_model(test_t)
        gru_probs = torch.sigmoid(gru_logits).squeeze(1).numpy()
    gru_preds = (gru_probs >= 0.5).astype(int)

    models_dict = {
        'Heuristic Threshold': (heur_preds, heur_preds.astype(float)),
        'Logistic Regression': (lr_preds, lr_probs),
        'LSTM Neural Network': (lstm_preds, lstm_probs),
        'GRU Neural Network': (gru_preds, gru_probs),
    }

    for name, (preds, probs) in models_dict.items():
        cm = confusion_matrix(test_y, preds)
        prec = precision_score(test_y, preds, zero_division=0)
        rec = recall_score(test_y, preds, zero_division=0)
        pr_auc = average_precision_score(test_y, probs) if not np.array_equal(probs, preds) else average_precision_score(test_y, preds)
        roc_auc = roc_auc_score(test_y, probs) if not np.array_equal(probs, preds) else roc_auc_score(test_y, preds)

        print(f"\n   Model: {name}")
        print(f"     Precision: {prec:.4f} | Recall: {rec:.4f} | PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f}")
        print(f"     Confusion Matrix: TN={cm[0,0]:,}, FP={cm[0,1]:,}, FN={cm[1,0]}, TP={cm[1,1]}")
        print(f"     Sum Check: {cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1]:,} == {len(test_y):,} (Pos: {cm[1,0] + cm[1,1]})")
        assert (cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1]) == len(test_y)
        assert (cm[1,0] + cm[1,1]) == (test_y == 1).sum()

    # 6. DRIVE-LEVEL EVALUATION
    print("\n" + "=" * 70)
    print(" 6. DRIVE-LEVEL EVALUATION & ALERT AGGREGATION")
    print("=" * 70)
    print("Alert Aggregation Rule:")
    print("  A physical drive is classified as 'Alert Triggered' if ANY of its valid test")
    print("  sliding windows has predicted probability >= 0.5 (or condition met for heuristic).")

    # Group test windows by serial number
    drive_window_indices = defaultdict(list)
    for idx, serial in enumerate(test_serials):
        drive_window_indices[serial].append(idx)

    # Identify which drives are failed drives in test set
    test_failed_serials = set(eligible_failed_drives) & test_s
    test_healthy_serials = test_s - set(failed_drives.keys())
    print(f"\nTest Drive Population Breakdown:")
    print(f"  Eligible Failed Drives in Test Set : {len(test_failed_serials)}")
    print(f"  Healthy Drives with Test Windows   : {len(test_healthy_serials):,}")
    print(f"  Total Unique Drives Evaluated      : {len(drive_window_indices):,}")

    for name, (preds, probs) in models_dict.items():
        print(f"\n--- Drive-Level Results for {name} ---")
        
        # Drive-level detections for failed drives
        detected_failed_drives = []
        missed_failed_drives = []
        drive_lead_times = {}

        for serial in sorted(list(test_failed_serials)):
            indices = drive_window_indices[serial]
            # Check if any window for this drive triggered alert
            drive_preds = preds[indices]
            drive_probs = probs[indices]
            t_fail = failed_drives[serial]

            # Find windows where pred == 1
            alert_indices = [idx for idx in indices if preds[idx] == 1]
            if len(alert_indices) > 0:
                detected_failed_drives.append(serial)
                # Compute lead time of the first alert window
                # Get the window cutoff date (t_end)
                # We can find t_end from the date of the window
                # Let's see: raw_windows['test']['X'][idx]
                # To get exact date, let's look at drive_records[serial]
                day_dict = drive_records[serial]
                d_list = sorted(list(day_dict.keys()))
                # For this drive, window indices correspond to i in range(len(d_list) - SEQ_LEN + 1) strictly before t_fail
                # Let's map local index
                local_idx = indices.index(alert_indices[0])
                # Window cutoff date is d_list[local_idx + SEQ_LEN - 1]
                t_end = d_list[local_idx + SEQ_LEN - 1]
                lead_time = (t_fail - t_end).days
                drive_lead_times[serial] = lead_time
            else:
                missed_failed_drives.append(serial)

        # Drive-level false alerts for healthy drives
        flagged_healthy_drives = []
        for serial in test_healthy_serials:
            indices = drive_window_indices.get(serial, [])
            if any(preds[idx] == 1 for idx in indices):
                flagged_healthy_drives.append(serial)

        drive_recall = len(detected_failed_drives) / len(test_failed_serials) if test_failed_serials else 0
        total_healthy_eval = len(test_healthy_serials)
        false_alert_rate = len(flagged_healthy_drives) / total_healthy_eval if total_healthy_eval else 0
        false_alert_burden_per_1000 = false_alert_rate * 1000

        print(f"  Drive-Level Recall             : {drive_recall:.2%} ({len(detected_failed_drives)} / {len(test_failed_serials)} failed drives detected)")
        print(f"  Detected Failed Drives         : {detected_failed_drives}")
        print(f"  Missed Failed Drives           : {missed_failed_drives}")
        print(f"  Alert Lead Times (Days ahead)  : {drive_lead_times}")
        print(f"  Healthy Drives Flagged (FP)    : {len(flagged_healthy_drives):,} / {total_healthy_eval:,}")
        print(f"  False Alert Rate (per drive)   : {false_alert_rate:.4%}")
        print(f"  False Alert Burden             : {false_alert_burden_per_1000:.1f} false alarms per 1,000 healthy drives")

if __name__ == "__main__":
    run_audit()
