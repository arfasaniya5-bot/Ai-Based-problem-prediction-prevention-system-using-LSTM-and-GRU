# Model Experiments and Comparative Evaluation Report

**Project**: DeviceGuard AI — AI-Based Problem Prediction & Prevention System Using LSTM and GRU  
**Phase**: Backend Part 4 — Training Data & Model Experiments Review  
**Date**: 2026-10-24  
**Status**: Experimental Comparison Completed & Audited  

---

## 1. Executive Summary

This report documents the experimental comparison of four failure-forecasting approaches for predicting hard-drive failure within a two-day horizon ($H = 2$ calendar days) using three consecutive days of historical SMART telemetry ($T = 3$ calendar days):

1. **Heuristic Threshold Baseline**: Rule-based alert if any critical SMART counter exceeds zero (`smart_5_raw > 0`, `smart_187_raw > 0`, or `smart_197_raw > 0` on Day 3).
2. **Logistic Regression Baseline**: Standard linear classifier trained on 33 summary features (mean, max, trend delta across 3 days) with balanced class weights.
3. **Long Short-Term Memory (LSTM)**: Recurrent neural network capturing temporal feature transitions across the 3-day sequence.
4. **Gated Recurrent Unit (GRU)**: Gated recurrent neural network with a simplified hidden gating mechanism.

### Key Experimental Observations:
- **Severe Class Imbalance**: In the un-downsampled test partition, positive failure windows account for only **0.0126%** of all samples (1 positive window per 7,963 negative windows).
- **False-Alarm Reduction**: The sequential LSTM and GRU architectures showed a substantial reduction in false alarms compared to the linear Logistic Regression baseline:
  - **Logistic Regression**: Falsely flagged **3,629 healthy drives** (a 31.49% drive false alert rate; 314.9 false alarms per 1,000 healthy drives).
  - **LSTM Neural Network**: Falsely flagged only **27 healthy drives** (a 0.23% drive false alert rate; 2.3 false alarms per 1,000 healthy drives).
- **Academic Candor & Characterization**: Based on these initial experiments, the LSTM architecture is **promising** for reducing operational alert noise in imbalanced telemetry. However, with only 7 unique failed drives in the test set, these results **do not constitute definitive proof** that recurrent networks capture general wear dynamics better than all other approaches. Point estimates carry wide statistical uncertainty, and these results reflect an exploratory academic benchmark on a 13-day dataset slice.

---

## 2. Experimental Setup & Leakage Safeguards

To prevent data contamination and overly optimistic evaluation, five safeguards were enforced:

| Parameter | Configuration Value | Methodological Rationale |
|---|---|---|
| **Sequence Input Length ($T$)** | `3` Consecutive Calendar Days | Preserves 70.8% of available failures without requiring multi-week history |
| **Prediction Horizon ($H$)** | `2` Days ($24\text{h} - 48\text{h}$) | Actionable operational window for backup migration |
| **Partitioning Strategy** | Stratified Group Split by `serial_number` | Prevents drive observation leakage between train, val, and test |
| **Partition Allocation** | 60% Train / 20% Val / 20% Test | Allocates 20 failed drives to Train, 7 to Val, 7 to Test |
| **Input Features (11 attributes)** | Tier 1 & 2 SMART Physical Raw Counters | Excludes serial numbers, model strings, and future failure flags |
| **Imputation & Scaling** | Median Imputer + RobustScaler | Fitted exclusively on training drives; applied to Val & Test |
| **Class Balancing** | Negative Downsampling (25:1) | **Applied exclusively to Training set**. Val and Test are untouched |

---

## 3. Window-Level Evaluation Results (Untouched Test Partition)

Evaluated across all **103,526 test sequences** (103,513 negative windows, 13 positive windows from 7 unique failed drives):

| Model Architecture | Precision | Recall | PR-AUC | ROC-AUC | Decision Threshold | Test Failed Drives |
|---|---|---|---|---|---|---|
| **Heuristic Threshold** | `0.0054` | **`0.8462`** | `0.0046` | `0.9134` | SMART 5/187/197 > 0 | `7` |
| **Logistic Regression (Summary)** | `0.0006` | `0.7692` | `0.0607` | `0.8253` | `p >= 0.50` | `7` |
| **LSTM Neural Network** | **`0.0329`** | `0.6154` | **`0.3369`** | **`0.9261`** | `p >= 0.50` | `7` |
| **GRU Neural Network** | `0.0272` | `0.6154` | `0.1632` | `0.9209` | `p >= 0.50` | `7` |

### Confusion Matrices (103,526 Test Windows):

#### 1. Heuristic Rule Baseline:
```
True Negative (TN) : 101,504  |  False Positive (FP) :  2,009
False Negative (FN):       2  |  True Positive (TP)  :     11
Sum: 103,526 (Positives: 13, Negatives: 101,504)
```

#### 2. Logistic Regression (33 Summary Features):
```
True Negative (TN) :  87,681  |  False Positive (FP) : 15,832
False Negative (FN):       3  |  True Positive (TP)  :     10
Sum: 103,526 (Positives: 13, Negatives: 103,513)
```

#### 3. LSTM Sequential Model:
```
True Negative (TN) : 103,278  |  False Positive (FP) :    235
False Negative (FN):       5  |  True Positive (TP)  :      8
Sum: 103,526 (Positives: 13, Negatives: 103,513)
```

#### 4. GRU Sequential Model:
```
True Negative (TN) : 103,227  |  False Positive (FP) :    286
False Negative (FN):       5  |  True Positive (TP)  :      8
Sum: 103,526 (Positives: 13, Negatives: 103,513)
```

---

## 4. Drive-Level Evaluation & False-Alert Burden

Because operational failure alerts are acted upon per physical storage unit rather than per sliding window, drive-level metrics were calculated using the following rule:

> **Alert Aggregation Rule**: A physical drive triggers an alert if **at least one** of its test sliding windows produces a prediction $\ge 0.50$ (or meets the non-zero condition for the heuristic baseline). Overlapping windows from the same drive are not treated as independent failure events.

### Drive-Level Performance Summary:
- **Eligible Failed Drives in Test Partition**: `7` drives
- **Healthy Drives Evaluated in Test Partition**: `11,523` drives (denominator for false alert rates)

| Model Architecture | Failed Drives Detected | Drive Recall | Healthy Drives Flagged (FP) | False Alert Rate (%) | False Alert Burden (per 1,000 Healthy Drives) |
|---|---|---|---|---|---|
| **Heuristic Threshold** | **`6 / 7`** | **`85.71%`** | `226` | `1.96%` | **`19.6`** per 1,000 drives |
| **Logistic Regression** | `5 / 7` | `71.43%` | `3,629` | `31.49%` | **`314.9`** per 1,000 drives |
| **LSTM Neural Network** | `4 / 7` | `57.14%` | **`27`** | **`0.23%`** | **`2.3`** per 1,000 drives |
| **GRU Neural Network** | `4 / 7` | `57.14%` | `34` | `0.30%` | `3.0` per 1,000 drives |

### Drive Detection Breakdown & Advance Lead Times:

1. **Detected Failed Drives**:
   - **LSTM & GRU (4 / 7 drives)**:
     - `W30068RN` (ST4000DM000): First alert at **3 days** before failure date.
     - `W300H36V` (ST4000DM000): First alert at **5 days** before failure date.
     - `Z300WGAW` (ST4000DM000): First alert at **3 days** before failure date.
     - `Z302FZTY` (ST4000DM000): First alert at **9 days** before failure date.
   - **Logistic Regression (5 / 7 drives)**: Detected the above 4 drives plus `MJ1311YNG3AUYA` (lead time: 4 days).
   - **Heuristic Baseline (6 / 7 drives)**: Detected the above 5 drives plus `Z304JD7V` (lead time: 1 day).

2. **Missed Failed Drives**:
   - `PL2331LAGUJLMJ` (Hitachi HDS5C4040ALE630): Missed by all 4 models. This drive experienced sudden failure without prior elevation in tracked SMART raw counters (a sudden mechanical or electronic failure rather than progressive wear).
   - `MJ1311YNG3AUYA` & `Z304JD7V`: Detected by simpler baselines but fell below the 0.50 probability threshold for LSTM/GRU.

3. **False-Alert Burden Analysis**:
   - While the Heuristic and Logistic Regression models achieved higher raw drive recall (6/7 and 5/7 vs 4/7), Logistic Regression falsely flagged **3,629 healthy drives** out of 11,523. In a medium data center fleet of 10,000 drives, this translates to over 3,100 false alarms.
   - In contrast, the LSTM model generated only **27 false drive alerts** across 11,523 healthy drives (2.3 per 1,000 drives), representing a **99.3% reduction in false alert burden** compared to Logistic Regression.

---

## 5. Statistical Uncertainty & Dataset Limitations

1. **Small Failure Sample Size**:
   - With only 7 failed drives in the test set, each drive corresponds to **14.29 percentage points** of drive recall.
   - 95% Wilson score confidence intervals for drive recall:
     - LSTM / GRU (4 / 7): $57.14\% \quad [25.0\%, 84.2\%]$
     - Logistic Regression (5 / 7): $71.43\% \quad [35.9\%, 91.8\%]$
     - Heuristic Threshold (6 / 7): $85.71\% \quad [48.7\%, 97.4\%]$
   - Because these confidence intervals overlap substantially, the observed difference in recall is within expected small-sample binomial variance.
2. **Observation Window Constraint (13 Days)**:
   - The dataset snapshot spans 13 calendar days (`01-01-2016` to `13-01-2016`). While sufficient to validate sequence construction ($T = 3$) and lead-time forecasting ($H = 2$), long-term wear trends across months or years cannot be evaluated on this slice.
3. **Vendor Specificity**:
   - Attribute `smart_187_raw` is reported primarily by Seagate drives (0% missing on Seagate ST4000DM000, 45% missing fleet-wide). For non-reporting vendors, median imputation assigns 0.0, which accurately reflects zero reported uncorrectable errors but limits cross-vendor feature parity.

---

## 6. Artifacts & Reproduction Commands

All experiment artifacts are saved under `backend/models/`:
- `preprocessor_config.json`: Median imputer statistics and RobustScaler centering/scaling parameters.
- `baseline_model.joblib`: Trained Logistic Regression baseline.
- `best_lstm.pt`: Best-performing LSTM state dictionary checkpoint.
- `best_gru.pt`: Best-performing GRU state dictionary checkpoint.
- `evaluation_summary.json`: Serialized window-level and drive-level evaluation metrics.

### Reproduction Command:
```powershell
.\backend\.venv\Scripts\python.exe backend/src/models/train.py
```
