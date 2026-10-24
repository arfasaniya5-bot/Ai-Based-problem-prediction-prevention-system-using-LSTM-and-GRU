# Backblaze Dataset Inspection Report

**Project**: DeviceGuard AI — Problem Prediction & Prevention System Using LSTM and GRU  
**Phase**: Backend Part 2 — Dataset Inspection & Validation  
**Date**: October 2026  
**Status**: Completed and Verified  

---

## 1. Dataset File and Size

| Metric | Measured Value |
|---|---|
| **File Location** | `backend/data/raw/harddrive.csv` |
| **Exact File Size** | `294,074,825 bytes` (~280.45 MB) |
| **Integrity Status** | Valid, uncorrupted, successfully readable |
| **Inspection Methodology** | Streaming chunked scan (`chunksize = 50,000`); file was never fully loaded into RAM |

---

## 2. Actual Dataset Schema

- **Total Detected Columns**: `95`
- **Duplicate Column Names**: `0`
- **Schema Breakdown**:
  - **Core Hardware Identifiers & Labels (5 columns)**:
    - `date`: Daily observation date (formatted as string `DD-MM-YYYY`, e.g., `01-01-2016`)
    - `serial_number`: Unique drive serial identifier (string)
    - `model`: Drive manufacturer model string (e.g., `ST4000DM000`, `Hitachi HDS5C3030ALA630`)
    - `capacity_bytes`: Total drive storage capacity in bytes (float64)
    - `failure`: Binary failure label indicator (`0` = healthy, `1` = failed hardware event)
  - **Raw SMART Telemetry**: `45` columns ending with `_raw` (e.g., `smart_1_raw`, `smart_5_raw`, `smart_9_raw`)
  - **Normalized SMART Telemetry**: `45` columns ending with `_normalized` (e.g., `smart_1_normalized`, `smart_5_normalized`)

---

## 3. Sample Inspection Results (First 1,000 Rows)

A non-buffered sample of the first 1,000 rows was loaded using `pandas.read_csv(..., nrows=1000)`:

- **Dimensions**: 1,000 rows × 95 columns
- **Core Fields**: All 5 expected core fields (`date`, `serial_number`, `model`, `capacity_bytes`, `failure`) were verified present.
- **Date Parsing**: Format verified as `DD-MM-YYYY`. 100% of sample dates parsed successfully with zero unparseable values.
- **Sample Missing Values**:
  - Core fields: `0 / 1,000` missing (0.0%)
  - `smart_1_raw` (Raw Read Error Rate): `0 / 1,000` missing (0.0%)
  - `smart_5_raw` (Reallocated Sectors Count): `0 / 1,000` missing (0.0%)
  - `smart_9_raw` (Power-On Hours): `0 / 1,000` missing (0.0%)
  - `smart_194_raw` (Temperature): `0 / 1,000` missing (0.0%)
  - `smart_197_raw` (Current Pending Sector Count): `0 / 1,000` missing (0.0%)
  - `smart_187_raw` (Reported Uncorrectable Errors): `464 / 1,000` missing (46.4%), attributed to non-Seagate drives in the sample that do not support attribute 187.

### Preview of Sample Records:
```text
      date   serial_number                   model  failure  smart_5_raw  smart_197_raw
01-01-2016  MJ0351YNG9Z0XA Hitachi HDS5C3030ALA630        0            0              0
01-01-2016        Z305B2QN             ST4000DM000        0            0              0
01-01-2016  MJ0351YNG9Z7LA Hitachi HDS5C3030ALA630        0            0              0
01-01-2016  MJ0351YNGABYAA Hitachi HDS5C3030ALA630        0            0              0
01-01-2016 WD-WMC4N2899475            WDC WD30EFRX        0            0              0
```

---

## 4. Full-Scan Dataset Statistics

The full dataset was processed across 16 streaming chunks of 50,000 rows in **2.26 seconds**:

| Metric | Full Dataset Count |
|---|---|
| **Total Rows** | `757,765` |
| **Unique Hard Drives** | `57,659` |
| **Distinct Drive Models** | `60` |
| **Temporal Coverage** | `01-01-2016` to `13-01-2016` (13 consecutive days) |
| **Healthy Observations (`failure = 0`)** | `757,713` (99.99314%) |
| **Failure Events (`failure = 1`)** | `52` (0.00686%) |
| **Unique Failed Storage Drives** | `48` |
| **Drives with Repeated Observations** | `57,655` / `57,659` (**99.99%**) |
| **Observations per Drive (Min / Max / Mean / Median)** | `1 / 14 / 13.14 / 13.0` |

### Top Drive Models by Fleet Volume:

| Model | Observations | Percentage | Failures Recorded | Failure Rate |
|---|---|---|---|---|
| **ST4000DM000** (Seagate 4TB) | `382,882` | 50.53% | `36` | 0.009% |
| **HGST HMS5C4040ALE640** (HGST 4TB) | `93,284` | 12.31% | `2` | 0.002% |
| **Hitachi HDS5C3030ALA630** (Hitachi 3TB) | `60,046` | 7.92% | `2` | 0.003% |
| **Hitachi HDS722020ALA330** (Hitachi 2TB) | `59,081` | 7.80% | `6` | 0.010% |
| **HGST HMS5C4040BLE640** (HGST 4TB) | `40,738` | 5.38% | `0` | 0.000% |
| **Hitachi HDS5C4040ALE630** (Hitachi 4TB) | `34,725` | 4.58% | `1` | 0.003% |
| **ST6000DX000** (Seagate 6TB) | `24,775` | 3.27% | `0` | 0.000% |
| **WDC WD30EFRX** (Western Digital 3TB) | `13,776` | 1.82% | `0` | 0.000% |

---

## 5. Missing-Value Summary (Full Dataset)

Missing values were computed across all 757,765 records:

### Core Telemetry Columns:
- `date`: **0 missing (0.00%)**
- `serial_number`: **0 missing (0.00%)**
- `model`: **0 missing (0.00%)**
- `capacity_bytes`: **0 missing (0.00%)**
- `failure`: **0 missing (0.00%)**

### Key SMART Failure Predictor Attributes:
- `smart_1_raw` (Raw Read Error Rate): **0 missing (0.00%)**
- `smart_5_raw` (Reallocated Sectors Count): **0 missing (0.00%)**
- `smart_9_raw` (Power-On Hours): **0 missing (0.00%)**
- `smart_12_raw` (Power Cycle Count): **0 missing (0.00%)**
- `smart_188_raw` (Command Timeout): **0 missing (0.00%)**
- `smart_194_raw` (Temperature): **13 missing (0.00%)**
- `smart_197_raw` (Current Pending Sector Count): **0 missing (0.00%)**
- `smart_198_raw` (Offline Uncorrectable Sector Count): **0 missing (0.00%)**
- `smart_187_raw` (Reported Uncorrectable Errors): **341,723 missing (45.10%)**  
  *Finding: Attribute 187 is proprietary to specific vendors (e.g., Seagate `ST4000DM000` has 100% reporting for SMART 187, whereas HGST/Hitachi report null).*

---

## 6. Failure-Label Distribution & Class Imbalance

- **Class 0 (Healthy Daily Reading)**: `757,713` (99.99314%)
- **Class 1 (Drive Failure Event)**: `52` (0.00686%)
- **Class Imbalance Ratio**: Approximately **1 : 14,571**

### Implication for Deep Learning:
Predicting failure purely as a single point-in-time classification would suffer from extreme class imbalance. In Part 3/4 preprocessing, the objective must be formulated as a **lead-time prediction window** (e.g., labeling drives within $K \in [3, 7]$ days prior to failure as an at-risk degrading state), combined with negative downsampling and class-weighted cross-entropy loss.

---

## 7. Temporal Coverage & Time-Series Suitability

### Repeated Observation Findings:
- **99.99%** of storage drives (`57,655` out of `57,659`) have repeated daily readings.
- The average drive has **13.14 daily readings** across the 13-day period (median: **13.0**).
- Failed drives possess on average **6.92 consecutive daily observations** preceding the failure timestamp.

### Pre-Failure Telemetry Progression (Case Study):
Inspection of failed drive `WD-WCC4M3TTXF1Y`:
- `01-01-2016`: Failure = 0, `smart_5_raw` = 0, `smart_197_raw` = 0
- `02-01-2016`: Failure = 0, `smart_5_raw` = 0, `smart_197_raw` = 0
- `03-01-2016`: **Failure = 1**, `smart_5_raw` = 0, **`smart_197_raw` = 1** (Pending sector spike detected!)

### Suitability Verdict:
**Confirmed Suitable for Sequential Modeling (LSTM & GRU)**:
The existence of consistent daily time-steps per serial number validates that temporal sequence tensors of shape `(batch_size, sequence_length, feature_count)` can be constructed using sliding windows of length $T \in [3, 7]$ days.

---

## 8. Limitations & Recommended Next Steps

### Limitations:
1. **Short Observation Duration**: The dataset spans 13 consecutive days. While adequate for short sequence lengths ($T \le 7$), longer 30-day single-drive windows cannot be formed from this slice without lead-time assumption framing.
2. **Extreme Class Imbalance**: Only 52 failure records exist across 757,765 rows.
3. **Vendor Telemetry Variations**: Different manufacturers populate different SMART metrics.

### Recommended Next Steps for Preprocessing (Part 3):
1. **Model Stratification**: Focus initial sequence modeling on the dominant model: Seagate `ST4000DM000` (382,882 rows, 36 failures, consistent SMART sensor definitions).
2. **Universal Feature Selection**: Select the universally populated core SMART attributes (`smart_1_raw`, `smart_5_raw`, `smart_9_raw`, `smart_12_raw`, `smart_187_raw` [for Seagate], `smart_188_raw`, `smart_194_raw`, `smart_197_raw`, `smart_198_raw`).
3. **Sliding-Window Sequence Construction**: Construct temporal sliding windows with a degradation warning label applied to pre-failure days.
4. **Data Normalization & Scaling**: Standardize continuous raw counts (Z-score or MinMax per feature) and perform stratified train/validation/test splitting by `serial_number` to prevent temporal data leakage.
