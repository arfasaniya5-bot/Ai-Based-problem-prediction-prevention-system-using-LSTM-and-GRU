# DeviceGuard AI — Part 10: Final Integration, End-to-End Verification & Demonstration Readiness Report

**Project Title:** AI-Based Problem Prediction & Prevention System Using LSTM and GRU  
**Phase:** Backend Part 10 — Final Integration, End-to-End Verification & Demonstration Readiness  
**Date:** 2026-10-24  
**Status:** Verification Completed — System Demonstration-Ready  

---

## 1. Project Overview

**DeviceGuard AI** is an academic deep learning and full-stack software system designed to forecast storage drive hardware failures within a 24-to-48 hour advance-warning window ($H = 2$ calendar days) using three consecutive days of historical SMART telemetry ($T = 3$ calendar days).

The system addresses the severe operational challenges of storage telemetry:
1. **Extreme Class Imbalance**: In raw enterprise fleet telemetry, hard-drive failures account for less than $0.015\%$ of daily observations.
2. **False-Alarm Mitigation**: Traditional rule-based threshold heuristics produce unmanageable alert volumes (hundreds of false alarms per 1,000 healthy drives). By modeling the temporal sequence of degradation using recurrent neural architectures (LSTM and GRU), false alarms are dramatically curtailed ($0.23\%$ drive false alarm rate on the test partition).
3. **End-to-End Operational Lifecycle**: Telemetry sequences are validated via Pydantic, normalized with training-fitted medians and RobustScaling, evaluated via PyTorch LSTM, persisted atomically to SQLite, exposed via REST APIs, and surfaced through a responsive React + Vite operations dashboard.

---

## 2. Architecture Summary

The complete system architecture comprises four loosely coupled layers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        React 18 + Vite Frontend                        │
│   Dashboard (KPIs)  │  Devices Registry  │  AI Predict  │  Alert Queue │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │ HTTP / JSON (REST)
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                          FastAPI Backend Core                          │
│  Request Schemas │ Route Handlers │ CORS Guard │ Global Error Handler  │
└──────────────────┬─────────────────────────────────┬───────────────────┘
                   │                                 │
                   ▼                                 ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────┐
│       PyTorch Inference Engine       │  │    SQLAlchemy ORM + SQLite   │
│  - Training-fitted RobustScaler      │  │  - Devices                   │
│  - Training median Imputer           │  │  - Prediction Records        │
│  - LSTM Classifier (best_lstm.pt)    │  │  - Alert Records             │
│  - GRU & Baseline Fallbacks          │  │  - Telemetry Readings        │
└──────────────────────────────────────┘  └──────────────────────────────┘
```

- **Frontend**: Single-page application built with React 18, React Router v6, Recharts, and Lucide Icons, served via Vite.
- **Backend**: Asynchronous REST API server built with FastAPI 0.110+ and Uvicorn.
- **Inference Service**: PyTorch 2.2+ recurrent neural network evaluating sequential tensors of shape `(1, 3, 11)`.
- **Persistence**: Relational SQLite database (`backend/data/deviceguard.db`) managed via SQLAlchemy 2.0+ ORM with atomic transactions and cascade relations.

---

## 3. Frontend Routes and Backend API Endpoints

The implemented routing contracts were verified directly from source code:

### A. React Frontend Routes (`src/App.jsx`)
| Route Path | Page Component | Functional Purpose |
|---|---|---|
| `/` | `Navigate to="/dashboard"` | Root redirect to main operational overview. |
| `/dashboard` | `Dashboard.jsx` | Live fleet metrics, health status, model diagnostics, and chronological risk trends. |
| `/devices` | `Devices.jsx` | Hardware drive inventory with search, status filtering, and pagination. |
| `/predictions` | `Predictions.jsx` | Interactive SMART telemetry input form, risk gauge, recommendations, and prediction log. |
| `/alerts` | `Alerts.jsx` | Proactive failure alert queue with severity filtering and one-click operator acknowledgement. |

### B. FastAPI Backend Endpoints (`backend/src/main.py`)
| HTTP Method | Route Path | Handler Module | Response Status | Purpose |
|---|---|---|---|---|
| `GET` | `/` | `main.py` | `200 OK` | Service metadata, environment, OpenAPI docs link. |
| `GET` | `/health` | `routes_health.py` | `200 OK` | Liveness probe and service version check. |
| `GET` | `/health/models` | `routes_health.py` | `200 OK` | In-memory model status, feature count, threshold. |
| `POST` | `/api/predict` | `routes_prediction.py` | `200 OK` | Submit 3-day sequence, compute inference, persist record & alerts. |
| `GET` | `/api/predictions` | `routes_prediction.py` | `200 OK` | Retrieve recent prediction audit log from SQLite. |
| `GET` | `/api/devices` | `routes_devices.py` | `200 OK` | Paginated device registry with substring search and status filter. |
| `GET` | `/api/alerts` | `routes_alerts.py` | `200 OK` | Paginated alerts queue with severity and acknowledgement filters. |
| `PATCH` | `/api/alerts/{id}/acknowledge` | `routes_alerts.py` | `200 OK` | Mark alert as acknowledged by operator (idempotent). |

---

## 4. Prediction Workflow Verification

The end-to-end inference lifecycle was traced through all execution stages:

1. **Input Submission**: Operator or automated telemetry agent provides a device serial number, model string, and an array of exactly 3 consecutive daily SMART readings ($T = 3$).
2. **Schema & Sequence Validation**:
   - Pydantic enforces `len(readings) == 3`. Sequences with length $< 3$ or $> 3$ are rejected with `HTTP 422 Unprocessable Entity`.
   - All 11 physical SMART counters are validated as numeric floats within $[-10^{15}, 10^{15}]$.
   - Strings in `serial_number` and `model` are bounded to $\le 64$ characters.
3. **Imputation & Feature Scaling**:
   - Missing or null counters are imputed using the pre-computed training medians stored in `backend/models/preprocessor_config.json`.
   - Feature ordering is locked to the 11 verified physical counters: `smart_1_raw`, `smart_4_raw`, `smart_5_raw`, `smart_7_raw`, `smart_9_raw`, `smart_12_raw`, `smart_187_raw`, `smart_188_raw`, `smart_194_raw`, `smart_197_raw`, `smart_198_raw`.
   - Values are normalized using the training-fitted RobustScaler: $z = \frac{x - \text{center}}{\text{scale}}$.
4. **PyTorch Neural Forward Pass**:
   - The normalized array is reshaped to tensor shape `(1, 3, 11)` and passed to `LSTMClassifier`.
   - Output logit is mapped to failure probability via sigmoid: $p = \sigma(z) \in [0.0, 1.0]$.
5. **Decision & Risk Classification**:
   - Threshold $p \ge 0.50 \rightarrow \text{Class 1 (Failure Predicted)}$, Risk Level `CRITICAL`.
   - $0.25 \le p < 0.50 \rightarrow \text{Class 0 (Healthy)}$, Risk Level `MODERATE`.
   - $p < 0.25 \rightarrow \text{Class 0 (Healthy)}$, Risk Level `LOW`.
6. **Atomic Persistence & Alert Lifecycle**:
   - `Device` entity is created or updated (setting status to `CRITICAL`, `WARNING`, or `HEALTHY`).
   - `PredictionRecord` is inserted.
   - All 3 daily observations are inserted into `TelemetryReading`.
   - If risk is `CRITICAL` or `MODERATE`, an `AlertRecord` is created or existing unacknowledged alert is updated in place.
   - Transaction commits atomically; rollbacks on any SQLite error with safe `HTTP 500` error propagation.
7. **UI Presentation**:
   - React updates the risk gauge, recommendation text, and refreshes the historical prediction log.

---

## 5. Device Registry Verification

Verified against [`routes_devices.py`](file:///c:/Users/arfas/OneDrive/Desktop%20-%20Copy/DL_PROJECT/backend/src/api/routes_devices.py) and [`Devices.jsx`](file:///c:/Users/arfas/OneDrive/Desktop%20-%20Copy/DL_PROJECT/src/pages/Devices.jsx):
- **Data Source**: Live queries to `GET /api/devices`.
- **Search Capabilities**: Case-insensitive substring search on `serial_number` (e.g. `?search=ZA1` matches `ZA160000`). Bounded to $\le 100$ characters.
- **Status Filters**: Whitelist enforced (`HEALTHY`, `WARNING`, `CRITICAL`, `FAILED`). Unsupported status strings return `HTTP 400 Bad Request`.
- **Pagination**: Zero-indexed or negative pages (`page=0`) and excessive page sizes (`page_size=500`) are rejected with `HTTP 422`.
- **Empty State**: Searches yielding 0 records display an informative empty state without breaking the UI.
- **Read-Only Verification**: The 13 registered drives in `backend/data/deviceguard.db` were verified without modification.

---

## 6. Alert Lifecycle Verification

Verified against [`routes_alerts.py`](file:///c:/Users/arfas/OneDrive/Desktop%20-%20Copy/DL_PROJECT/backend/src/api/routes_alerts.py) and [`Alerts.jsx`](file:///c:/Users/arfas/OneDrive/Desktop%20-%20Copy/DL_PROJECT/src/pages/Alerts.jsx):
- **Listing & Filters**: Filterable by `severity` (`INFO`, `WARNING`, `CRITICAL`) and `acknowledged` boolean flag.
- **Alert Deduplication**: Repeated predictions on degraded drives update the existing unacknowledged alert's timestamp and recommendation rather than generating duplicate unacknowledged rows.
- **Acknowledgement**:
  - Handled via `PATCH /api/alerts/{id}/acknowledge`.
  - Sets `acknowledged = True` and records `acknowledged_at`.
  - Idempotent: repeated calls safely return `HTTP 200` with the persisted timestamp.
  - Non-existent IDs return `HTTP 404 Not Found`.
- **UI State**: The frontend disables the acknowledge button during mutation and updates local state only upon backend confirmation.

---

## 7. Dashboard Verification

Verified against [`Dashboard.jsx`](file:///c:/Users/arfas/OneDrive/Desktop%20-%20Copy/DL_PROJECT/src/pages/Dashboard.jsx):
- **Live KPI Metrics**:
  - Total registered devices (fetched from `GET /api/devices`).
  - Active unacknowledged alerts count (fetched from `GET /api/alerts?acknowledged=false`).
  - Active model status indicator (fetched from `GET /health/models`).
  - Inferences completed count (computed from live `GET /api/predictions` records).
- **No Mock Masquerading**: If the backend server is unreachable, the Dashboard displays an explicit connection failure banner and does not substitute hardcoded mock statistics.
- **Trend Charts**: Visualizes risk score distribution and chronological prediction risk scores dynamically from SQLite prediction records.

---

## 8. Backend Regression Test Results

The backend regression test suite was executed in an isolated SQLite environment (`test_isolated.db` via `conftest.py` dependency override):

- **Execution Command**:
  ```powershell
  .\backend\.venv\Scripts\python.exe -m pytest backend/tests/ -v
  ```
- **Execution Duration**: `1.03 seconds`
- **Results**: **29 passed, 0 failed, 1 warning (Starlette testclient deprecation notice)**
- **Test Pass Rate**: **100%**

```text
backend/tests/test_api.py::test_root_endpoint PASSED                     [  3%]
backend/tests/test_api.py::test_api_health_endpoint PASSED               [  6%]
backend/tests/test_api.py::test_models_health_endpoint PASSED            [ 10%]
backend/tests/test_api.py::test_valid_prediction PASSED                  [ 13%]
backend/tests/test_api.py::test_prediction_with_null_features PASSED     [ 17%]
backend/tests/test_api.py::test_invalid_sequence_length_too_short PASSED [ 20%]
backend/tests/test_api.py::test_invalid_sequence_length_too_long PASSED  [ 24%]
backend/tests/test_api.py::test_invalid_non_numeric_feature PASSED       [ 27%]
backend/tests/test_api.py::test_database_persistence_and_history PASSED  [ 31%]
backend/tests/test_api.py::test_database_error_handling_predict PASSED   [ 34%]
backend/tests/test_api.py::test_predict_oversized_serial_number PASSED   [ 37%]
backend/tests/test_devices_alerts.py::test_list_devices_pagination PASSED [ 41%]
backend/tests/test_devices_alerts.py::test_list_devices_search PASSED    [ 44%]
backend/tests/test_devices_alerts.py::test_list_devices_status_filter PASSED [ 48%]
backend/tests/test_devices_alerts.py::test_list_devices_empty_search PASSED [ 51%]
backend/tests/test_devices_alerts.py::test_list_alerts_structure PASSED  [ 55%]
backend/tests/test_devices_alerts.py::test_list_alerts_severity_filter PASSED [ 58%]
backend/tests/test_devices_alerts.py::test_list_alerts_acknowledged_filter PASSED [ 62%]
backend/tests/test_devices_alerts.py::test_acknowledge_alert_success PASSED [ 65%]
backend/tests/test_devices_alerts.py::test_acknowledge_alert_idempotency PASSED [ 68%]
backend/tests/test_devices_alerts.py::test_acknowledge_alert_not_found PASSED [ 72%]
backend/tests/test_devices_alerts.py::test_unsupported_device_status_filter PASSED [ 75%]
backend/tests/test_devices_alerts.py::test_unsupported_alert_severity_filter PASSED [ 79%]
backend/tests/test_devices_alerts.py::test_invalid_device_pagination_page_zero PASSED [ 82%]
backend/tests/test_devices_alerts.py::test_invalid_device_pagination_page_size_exceeded PASSED [ 86%]
backend/tests/test_devices_alerts.py::test_duplicate_alert_prevention_on_repeated_predict PASSED [ 89%]
backend/tests/test_devices_alerts.py::test_device_status_consistency_warning PASSED [ 93%]
backend/tests/test_devices_alerts.py::test_devices_oversized_search_string PASSED [ 96%]
backend/tests/test_devices_alerts.py::test_acknowledge_alert_database_error PASSED [100%]

======================== 29 passed, 1 warning in 1.03s ========================
```

---

## 9. Frontend Production Build Results

The React frontend was compiled for production deployment using Vite:

- **Execution Command**:
  ```powershell
  npm run build
  ```
- **Build Duration**: `9.32 seconds`
- **Output Artifacts**:
  - `dist/index.html`: `0.69 kB` (gzip: `0.40 kB`)
  - `dist/assets/index-DWGrTvuX.css`: `22.21 kB` (gzip: `3.97 kB`)
  - `dist/assets/index-CcsgX0aF.js`: `595.35 kB` (gzip: `175.32 kB`)
- **Compilation Status**: **0 compilation errors, 0 lint warnings**.

---

## 10. Browser-Level Test Results or Limitations

> **Explicit Disclosure:**  
> **Browser-level verification was not performed. Verification was limited to the tests and build checks listed in this report.**  
> Interactive browser automation tooling (such as Playwright, Selenium, or Puppeteer) is not equipped in this environment. Verification of the user-facing application was conducted through end-to-end API client contract tests, JSX component inspection, and production bundle compilation.

---

## 11. Final Model, Dataset, and Database Integrity

All core machine learning assets, datasets, and development records were verified against cryptographic baselines:

| Artifact | Location | Size (Bytes) | SHA-256 Hash | Integrity Status |
|---|---|---|---|---|
| **Raw Telemetry Dataset** | `backend/data/raw/harddrive.csv` | `294,074,825` | `940D3C27E444393472EEA35B1F5A54A0A06F62E8B9D10F1D0DAAD23B37F7D7B6` | **Verified Unchanged** |
| **LSTM Checkpoint** | `backend/models/best_lstm.pt` | `28,841` | `494CB52C54A90C7E81AF94E14903A4E21126E5A27F293B7477372E4E723E242E` | **Verified Unchanged** |
| **GRU Checkpoint** | `backend/models/best_gru.pt` | `23,067` | `952CDCD595C18E81B0088DDB2196CE32A1E80C0966339F14BB1C369B468E2797` | **Verified Unchanged** |
| **Baseline Model** | `backend/models/baseline_model.joblib` | `987` | `5CE6800CD212370E72B19EC3DCC4992CF399FAD89EB09DE68EDCE9A2FA401095` | **Verified Unchanged** |
| **Preprocessor Config** | `backend/models/preprocessor_config.json` | `1,395` | `2E79BF5656236E583B7081D10B9D14D3974F7C5B19262EFD597BE81316CF17B8` | **Verified Unchanged** |

---

## 12. Database Preservation

The development database [`backend/data/deviceguard.db`](file:///c:/Users/arfas/OneDrive/Desktop%20-%20Copy/DL_PROJECT/backend/data/deviceguard.db) was inspected and confirmed completely preserved:
- **Device Count**: `13` registered storage drives.
- **Historical Prediction Records**: `23` records.
- **Alert Records**: `9` proactive alerts.
- **Test Database Pollution**: `0` mock or test records created in the user database. Automated tests executed exclusively against isolated fixtures.

---

## 13. Files Changed and Reasons

Across the stabilization and audit phases (Parts 8–10), the following files were modified or created to harden reliability:

| File Path | Nature of Change | Engineering Rationale |
|---|---|---|
| `backend/tests/conftest.py` | **Created** | Created isolated SQLite fixture (`test_isolated.db`) and `get_db` dependency override, eliminating test pollution of development data. |
| `backend/src/schemas/prediction.py` | **Modified** | Added `max_length=64` to `serial_number` and `model` in `PredictionRequest`. |
| `backend/src/api/routes_prediction.py` | **Modified** | Implemented alert deduplication on repeated predict, updated device status to `WARNING` on moderate risk, added telemetry history persistence, and sanitized HTTP 500 error messages with internal `logger.error()`. |
| `backend/src/api/routes_devices.py` | **Modified** | Added query bounds (`max_length=100` on search, `max_length=50` on status) and fixed `status` parameter shadowing `fastapi.status`. |
| `backend/src/api/routes_alerts.py` | **Modified** | Added query bounds (`max_length=50` on severity) and wrapped acknowledgement `db.commit()` in a try-except rollback block. |
| `.gitignore` | **Modified** | Added ignore patterns for `*.db`, `*.sqlite`, `*.sqlite3`, `.pytest_cache/`, and `.env`. |
| `backend/tests/test_api.py` | **Modified** | Added regression tests for database error handling (HTTP 500), telemetry history persistence, and oversized serial number validation (HTTP 422). |
| `backend/tests/test_devices_alerts.py` | **Modified** | Added regression tests for alert deduplication, status consistency (`WARNING`), oversized search strings, and acknowledgement database error handling. |
| `backend/reports/part8_integration_report.md` | **Created/Reconciled** | Integration report with corrected `PATCH` route, January 2016 date period, and `/predictions` React route. |
| `backend/reports/part9_security_reliability_report.md` | **Created** | Comprehensive Part 9 security and reliability audit report. |
| `backend/reports/part10_final_integration_report.md` | **Created** | Comprehensive Part 10 final integration, verification, and demonstration report. |

---

## 14. Remaining Bugs and Limitations

1. **Academic Demonstration Prototype**:
   - DeviceGuard AI is an academic research prototype. It is not certified for production critical storage systems.
2. **Short Evaluation Time Span & Sample Size**:
   - The dataset represents a 13-day slice of Backblaze January 2016 telemetry (`2016-01-01` to `2016-01-13`) with only 7 failed drives in the held-out test partition. Point estimates carry wide statistical confidence intervals.
3. **Threshold Calibration**:
   - The decision threshold ($p \ge 0.50$) is an operational heuristic configured to balance precision and recall. Risk scores should be interpreted as model-relative scores rather than calibrated Bayesian probabilities.
4. **Authentication & Authorization**:
   - The service does not implement user authentication or role-based access control. In a production environment, an API gateway or reverse proxy must enforce authentication.
5. **Database Concurrency**:
   - SQLite provides excellent file-based persistence for single-node prototypes; multi-node concurrent enterprise workloads would require migrating to PostgreSQL.

---

## 15. Instructions for Running the Project

### Prerequisites:
- Python 3.10+ (Python 3.13 tested) with virtual environment support.
- Node.js 18+ and npm.

### Step 1: Start the FastAPI Backend
Open a terminal in the project root:
```powershell
# Navigate to project directory
cd "c:\Users\arfas\OneDrive\Desktop - Copy\DL_PROJECT"

# Run Uvicorn with the virtual environment Python
.\backend\.venv\Scripts\python.exe -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Base URL: `http://127.0.0.1:8000`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`

### Step 2: Start the React Frontend
Open a second terminal in the project root:
```powershell
# Start Vite development server
npm run dev
```
- Access the web application at: `http://localhost:3000/` (or `http://localhost:5173/`).

### Step 3: Run Automated Verification Tests
```powershell
# Execute the complete backend test suite
.\backend\.venv\Scripts\python.exe -m pytest backend/tests/ -v

# Execute frontend production build
npm run build
```

---

## 16. Academic Demonstration Sequence

A recommended 10-step sequence for an academic presentation, college project review, or viva:

1. **Introduction & Motivation (Slide / Verbal)**:
   - Introduce hard-drive failure prediction in cloud datacenters. Explain why single-day threshold heuristics fail (excessive false alarms) and why modeling temporal trends over consecutive days ($T = 3$) provides advance warning ($H = 2$ days).
2. **Architecture & Pipeline Walkthrough**:
   - Present the end-to-end pipeline: SMART Telemetry $\rightarrow$ Median Imputation $\rightarrow$ RobustScaler $\rightarrow$ PyTorch LSTM $\rightarrow$ SQLite $\rightarrow$ FastAPI $\rightarrow$ React UI.
3. **Dashboard Overview (`/dashboard`)**:
   - Demonstrate the live dashboard. Point out the KPI metric cards (Total Devices, Monitored Devices, Active Alerts), the model health badge, and recent prediction records.
4. **Device Registry Inspection (`/devices`)**:
   - Navigate to Devices. Demonstrate substring search (e.g. search for a Seagate serial number) and operational status filtering (`HEALTHY`, `WARNING`, `CRITICAL`).
5. **Healthy Drive Inference Demonstration (`/predictions`)**:
   - Navigate to AI Predictions. Select the "Healthy Profile" preset or enter normal SMART readings (`smart_5_raw = 0`, `smart_197_raw = 0`).
   - Click "Submit Telemetry for Inference". Show that the LSTM model predicts Class 0 (`LOW` risk), displaying a green nominal badge and no alert generated.
6. **Degraded Drive Warning Demonstration (`/predictions`)**:
   - Enter degraded telemetry (e.g., reallocated sectors `smart_5_raw = 180`, pending sectors `smart_197_raw = 60`, seek errors increasing).
   - Click "Submit Telemetry for Inference". Show that the risk score rises ($\ge 0.50$, `CRITICAL`), triggering an immediate data backup recommendation.
7. **Prediction History Audit**:
   - Scroll down to the Prediction History table on the Predictions page to demonstrate that the new prediction was persisted to SQLite with model provenance.
8. **Proactive Alert Queue & Operator Acknowledgement (`/alerts`)**:
   - Navigate to the Prevention & Alerts page. Show the newly generated `CRITICAL` alert linked to the drive.
   - Click the "Acknowledge" button. Demonstrate that the alert transitions to acknowledged, records the timestamp, and decrements the unacknowledged counter.
9. **Model Architecture & Evaluation Honest Review**:
   - Highlight the experimental comparison: LSTM vs GRU vs Logistic Regression vs Heuristic Baseline. Show how LSTM reduced the drive false alarm rate from $31.49\%$ (Logistic Regression) to $0.23\%$ (LSTM).
10. **Limitations & Candor**:
    - Conclude with academic honesty: discuss the 13-day dataset window, 7 failed test drives, and emphasize that this is a research prototype rather than a production-certified system.

---

## 17. Final Project Status

Backend Part 10 is **complete, hardened, and verified**.
- Complete workflow (telemetry $\rightarrow$ model $\rightarrow$ database $\rightarrow$ API $\rightarrow$ UI) functions deterministically.
- All 29 automated regression tests pass in 1.03s.
- React frontend builds cleanly in 9.32s.
- Model checkpoints and raw dataset remain 100% unaltered.
- Development database records are preserved intact.

**STOP CONDITION ENFORCED**: All development is halted. Backend Part 10 is submitted for user review and approval. No further phases will be started.
