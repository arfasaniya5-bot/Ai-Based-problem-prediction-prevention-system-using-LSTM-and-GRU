# DeviceGuard AI — Backend Documentation

Backend service, database layer, and deep learning pipeline for:
**AI-Based Problem Prediction & Prevention System Using LSTM and GRU**.

---

## 1. Overview of Backend Capabilities

The backend provides:
- **Memory-Safe Preprocessing**: Streaming ingestion and verification of hard-drive telemetry from the Backblaze dataset (`backend/data/raw/harddrive.csv`).
- **Feature Pipeline**: Selection and scaling of 11 verified physical SMART raw counters with training-fitted median imputation and RobustScaling.
- **Deep Learning Sequential Models**: Pre-trained LSTM and GRU neural networks forecasting drive failure within a 2-day lead time ($H = 2$) using 3 consecutive days of telemetry ($T = 3$).
- **FastAPI Model Serving**: Low-latency REST API service with input validation, model inference, and persistence.
- **Database & Alert Layer**: SQLAlchemy ORM with SQLite for tracking devices, prediction logs, and failure alerts.

---

## 2. Environment Setup & Dependency Installation

Ensure Python 3.10+ (tested with Python 3.13) is installed.

### Step 1: Open PowerShell and navigate to the backend directory
```powershell
cd backend
```

### Step 2: Create and activate virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

*Note: If PowerShell displays an execution policy restriction, run:*
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

### Step 3: Install dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Starting the FastAPI Application

Launch the development server using Uvicorn:

```powershell
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

Once started:
- **Swagger Interactive API Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Alternative Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Root Service Discovery**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

---

## 4. API Endpoints & Request/Response Schemas

### A. Health & System Discovery
- `GET /`: General service metadata and status.
- `GET /health`: Service liveness probe.
  ```json
  {
    "status": "healthy",
    "timestamp": "2026-10-24T10:55:00Z",
    "version": "1.0.0",
    "environment": "development"
  }
  ```
- `GET /health/models`: Diagnostic check confirming model checkpoints and preprocessors are loaded in memory.
  ```json
  {
    "status": "loaded",
    "models_loaded": {
      "lstm": true,
      "gru": true,
      "baseline": true
    },
    "feature_count": 11,
    "sequence_length": 3,
    "horizon_days": 2,
    "decision_threshold": 0.50,
    "device": "cpu"
  }
  ```

### B. Model Inference
- `POST /api/predict`: Evaluates 3 consecutive daily telemetry readings and returns failure risk score and alert recommendations.

#### Example Request:
```json
{
  "serial_number": "W30068RN",
  "model": "ST4000DM000",
  "readings": [
    {
      "smart_1_raw": 114175000,
      "smart_4_raw": 4,
      "smart_5_raw": 0,
      "smart_7_raw": 32000,
      "smart_9_raw": 8760,
      "smart_12_raw": 4,
      "smart_187_raw": 0,
      "smart_188_raw": 0,
      "smart_194_raw": 24,
      "smart_197_raw": 0,
      "smart_198_raw": 0
    },
    {
      "smart_1_raw": 114210000,
      "smart_4_raw": 4,
      "smart_5_raw": 0,
      "smart_7_raw": 32100,
      "smart_9_raw": 8784,
      "smart_12_raw": 4,
      "smart_187_raw": 0,
      "smart_188_raw": 0,
      "smart_194_raw": 25,
      "smart_197_raw": 0,
      "smart_198_raw": 0
    },
    {
      "smart_1_raw": 114250000,
      "smart_4_raw": 4,
      "smart_5_raw": 8,
      "smart_7_raw": 32200,
      "smart_9_raw": 8808,
      "smart_12_raw": 4,
      "smart_187_raw": 2,
      "smart_188_raw": 0,
      "smart_194_raw": 25,
      "smart_197_raw": 4,
      "smart_198_raw": 0
    }
  ]
}
```

#### Example Response:
```json
{
  "serial_number": "W30068RN",
  "model_name": "LSTM Neural Network",
  "model_version": "1.0.0",
  "prediction_class": 1,
  "is_failure_predicted": true,
  "risk_score": 0.8142,
  "risk_level": "CRITICAL",
  "threshold": 0.50,
  "horizon_days": 2,
  "sequence_length_days": 3,
  "recommendation": "Critical failure risk detected within 48 hours. Initiate immediate automated data backup and schedule storage unit replacement.",
  "timestamp": "2026-10-24T10:55:00Z",
  "disclaimer": "Academic research prototype; not certified for production deployment."
}
```

### C. Device Registry & Alert Lifecycle (Part 7)

#### 1. Device Registry (`GET /api/devices`)
Lists registered storage drives with pagination, serial number search, and status filtering.
- **Query Parameters**:
  - `page` (int, default `1`): 1-indexed page number.
  - `page_size` (int, default `20`, max `100`): Items per page.
  - `search` (string, optional): Case-insensitive serial number substring filter.
  - `status` (string, optional): Filter by operational health status (`HEALTHY`, `WARNING`, `CRITICAL`, `FAILED`).
- **Response Format**:
  ```json
  {
    "items": [
      {
        "id": 1,
        "serial_number": "ST4000DM000_LIVE_01",
        "model": "ST4000DM000",
        "capacity_bytes": null,
        "status": "HEALTHY",
        "first_seen": "2026-10-24T11:20:00Z",
        "last_seen": "2026-10-24T11:25:00Z"
      }
    ],
    "total": 1,
    "page": 1,
    "page_size": 20,
    "total_pages": 1
  }
  ```

#### 2. Failure Prevention Alerts (`GET /api/alerts`)
Lists proactive hardware failure alerts stored in SQLite with associated device metadata.
- **Query Parameters**:
  - `page` (int, default `1`): 1-indexed page number.
  - `page_size` (int, default `20`, max `100`): Items per page.
  - `severity` (string, optional): Filter by alert severity (`INFO`, `WARNING`, `CRITICAL`).
  - `acknowledged` (boolean, optional): Filter by acknowledgement status (`true` or `false`).
- **Response Format**:
  ```json
  {
    "items": [
      {
        "id": 1,
        "prediction_id": 15,
        "device_id": 9,
        "device_serial": "ALERT_TEST_DRIVE",
        "device_model": "ST4000DM000",
        "timestamp": "2026-10-24T11:24:09Z",
        "severity": "CRITICAL",
        "recommendation": "Critical failure risk detected within 48 hours. Initiate immediate automated data backup.",
        "acknowledged": false,
        "acknowledged_at": null
      }
    ],
    "total": 1,
    "page": 1,
    "page_size": 20,
    "total_pages": 1
  }
  ```

#### 3. Alert Acknowledgement (`PATCH /api/alerts/{id}/acknowledge`)
Marks a proactive failure alert as acknowledged by a system administrator or operator. Idempotent on repeated calls.
- **Path Parameter**: `id` (integer) - Alert record ID.
- **Response Format**:
  ```json
  {
    "id": 1,
    "acknowledged": true,
    "acknowledged_at": "2026-10-24T11:24:18Z",
    "message": "Alert #1 successfully acknowledged."
  }
  ```

### D. Prediction History
- `GET /api/predictions?limit=20`: Lists recent stored prediction history records.

---

## 5. Database Setup & Architecture

### Database Selection: SQLite (with SQLAlchemy ORM)
- **Trade-offs Considered**:
  - **SQLite**: Local, file-based, zero background daemon requirement, fully ACID-compliant, seamless for local academic testing and development. Selected as the default development database.
  - **PostgreSQL**: Production-grade concurrency; easily supported by changing `DATABASE_URL`.
- **Configuration**:
  - Configurable via environment variable:
    ```text
    DATABASE_URL=sqlite:///./backend/data/deviceguard.db
    ```
- **Schema Entities**:
  - `Device`: Hardware identity, model, capacity, status (`HEALTHY`, `WARNING`, `CRITICAL`), timestamps.
  - `TelemetryReading`: Daily SMART reading observations linked to device.
  - `PredictionRecord`: Historical model inferences (score, class, threshold, model version, status).
  - `AlertRecord`: Actionable alerts and recommendations triggered by critical predictions.
  - `ApiLog`: API audit log for tracking request response times and errors.

---

## 6. Running Tests

Execute the comprehensive test suite verifying health endpoints, model inference, validation rules, database persistence, device registry, and alert lifecycle:

```powershell
python -m pytest backend/tests/ -v
```

---

## 7. Model Training & Preprocessing Reproduction

- **Inspect Raw Dataset**:
  ```powershell
  python src/data/inspect_dataset.py
  ```
- **Execute Preprocessing & Sequence Feasibility Assessment**:
  ```powershell
  python src/data/preprocess_dataset.py
  ```
- **Reproduce Model Experiments (Heuristic, Logistic Regression, LSTM, GRU)**:
  ```powershell
  python src/models/train.py
  ```

---

## 8. Directory Structure

```text
backend/
├── data/
│   ├── raw/                  # Expected location for harddrive.csv (unmodified)
│   ├── processed/            # Derived preprocessing artifacts
│   └── deviceguard.db        # SQLite database file
├── models/                   # Saved model checkpoints and preprocessor configs
│   ├── baseline_model.joblib # Logistic Regression baseline
│   ├── best_gru.pt           # Trained GRU PyTorch checkpoint
│   ├── best_lstm.pt          # Trained LSTM PyTorch checkpoint
│   ├── evaluation_summary.json # Serialized evaluation metrics
│   └── preprocessor_config.json # Imputer and scaler parameters
├── reports/                  # Assessment and experiment documentation
│   ├── dataset_inspection.md # Dataset schema and row counts
│   ├── preprocessing_assessment.md # Duplicate audit and sequence feasibility
│   └── model_experiments.md # Model comparison and drive-level evaluation
├── src/
│   ├── api/                  # FastAPI router definitions
│   │   ├── __init__.py
│   │   ├── routes_health.py
│   │   └── routes_prediction.py
│   ├── database/             # SQLAlchemy connection and ORM models
│   │   ├── __init__.py
│   │   ├── connection.py
│   │   └── models.py
│   ├── models/               # Model architectures and training scripts
│   │   ├── __init__.py
│   │   ├── architectures.py
│   │   └── train.py
│   ├── schemas/              # Pydantic request and response schemas
│   │   ├── __init__.py
│   │   ├── health.py
│   │   └── prediction.py
│   ├── services/             # Inference and preprocessing services
│   │   ├── __init__.py
│   │   └── inference_service.py
│   ├── config.py             # Pydantic Settings configuration
│   └── main.py               # FastAPI application entrypoint
├── tests/                    # Backend unit and integration test suite
│   ├── __init__.py
│   └── test_api.py
├── .gitignore                # Excludes .venv and raw/processed dataset files
├── requirements.txt          # Python dependencies
└── README.md                 # Setup and execution instructions
```

---

## 9. Known Limitations & Research Disclosure

1. **Academic Prototype Status**: The system is an academic research prototype. Predictions and alert recommendations must not be used as the sole basis for safety-critical hardware decommissions without human oversight.
2. **Observation Period (13 Days)**: The underlying dataset covers 13 calendar days (`01-01-2016` to `13-01-2016`). While valid for $H = 2$ lead-time forecasting, multi-quarter degradation patterns require longer historical data.
3. **Severe Class Imbalance**: With ~8,000 healthy drive windows per failure window, precision point estimates are naturally constrained. Sequential models significantly reduce false alarms compared to linear models, but small-sample variance exists.
