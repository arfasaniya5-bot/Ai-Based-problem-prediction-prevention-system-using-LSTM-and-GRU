# Backend Part 9: Security Audit, Reliability Hardening & Regression Testing Report

**Project Title:** AI-Based Problem Prediction & Prevention System Using LSTM and GRU (DeviceGuard AI)  
**Phase:** Backend Part 9 — Security Audit, Reliability Hardening & Regression Testing  
**Date:** 2026-10-24  
**Status:** Audit & Hardening Completed — Ready for Review  

---

## 1. Executive Summary

This report documents the security audit, reliability hardening, and regression verification for **DeviceGuard AI**. The audit inspected the FastAPI backend service, Pydantic input validation schemas, database transaction atomicity, React client API integration, configuration defaults, and dependency profiles.

### Primary Objectives & Outcomes:
- **Security Audit Completed**: Inspected the codebase for potential secrets, credential leaks, stack trace exposure, unsafe CORS headers, SQL injection vectors, and unbounded payloads. Zero credentials or secret keys were detected in the tracked codebase.
- **Exception Sanitization Implemented**: Replaced raw internal exception strings in HTTP 500 error responses with generic client-safe messages, while logging full operational error details securely on the server.
- **Input Boundaries Hardened**: Implemented `max_length` constraints on Pydantic models (`serial_number: max_length=64`, `model: max_length=64`) and FastAPI query parameters (`search: max_length=100`, `status: max_length=50`, `severity: max_length=50`) to prevent oversized payload abuse.
- **Transaction Rollback Protection**: Added explicit try-except rollback blocks to alert acknowledgement mutations (`PATCH /api/alerts/{id}/acknowledge`), ensuring SQLite database locks or write failures do not leave database sessions in an uncommitted state.
- **Repository Hygiene Hardened**: Updated `.gitignore` to prevent tracking of local SQLite databases (`*.db`, `*.sqlite`, `*.sqlite3`), local environment files (`.env`, `.env*.local`), and test caches (`.pytest_cache/`).
- **Comprehensive Regression Suite**: 29 out of 29 automated backend tests passed in **1.29 seconds** (100% pass rate). The React frontend production build succeeded cleanly in **9.32 seconds** with zero compilation or lint errors.
- **Dataset and Checkpoint Integrity**: Strict cryptographic verification confirmed that `backend/data/raw/harddrive.csv` and all saved model checkpoints (`best_lstm.pt`, `best_gru.pt`, `baseline_model.joblib`, `preprocessor_config.json`) match their baseline SHA-256 hashes to the exact bit.

---

## 2. Security Audit Findings

| Audit Check Category | Evaluation Status | Finding Details | Action Taken / Status |
|---|---|---|---|
| **Hardcoded Secrets & API Keys** | **Passed** | Automated regex scans for patterns matching `api_key`, `secret`, `password`, `bearer`, or `token` across `backend/src`, `src`, and root configs found zero hardcoded credentials. | Confirmed clean; no secrets tracked in Git. |
| **Stack Trace & Path Exposure** | **Discovered & Fixed** | Handlers in `routes_prediction.py` previously interpolated raw exception strings (`f"Database persistence failed: {db_err}"` and `f"Inference execution failed: {e}"`) into HTTP 500 responses, risking leakage of internal file paths or SQLite table errors. | Sanitized client responses to generic messages and routed detailed errors to `logger.error()`. |
| **CORS Configuration** | **Passed** | FastAPI CORS middleware restricts `allow_origins` to explicitly whitelisted development addresses (`http://localhost:3000`, `http://127.0.0.1:3000`, `http://localhost:5173`, `http://127.0.0.1:5173`). Wildcard `*` origins are not permitted. | Confirmed secure for development/prototype deployment. |
| **SQL Injection Risks** | **Passed** | All database queries utilize SQLAlchemy ORM with bound parameters. Search queries utilize `.ilike(...)` parameterized bindings. No raw SQL query string formatting exists. | Confirmed secure. |
| **Unbounded Payloads & Inputs** | **Discovered & Fixed** | Serial numbers, models, and query strings lacked upper boundary constraints, allowing arbitrarily large strings to be sent to database columns. | Added `max_length=64` to `serial_number`/`model` and `max_length=100`/`50` to search/filter query parameters. |
| **PyTorch Checkpoint Loading** | **Passed** | `inference_service.py` loads model weights using `torch.load(..., weights_only=True)`. This mitigates arbitrary code execution vulnerabilities via pickled model files. | Confirmed secure. |
| **Authentication & Authorization** | **Limitation (Documented)** | The application currently operates as an open internal prototype without multi-user authentication (JWT, OAuth2, or sessions) or role-based access control. | Preserved without new dependencies; documented as an architectural limitation for future production hardening. |

---

## 3. API Validation and Error Handling

All seven core endpoints were audited and tested against invalid, malformed, or boundary inputs:

| Endpoint | Method | Input Validation Tested | Expected Status | Result |
|---|---|---|---|---|
| `/health` | `GET` | Service liveness probe | `HTTP 200` | **Passed** |
| `/health/models` | `GET` | In-memory artifact validation | `HTTP 200` | **Passed** |
| `/api/predict` | `POST` | Valid 3-day telemetry sequence | `HTTP 200` | **Passed** |
| `/api/predict` | `POST` | Sequence length < 3 (2 days) | `HTTP 422` | **Passed** (validation error returned) |
| `/api/predict` | `POST` | Sequence length > 3 (4 days) | `HTTP 422` | **Passed** (validation error returned) |
| `/api/predict` | `POST` | Non-numeric SMART features | `HTTP 422` | **Passed** (validation error returned) |
| `/api/predict` | `POST` | Serial number > 64 chars | `HTTP 422` | **Passed** (string length bounded) |
| `/api/predict` | `POST` | Missing/null features | `HTTP 200` | **Passed** (median imputation applied) |
| `/api/predict` | `POST` | Injected database write error | `HTTP 500` | **Passed** (clean error, rollback executed) |
| `/api/predictions` | `GET` | Pagination limit (`ge=1, le=100`) | `HTTP 200` / `422` | **Passed** (bounds enforced) |
| `/api/devices` | `GET` | Valid pagination & substring search | `HTTP 200` | **Passed** |
| `/api/devices` | `GET` | Invalid page index (`page=0`) | `HTTP 422` | **Passed** |
| `/api/devices` | `GET` | Invalid page size (`page_size=500`) | `HTTP 422` | **Passed** |
| `/api/devices` | `GET` | Search string > 100 chars | `HTTP 422` | **Passed** |
| `/api/devices` | `GET` | Unsupported status filter | `HTTP 400` | **Passed** (whitelist error message) |
| `/api/alerts` | `GET` | Valid pagination & severity filter | `HTTP 200` | **Passed** |
| `/api/alerts` | `GET` | Unsupported severity filter | `HTTP 400` | **Passed** (whitelist error message) |
| `/api/alerts/{id}/acknowledge` | `PATCH` | Valid alert ID acknowledgement | `HTTP 200` | **Passed** (persisted timestamp) |
| `/api/alerts/{id}/acknowledge` | `PATCH` | Repeated call (idempotency) | `HTTP 200` | **Passed** (safe repeated acknowledgement) |
| `/api/alerts/{id}/acknowledge` | `PATCH` | Non-existent alert ID (`999999`) | `HTTP 404` | **Passed** (resource not found) |
| `/api/alerts/{id}/acknowledge` | `PATCH` | Injected database commit error | `HTTP 500` | **Passed** (rollback executed, error sanitized) |

---

## 4. Database Integrity and Transaction Safety

The persistence layer was verified against SQLAlchemy ORM models:
- **Foreign Key Relationships**:
  - `PredictionRecord.device_id` correctly references `Device.id` (`ondelete="SET NULL"`).
  - `AlertRecord.device_id` references `Device.id` and `AlertRecord.prediction_id` references `PredictionRecord.id` (`ondelete="CASCADE"`).
  - `TelemetryReading.device_id` references `Device.id` (`ondelete="CASCADE"`).
- **Rollback Safety**:
  - In `POST /api/predict`, commit failures execute `db.rollback()` and raise `HTTP 500`. No orphaned records are created.
  - In `PATCH /api/alerts/{id}/acknowledge`, commit failures execute `db.rollback()` and raise `HTTP 500`.
- **Test Database Isolation**:
  - All tests execute against an isolated temporary SQLite database configured in `backend/tests/conftest.py`.
  - The live application database (`backend/data/deviceguard.db`) remained completely unpolluted throughout test execution:
    - Devices: **13** (identical to pre-test baseline).
    - Predictions: **23** (identical to pre-test baseline).
    - Alerts: **9** (identical to pre-test baseline).

---

## 5. Alert Lifecycle Verification

The alert management subsystem was confirmed reliable:
- **Severity Classification**:
  - Failure probability $\ge 0.50 \rightarrow \text{CRITICAL}$ alert.
  - Failure probability between $0.20$ and $0.4999 \rightarrow \text{WARNING}$ alert.
  - Failure probability $< 0.20 \rightarrow$ No alert generated.
- **Alert Deduplication**:
  - Repeated predictions for an at-risk drive update the existing unacknowledged alert row (`prediction_id`, `timestamp`, and `recommendation`) rather than accumulating redundant unacknowledged alerts.
- **Acknowledgement Flow**:
  - An operator acknowledgement sets `acknowledged = True` and sets `acknowledged_at` to the current UTC timestamp.
  - Idempotency was verified: multiple calls to acknowledge an already-acknowledged alert safely return `HTTP 200` without modifying the original acknowledgement timestamp.

---

## 6. Frontend Reliability Findings

The React frontend components were inspected across `src/pages/Dashboard.jsx`, `src/pages/Devices.jsx`, `src/pages/Predictions.jsx`, and `src/pages/Alerts.jsx`:
- **Network & Server Disconnections**: When the backend is offline or unreachable, clear error notifications are rendered with the server URL and retry options.
- **Loading & Empty State Differentiation**: Tables distinguish between active loading spinners, zero matching records (empty states), and network errors.
- **Double Submission Prevention**: The AI Predictions form sets `isSubmitting = true` on submit and blocks duplicate triggers while an inference request is in-flight.
- **State Synchronization on Mutation**: Alert acknowledgement buttons update local UI state only upon receiving HTTP 200 confirmation from the backend API.
- **No Mock Data Masquerading**: Dashboard KPIs, device registries, and alert queues are populated exclusively from live API queries; mock fallback arrays are not silently displayed when live queries fail.
- **Production Build**: Executed `npm run build`:
  - `dist/index.html`: 0.69 kB
  - `dist/assets/index-DWGrTvuX.css`: 22.21 kB
  - `dist/assets/index-CcsgX0aF.js`: 595.35 kB
  - Zero compilation errors, 0 lint warnings.

---

## 7. Dependency and Configuration Review

- **Python Dependencies (`backend/requirements.txt`)**:
  - Inspected 11 tracked packages (`pandas`, `numpy`, `scikit-learn`, `torch`, `fastapi`, `uvicorn`, `pydantic`, `pydantic-settings`, `sqlalchemy`, `httpx`, `pytest`). All packages are actively used. No bloated or unnecessary libraries are present.
- **Node Dependencies (`package.json`)**:
  - Minimal dependencies (`react`, `react-dom`, `react-router-dom`, `recharts`, `lucide-react`, `vite`).
- **Environment Configuration**:
  - `.env` and `.env.example` configure `VITE_API_BASE_URL=http://127.0.0.1:8000`. No sensitive values or credentials exist.
- **Git Ignore Hygiene (`.gitignore`)**:
  - Added missing ignore rules for `.env`, `*.db`, `*.sqlite`, `*.sqlite3`, and `.pytest_cache/` to ensure development databases and environment files are not accidentally committed to version control.

---

## 8. Automated Test Results

The full backend test suite was executed via PyTest:
- **Command**: `.\backend\.venv\Scripts\python.exe -m pytest backend/tests/ -v`
- **Duration**: `1.29 seconds`
- **Results**: **29 passed, 0 failed, 1 deprecation warning (Starlette testclient)**

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

======================== 29 passed, 1 warning in 1.29s ========================
```

- **Frontend Production Build**:
  - **Command**: `npm run build`
  - **Result**: `✓ built in 9.32s` with 0 errors.

---

## 9. Files Created or Modified

| File Path | Nature of Change | Rationale |
|---|---|---|
| `backend/src/schemas/prediction.py` | **Modified** | Added `max_length=64` to `serial_number` and `model` to bound payload inputs. |
| `backend/src/api/routes_prediction.py` | **Modified** | Added server-side error logging via `logger.error` and sanitized HTTP 500 error responses to prevent internal path/exception exposure. |
| `backend/src/api/routes_devices.py` | **Modified** | Added `max_length=100` for search queries and `max_length=50` for status filters. |
| `backend/src/api/routes_alerts.py` | **Modified** | Added `max_length=50` for severity filters and wrapped acknowledgement `db.commit()` in a try-except rollback block. |
| `.gitignore` | **Modified** | Added ignore patterns for `*.db`, `*.sqlite`, `*.sqlite3`, `.pytest_cache/`, and `.env`. |
| `backend/tests/test_api.py` | **Modified** | Added `test_predict_oversized_serial_number` regression test. |
| `backend/tests/test_devices_alerts.py` | **Modified** | Added `test_devices_oversized_search_string` and `test_acknowledge_alert_database_error` regression tests. |
| `backend/reports/part9_security_reliability_report.md` | **Created** | Comprehensive Part 9 completion report. |

---

## 10. Model and Dataset Integrity

Cryptographic verification was performed before and after Part 9 modifications. All SHA-256 hashes and file byte sizes remain identical to baseline:

| Artifact | Location | Size (Bytes) | SHA-256 Checksum | Integrity Status |
|---|---|---|---|---|
| **Raw Telemetry Dataset** | `backend/data/raw/harddrive.csv` | `294,074,825` | `940D3C27E444393472EEA35B1F5A54A0A06F62E8B9D10F1D0DAAD23B37F7D7B6` | **Verified Unchanged** |
| **LSTM Checkpoint** | `backend/models/best_lstm.pt` | `28,841` | `494CB52C54A90C7E81AF94E14903A4E21126E5A27F293B7477372E4E723E242E` | **Verified Unchanged** |
| **GRU Checkpoint** | `backend/models/best_gru.pt` | `23,067` | `952CDCD595C18E81B0088DDB2196CE32A1E80C0966339F14BB1C369B468E2797` | **Verified Unchanged** |
| **Baseline Model** | `backend/models/baseline_model.joblib` | `987` | `5CE6800CD212370E72B19EC3DCC4992CF399FAD89EB09DE68EDCE9A2FA401095` | **Verified Unchanged** |
| **Preprocessor Config** | `backend/models/preprocessor_config.json` | `1,395` | `2E79BF5656236E583B7081D10B9D14D3974F7C5B19262EFD597BE81316CF17B8` | **Verified Unchanged** |

---

## 11. Remaining Risks and Limitations

1. **Authentication Absence**: The application does not feature authentication, authorization, or session management. It is designed for internal development or demonstration environments and should not be exposed to the public Internet without a reverse proxy / API gateway enforcing authentication.
2. **Browser Automation Testing**: Interactive browser-level automation (e.g. Playwright or Selenium) was not performed due to the lack of headless browser drivers in this environment. API-to-UI contract testing was verified through unit, integration, and build checks.
3. **Dataset Temporal Horizon**: The models were trained and benchmarked on a 13-day slice of Backblaze January 2016 telemetry (`2016-01-01` to `2016-01-13`) with only 7 failed drives in the test set. Real-world predictive accuracy on newer hard drive families requires broader multi-year datasets.
4. **SQLite Concurrency**: SQLite is suitable for single-node development prototypes. Heavy concurrent write workloads would require transitioning to PostgreSQL.

---

## 12. Final Recommendation

Backend Part 9 is **complete, hardened, and verified**:
- Confirmed security vulnerabilities and exception detail leaks were eliminated.
- Input bounds and database transaction rollbacks were enforced.
- 29 backend regression tests and frontend build pass with 100% success.
- Model weights, training parameters, and raw datasets were preserved strictly untouched.

**Stop Condition**: All implementation work is halted. Backend Part 9 is submitted for user review. Part 10 will not be started without explicit approval.
