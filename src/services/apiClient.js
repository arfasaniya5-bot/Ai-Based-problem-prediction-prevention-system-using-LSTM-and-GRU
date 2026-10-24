/**
 * Centralized API client for DeviceGuard AI frontend.
 * Interacts with the FastAPI backend service and handles network timeouts,
 * HTTP error states, and schema validations.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
const DEFAULT_TIMEOUT_MS = 10000;

export class ApiError extends Error {
  constructor(message, status = null, detail = null, isNetworkError = false) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
    this.isNetworkError = isNetworkError;
  }
}

/**
 * Internal fetch wrapper with timeout and standardized error handling.
 */
async function apiFetch(endpoint, options = {}) {
  const url = `${API_BASE_URL.replace(/\/$/, '')}${endpoint}`;
  const timeoutMs = options.timeout || DEFAULT_TIMEOUT_MS;
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);

  const fetchOptions = {
    ...options,
    signal: controller.signal,
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
      ...(options.headers || {}),
    },
  };

  try {
    const response = await fetch(url, fetchOptions);
    clearTimeout(id);

    let data = null;
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      try {
        data = await response.json();
      } catch (jsonErr) {
        throw new ApiError('Failed to parse JSON response from backend server.', response.status, null);
      }
    } else {
      const text = await response.text();
      data = text ? { message: text } : null;
    }

    if (!response.ok) {
      const errorMsg =
        (data && (data.detail || data.message)) ||
        `HTTP Error ${response.status}: ${response.statusText}`;
      throw new ApiError(
        typeof errorMsg === 'string' ? errorMsg : JSON.stringify(errorMsg),
        response.status,
        data?.detail || null
      );
    }

    return data;
  } catch (error) {
    clearTimeout(id);
    if (error instanceof ApiError) {
      throw error;
    }

    if (error.name === 'AbortError') {
      throw new ApiError(
        `Request timed out after ${timeoutMs / 1000}s while contacting ${endpoint}.`,
        408,
        null,
        true
      );
    }

    // Network connection errors (e.g. backend server offline, refused connection)
    throw new ApiError(
      `Cannot connect to backend server at ${API_BASE_URL}. Ensure the FastAPI server is running.`,
      null,
      error.message,
      true
    );
  }
}

export const apiClient = {
  getBaseUrl() {
    return API_BASE_URL;
  },

  /**
   * Root service discovery information
   * GET /
   */
  async getRootInfo() {
    return apiFetch('/');
  },

  /**
   * Basic application health status
   * GET /health
   */
  async checkHealth() {
    return apiFetch('/health');
  },

  /**
   * Model loading and inference configuration status
   * GET /health/models
   */
  async checkModelsHealth() {
    return apiFetch('/health/models');
  },

  /**
   * Submit 3-day SMART telemetry sequence for failure early-warning inference
   * POST /api/predict
   */
  async predictFailure(payload) {
    if (!payload || !Array.isArray(payload.readings)) {
      throw new ApiError('Prediction request must contain a readings array.', 400);
    }
    if (payload.readings.length !== 3) {
      throw new ApiError(
        `Inference requires exactly 3 consecutive daily telemetry readings (received ${payload.readings.length}).`,
        422
      );
    }

    return apiFetch('/api/predict', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  /**
   * Query historical prediction records stored in SQLite
   * GET /api/predictions?limit={limit}
   */
  async getPredictions(limit = 20) {
    return apiFetch(`/api/predictions?limit=${encodeURIComponent(limit)}`);
  },

  /**
   * Query paginated device registry from SQLite
   * GET /api/devices?page={page}&page_size={pageSize}&search={search}&status={status}
   */
  async getDevices({ page = 1, pageSize = 20, search = '', status = '' } = {}) {
    const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
    if (search && search.trim()) {
      params.append('search', search.trim());
    }
    if (status && status.trim() && status !== 'All') {
      params.append('status', status.trim());
    }
    return apiFetch(`/api/devices?${params.toString()}`);
  },

  /**
   * Query paginated hardware failure alerts from SQLite
   * GET /api/alerts?page={page}&page_size={pageSize}&severity={severity}&acknowledged={acknowledged}
   */
  async getAlerts({ page = 1, pageSize = 20, severity = '', acknowledged = null } = {}) {
    const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
    if (severity && severity.trim() && severity !== 'All') {
      params.append('severity', severity.trim());
    }
    if (acknowledged !== null && acknowledged !== undefined && acknowledged !== 'All') {
      params.append('acknowledged', String(acknowledged));
    }
    return apiFetch(`/api/alerts?${params.toString()}`);
  },

  /**
   * Acknowledge an active failure alert
   * PATCH /api/alerts/{id}/acknowledge
   */
  async acknowledgeAlert(alertId) {
    if (!alertId) {
      throw new ApiError('Alert ID is required for acknowledgement.', 400);
    }
    return apiFetch(`/api/alerts/${encodeURIComponent(alertId)}/acknowledge`, {
      method: 'PATCH',
    });
  },
};

export default apiClient;
