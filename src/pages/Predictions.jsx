import React, { useState, useEffect, useCallback } from 'react';
import {
  Cpu,
  Activity,
  HardDrive,
  Info,
  Clock,
  ShieldAlert,
  CheckCircle2,
  AlertTriangle,
  Send,
  RefreshCw,
  AlertCircle,
  FileText,
  Sliders,
  Check
} from 'lucide-react';
import { apiClient, ApiError } from '../services/apiClient';

// Verified 11 SMART raw features expected by the PyTorch model
const SMART_FEATURES = [
  { key: 'smart_1_raw', label: 'Raw Read Error Rate (SMART 1)' },
  { key: 'smart_4_raw', label: 'Start/Stop Count (SMART 4)' },
  { key: 'smart_5_raw', label: 'Reallocated Sectors (SMART 5)' },
  { key: 'smart_7_raw', label: 'Seek Error Rate (SMART 7)' },
  { key: 'smart_9_raw', label: 'Power-On Hours (SMART 9)' },
  { key: 'smart_12_raw', label: 'Power Cycle Count (SMART 12)' },
  { key: 'smart_187_raw', label: 'Reported Uncorrectable (SMART 187)' },
  { key: 'smart_188_raw', label: 'Command Timeout (SMART 188)' },
  { key: 'smart_194_raw', label: 'Temperature Celsius (SMART 194)' },
  { key: 'smart_197_raw', label: 'Current Pending Sectors (SMART 197)' },
  { key: 'smart_198_raw', label: 'Offline Uncorrectable (SMART 198)' },
];

// Realistic demonstration 3-day sample sequences
const SAMPLE_HEALTHY_SEQUENCE = [
  { smart_1_raw: 120534, smart_4_raw: 8, smart_5_raw: 0, smart_7_raw: 234190, smart_9_raw: 15200, smart_12_raw: 8, smart_187_raw: 0, smart_188_raw: 0, smart_194_raw: 26, smart_197_raw: 0, smart_198_raw: 0 },
  { smart_1_raw: 120560, smart_4_raw: 8, smart_5_raw: 0, smart_7_raw: 234250, smart_9_raw: 15224, smart_12_raw: 8, smart_187_raw: 0, smart_188_raw: 0, smart_194_raw: 27, smart_197_raw: 0, smart_198_raw: 0 },
  { smart_1_raw: 120590, smart_4_raw: 8, smart_5_raw: 0, smart_7_raw: 234310, smart_9_raw: 15248, smart_12_raw: 8, smart_187_raw: 0, smart_188_raw: 0, smart_194_raw: 26, smart_197_raw: 0, smart_198_raw: 0 },
];

const SAMPLE_CRITICAL_SEQUENCE = [
  { smart_1_raw: 8941020, smart_4_raw: 14, smart_5_raw: 16, smart_7_raw: 890120, smart_9_raw: 28400, smart_12_raw: 14, smart_187_raw: 12, smart_188_raw: 4, smart_194_raw: 34, smart_197_raw: 8, smart_198_raw: 8 },
  { smart_1_raw: 14502100, smart_4_raw: 14, smart_5_raw: 64, smart_7_raw: 1450300, smart_9_raw: 28424, smart_12_raw: 14, smart_187_raw: 45, smart_188_raw: 9, smart_194_raw: 38, smart_197_raw: 24, smart_198_raw: 24 },
  { smart_1_raw: 28941500, smart_4_raw: 14, smart_5_raw: 180, smart_7_raw: 2894000, smart_9_raw: 28448, smart_12_raw: 14, smart_187_raw: 128, smart_188_raw: 22, smart_194_raw: 42, smart_197_raw: 60, smart_198_raw: 60 },
];

export default function Predictions() {
  const [modelHealth, setModelHealth] = useState(null);
  const [serialNumber, setSerialNumber] = useState('ST4000DM000_LIVE_01');
  const [modelName, setModelName] = useState('ST4000DM000');
  const [readings, setReadings] = useState(SAMPLE_HEALTHY_SEQUENCE);
  const [activeProfile, setActiveProfile] = useState('healthy');

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [submitError, setSubmitError] = useState(null);

  const [historyLoading, setHistoryLoading] = useState(true);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [historyError, setHistoryError] = useState(null);

  // Fetch model health and historical predictions on mount
  const loadHistory = useCallback(async () => {
    setHistoryLoading(true);
    setHistoryError(null);
    try {
      const records = await apiClient.getPredictions(20);
      setPredictionHistory(Array.isArray(records) ? records : []);
    } catch (err) {
      setHistoryError(err.message);
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    apiClient
      .checkModelsHealth()
      .then(setModelHealth)
      .catch((err) => console.warn('Model health check:', err.message));

    loadHistory();
  }, [loadHistory]);

  const handleSelectProfile = (type) => {
    setActiveProfile(type);
    if (type === 'healthy') {
      setSerialNumber('ST4000_HEALTHY_DEMO');
      setModelName('ST4000DM000');
      setReadings(SAMPLE_HEALTHY_SEQUENCE);
    } else if (type === 'critical') {
      setSerialNumber('ST4000_CRITICAL_DEMO');
      setModelName('ST4000DM000');
      setReadings(SAMPLE_CRITICAL_SEQUENCE);
    }
    setSubmitError(null);
  };

  const handleReadingChange = (dayIndex, fieldKey, val) => {
    setActiveProfile('custom');
    const numericVal = val === '' ? null : Number(val);
    setReadings((prev) => {
      const updated = [...prev];
      updated[dayIndex] = {
        ...updated[dayIndex],
        [fieldKey]: isNaN(numericVal) ? null : numericVal,
      };
      return updated;
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (isSubmitting) return; // Prevent duplicate submissions

    setSubmitError(null);
    setPredictionResult(null);

    // Client-side sequence length validation
    if (!Array.isArray(readings) || readings.length !== 3) {
      setSubmitError('Inference requires exactly 3 consecutive daily telemetry readings.');
      return;
    }

    if (!serialNumber.trim()) {
      setSubmitError('Please specify a hard-drive serial number or device identifier.');
      return;
    }

    setIsSubmitting(true);

    try {
      const payload = {
        serial_number: serialNumber.trim(),
        model: modelName.trim() || undefined,
        readings: readings,
      };

      const result = await apiClient.predictFailure(payload);
      setPredictionResult(result);

      // Refresh history table with the newly saved prediction
      loadHistory();
    } catch (err) {
      setSubmitError(err.message || 'Failed to generate prediction.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="predictions-container">
      {/* 1. Academic Prototype Disclosure Banner */}
      <aside className="demo-notice-banner" aria-label="Academic Prototype Disclosure">
        <Info size={16} className="demo-notice-icon" />
        <div>
          <span className="demo-notice-title">Academic Research Prototype:</span>
          Predictions are computed via a pre-trained PyTorch LSTM neural network evaluating a 3-day sequence ($T=3$) for failure risk over the next 48 hours ($H=2$). Decision threshold is $0.50$.
        </div>
      </aside>

      {/* 2. Model Operational Status Card */}
      <section className="chart-card" style={{ padding: '16px 20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div className="model-pred-icon lstm" style={{ width: '38px', height: '38px' }}>
              <Cpu size={20} />
            </div>
            <div>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, margin: 0, color: 'var(--text-main)' }}>
                PyTorch LSTM Inference Engine
              </h3>
              <span style={{ fontSize: '0.775rem', color: 'var(--text-muted)' }}>
                Target: Hard-drive failure early warning within 2 days ($H=2$)
              </span>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span className={`status-badge ${modelHealth?.status === 'loaded' ? 'healthy' : 'warning'}`}>
              {modelHealth?.status === 'loaded' ? 'Model Weights Loaded (CPU)' : 'Status: Checking...'}
            </span>
            <span className="status-badge" style={{ backgroundColor: '#f1f5f9', color: 'var(--text-secondary)' }}>
              11 SMART Raw Inputs
            </span>
          </div>
        </div>
      </section>

      {/* 3. Interactive Telemetry Submission Form */}
      <section className="table-card" aria-label="Prediction Submission Form">
        <div className="table-header">
          <div className="chart-title-area">
            <h2 className="table-title">
              <Send size={18} color="var(--primary)" />
              Submit SMART Telemetry Sequence
            </h2>
            <p className="chart-subtitle">
              Supply exactly 3 consecutive days of hardware readings for device failure probability evaluation
            </p>
          </div>

          {/* Quick-load Profile Buttons */}
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <button
              type="button"
              className={`action-btn-placeholder ${activeProfile === 'healthy' ? 'active-profile' : ''}`}
              style={{
                backgroundColor: activeProfile === 'healthy' ? 'var(--status-healthy-bg)' : '#ffffff',
                borderColor: activeProfile === 'healthy' ? 'var(--status-healthy-border)' : 'var(--border-color)',
                color: activeProfile === 'healthy' ? 'var(--status-healthy)' : 'var(--text-secondary)',
                fontWeight: activeProfile === 'healthy' ? 600 : 500,
              }}
              onClick={() => handleSelectProfile('healthy')}
              disabled={isSubmitting}
            >
              <CheckCircle2 size={13} />
              <span>Load Healthy Sample</span>
            </button>

            <button
              type="button"
              className={`action-btn-placeholder ${activeProfile === 'critical' ? 'active-profile' : ''}`}
              style={{
                backgroundColor: activeProfile === 'critical' ? 'var(--status-danger-bg)' : '#ffffff',
                borderColor: activeProfile === 'critical' ? 'var(--status-danger-border)' : 'var(--border-color)',
                color: activeProfile === 'critical' ? 'var(--status-danger)' : 'var(--text-secondary)',
                fontWeight: activeProfile === 'critical' ? 600 : 500,
              }}
              onClick={() => handleSelectProfile('critical')}
              disabled={isSubmitting}
            >
              <ShieldAlert size={13} />
              <span>Load Critical Sample</span>
            </button>
          </div>
        </div>

        <form onSubmit={handleSubmit}>
          {/* Hardware Identity Inputs */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '16px', marginBottom: '20px' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '6px' }}>
                Device Serial Number <span style={{ color: 'var(--status-danger)' }}>*</span>
              </label>
              <input
                type="text"
                className="search-input-field"
                style={{ width: '100%', paddingLeft: '12px' }}
                value={serialNumber}
                onChange={(e) => setSerialNumber(e.target.value)}
                placeholder="e.g. W30068RN"
                required
                disabled={isSubmitting}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '6px' }}>
                Hard Drive Model
              </label>
              <input
                type="text"
                className="search-input-field"
                style={{ width: '100%', paddingLeft: '12px' }}
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                placeholder="e.g. ST4000DM000"
                disabled={isSubmitting}
              />
            </div>
          </div>

          {/* 3-Day Telemetry Matrix */}
          <div style={{ marginBottom: '20px' }}>
            <h4 style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-main)', marginBottom: '8px' }}>
              Sequential Telemetry Readings ($T = 3$ consecutive days)
            </h4>
            
            <div style={{ overflowX: 'auto' }}>
              <table className="device-table" style={{ fontSize: '0.8rem' }}>
                <thead>
                  <tr>
                    <th style={{ width: '260px' }}>Physical SMART Raw Counter</th>
                    <th style={{ textAlign: 'center' }}>Day t-2 (2 Days Prior)</th>
                    <th style={{ textAlign: 'center' }}>Day t-1 (1 Day Prior)</th>
                    <th style={{ textAlign: 'center' }}>Day t (Cutoff Observation)</th>
                  </tr>
                </thead>
                <tbody>
                  {SMART_FEATURES.map((feat) => (
                    <tr key={feat.key}>
                      <td style={{ fontWeight: 500, color: 'var(--text-main)' }}>{feat.label}</td>
                      {[0, 1, 2].map((dayIdx) => (
                        <td key={dayIdx} style={{ padding: '6px' }}>
                          <input
                            type="number"
                            className="search-input-field"
                            style={{ width: '100%', padding: '4px 8px', fontSize: '0.8rem', textAlign: 'right' }}
                            value={readings[dayIdx]?.[feat.key] ?? ''}
                            onChange={(e) => handleReadingChange(dayIdx, feat.key, e.target.value)}
                            disabled={isSubmitting}
                          />
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Form Error Banner */}
          {submitError && (
            <div className="demo-notice-banner" style={{ backgroundColor: 'var(--status-danger-bg)', borderColor: 'var(--status-danger-border)', color: 'var(--status-danger)', marginBottom: '16px' }}>
              <AlertCircle size={16} />
              <span>{submitError}</span>
            </div>
          )}

          {/* Action Button */}
          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
            <button
              type="submit"
              className="action-btn-placeholder"
              style={{
                backgroundColor: 'var(--primary)',
                color: '#ffffff',
                padding: '10px 24px',
                fontSize: '0.875rem',
                fontWeight: 600,
                opacity: isSubmitting ? 0.7 : 1,
                cursor: isSubmitting ? 'not-allowed' : 'pointer',
              }}
              disabled={isSubmitting}
            >
              {isSubmitting ? (
                <>
                  <Clock size={16} className="spin-icon" />
                  <span>Executing PyTorch LSTM Inference...</span>
                </>
              ) : (
                <>
                  <Send size={16} />
                  <span>Run Failure Prediction</span>
                </>
              )}
            </button>
          </div>
        </form>
      </section>

      {/* 4. Live Prediction Output Card */}
      {predictionResult && (
        <section
          className="table-card"
          style={{
            borderLeft: `4px solid ${
              predictionResult.prediction_class === 1 ? 'var(--status-danger)' : 'var(--status-healthy)'
            }`,
          }}
          aria-label="Prediction Inference Result"
        >
          <div className="table-header">
            <div className="chart-title-area">
              <h2 className="table-title">
                {predictionResult.prediction_class === 1 ? (
                  <ShieldAlert size={20} color="var(--status-danger)" />
                ) : (
                  <CheckCircle2 size={20} color="var(--status-healthy)" />
                )}
                Inference Result: {predictionResult.serial_number}
              </h2>
              <p className="chart-subtitle">
                Generated via {predictionResult.model_name} (v{predictionResult.model_version}) at{' '}
                {new Date(predictionResult.timestamp).toLocaleString()}
              </p>
            </div>

            <span
              className={`status-badge ${
                predictionResult.prediction_class === 1 ? 'high-risk' : 'healthy'
              }`}
              style={{ fontSize: '0.85rem', padding: '6px 14px' }}
            >
              {predictionResult.prediction_class === 1 ? 'CRITICAL FAILURE PREDICTED' : 'HEALTHY / LOW RISK'}
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', margin: '16px 0' }}>
            <div style={{ padding: '12px 14px', backgroundColor: '#f8fafc', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block' }}>Failure Probability</span>
              <strong style={{ fontSize: '1.25rem', color: predictionResult.risk_score >= 0.5 ? 'var(--status-danger)' : 'var(--status-healthy)' }}>
                {predictionResult.risk_score.toFixed(4)}
              </strong>
            </div>

            <div style={{ padding: '12px 14px', backgroundColor: '#f8fafc', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block' }}>Risk Category</span>
              <strong style={{ fontSize: '1.1rem', color: 'var(--text-main)' }}>
                {predictionResult.risk_level}
              </strong>
            </div>

            <div style={{ padding: '12px 14px', backgroundColor: '#f8fafc', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block' }}>Advance Lead Time</span>
              <strong style={{ fontSize: '1.1rem', color: 'var(--text-main)' }}>
                {predictionResult.horizon_days} Days (48h)
              </strong>
            </div>

            <div style={{ padding: '12px 14px', backgroundColor: '#f8fafc', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block' }}>Decision Threshold</span>
              <strong style={{ fontSize: '1.1rem', color: 'var(--text-main)' }}>
                {predictionResult.threshold.toFixed(2)}
              </strong>
            </div>
          </div>

          {/* Actionable Preventative Recommendation */}
          <div
            style={{
              padding: '14px 18px',
              backgroundColor: predictionResult.prediction_class === 1 ? 'var(--status-danger-bg)' : 'var(--status-healthy-bg)',
              borderRadius: 'var(--radius-sm)',
              border: `1px solid ${
                predictionResult.prediction_class === 1 ? 'var(--status-danger-border)' : 'var(--status-healthy-border)'
              }`,
            }}
          >
            <strong style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-main)', marginBottom: '4px' }}>
              Actionable Recommendation:
            </strong>
            <p style={{ margin: 0, fontSize: '0.825rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              {predictionResult.recommendation}
            </p>
          </div>
        </section>
      )}

      {/* 5. Historical Predictions Audit Log */}
      <section className="table-card" aria-label="Historical Predictions Audit Log">
        <div className="table-header">
          <div className="chart-title-area">
            <h2 className="table-title">
              <FileText size={18} color="var(--primary)" />
              Database Prediction Audit History
            </h2>
            <p className="chart-subtitle">
              Live prediction records persisted in SQLite database (<code>prediction_records</code> table)
            </p>
          </div>

          <button
            type="button"
            className="action-btn-placeholder"
            onClick={loadHistory}
            disabled={historyLoading}
            title="Refresh prediction history"
          >
            <RefreshCw size={13} className={historyLoading ? 'spin-icon' : ''} />
            <span>Refresh</span>
          </button>
        </div>

        {historyLoading ? (
          <div className="smart-empty-container-compact">
            <Clock size={20} className="spin-icon" color="var(--primary)" />
            <p className="smart-empty-desc">Loading database history...</p>
          </div>
        ) : historyError ? (
          <div className="demo-notice-banner" style={{ backgroundColor: 'var(--status-danger-bg)', borderColor: 'var(--status-danger-border)', color: 'var(--status-danger)' }}>
            <AlertCircle size={16} />
            <span>{historyError}</span>
          </div>
        ) : predictionHistory.length === 0 ? (
          <div className="smart-empty-container-compact">
            <Info size={20} color="var(--text-muted)" />
            <p className="smart-empty-desc">No predictions recorded in database yet.</p>
          </div>
        ) : (
          <div className="table-wrapper">
            <table className="device-table">
              <thead>
                <tr>
                  <th>Timestamp (UTC)</th>
                  <th>Record ID</th>
                  <th>Device ID</th>
                  <th>Model</th>
                  <th>Probability</th>
                  <th>Risk Level</th>
                  <th>Prediction Class</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {predictionHistory.map((pred) => {
                  let badgeClass = 'healthy';
                  if (pred.prediction_class === 1 || pred.risk_level === 'CRITICAL') {
                    badgeClass = 'high-risk';
                  } else if (pred.risk_level === 'MODERATE') {
                    badgeClass = 'warning';
                  }

                  return (
                    <tr key={pred.id}>
                      <td style={{ fontSize: '0.8rem', whiteSpace: 'nowrap' }}>
                        {new Date(pred.timestamp).toLocaleString()}
                      </td>
                      <td>#{pred.id}</td>
                      <td>{pred.device_id ? `#${pred.device_id}` : 'General'}</td>
                      <td>{pred.model_name}</td>
                      <td>
                        <span className={`risk-score-pill ${badgeClass === 'high-risk' ? 'high' : badgeClass === 'warning' ? 'warning' : 'healthy'}`}>
                          {pred.risk_score.toFixed(4)}
                        </span>
                      </td>
                      <td>
                        <span className={`status-badge ${badgeClass}`}>{pred.risk_level}</span>
                      </td>
                      <td>
                        {pred.prediction_class === 1 ? (
                          <span className="status-badge high-risk">CRITICAL (1)</span>
                        ) : (
                          <span className="status-badge healthy">HEALTHY (0)</span>
                        )}
                      </td>
                      <td>
                        <span className="status-badge resolved" style={{ fontSize: '0.7rem' }}>
                          {pred.status}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
