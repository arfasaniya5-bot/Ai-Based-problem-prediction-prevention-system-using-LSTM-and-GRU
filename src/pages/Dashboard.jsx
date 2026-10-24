import React, { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import {
  HardDrive,
  CheckCircle2,
  AlertTriangle,
  ShieldAlert,
  Info,
  TrendingUp,
  Activity,
  RefreshCw,
  Server,
  Cpu,
  AlertCircle
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine
} from 'recharts';
import { apiClient } from '../services/apiClient';

export default function Dashboard() {
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [apiHealth, setApiHealth] = useState(null);
  const [modelHealth, setModelHealth] = useState(null);
  const [predictions, setPredictions] = useState([]);

  const [deviceCount, setDeviceCount] = useState(0);
  const [activeAlertsCount, setActiveAlertsCount] = useState(0);

  const fetchDashboardData = useCallback(async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    else setLoading(true);
    setError(null);

    try {
      // Execute health checks, prediction queries, device registry, and alert counts in parallel
      const [healthRes, modelsRes, predsRes, devicesRes, alertsRes] = await Promise.all([
        apiClient.checkHealth().catch((err) => {
          console.warn('API health check failed:', err);
          return null;
        }),
        apiClient.checkModelsHealth().catch((err) => {
          console.warn('Models health check failed:', err);
          return null;
        }),
        apiClient.getPredictions(50).catch((err) => {
          console.warn('Predictions query failed:', err);
          return [];
        }),
        apiClient.getDevices({ page: 1, pageSize: 1 }).catch((err) => {
          console.warn('Devices query failed:', err);
          return null;
        }),
        apiClient.getAlerts({ page: 1, pageSize: 1, acknowledged: false }).catch((err) => {
          console.warn('Alerts query failed:', err);
          return null;
        }),
      ]);

      if (!healthRes && !modelsRes && predsRes.length === 0) {
        throw new Error(
          `Unable to connect to the FastAPI backend at ${apiClient.getBaseUrl()}. Please ensure the backend server is running.`
        );
      }

      setApiHealth(healthRes);
      setModelHealth(modelsRes);
      setPredictions(Array.isArray(predsRes) ? predsRes : []);
      if (devicesRes && typeof devicesRes.total === 'number') {
        setDeviceCount(devicesRes.total);
      }
      if (alertsRes && typeof alertsRes.total === 'number') {
        setActiveAlertsCount(alertsRes.total);
      }
    } catch (err) {
      setError(err.message || 'Failed to fetch dashboard data.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchDashboardData();
  }, [fetchDashboardData]);

  // Derived real metrics from actual prediction records
  const totalInferences = predictions.length;
  const healthyCount = predictions.filter((p) => p.risk_level === 'LOW').length;
  const warningCount = predictions.filter((p) => p.risk_level === 'MODERATE').length;
  const criticalCount = predictions.filter(
    (p) => p.risk_level === 'CRITICAL' || p.prediction_class === 1
  ).length;

  // Unique devices observed in live predictions
  const uniqueDevicesObserved = new Set(
    predictions.map((p) => p.device_id).filter((id) => id !== null && id !== undefined)
  ).size;

  // Prepare chronological trend data from real prediction records (oldest to newest for plotting)
  const chartData = [...predictions]
    .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))
    .slice(-15) // Show last 15 inferences
    .map((p, idx) => {
      const dateObj = new Date(p.timestamp);
      const timeStr = dateObj.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      const dateStr = dateObj.toLocaleDateString([], { month: 'short', day: 'numeric' });
      return {
        index: idx + 1,
        timeLabel: `${dateStr} ${timeStr}`,
        riskScore: parseFloat(p.risk_score.toFixed(4)),
        threshold: p.threshold || 0.5,
        model: p.model_name,
        deviceRef: p.device_id ? `Device #${p.device_id}` : 'General',
      };
    });

  return (
    <div className="dashboard-container">
      {/* 1. System Health & Integration Status Header */}
      <section className="chart-card" style={{ padding: '16px 20px', marginBottom: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Server size={18} color="var(--primary)" />
              <strong style={{ fontSize: '0.9rem', color: 'var(--text-main)' }}>FastAPI Backend:</strong>
              {apiHealth ? (
                <span className="status-badge healthy" style={{ fontSize: '0.75rem' }}>
                  Online (v{apiHealth.version})
                </span>
              ) : (
                <span className="status-badge high-risk" style={{ fontSize: '0.75rem' }}>
                  Disconnected
                </span>
              )}
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Cpu size={18} color="var(--purple-accent, #9333ea)" />
              <strong style={{ fontSize: '0.9rem', color: 'var(--text-main)' }}>PyTorch Models:</strong>
              {modelHealth?.status === 'loaded' ? (
                <span className="status-badge healthy" style={{ fontSize: '0.75rem' }}>
                  LSTM Loaded (11 Features, H=2d)
                </span>
              ) : (
                <span className="status-badge warning" style={{ fontSize: '0.75rem' }}>
                  {modelHealth?.status || 'Awaiting Warmup'}
                </span>
              )}
            </div>
          </div>

          <button
            type="button"
            className="action-btn-placeholder"
            style={{ cursor: 'pointer', padding: '6px 14px' }}
            onClick={() => fetchDashboardData(true)}
            disabled={loading || refreshing}
            title="Refresh telemetry and prediction history from database"
            aria-label="Refresh dashboard data"
          >
            <RefreshCw size={13} className={refreshing ? 'spin-icon' : ''} />
            <span>{refreshing ? 'Refreshing...' : 'Refresh Data'}</span>
          </button>
        </div>
      </section>

      {/* 2. Error / Offline Banner */}
      {error && (
        <aside className="demo-notice-banner" style={{ backgroundColor: 'var(--status-danger-bg)', borderColor: 'var(--status-danger-border)' }} aria-label="Backend Connection Error">
          <AlertCircle size={18} color="var(--status-danger)" className="demo-notice-icon" />
          <div style={{ color: 'var(--status-danger)' }}>
            <strong>Backend Connection Notice:</strong> {error}
            <div style={{ fontSize: '0.775rem', marginTop: '4px', color: 'var(--text-secondary)' }}>
              Start the backend with: <code>python -m uvicorn src.main:app --host 127.0.0.1 --port 8000</code>
            </div>
          </div>
        </aside>
      )}

      {/* 3. Academic Prototype Disclosure Banner */}
      <aside className="demo-notice-banner" aria-label="Academic Prototype Notice">
        <Info size={16} className="demo-notice-icon" />
        <div>
          <span className="demo-notice-title">Academic Prototype:</span>
          Predictions are generated by PyTorch LSTM neural networks trained on historical Backblaze telemetry for 48-hour failure advance warning ($T=3$, $H=2$).
        </div>
      </aside>

      {/* 4. Four Summary Metric Cards */}
      <section className="stats-grid" aria-label="Hardware Health Summary">
        {/* Total Registered Drives */}
        <div className="stat-card">
          <div className="stat-icon-wrapper blue">
            <Server size={22} />
          </div>
          <div className="stat-details">
            <span className="stat-value">{loading ? '...' : deviceCount}</span>
            <span className="stat-label">Registered Drives</span>
            <span className="stat-trend blue" style={{ fontSize: '0.75rem' }}>
              SQLite Registry Table
            </span>
          </div>
        </div>

        {/* Total Inferences */}
        <div className="stat-card">
          <div className="stat-icon-wrapper green">
            <HardDrive size={22} />
          </div>
          <div className="stat-details">
            <span className="stat-value">{loading ? '...' : totalInferences}</span>
            <span className="stat-label">Total Inferences</span>
            <span className="stat-trend green" style={{ fontSize: '0.75rem' }}>
              <TrendingUp size={12} /> {healthyCount} Low Risk
            </span>
          </div>
        </div>

        {/* Needs Inspection / Moderate Risk */}
        <div className="stat-card">
          <div className="stat-icon-wrapper amber">
            <AlertTriangle size={22} />
          </div>
          <div className="stat-details">
            <span className="stat-value">{loading ? '...' : warningCount}</span>
            <span className="stat-label">Moderate Risk</span>
            <span className="stat-trend amber">
              <TrendingUp size={12} /> 0.20 &le; Prob &lt; 0.50
            </span>
          </div>
        </div>

        {/* Active Unacknowledged Alerts */}
        <div className="stat-card">
          <div className="stat-icon-wrapper red">
            <ShieldAlert size={22} />
          </div>
          <div className="stat-details">
            <span className="stat-value">{loading ? '...' : activeAlertsCount}</span>
            <span className="stat-label">Active Warnings</span>
            <span className="stat-trend red">
              <TrendingUp size={12} /> Pending Acknowledgement
            </span>
          </div>
        </div>
      </section>

      {/* 5. Failure Risk Trend Chart Area */}
      <section className="chart-card" aria-label="Risk Forecast Chart">
        <div className="chart-header">
          <div className="chart-title-area">
            <h2 className="chart-title">
              <Activity size={18} color="var(--primary)" />
              Recent Inference Trajectory — PyTorch LSTM
            </h2>
            <p className="chart-subtitle">
              Live sequential failure probabilities recorded in the SQLite database (Last 15 inferences)
            </p>
          </div>
          <div className="chart-legend-custom">
            <div className="legend-item">
              <span className="legend-dot lstm" />
              <span>LSTM Failure Probability</span>
            </div>
            <div className="legend-item">
              <span style={{ display: 'inline-block', width: '12px', height: '2px', backgroundColor: '#dc2626', marginRight: '4px' }} />
              <span>Decision Threshold (0.50)</span>
            </div>
          </div>
        </div>

        {chartData.length === 0 ? (
          <div className="smart-empty-container-compact" style={{ padding: '40px 20px' }}>
            <Activity size={28} color="var(--text-muted)" />
            <div>
              <strong style={{ display: 'block', color: 'var(--text-main)', marginBottom: '4px' }}>
                No Inference Data Recorded Yet
              </strong>
              <p className="smart-empty-desc">
                Execute a prediction on the <Link to="/predictions" style={{ color: 'var(--primary)', textDecoration: 'underline' }}>AI Predictions</Link> page to generate live sequential risk trajectories.
              </p>
            </div>
          </div>
        ) : (
          <div className="chart-container-inner">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                <XAxis
                  dataKey="timeLabel"
                  stroke="#94a3b8"
                  fontSize={11}
                  tickLine={false}
                  axisLine={{ stroke: '#e2e8f0' }}
                />
                <YAxis
                  domain={[0, 1]}
                  ticks={[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]}
                  stroke="#94a3b8"
                  fontSize={11}
                  tickLine={false}
                  axisLine={{ stroke: '#e2e8f0' }}
                  tickFormatter={(val) => val.toFixed(1)}
                />
                <ReferenceLine y={0.5} stroke="#dc2626" strokeDasharray="4 4" label={{ value: 'Critical Threshold (0.50)', fill: '#dc2626', fontSize: 10, position: 'insideTopRight' }} />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="chart-tooltip">
                          <div className="chart-tooltip-date">{data.timeLabel} ({data.deviceRef})</div>
                          <div className="chart-tooltip-row" style={{ color: 'var(--primary)' }}>
                            <span>LSTM Probability:</span>
                            <strong>{data.riskScore}</strong>
                          </div>
                          <div className="chart-tooltip-row" style={{ color: 'var(--text-muted)', fontSize: '0.72rem' }}>
                            <span>Model: {data.model}</span>
                          </div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="riskScore"
                  stroke="#2563eb"
                  strokeWidth={2.5}
                  dot={{ r: 4, fill: '#2563eb', strokeWidth: 1, stroke: '#ffffff' }}
                  activeDot={{ r: 6 }}
                  name="LSTM Probability"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </section>

      {/* 6. Recent Prediction Records Table */}
      <section className="table-card" aria-label="Recent Predictions Table">
        <div className="table-header">
          <div className="chart-title-area">
            <h2 className="table-title">
              <ShieldAlert size={18} color="var(--primary)" />
              Recent Inference History
            </h2>
            <p className="chart-subtitle">
              Live audit trail of model predictions recorded in the SQLite database
            </p>
          </div>
          <span className="table-badge-demo" style={{ backgroundColor: 'var(--primary-light)', color: 'var(--primary)' }}>
            Live SQLite Records
          </span>
        </div>

        {predictions.length === 0 ? (
          <div className="smart-empty-container-compact">
            <Info size={20} color="var(--text-muted)" />
            <p className="smart-empty-desc">
              No historical prediction records found in the database. Submit a prediction via the <Link to="/predictions" style={{ color: 'var(--primary)', textDecoration: 'underline' }}>AI Predictions page</Link>.
            </p>
          </div>
        ) : (
          <div className="table-wrapper">
            <table className="device-table">
              <thead>
                <tr>
                  <th>Timestamp (UTC)</th>
                  <th>Device Reference</th>
                  <th>Model</th>
                  <th>Risk Score</th>
                  <th>Risk Level</th>
                  <th>Failure Predicted</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {predictions.slice(0, 10).map((pred) => {
                  let badgeClass = 'healthy';
                  if (pred.risk_level === 'CRITICAL' || pred.prediction_class === 1) {
                    badgeClass = 'high-risk';
                  } else if (pred.risk_level === 'MODERATE') {
                    badgeClass = 'warning';
                  }

                  const formattedDate = new Date(pred.timestamp).toLocaleString();

                  return (
                    <tr key={pred.id}>
                      <td style={{ fontSize: '0.8rem', whiteSpace: 'nowrap' }}>{formattedDate}</td>
                      <td className="serial-cell">
                        {pred.device_id ? `Device #${pred.device_id}` : 'General Telemetry'}
                      </td>
                      <td>{pred.model_name}</td>
                      <td>
                        <span className={`risk-score-pill ${badgeClass === 'high-risk' ? 'high' : badgeClass === 'warning' ? 'warning' : 'healthy'}`}>
                          {pred.risk_score.toFixed(4)}
                        </span>
                      </td>
                      <td>
                        <span className={`status-badge ${badgeClass}`}>
                          {pred.risk_level}
                        </span>
                      </td>
                      <td>
                        {pred.prediction_class === 1 ? (
                          <span className="status-badge high-risk">CRITICAL (Yes)</span>
                        ) : (
                          <span className="status-badge healthy">HEALTHY (No)</span>
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
