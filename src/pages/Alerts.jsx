import React, { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import {
  ShieldAlert,
  AlertTriangle,
  Bell,
  Info,
  ShieldCheck,
  RefreshCw,
  Clock,
  AlertCircle,
  CheckCircle2,
  Check,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';
import { apiClient } from '../services/apiClient';

export default function Alerts() {
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);

  const [alerts, setAlerts] = useState([]);
  const [totalAlerts, setTotalAlerts] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(15);
  const [totalPages, setTotalPages] = useState(1);

  const [severityFilter, setSeverityFilter] = useState('All');
  const [ackFilter, setAckFilter] = useState('All'); // 'All', 'false', 'true'
  const [acknowledgingId, setAcknowledgingId] = useState(null);

  // Overall metric counts across SQLite alert_records
  const [summaryCounts, setSummaryCounts] = useState({
    total: 0,
    unacknowledged: 0,
    critical: 0,
  });

  const fetchAlerts = useCallback(
    async (targetPage = 1, isRefresh = false) => {
      if (isRefresh) setRefreshing(true);
      else setLoading(true);
      setError(null);

      try {
        const ackParam = ackFilter === 'All' ? null : ackFilter === 'true';
        const response = await apiClient.getAlerts({
          page: targetPage,
          pageSize,
          severity: severityFilter,
          acknowledged: ackParam,
        });

        setAlerts(response.items || []);
        setTotalAlerts(response.total || 0);
        setPage(response.page || 1);
        setTotalPages(response.total_pages || 1);

        // Fetch counts without filters for top summary cards
        if (targetPage === 1 && severityFilter === 'All' && ackFilter === 'All') {
          const unackRes = await apiClient.getAlerts({ page: 1, pageSize: 1, acknowledged: false });
          const critRes = await apiClient.getAlerts({ page: 1, pageSize: 1, severity: 'CRITICAL' });

          setSummaryCounts({
            total: response.total || 0,
            unacknowledged: unackRes.total || 0,
            critical: critRes.total || 0,
          });
        }
      } catch (err) {
        setError(err.message || 'Failed to fetch hardware failure alerts.');
      } finally {
        setLoading(false);
        setRefreshing(false);
      }
    },
    [pageSize, severityFilter, ackFilter]
  );

  useEffect(() => {
    fetchAlerts(1);
  }, [fetchAlerts]);

  // Real interactive alert acknowledgement persisting to SQLite
  const handleAcknowledge = async (alertId) => {
    if (acknowledgingId) return;
    setAcknowledgingId(alertId);
    setError(null);
    setSuccessMessage(null);

    try {
      const res = await apiClient.acknowledgeAlert(alertId);

      // Update state locally upon backend confirmation
      setAlerts((prev) =>
        prev.map((a) =>
          a.id === alertId
            ? { ...a, acknowledged: true, acknowledged_at: res.acknowledged_at }
            : a
        )
      );

      setSummaryCounts((prev) => ({
        ...prev,
        unacknowledged: Math.max(0, prev.unacknowledged - 1),
      }));

      setSuccessMessage(res.message || `Alert #${alertId} acknowledged.`);
      setTimeout(() => setSuccessMessage(null), 4000);
    } catch (err) {
      setError(err.message || `Failed to acknowledge Alert #${alertId}.`);
    } finally {
      setAcknowledgingId(null);
    }
  };

  return (
    <div className="alerts-container">
      {/* 1. Header with Refresh */}
      <section className="chart-card" style={{ padding: '16px 20px', marginBottom: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <h2 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-main)', margin: 0 }}>
              Hardware Prevention &amp; Alert Queue
            </h2>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: '2px 0 0' }}>
              Actionable failure alerts stored in SQLite (<code>alert_records</code> table) with operator acknowledgement lifecycle
            </p>
          </div>

          <button
            type="button"
            className="action-btn-placeholder"
            style={{ cursor: 'pointer', padding: '6px 14px' }}
            onClick={() => fetchAlerts(page, true)}
            disabled={loading || refreshing}
            title="Refresh alerts from database"
            aria-label="Refresh alerts"
          >
            <RefreshCw size={13} className={refreshing ? 'spin-icon' : ''} />
            <span>{refreshing ? 'Refreshing...' : 'Refresh Alerts'}</span>
          </button>
        </div>
      </section>

      {/* 2. Success Banner */}
      {successMessage && (
        <aside className="demo-notice-banner" style={{ backgroundColor: 'var(--status-healthy-bg)', borderColor: 'var(--status-healthy-border)', color: 'var(--status-healthy)' }} aria-label="Success Notification">
          <CheckCircle2 size={18} className="demo-notice-icon" />
          <span>{successMessage}</span>
        </aside>
      )}

      {/* 3. Error Banner */}
      {error && (
        <aside className="demo-notice-banner" style={{ backgroundColor: 'var(--status-danger-bg)', borderColor: 'var(--status-danger-border)' }} aria-label="Alerts Error Notification">
          <AlertCircle size={18} color="var(--status-danger)" className="demo-notice-icon" />
          <div style={{ color: 'var(--status-danger)' }}>
            <strong>Alert Error:</strong> {error}
          </div>
        </aside>
      )}

      {/* 4. Three Summary Cards */}
      <section className="alerts-stats-grid" aria-label="Alerts Summary Statistics">
        {/* Total Alerts */}
        <div className="stat-card">
          <div className="stat-icon-wrapper blue">
            <Bell size={22} />
          </div>
          <div className="stat-details">
            <span className="stat-value">{loading ? '...' : summaryCounts.total}</span>
            <span className="stat-label">Total Generated Alerts</span>
            <span className="stat-trend blue">Stored in SQLite</span>
          </div>
        </div>

        {/* Unacknowledged Alerts */}
        <div className="stat-card">
          <div className="stat-icon-wrapper amber">
            <AlertTriangle size={22} />
          </div>
          <div className="stat-details">
            <span className="stat-value">{loading ? '...' : summaryCounts.unacknowledged}</span>
            <span className="stat-label">Unacknowledged Warnings</span>
            <span className="stat-trend amber">Pending Operator Review</span>
          </div>
        </div>

        {/* Critical Risk */}
        <div className="stat-card">
          <div className="stat-icon-wrapper red">
            <ShieldAlert size={22} />
          </div>
          <div className="stat-details">
            <span className="stat-value">{loading ? '...' : summaryCounts.critical}</span>
            <span className="stat-label">Critical Severity Alerts</span>
            <span className="stat-trend red">Prob &ge; 0.50 (48h Lead Time)</span>
          </div>
        </div>
      </section>

      {/* 5. Filter Toolbar */}
      <section className="devices-filters-card" aria-label="Alert Filter Controls">
        <div className="filter-inputs-group">
          {/* Severity Filter */}
          <select
            className="filter-select-field"
            value={severityFilter}
            onChange={(e) => setSeverityFilter(e.target.value)}
            aria-label="Filter by severity level"
          >
            <option value="All">All Severities</option>
            <option value="CRITICAL">CRITICAL Only</option>
            <option value="WARNING">WARNING Only</option>
            <option value="INFO">INFO Only</option>
          </select>

          {/* Acknowledgement Status Filter */}
          <select
            className="filter-select-field"
            value={ackFilter}
            onChange={(e) => setAckFilter(e.target.value)}
            aria-label="Filter by acknowledgement status"
          >
            <option value="All">All Acknowledgement States</option>
            <option value="false">Unacknowledged Only</option>
            <option value="true">Acknowledged Only</option>
          </select>
        </div>
      </section>

      {/* 6. Active Alerts Table */}
      <section className="alerts-table-card" aria-label="Failure Alerts Table">
        <div className="alerts-table-header">
          <div className="chart-title-area">
            <h2 className="table-title">
              <ShieldAlert size={18} color="var(--primary)" />
              Failure Prevention &amp; Alert Queue
            </h2>
            <p className="chart-subtitle">
              Prioritized device warnings and actionable preventative recommendations
            </p>
          </div>
          <span className="table-badge-demo" style={{ backgroundColor: 'var(--primary-light)', color: 'var(--primary)' }}>
            Live SQLite Records ({totalAlerts} Total)
          </span>
        </div>

        {loading ? (
          <div className="smart-empty-container-compact">
            <Clock size={20} className="spin-icon" color="var(--primary)" />
            <p className="smart-empty-desc">Loading alerts queue from database...</p>
          </div>
        ) : alerts.length === 0 ? (
          <div className="devices-empty-state" style={{ padding: '36px 20px', textAlign: 'center' }}>
            <div className="empty-icon-box" style={{ margin: '0 auto 12px' }}>
              <ShieldCheck size={32} color="var(--status-healthy)" />
            </div>
            <h3 className="empty-state-title" style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-main)', marginBottom: '4px' }}>
              No Active Alerts Found
            </h3>
            <p className="empty-state-desc" style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', maxWidth: '500px', margin: '0 auto' }}>
              {severityFilter !== 'All' || ackFilter !== 'All'
                ? 'No alerts match the selected severity or acknowledgement filters.'
                : 'All monitored hard drives in the database are operating within nominal parameters. Proactive alerts are generated when telemetry indicates elevated degradation.'}
            </p>
          </div>
        ) : (
          <>
            <div className="table-wrapper">
              <table className="device-table">
                <thead>
                  <tr>
                    <th>Alert ID</th>
                    <th>Device Serial / Model</th>
                    <th>Trigger Timestamp (UTC)</th>
                    <th>Severity</th>
                    <th>Actionable Preventative Recommendation</th>
                    <th>Operator Acknowledgement</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {alerts.map((alert) => {
                    const isCritical = alert.severity === 'CRITICAL';
                    const badgeClass = isCritical ? 'high-risk' : 'warning';
                    const isPendingAck = acknowledgingId === alert.id;

                    return (
                      <tr key={alert.id}>
                        <td style={{ fontWeight: 600, color: 'var(--text-main)' }}>#{alert.id}</td>
                        <td className="serial-cell">
                          <div>{alert.device_serial || `Device #${alert.device_id || 'Unknown'}`}</div>
                          {alert.device_model && (
                            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{alert.device_model}</div>
                          )}
                        </td>
                        <td style={{ fontSize: '0.8rem', whiteSpace: 'nowrap' }}>
                          {new Date(alert.timestamp).toLocaleString()}
                        </td>
                        <td>
                          <span className={`status-badge ${badgeClass}`}>
                            {alert.severity}
                          </span>
                        </td>
                        <td className="recommended-action-text" style={{ maxWidth: '380px' }}>
                          {alert.recommendation}
                        </td>
                        <td>
                          {alert.acknowledged ? (
                            <span className="status-badge resolved" style={{ fontSize: '0.725rem' }}>
                              <Check size={12} style={{ marginRight: '4px' }} />
                              Acknowledged
                              {alert.acknowledged_at && (
                                <div style={{ fontSize: '0.675rem', color: 'var(--text-muted)' }}>
                                  {new Date(alert.acknowledged_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                </div>
                              )}
                            </span>
                          ) : (
                            <span className="status-badge open" style={{ fontSize: '0.725rem' }}>
                              Unacknowledged
                            </span>
                          )}
                        </td>
                        <td>
                          {!alert.acknowledged ? (
                            <button
                              type="button"
                              className="action-btn-placeholder"
                              style={{
                                backgroundColor: 'var(--primary)',
                                color: '#ffffff',
                                border: 'none',
                                padding: '6px 12px',
                                fontSize: '0.75rem',
                                fontWeight: 600,
                                opacity: isPendingAck ? 0.6 : 1,
                              }}
                              onClick={() => handleAcknowledge(alert.id)}
                              disabled={isPendingAck}
                              title={`Acknowledge Alert #${alert.id}`}
                              aria-label={`Acknowledge Alert #${alert.id}`}
                            >
                              {isPendingAck ? (
                                <>
                                  <Clock size={12} className="spin-icon" />
                                  <span>Saving...</span>
                                </>
                              ) : (
                                <>
                                  <Check size={12} />
                                  <span>Acknowledge</span>
                                </>
                              )}
                            </button>
                          ) : (
                            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Verified</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Pagination Controls */}
            {totalPages > 1 && (
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  paddingTop: '16px',
                  marginTop: '12px',
                  borderTop: '1px solid var(--border-color)',
                  fontSize: '0.8rem',
                  color: 'var(--text-secondary)',
                }}
              >
                <span>
                  Showing page <strong>{page}</strong> of <strong>{totalPages}</strong> ({totalAlerts} total)
                </span>

                <div style={{ display: 'flex', gap: '8px' }}>
                  <button
                    type="button"
                    className="action-btn-placeholder"
                    onClick={() => fetchAlerts(page - 1)}
                    disabled={page <= 1 || loading}
                    style={{ padding: '6px 12px', opacity: page <= 1 ? 0.5 : 1 }}
                  >
                    <ChevronLeft size={14} />
                    <span>Previous</span>
                  </button>

                  <button
                    type="button"
                    className="action-btn-placeholder"
                    onClick={() => fetchAlerts(page + 1)}
                    disabled={page >= totalPages || loading}
                    style={{ padding: '6px 12px', opacity: page >= totalPages ? 0.5 : 1 }}
                  >
                    <span>Next</span>
                    <ChevronRight size={14} />
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </section>
    </div>
  );
}
