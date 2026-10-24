import React, { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import {
  HardDrive,
  CheckCircle2,
  AlertTriangle,
  Search,
  RefreshCw,
  Server,
  Info,
  ShieldAlert,
  AlertCircle,
  Clock,
  ChevronLeft,
  ChevronRight,
  X
} from 'lucide-react';
import { apiClient } from '../services/apiClient';

export default function Devices() {
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);

  const [devices, setDevices] = useState([]);
  const [totalDevices, setTotalDevices] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(15);
  const [totalPages, setTotalPages] = useState(1);

  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('All');

  // Summary counts across all devices
  const [summaryCounts, setSummaryCounts] = useState({
    total: 0,
    healthy: 0,
    warning: 0,
    critical: 0,
  });

  const fetchDevices = useCallback(
    async (targetPage = 1, isRefresh = false) => {
      if (isRefresh) setRefreshing(true);
      else setLoading(true);
      setError(null);

      try {
        const response = await apiClient.getDevices({
          page: targetPage,
          pageSize,
          search: searchTerm,
          status: statusFilter,
        });

        setDevices(response.items || []);
        setTotalDevices(response.total || 0);
        setPage(response.page || 1);
        setTotalPages(response.total_pages || 1);

        // Fetch overall counts without search filter if viewing unfiltered page 1
        if (targetPage === 1 && !searchTerm.trim() && statusFilter === 'All') {
          const healthyRes = await apiClient.getDevices({ page: 1, pageSize: 1, status: 'HEALTHY' });
          const warningRes = await apiClient.getDevices({ page: 1, pageSize: 1, status: 'WARNING' });
          const criticalRes = await apiClient.getDevices({ page: 1, pageSize: 1, status: 'CRITICAL' });

          setSummaryCounts({
            total: response.total || 0,
            healthy: healthyRes.total || 0,
            warning: warningRes.total || 0,
            critical: criticalRes.total || 0,
          });
        }
      } catch (err) {
        setError(err.message || 'Failed to connect to backend device registry.');
      } finally {
        setLoading(false);
        setRefreshing(false);
      }
    },
    [pageSize, searchTerm, statusFilter]
  );

  useEffect(() => {
    fetchDevices(1);
  }, [fetchDevices]);

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    fetchDevices(1);
  };

  const handleClearSearch = () => {
    setSearchTerm('');
    // fetchDevices will re-run via effect dependency
  };

  return (
    <div className="devices-container">
      {/* 1. Header with Refresh */}
      <section className="chart-card" style={{ padding: '16px 20px', marginBottom: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <h2 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-main)', margin: 0 }}>
              Device Registry
            </h2>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: '2px 0 0' }}>
              Persistent storage drive inventory managed by FastAPI and SQLite (<code>devices</code> table)
            </p>
          </div>

          <button
            type="button"
            className="action-btn-placeholder"
            style={{ cursor: 'pointer', padding: '6px 14px' }}
            onClick={() => fetchDevices(page, true)}
            disabled={loading || refreshing}
            title="Refresh device records from database"
            aria-label="Refresh devices"
          >
            <RefreshCw size={13} className={refreshing ? 'spin-icon' : ''} />
            <span>{refreshing ? 'Refreshing...' : 'Refresh Devices'}</span>
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
              Cannot retrieve live device records while the FastAPI service is offline.
            </div>
          </div>
        </aside>
      )}

      {/* 3. Three Summary Metric Cards */}
      <section className="devices-stats-grid" aria-label="Device Summary Statistics">
        {/* Total Devices */}
        <div className="stat-card">
          <div className="stat-icon-wrapper blue">
            <Server size={22} />
          </div>
          <div className="stat-details">
            <span className="stat-value">{loading ? '...' : summaryCounts.total}</span>
            <span className="stat-label">Total Registered Drives</span>
            <span className="stat-trend blue">Stored in SQLite</span>
          </div>
        </div>

        {/* Healthy Devices */}
        <div className="stat-card">
          <div className="stat-icon-wrapper green">
            <CheckCircle2 size={22} />
          </div>
          <div className="stat-details">
            <span className="stat-value">{loading ? '...' : summaryCounts.healthy}</span>
            <span className="stat-label">Healthy Condition</span>
            <span className="stat-trend green">Status: HEALTHY</span>
          </div>
        </div>

        {/* Attention Recommended */}
        <div className="stat-card">
          <div className="stat-icon-wrapper amber">
            <AlertTriangle size={22} />
          </div>
          <div className="stat-details">
            <span className="stat-value">{loading ? '...' : summaryCounts.warning + summaryCounts.critical}</span>
            <span className="stat-label">Elevated Risk</span>
            <span className="stat-trend amber">WARNING or CRITICAL</span>
          </div>
        </div>
      </section>

      {/* 4. Search and Filters Toolbar */}
      <section className="devices-filters-card" aria-label="Search and Filter Controls">
        <form onSubmit={handleSearchSubmit} className="filter-inputs-group">
          {/* Search Input */}
          <div className="search-box-wrapper" style={{ flex: 1 }}>
            <Search size={16} className="search-icon-inside" />
            <input
              type="text"
              className="search-input-field"
              placeholder="Search by serial number (e.g. ST4000, W30)..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              aria-label="Search by serial number"
            />
            {searchTerm && (
              <button
                type="button"
                className="search-clear-btn"
                onClick={handleClearSearch}
                aria-label="Clear search input"
              >
                <X size={14} />
              </button>
            )}
          </div>

          {/* Status Filter Dropdown */}
          <select
            className="filter-select-field"
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            aria-label="Filter by operational status"
          >
            <option value="All">All Operational Statuses</option>
            <option value="HEALTHY">HEALTHY</option>
            <option value="WARNING">WARNING</option>
            <option value="CRITICAL">CRITICAL</option>
            <option value="FAILED">FAILED</option>
          </select>

          <button
            type="submit"
            className="action-btn-placeholder"
            style={{ backgroundColor: 'var(--primary)', color: '#ffffff', padding: '8px 16px', fontWeight: 600 }}
          >
            Search
          </button>
        </form>
      </section>

      {/* 5. Devices Registry Table */}
      <section className="table-card" aria-label="Device Registry Table">
        <div className="table-header">
          <div className="chart-title-area">
            <h2 className="table-title">
              <HardDrive size={18} color="var(--primary)" />
              Storage Drive Registry
            </h2>
            <p className="chart-subtitle">
              Live device entities persisted in SQLite with first and last telemetry observation timestamps
            </p>
          </div>
          <span className="table-badge-demo" style={{ backgroundColor: 'var(--primary-light)', color: 'var(--primary)' }}>
            Live SQLite Records ({totalDevices} Total)
          </span>
        </div>

        {loading ? (
          <div className="smart-empty-container-compact">
            <Clock size={20} className="spin-icon" color="var(--primary)" />
            <p className="smart-empty-desc">Loading device registry from backend...</p>
          </div>
        ) : devices.length === 0 ? (
          <div className="smart-empty-container-compact" style={{ padding: '36px 20px' }}>
            <HardDrive size={28} color="var(--text-muted)" />
            <div>
              <strong style={{ display: 'block', color: 'var(--text-main)', marginBottom: '4px' }}>
                No Devices Found
              </strong>
              <p className="smart-empty-desc">
                {searchTerm || statusFilter !== 'All'
                  ? 'No registered drives match the current search or status filter.'
                  : 'The device registry is empty. Submit 3-day telemetry on the AI Predictions page to register a drive.'}
              </p>
            </div>
          </div>
        ) : (
          <>
            <div className="table-wrapper">
              <table className="device-table">
                <thead>
                  <tr>
                    <th>Serial Number</th>
                    <th>Hardware Model</th>
                    <th>Operational Status</th>
                    <th>First Telemetry (UTC)</th>
                    <th>Latest Telemetry (UTC)</th>
                    <th>Database ID</th>
                  </tr>
                </thead>
                <tbody>
                  {devices.map((device) => {
                    let badgeClass = 'healthy';
                    if (device.status === 'CRITICAL' || device.status === 'FAILED') {
                      badgeClass = 'high-risk';
                    } else if (device.status === 'WARNING') {
                      badgeClass = 'warning';
                    }

                    return (
                      <tr key={device.id}>
                        <td className="serial-cell" style={{ fontWeight: 600 }}>
                          {device.serial_number}
                        </td>
                        <td>{device.model || 'Unknown Model'}</td>
                        <td>
                          <span className={`status-badge ${badgeClass}`}>
                            {device.status}
                          </span>
                        </td>
                        <td style={{ fontSize: '0.8rem', whiteSpace: 'nowrap' }}>
                          {new Date(device.first_seen).toLocaleString()}
                        </td>
                        <td style={{ fontSize: '0.8rem', whiteSpace: 'nowrap' }}>
                          {new Date(device.last_seen).toLocaleString()}
                        </td>
                        <td style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                          #{device.id}
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
                  Showing page <strong>{page}</strong> of <strong>{totalPages}</strong> ({totalDevices} total)
                </span>

                <div style={{ display: 'flex', gap: '8px' }}>
                  <button
                    type="button"
                    className="action-btn-placeholder"
                    onClick={() => fetchDevices(page - 1)}
                    disabled={page <= 1 || loading}
                    style={{ padding: '6px 12px', opacity: page <= 1 ? 0.5 : 1 }}
                  >
                    <ChevronLeft size={14} />
                    <span>Previous</span>
                  </button>

                  <button
                    type="button"
                    className="action-btn-placeholder"
                    onClick={() => fetchDevices(page + 1)}
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
