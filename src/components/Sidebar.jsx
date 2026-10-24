import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  HardDrive,
  LayoutDashboard,
  Cpu,
  ShieldAlert,
  Server,
  X
} from 'lucide-react';

const navItems = [
  {
    path: '/dashboard',
    label: 'Dashboard',
    icon: LayoutDashboard,
  },
  {
    path: '/devices',
    label: 'Devices',
    icon: HardDrive,
  },
  {
    path: '/predictions',
    label: 'AI Predictions',
    icon: Cpu,
  },
  {
    path: '/alerts',
    label: 'Prevention & Alerts',
    icon: ShieldAlert,
  },
];

export default function Sidebar({ mobileOpen, onCloseMobile }) {
  return (
    <>
      {/* Mobile backdrop */}
      <div
        className={`sidebar-backdrop ${mobileOpen ? 'mobile-open' : ''}`}
        onClick={onCloseMobile}
        aria-hidden="true"
      />

      <aside className={`sidebar ${mobileOpen ? 'mobile-open' : ''}`}>
        {/* Brand Header */}
        <div className="sidebar-header">
          <div className="sidebar-logo-icon" aria-label="DeviceGuard Logo">
            <Server size={20} strokeWidth={2.2} />
          </div>
          <div className="sidebar-logo-text">
            <span className="sidebar-logo-title">DeviceGuard AI</span>
            <span className="sidebar-logo-subtitle">Hard Drive Health Prediction</span>
          </div>
          {mobileOpen && (
            <button
              className="mobile-menu-btn"
              style={{ marginLeft: 'auto' }}
              onClick={onCloseMobile}
              aria-label="Close sidebar"
            >
              <X size={18} />
            </button>
          )}
        </div>

        {/* Navigation Links */}
        <nav className="sidebar-nav" aria-label="Main Navigation">
          {navItems.map((item) => {
            const IconComponent = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  `nav-item ${isActive ? 'active' : ''}`
                }
                onClick={onCloseMobile}
              >
                <IconComponent size={18} className="nav-icon" />
                <span>{item.label}</span>
              </NavLink>
            );
          })}
        </nav>

        {/* Sidebar Footer info */}
        <div className="sidebar-footer">
          <div>Deep Learning Lab System</div>
          <div className="sidebar-footer-badge">
            <Cpu size={13} />
            <span>LSTM &amp; GRU Architecture</span>
          </div>
        </div>
      </aside>
    </>
  );
}
