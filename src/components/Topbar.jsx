import React from 'react';
import { Menu, Bell } from 'lucide-react';

export default function Topbar({ title, subtitle, onToggleMobile }) {
  return (
    <header className="topbar">
      <div className="topbar-left">
        <button
          className="mobile-menu-btn"
          onClick={onToggleMobile}
          aria-label="Toggle navigation menu"
        >
          <Menu size={20} />
        </button>
        <div className="page-header-text">
          <h1 className="page-title">{title}</h1>
          {subtitle && <p className="page-subtitle">{subtitle}</p>}
        </div>
      </div>

      <div className="topbar-right">
        {/* Notification Bell (Placeholder) */}
        <button
          type="button"
          className="topbar-action-btn"
          title="Notifications"
          aria-label="Notifications"
        >
          <Bell size={18} />
          <span className="notification-dot" />
        </button>

        {/* User Profile Badge (Placeholder) */}
        <div className="user-profile-badge">
          <div className="user-avatar" aria-label="User avatar">
            AS
          </div>
          <div className="user-info">
            <span className="user-name">Arfa Saniya</span>
            <span className="user-role">Admin</span>
          </div>
        </div>
      </div>
    </header>
  );
}
