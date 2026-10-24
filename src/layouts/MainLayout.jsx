import React, { useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import Sidebar from '../components/Sidebar';
import Topbar from '../components/Topbar';

const routeMeta = {
  '/dashboard': {
    title: 'Dashboard',
    subtitle: 'Overview of computer lab hard drive health and predictions',
  },
  '/devices': {
    title: 'Devices',
    subtitle: 'View and manage computer lab hard drives',
  },
  '/predictions': {
    title: 'AI Predictions',
    subtitle: 'View hard-drive failure prediction demonstrations.',
  },
  '/alerts': {
    title: 'Prevention & Alerts',
    subtitle: 'Review device warnings and recommended preventive actions.',
  },
};

export default function MainLayout() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();

  const currentMeta = routeMeta[location.pathname] || {
    title: 'DeviceGuard AI',
    subtitle: 'Hard Drive Health Prediction System',
  };

  const handleToggleMobile = () => {
    setMobileOpen((prev) => !prev);
  };

  const handleCloseMobile = () => {
    setMobileOpen(false);
  };

  return (
    <div className="app-layout">
      <Sidebar
        mobileOpen={mobileOpen}
        onCloseMobile={handleCloseMobile}
      />
      <div className="main-wrapper">
        <Topbar
          title={currentMeta.title}
          subtitle={currentMeta.subtitle}
          onToggleMobile={handleToggleMobile}
        />
        <main className="page-content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
