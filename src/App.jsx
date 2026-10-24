import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import MainLayout from './layouts/MainLayout';
import Dashboard from './pages/Dashboard';
import Devices from './pages/Devices';
import Predictions from './pages/Predictions';
import Alerts from './pages/Alerts';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MainLayout />}>
          {/* Root route redirects to /dashboard */}
          <Route index element={<Navigate to="/dashboard" replace />} />
          
          {/* Exactly 4 application routes */}
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="devices" element={<Devices />} />
          <Route path="predictions" element={<Predictions />} />
          <Route path="alerts" element={<Alerts />} />
          
          {/* Fallback for any unknown route */}
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
