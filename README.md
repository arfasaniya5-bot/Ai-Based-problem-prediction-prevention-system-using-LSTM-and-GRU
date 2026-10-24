# DeviceGuard AI — Frontend Application

AI-Based Problem Prediction & Prevention System Using LSTM and GRU for computer hard drive failure forecasting.

## Overview

DeviceGuard AI is an academic Deep Learning project aimed at proactively monitoring hard drive telemetry (SMART attributes) to forecast failure risks and prescribe preventive actions before catastrophic hardware loss occurs.

This repository contains the **complete React frontend UI**:
- **Lightweight, clean academic UI** built with React.js and Vite following an accessible blue-and-white theme.
- **Responsive application layout** with persistent left sidebar, dynamic top navigation bar, and mobile drawer support.
- **Four core navigation routes**:
  - `/dashboard` — Health counters, comparative LSTM vs GRU 30-day failure risk trend chart, recent high-risk drives table.
  - `/devices` — Inventory summary, live local search & filter toolbar (by risk level & model), responsive hardware table, and empty state.
  - `/predictions` — Device selector, detailed hardware profile, neutral LSTM and GRU model status cards, and SMART trends placeholder.
  - `/alerts` — Prevention & Alerts queue, 3 compact metric cards (Total Alerts, Needs Inspection, High Risk), and prioritized actions table.
- **Demonstration Data Isolation**: All mock data is clearly separated in `src/data/mockDevices.js` and `src/data/mockAlerts.js` with prominent demonstration notice banners.
- **Strict Scope Control**: No backend, database, dataset processing, or real ML inference is executed in this frontend phase.

---

## Technology Stack

- **React.js** (v18.3.1)
- **Vite** (v6.0.3)
- **React Router DOM** (v6.28.0)
- **Recharts** (v2.15.0) for lightweight trend charting
- **Lucide React** (v0.468.0) for standard iconography
- **CSS** (Custom responsive design system with academic styling)

---

## Project Structure

```text
DL_PROJECT/
├── public/
├── src/
│   ├── components/
│   │   ├── Sidebar.jsx       # Left navigation bar with logo and nav items
│   │   └── Topbar.jsx        # Top application bar with route info and user badge
│   ├── data/
│   │   ├── mockDevices.js    # Demonstration devices and summary metrics
│   │   └── mockAlerts.js     # Demonstration alerts and prevention actions
│   ├── layouts/
│   │   └── MainLayout.jsx    # Responsive shell combining Sidebar, Topbar, and Outlet
│   ├── pages/
│   │   ├── Dashboard.jsx     # /dashboard - Overview, trend chart & high-risk table
│   │   ├── Devices.jsx       # /devices - Filterable hard-drive inventory
│   │   ├── Predictions.jsx   # /predictions - Model inference cards & SMART trends
│   │   └── Alerts.jsx        # /alerts - Failure prevention & alerts queue
│   ├── styles/
│   │   └── global.css        # Academic light theme design tokens and responsive CSS
│   ├── App.jsx               # Router setup, route definitions, and root redirect
│   └── main.jsx              # React DOM entry point
├── index.html                # Application HTML template
├── package.json              # Project metadata, scripts, and dependencies
├── vite.config.js            # Vite configuration
├── .gitignore                # Git ignore configuration
└── README.md                 # Setup and run instructions
```

---

## Getting Started

### 1. Prerequisites
Ensure you have Node.js (v18+) and npm installed on your machine:
```bash
node -v
npm -v
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Run Development Server
To launch the Vite development server locally:
```bash
npm run dev
```
By default, the application will be accessible at:
```text
http://localhost:3000/
```

### 4. Build for Production
To verify and create a production build:
```bash
npm run build
```

To preview the production build locally:
```bash
npm run preview
```

---

## Application Routes

| Route Path | View Name | Description |
|---|---|---|
| `/` | *Redirect* | Automatically redirects to `/dashboard` |
| `/dashboard` | Dashboard | Summary metrics, comparative LSTM vs GRU chart, and priority risk table |
| `/devices` | Devices | Searchable and filterable laboratory hard-drive inventory |
| `/predictions` | AI Predictions | Device selector, hardware profile, neutral LSTM/GRU cards, and SMART trends placeholder |
| `/alerts` | Prevention & Alerts | Priority warnings, alert counters, and recommended preventive action queue |
