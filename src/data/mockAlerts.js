// Demonstration alerts dataset for Prevention & Alerts page UI evaluation
// NOTE: Simulated demonstration records only. Does not represent actual hardware conditions or live predictions.

export const demoAlertsSummary = {
  totalAlerts: 260,
  needsInspection: 180,
  highRisk: 80,
};

export const demoAlertsList = [
  {
    id: 'ALT-101',
    serial: 'W3029198620K',
    model: 'Hitachi HDS5C3',
    riskLevel: 'High Risk',
    recommendedAction: 'Back up important data and inspect drive.',
    status: 'Open',
    date: '25 Oct 2026',
  },
  {
    id: 'ALT-102',
    serial: 'PL1331LAG944W',
    model: 'Hitachi HDS5C4',
    riskLevel: 'High Risk',
    recommendedAction: 'Schedule drive replacement.',
    status: 'Open',
    date: '24 Oct 2026',
  },
  {
    id: 'ALT-103',
    serial: 'S30026H1',
    model: 'ST4000DM000',
    riskLevel: 'High Risk',
    recommendedAction: 'Back up important data immediately.',
    status: 'In Progress',
    date: '24 Oct 2026',
  },
  {
    id: 'ALT-104',
    serial: 'PL2331LAG0725U',
    model: 'HGST HMS5C40',
    riskLevel: 'Needs Inspection',
    recommendedAction: 'Review drive health indicators.',
    status: 'Open',
    date: '23 Oct 2026',
  },
  {
    id: 'ALT-105',
    serial: 'PL1331LAG9344K',
    model: 'HGST HMS5C40',
    riskLevel: 'Needs Inspection',
    recommendedAction: 'Schedule a hardware inspection.',
    status: 'In Progress',
    date: '22 Oct 2026',
  },
  {
    id: 'ALT-106',
    serial: 'W3018902123M',
    model: 'Hitachi HDS5C3',
    riskLevel: 'High Risk',
    recommendedAction: 'Perform offline diagnostic verification.',
    status: 'Open',
    date: '20 Oct 2026',
  },
  {
    id: 'ALT-107',
    serial: 'Z3032SH1',
    model: 'ST4000DM000',
    riskLevel: 'Low Risk',
    recommendedAction: 'Routine monitoring; normal operation.',
    status: 'Resolved',
    date: '20 Oct 2026',
  },
];
