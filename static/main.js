// ─────────────────────────────────────────────────────────────────────────────
//  SATMON  —  main.js
// ─────────────────────────────────────────────────────────────────────────────

// ── Feature groups & defaults ────────────────────────────────────────────────
const GROUPS = {
  'g-power':   ['BusVoltage (V)','BusCurrent (A)'],
  'g-battery': ['BatteryVoltage (V)','BatteryTemperature (°C)','BatterySOC (%)'],
  'g-solar':   ['SolarVoltage (V)','SolarCurrent (A)','Sunlight (0 or 1)'],
  'g-attitude':['WheelRPM (RPM)','WheelTemperature (°C)','GyroMagnitude (deg/s)'],
  'g-obc':     ['CPUUsage (%)','CPUTemperature (°C)','SignalStrength (dBm)'],
  'g-orbital': ['OrbitPhase (%)','Altitude (km)'],
};

const DEFAULTS = {
  'OrbitPhase (%)':45,'Sunlight (0 or 1)':1,'BusVoltage (V)':28.5,
  'BusCurrent (A)':3.2,'BatteryVoltage (V)':3.7,'BatteryTemperature (°C)':22,
  'BatterySOC (%)':85,'SolarVoltage (V)':32,'SolarCurrent (A)':2.5,
  'WheelRPM (RPM)':5000,'WheelTemperature (°C)':35,'CPUUsage (%)':40,
  'CPUTemperature (°C)':55,'SignalStrength (dBm)':-80,
  'GyroMagnitude (deg/s)':0.05,'Altitude (km)':500,
};

const PAGE_META = {
  dashboard: ['Overview',  'Dashboard',           'Live virtual replica of satellite state, fault prediction and explainability.'],
  subsystems:['Overview',  'Subsystem Health',    'Per-subsystem status derived from the most recent telemetry submission.'],
  analytics: ['Overview',  'Analytics',           'Confidence trend, fault distribution and session statistics.'],
  alerts:    ['Overview',  'Alerts',              'Auto-generated when a warning or critical fault is detected.'],
  manual:    ['Input',     'Manual Input',        'Enter 16 telemetry parameters and run an anomaly scan.'],
  simulate:  ['Input',     'Live Simulation',     'Continuously drifting synthetic telemetry scanned by XGBoost.'],
  quickload: ['Input',     'Quick Load',          'Load a preset scenario or randomize telemetry for fast testing.'],
  external:  ['Input',     'External Source',     'Live satellite data ingestion — currently inactive.'],
  result:    ['AI Results','Classification',      'XGBoost output of the most recent anomaly scan.'],
  xai:       ['AI Results','SHAP Analysis',       'Feature-level explanation via SHAP TreeExplainer.'],
  lime:      ['AI Results','LIME Analysis',       'Local interpretable model-agnostic explanations.'],
  compare:   ['AI Results','Model Comparison',    'XGBoost vs RandomForest on the same input.'],
  rul:       ['AI Results','Remaining Useful Life','Linear regression projection of parameter degradation.'],
  recommend: ['AI Results','Recommendations',     'Suggested operator response to the detected fault.'],
  timeline:  ['Records',   'Mission Timeline',    'Visual event log of all predictions this session.'],
  history:   ['Records',   'Prediction History',  'All anomaly scans run during this session.'],
  export:    ['Records',   'Export',              'Download full JSON including SHAP and LIME values.'],
};

const SEV_ICON  = { nominal:'✓', warning:'⚠', critical:'✕' };
const SEV_LABEL = { nominal:'Normal', warning:'Warning', critical:'Critical' };

let RANGES = {}, PRESETS = {}, LAST = null, HIST = [], MODEL = 'xgb';
let SIM_IV = null, SIM_ON = false, SIM_T = 0;

// ── Live chart ───────────────────────────────────────────────────────────────
const CHART_MAX = 60;
const CHART_SERIES = {
  'BusVoltage (V)':       { color:'#3b82f6', label:'Bus Voltage (V)',   ymin:10,   ymax:35   },
  'BatterySOC (%)':       { color:'#22c55e', label:'Battery SOC (%)',  ymin:0,    ymax:100  },
  'CPUTemperature (°C)':  { color:'#f59e0b', label:'CPU Temp (°C)',    ymin:0,    ymax:140  },
  'SignalStrength (dBm)': { color:'#ef4444', label:'Signal Str (dBm)', ymin:-120, ymax:-30  },
};
let CHART_BUF    = [];
let CHART_PAUSED = false;
let CHART_ACTIVE = Object.fromEntries(Object.keys(CHART_SERIES).map(k => [k, true]));

// ── Clock ────────────────────────────────────────────────────────────────────
setInterval(() => {
  const el = document.getElementById('clk');
  if (el) el.textContent = new Date().toUTCString().split(' ')[4] + ' UTC';
}, 1000);

// ── Build series toggle buttons ──────────────────────────────────────────────
const seriesContainer = document.getElementById('seriesBtns');
if (seriesContainer) {
  Object.entries(CHART_SERIES).forEach(([key, meta]) => {
    const btn = document.createElement('button');
    btn.className = 'series-btn';
    btn.style.setProperty('--sc', meta.color);
    btn.textContent = meta.label;
    btn.id = 'sbtn_' + key.replace(/[^a-zA-Z0-9]/g, '_');
    btn.onclick = () => toggleSeries(key);
    seriesContainer.appendChild(btn);
  });
}

// ── Navigation ───────────────────────────────────────────────────────────────
function go(page) {
  document.querySelectorAll('.page').forEach(e => e.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(e => e.classList.remove('active'));

  const pg = document.getElementById('page-' + page);
  if (pg) pg.classList.add('active');
  const nb = document.querySelector(`.nav-btn[data-page="${page}"]`);
  if (nb) nb.classList.add('active');

  // Dashboard uses the banner as its header — hide the global text header
  const hdr = document.getElementById('global-page-hdr');
  if (hdr) hdr.style.display = page === 'dashboard' ? 'none' : 'block';

  const m = PAGE_META[page];
  if (m) {
    const ey = document.getElementById('pg-eyebrow');
    const ti = document.getElementById('pg-title');
    const de = document.getElementById('pg-desc');
    if (ey) ey.textContent = m[0];
    if (ti) ti.textContent = m[1];
    if (de) de.textContent = m[2];
  }

  if (page === 'subsystems') fetchSubs();
  if (page === 'analytics')  fetchAnalytics();
  if (page === 'alerts')     fetchAlerts();
  if (page === 'rul')        renderRUL(LAST);
  if (page === 'timeline')   renderTimeline();

  const c = document.querySelector('.content');
  if (c) c.scrollTo({ top: 0, behavior: 'smooth' });
}

// Hide header on initial load since dashboard is the default page
document.addEventListener('DOMContentLoaded', () => {
  const hdr = document.getElementById('global-page-hdr');
  if (hdr) hdr.style.display = 'none';
});

// ── Build input fields ───────────────────────────────────────────────────────
Object.entries(GROUPS).forEach(([gid, names]) => {
  const el = document.getElementById(gid);
  if (!el) return;
  names.forEach(n => {
    const d = document.createElement('div');
    d.className = 'form-group';
    d.innerHTML = `
      <label>${n}</label>
      <input class="form-input" type="number" id="f_${n}" step="any" value="${DEFAULTS[n] ?? 0}" oninput="validateF('${n}')"/>
      <div class="form-hint" id="r_${n}"></div>`;
    el.appendChild(d);
  });
});

// ── Ranges ───────────────────────────────────────────────────────────────────
fetch('/ranges').then(r => r.json()).then(d => {
  RANGES = d;
  FEATURES.forEach(n => {
    const h = document.getElementById('r_' + n);
    const r = d[n];
    if (h && r) h.textContent = `Nominal: ${r.min} – ${r.max} ${r.unit}`;
  });
}).catch(() => {});

function validateF(n) {
  const el = document.getElementById('f_' + n);
  const r  = RANGES[n];
  if (!el || !r) return;
  const v = parseFloat(el.value);
  el.classList.toggle('oor', !isNaN(v) && (v < r.min || v > r.max));
}
function validateAll() { FEATURES.forEach(validateF); }

// ── Model toggle ─────────────────────────────────────────────────────────────
function setModel(m) {
  MODEL = m;
  document.getElementById('btn-xgb').classList.toggle('active', m === 'xgb');
  document.getElementById('btn-rf').classList.toggle('active', m === 'rf');
}

// ── Presets ───────────────────────────────────────────────────────────────────
fetch('/presets').then(r => r.json()).then(d => {
  PRESETS = d;
  const g = document.getElementById('pGrid');
  if (!g) return;
  g.innerHTML = Object.entries(d).map(([k, p]) => `
    <div class="preset-card">
      <div class="preset-card-header">
        <span class="badge ${p.severity}">${SEV_LABEL[p.severity] || p.severity}</span>
        <span class="preset-name">${p.label}</span>
        <span class="preset-tier ${p.tier}">${p.tier === 'high' ? 'High conf.' : 'Low conf.'}</span>
      </div>
      <p class="preset-desc">${p.desc}</p>
      <div class="preset-actions">
        <button class="btn btn-ghost" onclick="loadPreset('${k}')">Load Only</button>
        <button class="btn btn-primary" onclick="loadAndScan('${k}')">Load &amp; Scan</button>
      </div>
    </div>`).join('');
}).catch(() => {
  const g = document.getElementById('pGrid');
  if (g) g.innerHTML = '<div class="empty-state">Could not load presets.</div>';
});

function loadPreset(k) {
  if (!PRESETS[k]) return;
  FEATURES.forEach(n => {
    const el = document.getElementById('f_' + n);
    if (el && PRESETS[k].values[n] !== undefined) el.value = PRESETS[k].values[n];
  });
  validateAll(); go('manual');
}
function loadAndScan(k) {
  if (!PRESETS[k]) return;
  FEATURES.forEach(n => {
    const el = document.getElementById('f_' + n);
    if (el && PRESETS[k].values[n] !== undefined) el.value = PRESETS[k].values[n];
  });
  validateAll(); runScan();
}

// ── Randomize ─────────────────────────────────────────────────────────────────
function randFields(mode) {
  FEATURES.forEach(n => {
    const el = document.getElementById('f_' + n);
    const r  = RANGES[n];
    if (!el) return;
    if (!r) { el.value = DEFAULTS[n] ?? 0; return; }
    const sp = r.max - r.min;
    let v;
    if (mode === 'confident') {
      const pad = sp * 0.2;
      v = (r.min + pad) + Math.random() * (sp - 2 * pad);
    } else if (mode === 'borderline') {
      const eb = sp * 0.15;
      v = Math.random() < 0.5 ? r.min + Math.random() * eb : r.max - Math.random() * eb;
    } else {
      const os = sp * (0.3 + Math.random() * 0.5);
      v = Math.random() < 0.5 ? r.max + os : r.min - os;
    }
    el.value = Math.round(v * 100) / 100;
  });
  validateAll();
}
function resetFields() {
  FEATURES.forEach(n => {
    const el = document.getElementById('f_' + n);
    if (el) el.value = DEFAULTS[n] ?? 0;
  });
  validateAll();
}

// ── History & Timeline ────────────────────────────────────────────────────────
function renderHist() {
  const b = document.getElementById('hist-body');
  if (!b) return;
  if (!HIST.length) { b.innerHTML = '<div class="empty-state">No scans yet.</div>'; return; }
  b.innerHTML = `
    <table class="hist-table">
      <thead><tr><th>Time</th><th>Result</th><th>Confidence</th><th>Model</th></tr></thead>
      <tbody>
        ${HIST.slice().reverse().map(h => `
          <tr>
            <td class="mono">${h.time}</td>
            <td><span class="badge ${h.severity}">${h.prediction}</span></td>
            <td class="mono">${(h.confidence * 100).toFixed(1)}%</td>
            <td class="mono">${h.model || 'XGBoost'}</td>
          </tr>`).join('')}
      </tbody>
    </table>`;
}

function renderTimeline() {
  const b = document.getElementById('timeline-body');
  if (!b) return;
  if (!HIST.length) { b.innerHTML = '<div class="empty-state">No scans yet.</div>'; return; }
  b.innerHTML = `<div class="timeline">
    ${HIST.slice().reverse().map((h, i) => `
      <div class="tl-item">
        <div class="tl-line"></div>
        <div class="tl-dot ${h.severity}"></div>
        <div class="tl-card">
          <div class="tl-time">${h.time}</div>
          <div class="tl-pred ${h.severity}">${SEV_ICON[h.severity] || '·'} ${h.prediction}</div>
          <div class="tl-meta">Confidence: ${(h.confidence * 100).toFixed(1)}% · ${h.model || 'XGBoost'}</div>
        </div>
      </div>`).join('')}
  </div>`;
}

function clearHist() {
  HIST = [];
  renderHist(); renderTimeline();
  const ks = document.getElementById('kpi-scans');
  if (ks) ks.textContent = '0';
  fetch('/history/clear', { method: 'POST' }).catch(() => {});
}

// ── Alerts ────────────────────────────────────────────────────────────────────
function fetchAlerts() {
  fetch('/alerts').then(r => r.json()).then(d => {
    const b = document.getElementById('alerts-body');
    if (!b) return;
    if (!d.alerts || !d.alerts.length) {
      b.innerHTML = '<div class="empty-state">No alerts this session.</div>';
      return;
    }
    b.innerHTML = d.alerts.map(a => `
      <div class="alert-item">
        <div class="alert-dot ${a.severity}"></div>
        <div class="alert-body">
          <div class="alert-pred">${a.prediction}</div>
          <div class="alert-meta">${a.time.slice(11, 19)} UTC · Confidence ${(a.confidence * 100).toFixed(1)}%</div>
        </div>
        <span class="badge ${a.severity}">${SEV_LABEL[a.severity] || a.severity}</span>
      </div>`).join('');
    const nd = document.getElementById('nd-alerts');
    if (nd && d.alerts.length) nd.style.display = 'inline-block';
  }).catch(() => {});
}

function clearAlerts() {
  fetch('/alerts/clear', { method: 'POST' }).then(() => fetchAlerts()).catch(() => {});
  const nd = document.getElementById('nd-alerts');
  if (nd) nd.style.display = 'none';
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function showToast(prediction, severity) {
  const container = document.getElementById('toast-container');
  if (!container) return;
  const toast = document.createElement('div');
  toast.className = `toast ${severity}`;
  toast.innerHTML = `
    <div class="toast-icon">${severity === 'critical' ? '✕' : '⚠'}</div>
    <div>
      <div class="toast-title">${prediction}</div>
      <div class="toast-sub">${SEV_LABEL[severity] || severity} detected</div>
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">✕</button>`;
  container.appendChild(toast);
  setTimeout(() => { if (toast.parentElement) toast.remove(); }, 6000);
}

// ── RUL ───────────────────────────────────────────────────────────────────────
function renderRUL(data) {
  const panel = document.getElementById('rul-panel');
  if (!panel) return;
  if (!data || !data.rul || !Object.keys(data.rul).length) {
    panel.innerHTML = '<div class="empty-state">Run a few scans or start simulation to generate RUL estimates.</div>';
    return;
  }
  panel.innerHTML = Object.entries(data.rul).map(([param, r]) => {
    if (r.status === 'insufficient_data') return `
      <div class="rul-card nominal">
        <div class="rul-card-header"><span class="rul-param">${param}</span><span class="badge nominal">Monitoring</span></div>
        <div class="rul-ticks-display">Insufficient data — need 3+ data points</div>
        <div class="rul-meta"><span>Current: <strong>${r.current} ${r.unit}</strong></span></div>
      </div>`;
    const ticks = r.rul_ticks >= 9999 ? '∞' : r.rul_ticks;
    const pct   = r.rul_ticks >= 9999 ? 100 : Math.min(100, (r.rul_ticks / 50) * 100);
    const barColor = r.status === 'critical' ? 'var(--status-crit)'
                   : r.status === 'warning'  ? 'var(--status-warn)'
                   : 'var(--status-ok)';
    return `
      <div class="rul-card ${r.status}">
        <div class="rul-card-header">
          <span class="rul-param">${param}</span>
          <span class="badge ${r.status}">${SEV_LABEL[r.status] || r.status}</span>
        </div>
        <div class="rul-ticks-display">Est. remaining: <strong>${ticks}</strong> ticks until threshold</div>
        <div class="rul-bar-track"><div class="rul-bar-fill" style="width:${pct.toFixed(0)}%;background:${barColor}"></div></div>
        <div class="rul-meta">
          <span>Current: <strong>${r.current} ${r.unit}</strong></span>
          <span>Threshold: <strong>${r.threshold} ${r.unit}</strong></span>
          <span>Trend: <strong>${r.slope > 0 ? '+' : ''}${r.slope}/tick</strong></span>
        </div>
      </div>`;
  }).join('');
}

// ── Model comparison ──────────────────────────────────────────────────────────
async function runComparison() {
  const panel = document.getElementById('compare-panel');
  if (!panel) return;
  panel.innerHTML = '<div class="empty-state">Running comparison…</div>';
  const vals = FEATURES.map(n => {
    const el = document.getElementById('f_' + n);
    return el ? parseFloat(el.value) || 0 : 0;
  });
  try {
    const r = await fetch('/compare', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features: vals }),
    });
    const d = await r.json();
    if (d.error) { panel.innerHTML = `<div class="err-box">${d.error}</div>`; return; }

    const col = (label, res, accentColor) => {
      const sorted = Object.entries(res.probabilities).sort((a, b) => b[1] - a[1]);
      const topShap = (res.shap_values || []).slice(0, 5);
      return `
        <div class="cmp-col">
          <div class="cmp-model-name" style="border-bottom:2px solid ${accentColor}">${label}</div>
          <div class="cmp-pred ${res.severity}">${res.prediction}</div>
          <div class="cmp-conf">Confidence: ${(res.confidence * 100).toFixed(2)}%</div>
          <div class="cmp-sub-label">Probabilities</div>
          ${sorted.slice(0, 4).map(([lbl, p]) => `
            <div class="cmp-prob-row">
              <span class="cmp-prob-name">${lbl}</span>
              <div class="prob-track" style="margin:0"><div class="prob-fill" style="width:${(p*100).toFixed(1)}%;background:${accentColor}60"></div></div>
              <span class="cmp-prob-pct">${(p * 100).toFixed(1)}%</span>
            </div>`).join('')}
          <div class="cmp-sub-label">Top SHAP Features</div>
          ${topShap.map(f => `
            <div class="cmp-feat-row">
              <span class="cmp-feat-name">${f.name}</span>
              <span class="cmp-feat-val ${f.direction === 'positive' ? 'pos' : 'neg'}">${f.shap_value > 0 ? '+' : ''}${f.shap_value.toFixed(4)}</span>
            </div>`).join('')}
        </div>`;
    };

    const agree = d.xgb.prediction === d.rf.prediction;
    panel.innerHTML = `
      <div class="cmp-grid">
        ${col('XGBoost · Primary', d.xgb, '#f59e0b')}
        ${col('RandomForest · Fallback', d.rf, '#3b82f6')}
      </div>
      <div class="cmp-verdict ${agree ? 'agree' : 'disagree'}">
        ${agree
          ? `Both models agree: <strong>${d.xgb.prediction}</strong>`
          : `Models disagree — XGBoost: <strong>${d.xgb.prediction}</strong> · RandomForest: <strong>${d.rf.prediction}</strong>`}
      </div>`;
    const nd = document.getElementById('nd-compare');
    if (nd) nd.style.display = 'inline-block';
  } catch (e) {
    panel.innerHTML = `<div class="err-box">Network error: ${e.message}</div>`;
  }
}

// ── Export ────────────────────────────────────────────────────────────────────
function exportJSON() {
  const s = document.getElementById('export-status');
  if (!LAST) { if (s) s.textContent = 'No scan data yet.'; return; }
  const blob = new Blob([JSON.stringify(LAST, null, 2)], { type: 'application/json' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = `satmon_${Date.now()}.json`; a.click();
  URL.revokeObjectURL(url);
  if (s) s.textContent = '✓ JSON exported (includes SHAP + LIME values).';
}
function copySummary() {
  const s = document.getElementById('export-status');
  if (!LAST) { if (s) s.textContent = 'No scan data yet.'; return; }
  const top3 = (LAST.shap_values || []).slice(0, 3)
    .map(f => `  ${f.name}: ${f.shap_value > 0 ? '+' : ''}${f.shap_value.toFixed(4)}`).join('\n');
  const t = `SATMON Scan Result\nModel: ${LAST.model_used}\nPrediction: ${LAST.prediction}\nConfidence: ${(LAST.confidence * 100).toFixed(2)}%\nSeverity: ${LAST.severity}\nTop SHAP:\n${top3}\nTimestamp: ${LAST.timestamp}`;
  navigator.clipboard.writeText(t).then(() => { if (s) s.textContent = '✓ Copied to clipboard.'; });
}

// ── Core render ───────────────────────────────────────────────────────────────
function renderAll(data) {
  LAST = data;
  HIST.push({
    time:       new Date().toLocaleTimeString(),
    prediction: data.prediction,
    severity:   data.severity,
    confidence: data.confidence,
    model:      data.model_used || 'XGBoost',
  });
  renderHist();
  renderTimeline();

  // KPIs
  const ks  = document.getElementById('kpi-state');
  const kc  = document.getElementById('kpi-conf');
  const ksv = document.getElementById('kpi-sev');
  const kn  = document.getElementById('kpi-scans');
  if (ks)  ks.textContent  = data.prediction;
  if (kc)  kc.textContent  = (data.confidence * 100).toFixed(1) + '%';
  if (ksv) ksv.textContent = SEV_LABEL[data.severity] || data.severity;
  if (kn)  kn.textContent  = HIST.length;

  // KPI card border colour
  const kpiConf = document.querySelector('#kpi-conf')?.closest('.kpi-card');
  if (kpiConf) {
    kpiConf.className = 'kpi-card ' + (data.severity === 'nominal' ? 'ok' : data.severity === 'critical' ? 'crit' : 'warn');
  }

  // Subsystem count
  if (data.subsystem_health) {
    const ac = Object.values(data.subsystem_health).filter(s => s.status === 'alert').length;
    const ka = document.getElementById('kpi-alerts');
    if (ka) ka.textContent = `${ac} / ${Object.keys(data.subsystem_health).length}`;
    renderSubGrid(data.subsystem_health);
  }

  // Telemetry snapshot
  if (data.telemetry) {
    const dt = document.getElementById('dash-telemetry');
    if (dt) dt.innerHTML = `<div class="tel-grid">
      ${Object.entries(data.telemetry).map(([k, v]) => `
        <div class="tel-chip">
          <div class="tel-key">${k}</div>
          <div class="tel-val">${v}</div>
        </div>`).join('')}
    </div>`;
  }

  // Model tag
  const mtw = document.getElementById('model-tag-wrap');
  const mt  = document.getElementById('model-tag');
  if (mtw && mt) {
    mtw.style.display = 'block';
    mt.textContent    = data.model_used === 'XGBoost'
      ? 'XGBoost · Knowledge-Distilled · n=20,000'
      : 'RandomForest · n_estimators=300';
    mt.className = `model-tag ${data.model_used === 'XGBoost' ? 'xgb' : 'rf'}`;
  }

  // Classification result
  const rp = document.getElementById('result-panel');
  if (rp) {
    const sorted = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
    rp.innerHTML = `
      <div class="result-header">
        <span class="badge ${data.severity}">${SEV_LABEL[data.severity]}</span>
        <div class="result-prediction ${data.severity}">${data.prediction}</div>
        <div class="result-confidence">${(data.confidence * 100).toFixed(2)}% confidence</div>
      </div>
      <div class="prob-list">
        ${sorted.map(([lbl, p], i) => `
          <div class="prob-row">
            <span class="prob-name">${lbl}</span>
            <div class="prob-track"><div class="prob-fill${i === 0 ? ' top' : ''}" style="width:${(p * 100).toFixed(1)}%"></div></div>
            <span class="prob-pct">${(p * 100).toFixed(1)}%</span>
          </div>`).join('')}
      </div>`;
  }

  // SHAP
  renderXAI('xai-panel', data.shap_values, 'shap_value', 'SHAP Value');

  // LIME
  if (data.lime_values && data.lime_values.length) {
    renderXAI('lime-panel',
      data.lime_values.map(f => ({ name: f.feature, shap_value: f.weight, shap_abs: Math.abs(f.weight), direction: f.direction })),
      'shap_value', 'LIME Weight');
  } else {
    const lp = document.getElementById('lime-panel');
    if (lp && !data.lime_values?.length) {
      lp.innerHTML = '<div class="empty-state">LIME runs on manual scans only (not simulation ticks).</div>';
    }
  }

  // Narrative
  const ep = document.getElementById('explain-panel');
  if (ep) {
    ep.innerHTML = data.explanation
      ? `<div class="explain-box">${data.explanation}</div>`
      : '';
  }

  // RUL
  renderRUL(data);

  // Recommendations
  const rcp = document.getElementById('rec-panel');
  if (rcp && data.recommendations?.length) {
    rcp.innerHTML = `
      <div class="card-header">
        <div>
          <div class="card-title">${data.prediction}</div>
          <div class="card-subtitle">Recommended operator actions</div>
        </div>
        <span class="badge ${data.severity}">${SEV_LABEL[data.severity]}</span>
      </div>
      <div class="card-body">
        <ol style="padding-left:20px;display:flex;flex-direction:column;gap:10px">
          ${data.recommendations.map(r => `<li style="font-size:13px;color:var(--text-secondary);line-height:1.65">${r}</li>`).join('')}
        </ol>
      </div>`;
  }

  // Live chart update
  if (data.telemetry && !CHART_PAUSED) {
    const point = { time: new Date().toLocaleTimeString(), raw: {} };
    Object.keys(CHART_SERIES).forEach(k => {
      if (data.telemetry[k] !== undefined) point.raw[k] = parseFloat(data.telemetry[k]);
    });
    CHART_BUF.push(point);
    if (CHART_BUF.length > CHART_MAX) CHART_BUF.shift();
    drawLiveChart();
  }

  // Toast + alert refresh
  if (data.severity === 'warning' || data.severity === 'critical') {
    showToast(data.prediction, data.severity);
    fetchAlerts();
  }

  // Nav dots
  ['result', 'xai', 'lime', 'recommend', 'rul'].forEach(p => {
    const nd = document.getElementById('nd-' + p);
    if (nd) nd.style.display = 'inline-block';
  });
}

// Shared XAI renderer (used for both SHAP and LIME)
function renderXAI(panelId, values, valKey, valLabel) {
  const panel = document.getElementById(panelId);
  if (!panel || !values || !values.length) return;
  const top = values.slice(0, 10);
  const mx  = Math.max(...top.map(f => f.shap_abs), 0.0001);
  panel.innerHTML = `
    <div class="xai-legend">
      <div class="xai-legend-item"><div class="xai-swatch pos"></div>Supports this class</div>
      <div class="xai-legend-item"><div class="xai-swatch neg"></div>Contradicts this class</div>
    </div>
    ${top.map(f => {
      const pct = (f.shap_abs / mx * 100).toFixed(1);
      const cls = f.direction === 'positive' ? 'pos' : 'neg';
      return `
        <div class="xai-row">
          <span class="xai-name">${f.name}</span>
          <div class="xai-track"><div class="xai-fill ${cls}" style="width:${pct}%"></div></div>
          <span class="xai-val ${cls}">${f[valKey] > 0 ? '+' : ''}${f[valKey].toFixed(4)}</span>
          <span class="xai-raw">val: ${'value' in f ? f.value : '—'}</span>
        </div>`;
    }).join('')}`;
}

// ── Subsystem grid ────────────────────────────────────────────────────────────
function renderSubGrid(health) {
  const g = document.getElementById('subGrid');
  if (!g) return;
  const C = 188; // circumference for r=30
  g.innerHTML = Object.entries(health).map(([k, h]) => {
    const oor   = h.out_of_range || [];
    const total = ({ power:7, thermal:3, attitude:2, compute:2, comms:1, orbital:3 }[k]) || 3;
    const score = Math.max(0, Math.round((1 - oor.length / total) * 100));
    const offset = C - (score / 100) * C;
    const stroke = h.status === 'alert' ? 'var(--status-crit)'
                 : score >= 80           ? 'var(--status-ok)'
                 :                         'var(--status-warn)';
    return `
      <div class="sub-card ${h.status}">
        <div class="sub-ring-wrap">
          <svg class="sub-ring" viewBox="0 0 72 72">
            <circle class="bg" cx="36" cy="36" r="30"/>
            <circle class="fg" cx="36" cy="36" r="30"
              stroke="${stroke}" stroke-dasharray="${C}" stroke-dashoffset="${offset.toFixed(1)}"/>
          </svg>
          <div class="sub-ring-val" style="color:${stroke}">${score}%</div>
          <div class="sub-ic-badge ${h.status === 'alert' ? 'alert' : 'ok'}">${h.status === 'alert' ? '!' : '✓'}</div>
        </div>
        <div class="sub-label">${h.icon} ${h.label}</div>
        <div class="sub-status ${h.status === 'alert' ? 'crit' : 'ok'}">${h.status === 'alert' ? 'Alert' : 'Nominal'}</div>
        ${oor.length
          ? `<ul class="sub-oor-list">${oor.map(f => `<li>${f}</li>`).join('')}</ul>`
          : `<p class="sub-ok-text">All parameters nominal.</p>`}
      </div>`;
  }).join('');
}

function fetchSubs() {
  fetch('/subsystems').then(r => r.json()).then(d => {
    if (d.subsystems) renderSubGrid(d.subsystems);
  }).catch(() => {});
}

// ── Analytics ─────────────────────────────────────────────────────────────────
function fetchAnalytics() {
  fetch('/analytics').then(r => r.json()).then(d => {
    const at  = document.getElementById('ana-total');
    const ac  = document.getElementById('ana-conf');
    const acr = document.getElementById('ana-crit');
    const an  = document.getElementById('ana-nom');
    if (at)  at.textContent  = d.total || 0;
    if (ac)  ac.textContent  = d.total ? ((d.avg_confidence * 100).toFixed(1) + '%') : '—';
    const crit = (d.by_severity?.critical) || 0;
    const nom  = (d.by_severity?.nominal)  || 0;
    if (acr) acr.textContent = crit;
    if (an)  an.textContent  = d.total ? ((nom / d.total * 100).toFixed(0) + '%') : '—';

    // Trend chart
    const trend  = d.confidence_trend || [];
    const canvas = document.getElementById('trendChart');
    const empty  = document.getElementById('trendEmpty');
    if (!trend.length) {
      if (canvas) canvas.style.display = 'none';
      if (empty)  empty.style.display  = 'block';
    } else {
      if (canvas) { canvas.style.display = 'block'; drawTrend(canvas, trend); }
      if (empty)  empty.style.display  = 'none';
    }

    // Distribution
    const ad = document.getElementById('ana-dist');
    if (ad) {
      if (d.by_class && Object.keys(d.by_class).length) {
        const mx = Math.max(...Object.values(d.by_class), 1);
        ad.innerHTML = Object.entries(d.by_class).sort((a, b) => b[1] - a[1]).map(([lbl, cnt]) => `
          <div class="ana-row">
            <span class="ana-label">${lbl}</span>
            <div class="ana-track"><div class="ana-fill" style="width:${(cnt/mx*100).toFixed(0)}%"></div></div>
            <span class="ana-count">${cnt}</span>
          </div>`).join('');
      } else ad.innerHTML = '<div class="empty-state">No data yet.</div>';
    }

    // Severity
    const as = document.getElementById('ana-sev');
    if (as) {
      if (d.by_severity && Object.keys(d.by_severity).length) {
        const col = { nominal: 'var(--status-ok)', warning: 'var(--status-warn)', critical: 'var(--status-crit)' };
        as.innerHTML = Object.entries(d.by_severity).map(([s, c]) => `
          <div class="sev-row">
            <div class="sev-dot" style="background:${col[s] || '#aaa'}"></div>
            <span class="sev-name">${SEV_LABEL[s] || s}</span>
            <span class="sev-count">${c}</span>
          </div>`).join('');
      } else as.innerHTML = '<div class="empty-state">No data yet.</div>';
    }
  }).catch(() => {});
}

function drawTrend(canvas, trend) {
  const dpr = window.devicePixelRatio || 1;
  const W   = canvas.parentElement.clientWidth - 36;
  const H   = 110;
  canvas.width  = W * dpr; canvas.height = H * dpr;
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const PAD = { t: 10, r: 12, b: 28, l: 40 };
  const pw  = W - PAD.l - PAD.r;
  const ph  = H - PAD.t - PAD.b;
  const n   = trend.length;
  const vals = trend.map(p => p.c);
  const minV = Math.max(0, Math.min(...vals) - 5);
  const maxV = Math.min(100, Math.max(...vals) + 5);
  const toX  = i => PAD.l + (i / (n - 1 || 1)) * pw;
  const toY  = v => PAD.t + ph - ((v - minV) / (maxV - minV || 1)) * ph;

  // Grid
  ctx.strokeStyle = 'rgba(44,61,90,0.6)'; ctx.lineWidth = 1;
  [0, 25, 50, 75, 100].forEach(v => {
    if (v < minV || v > maxV) return;
    const y = toY(v);
    ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(PAD.l + pw, y); ctx.stroke();
    ctx.fillStyle = 'rgba(71,85,105,0.8)';
    ctx.font = `9px 'JetBrains Mono'`;
    ctx.fillText(v + '%', 2, y + 3);
  });

  // Fill
  const grad = ctx.createLinearGradient(0, PAD.t, 0, PAD.t + ph);
  grad.addColorStop(0, 'rgba(59,130,246,0.2)');
  grad.addColorStop(1, 'rgba(59,130,246,0.02)');
  ctx.beginPath();
  ctx.moveTo(toX(0), toY(vals[0]));
  vals.forEach((v, i) => { if (i > 0) ctx.lineTo(toX(i), toY(v)); });
  ctx.lineTo(toX(n - 1), PAD.t + ph);
  ctx.lineTo(toX(0), PAD.t + ph);
  ctx.closePath();
  ctx.fillStyle = grad; ctx.fill();

  // Line
  ctx.beginPath();
  ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 1.5; ctx.lineJoin = 'round';
  vals.forEach((v, i) => { i === 0 ? ctx.moveTo(toX(i), toY(v)) : ctx.lineTo(toX(i), toY(v)); });
  ctx.stroke();

  // Dots by severity
  const sc = { nominal: '#22c55e', warning: '#f59e0b', critical: '#ef4444' };
  trend.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(toX(i), toY(p.c), 3, 0, Math.PI * 2);
    ctx.fillStyle = sc[p.s] || '#3b82f6'; ctx.fill();
  });

  // X labels
  ctx.fillStyle = 'rgba(71,85,105,0.8)'; ctx.font = `9px 'JetBrains Mono'`; ctx.textAlign = 'center';
  [[0, trend[0].t], [Math.floor(n / 2), trend[Math.floor(n / 2)]?.t], [n - 1, trend[n - 1].t]].forEach(([i, lbl]) => {
    if (lbl) ctx.fillText(lbl.slice(11, 16), toX(i), H - 6);
  });
  ctx.textAlign = 'left';
}

// ── Manual scan ───────────────────────────────────────────────────────────────
async function runScan() {
  const btn  = document.getElementById('scanBtn');
  const anim = document.getElementById('scanAnim');
  if (btn)  btn.disabled = true;
  if (anim) anim.style.width = '14px';
  const vals = FEATURES.map(n => {
    const el = document.getElementById('f_' + n);
    return el ? parseFloat(el.value) || 0 : 0;
  });
  const incLime = document.getElementById('limeCheck');
  try {
    const r = await fetch('/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features: vals, model: MODEL, lime: incLime?.checked ?? true }),
    });
    const d = await r.json();
    const rp = document.getElementById('result-panel');
    if (d.error) {
      if (rp) rp.innerHTML = `<div class="err-box">${d.error}</div>`;
      go('result'); return;
    }
    renderAll(d); go('result');
  } catch (e) {
    const rp = document.getElementById('result-panel');
    if (rp) rp.innerHTML = `<div class="err-box">Network error: ${e.message}</div>`;
    go('result');
  } finally {
    if (btn)  btn.disabled = false;
    if (anim) anim.style.width = '0';
  }
}

// ── Simulation ────────────────────────────────────────────────────────────────
function toggleSim() { SIM_ON ? stopSim() : startSim(); }

function startSim() {
  SIM_ON = true;
  const btn = document.getElementById('simBtn');
  if (btn) { btn.textContent = '⏸ Stop Simulation'; btn.classList.add('active'); }
  const spd = parseInt(document.getElementById('simSpd')?.value) || 2000;
  doTick();
  SIM_IV = setInterval(doTick, spd);
}

function stopSim() {
  SIM_ON = false;
  clearInterval(SIM_IV); SIM_IV = null;
  const btn = document.getElementById('simBtn');
  if (btn) { btn.textContent = '▷ Start Simulation'; btn.classList.remove('active'); }
}

async function doTick() {
  try {
    const r = await fetch('/simulate/tick', { method: 'POST' });
    const d = await r.json();
    if (d.error) return;
    SIM_T++;
    const st = document.getElementById('simTicks');
    if (st) st.textContent = SIM_T;

    const feed = document.getElementById('simFeed');
    if (feed) {
      const entry = document.createElement('div');
      entry.className = `feed-item`;
      entry.innerHTML = `
        <span class="feed-time">${new Date().toLocaleTimeString()}</span>
        <span class="badge ${d.severity}" style="font-size:10px">${SEV_LABEL[d.severity]}</span>
        <span class="feed-pred">${d.prediction}</span>
        <span class="feed-conf">${(d.confidence * 100).toFixed(1)}%</span>`;
      if (feed.firstChild?.classList?.contains('empty-state')) feed.innerHTML = '';
      feed.prepend(entry);
      while (feed.children.length > 15) feed.removeChild(feed.lastChild);
    }
    renderAll(d);
  } catch (e) { /* silent */ }
}

// ── Live Chart ────────────────────────────────────────────────────────────────
function drawLiveChart() {
  const canvas = document.getElementById('liveChart');
  const empty  = document.getElementById('liveChartEmpty');
  const pill   = document.getElementById('live-pill');

  if (!CHART_BUF.length) {
    if (canvas) canvas.style.display = 'none';
    if (empty)  empty.style.display  = 'block';
    if (pill)   pill.style.display   = 'none';
    return;
  }
  if (canvas) canvas.style.display = 'block';
  if (empty)  empty.style.display  = 'none';
  if (pill)   pill.style.display   = CHART_PAUSED ? 'none' : 'inline-flex';

  const dpr  = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  const W    = Math.floor(rect.width) - 4;
  const H    = 160;
  canvas.width  = W * dpr; canvas.height = H * dpr;
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const PAD = { t: 14, r: 60, b: 32, l: 46 };
  const pw  = W - PAD.l - PAD.r;
  const ph  = H - PAD.t - PAD.b;

  ctx.fillStyle = 'var(--surface-0, #0e1117)';
  ctx.fillRect(0, 0, W, H);

  // Grid
  for (let i = 0; i <= 4; i++) {
    const y = PAD.t + (ph / 4) * i;
    ctx.strokeStyle = 'rgba(44,61,90,0.5)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(PAD.l + pw, y); ctx.stroke();
    ctx.fillStyle = 'rgba(71,85,105,0.8)';
    ctx.font = `10px 'JetBrains Mono'`; ctx.textAlign = 'right';
    ctx.fillText((100 - i * 25) + '%', PAD.l - 6, y + 3);
  }

  const toX = i => PAD.l + (i / (CHART_MAX - 1)) * pw;
  const toY = (v, meta) => PAD.t + ph - ((v - meta.ymin) / (meta.ymax - meta.ymin)) * ph;

  Object.entries(CHART_SERIES).forEach(([key, meta]) => {
    if (!CHART_ACTIVE[key]) return;
    const pts = CHART_BUF
      .map((p, i) => p.raw[key] !== undefined ? { x: toX(i), y: toY(p.raw[key], meta), v: p.raw[key] } : null)
      .filter(Boolean);
    if (pts.length < 2) return;

    // Gradient fill
    const grad = ctx.createLinearGradient(0, PAD.t, 0, PAD.t + ph);
    grad.addColorStop(0, meta.color + '25');
    grad.addColorStop(1, meta.color + '02');
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) {
      const cx = (pts[i - 1].x + pts[i].x) / 2;
      ctx.bezierCurveTo(cx, pts[i - 1].y, cx, pts[i].y, pts[i].x, pts[i].y);
    }
    ctx.lineTo(pts[pts.length - 1].x, PAD.t + ph);
    ctx.lineTo(pts[0].x, PAD.t + ph);
    ctx.closePath();
    ctx.fillStyle = grad; ctx.fill();

    // Line
    ctx.save();
    ctx.beginPath(); ctx.strokeStyle = meta.color; ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) {
      const cx = (pts[i - 1].x + pts[i].x) / 2;
      ctx.bezierCurveTo(cx, pts[i - 1].y, cx, pts[i].y, pts[i].x, pts[i].y);
    }
    ctx.stroke(); ctx.restore();

    // Latest dot
    const last = pts[pts.length - 1];
    ctx.beginPath(); ctx.arc(last.x, last.y, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = meta.color; ctx.fill();
    ctx.beginPath(); ctx.arc(last.x, last.y, 1.5, 0, Math.PI * 2);
    ctx.fillStyle = '#fff'; ctx.fill();

    // Latest value label (right side)
    ctx.fillStyle = meta.color; ctx.font = `10px 'JetBrains Mono'`; ctx.textAlign = 'left';
    ctx.fillText(last.v.toFixed(1), PAD.l + pw + 4, last.y + 3);
  });

  // X labels
  ctx.fillStyle = 'rgba(71,85,105,0.7)'; ctx.font = `9px 'JetBrains Mono'`; ctx.textAlign = 'center';
  const n = CHART_BUF.length;
  const labelCount = Math.min(6, n);
  for (let li = 0; li < labelCount; li++) {
    const di = Math.round((li / (labelCount - 1 || 1)) * (n - 1));
    if (CHART_BUF[di]) ctx.fillText(CHART_BUF[di].time.slice(-8), toX(di), H - 14);
  }
  ctx.textAlign = 'left';
}

function toggleChartPause() {
  CHART_PAUSED = !CHART_PAUSED;
  const btn = document.getElementById('chartPauseBtn');
  if (btn) btn.textContent = CHART_PAUSED ? '▷ Resume' : '⏸ Pause';
  drawLiveChart();
}

function toggleSeries(key) {
  CHART_ACTIVE[key] = !CHART_ACTIVE[key];
  const btn = document.getElementById('sbtn_' + key.replace(/[^a-zA-Z0-9]/g, '_'));
  if (btn) btn.classList.toggle('off', !CHART_ACTIVE[key]);
  drawLiveChart();
}

function clearChart() { CHART_BUF = []; drawLiveChart(); }

function exportChartCSV() {
  if (!CHART_BUF.length) { alert('No chart data yet.'); return; }
  const keys = Object.keys(CHART_SERIES);
  const header = ['Time', ...keys].join(',');
  const rows = CHART_BUF.map(p =>
    [p.time, ...keys.map(k => p.raw[k] !== undefined ? p.raw[k].toFixed(3) : '')].join(',')
  );
  const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = `satmon_telemetry_${Date.now()}.csv`; a.click();
  URL.revokeObjectURL(url);
}

window.addEventListener('resize', () => { if (CHART_BUF.length) drawLiveChart(); });