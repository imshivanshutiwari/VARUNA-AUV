// VARUNA-AUV Naval Ops Center – Real-time dashboard
'use strict';

const PALETTE = {
  accent:  '#00d4ff',
  accent2: '#00ff88',
  warn:    '#ff6b35',
  danger:  '#ff2244',
  grid:    'rgba(0,212,255,0.12)',
  text:    '#b8d8f0',
};

const CHART_DEFAULTS = {
  responsive: true,
  animation: false,
  plugins: { legend: { display: false } },
  scales: {
    x: { ticks: { color: PALETTE.text, font: { size: 9 } }, grid: { color: PALETTE.grid } },
    y: { ticks: { color: PALETTE.text, font: { size: 9 } }, grid: { color: PALETTE.grid } },
  },
};

function makeChart(id, type, data, extraOptions = {}) {
  const ctx = document.getElementById(id).getContext('2d');
  return new Chart(ctx, {
    type,
    data,
    options: deepMerge(JSON.parse(JSON.stringify(CHART_DEFAULTS)), extraOptions),
  });
}

function deepMerge(target, src) {
  for (const k of Object.keys(src)) {
    if (src[k] && typeof src[k] === 'object' && !Array.isArray(src[k])) {
      target[k] = deepMerge(target[k] || {}, src[k]);
    } else {
      target[k] = src[k];
    }
  }
  return target;
}

// ── Chart instances ─────────────────────────────────────────────
const charts = {};

function initCharts() {
  // Spectrum
  charts.spectrum = makeChart('chart-spectrum', 'line', {
    labels: [],
    datasets: [{ data: [], borderColor: PALETTE.accent, borderWidth: 1.5, fill: true,
      backgroundColor: 'rgba(0,212,255,0.08)', pointRadius: 0, tension: 0.3 }],
  }, { scales: { y: { title: { display: true, text: 'dB', color: PALETTE.text } } } });

  // LOFARgram bar
  charts.lofar = makeChart('chart-lofar', 'bar', {
    labels: [],
    datasets: [{ data: [], backgroundColor: 'rgba(0,255,136,0.6)', borderWidth: 0 }],
  });

  // DEMON
  charts.demon = makeChart('chart-demon', 'line', {
    labels: [],
    datasets: [{ data: [], borderColor: PALETTE.warn, borderWidth: 1.5, fill: false,
      pointRadius: 0, tension: 0.2 }],
  });

  // MFCC heatmap-style bar
  charts.mfcc = makeChart('chart-mfcc', 'bar', {
    labels: [],
    datasets: [{ data: [], backgroundColor: [], borderWidth: 0 }],
  }, { indexAxis: 'y' });

  // Classification
  charts.classify = makeChart('chart-classify', 'bar', {
    labels: ['Cargo','Tanker','Tug','Passenger','Submarine','Biological','Unknown'],
    datasets: [{ data: new Array(7).fill(0),
      backgroundColor: ['#00d4ff','#00ff88','#ffd700','#ff6b35','#ff2244','#aa44ff','#4a7090'],
      borderWidth: 0 }],
  }, { scales: { y: { min: 0, max: 1, title: { display: true, text: 'Prob', color: PALETTE.text } } } });

  // DOA gauge via doughnut
  charts.doa = makeChart('chart-doa', 'doughnut', {
    labels: ['DOA', 'Rest'],
    datasets: [{ data: [0, 180], backgroundColor: [PALETTE.accent, 'rgba(0,0,0,0.2)'],
      borderWidth: 0 }],
  }, { plugins: { legend: { display: false } }, cutout: '65%' });

  // CFAR scatter
  charts.cfar = makeChart('chart-cfar', 'scatter', {
    datasets: [{ data: [], backgroundColor: PALETTE.danger, pointRadius: 5 }],
  }, { scales: {
    x: { title: { display: true, text: 'Hz', color: PALETTE.text } },
    y: { title: { display: true, text: 'dB', color: PALETTE.text } },
  }});

  // Waterfall: canvas drawn manually
}

// Waterfall buffer
const WATERFALL_ROWS = 64;
const waterfallBuf = [];

function drawWaterfall(spectrumDb) {
  const canvas = document.getElementById('chart-waterfall');
  const ctx = canvas.getContext('2d');
  // Push new row
  waterfallBuf.push(spectrumDb.slice(0, 256));
  if (waterfallBuf.length > WATERFALL_ROWS) waterfallBuf.shift();

  const w = canvas.offsetWidth || 400;
  const h = 120;
  canvas.width  = w;
  canvas.height = h;

  const rowH = h / WATERFALL_ROWS;
  const colW = w / 256;

  for (let r = 0; r < waterfallBuf.length; r++) {
    const row = waterfallBuf[waterfallBuf.length - 1 - r];
    for (let c = 0; c < row.length; c++) {
      const norm = Math.min(1, Math.max(0, (row[c] + 80) / 80));
      const blue  = Math.round(255 * (1 - norm));
      const green = Math.round(180 * norm);
      const red   = Math.round(255 * Math.max(0, norm - 0.5) * 2);
      ctx.fillStyle = `rgb(${red},${green},${blue})`;
      ctx.fillRect(Math.round(c * colW), Math.round(r * rowH), Math.ceil(colW) + 1, Math.ceil(rowH) + 1);
    }
  }
}

// ── Update helpers ───────────────────────────────────────────────
function updateSpectrum(data) {
  const n = data.spectrum_db.length;
  const labels = data.spectrum_db.map((_, i) => (i * 22050 / n).toFixed(0));
  charts.spectrum.data.labels = labels.filter((_, i) => i % 4 === 0);
  charts.spectrum.data.datasets[0].data = data.spectrum_db.filter((_, i) => i % 4 === 0);
  charts.spectrum.update('none');
  drawWaterfall(data.spectrum_db);
}

function updateLofar(data) {
  const sl = data.lofar_slice;
  charts.lofar.data.labels = sl.map((_, i) => i);
  charts.lofar.data.datasets[0].data = sl;
  charts.lofar.update('none');
}

function updateDemon(data) {
  const pd = data.demon_power_db || [];
  charts.demon.data.labels = pd.map((_, i) => i);
  charts.demon.data.datasets[0].data = pd;
  charts.demon.update('none');
  document.getElementById('blade-rate').textContent =
    data.demon_blade_rate_hz ? data.demon_blade_rate_hz.toFixed(2) + ' Hz' : '—';
}

function updateMfcc(data) {
  const mf = data.mfcc_frame || [];
  const colors = mf.map(v => {
    const norm = Math.min(1, Math.max(0, (v + 20) / 40));
    return `hsl(${200 - norm * 160},80%,${30 + norm * 40}%)`;
  });
  charts.mfcc.data.labels = mf.map((_, i) => i);
  charts.mfcc.data.datasets[0].data = mf;
  charts.mfcc.data.datasets[0].backgroundColor = colors;
  charts.mfcc.update('none');
}

function updateClassify(data) {
  const clf = data.classification || {};
  const probs = clf.probabilities || new Array(7).fill(0);
  charts.classify.data.datasets[0].data = probs;
  charts.classify.update('none');
  document.getElementById('clf-label').textContent = clf.label || '—';
  document.getElementById('clf-conf').textContent =
    clf.confidence != null ? (clf.confidence * 100).toFixed(1) + '%' : '—';
  const alert = document.getElementById('submarine-alert');
  if (clf.is_submarine_alert) alert.classList.remove('hidden');
  else alert.classList.add('hidden');
}

function updateTracks(data) {
  const tbody = document.getElementById('track-tbody');
  tbody.innerHTML = '';
  (data.tracks || []).forEach(t => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${t.id}</td><td>${t.bearing_deg.toFixed(1)}</td><td>${t.range_m.toFixed(0)}</td><td>${t.label}</td><td>${(t.confidence*100).toFixed(0)}%</td>`;
    tbody.appendChild(tr);
  });
}

function updateDoa(data) {
  const doa = data.doa_deg || 0;
  const norm = ((doa + 90) / 180) * 180;
  charts.doa.data.datasets[0].data = [norm, 180 - norm];
  charts.doa.update('none');
  document.getElementById('doa-val').textContent = doa.toFixed(1) + '°';
}

function updateCfar(data) {
  const dets = data.cfar_detections || [];
  charts.cfar.data.datasets[0].data = dets.map(d => ({ x: d.frequency_hz, y: d.magnitude_db }));
  charts.cfar.update('none');
  document.getElementById('cfar-count').textContent = dets.length;
}

function updateStatus(data) {
  document.getElementById('ts-val').textContent = data.timestamp_ms + ' ms';
}

// ── WebSocket ────────────────────────────────────────────────────
function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);

  ws.onopen = () => {
    document.getElementById('conn-dot').className = 'dot green';
    document.getElementById('conn-label').textContent = 'Live';
  };

  ws.onmessage = (ev) => {
    let data;
    try { data = JSON.parse(ev.data); } catch { return; }
    updateSpectrum(data);
    updateLofar(data);
    updateDemon(data);
    updateMfcc(data);
    updateClassify(data);
    updateTracks(data);
    updateDoa(data);
    updateCfar(data);
    updateStatus(data);
  };

  ws.onclose = () => {
    document.getElementById('conn-dot').className = 'dot red';
    document.getElementById('conn-label').textContent = 'Disconnected';
    setTimeout(connect, 2000);
  };

  ws.onerror = () => ws.close();
}

// ── Clock ────────────────────────────────────────────────────────
function updateClock() {
  document.getElementById('clock').textContent =
    new Date().toISOString().substr(11, 8) + ' UTC';
}

// ── Boot ─────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  initCharts();
  connect();
  setInterval(updateClock, 1000);
  updateClock();
});
