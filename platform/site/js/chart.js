// Chart.js intensity chart: wind (kt, left axis) colored per-point by the
// Saffir-Simpson scale, pressure (mb, right axis) as a thinner line with
// gaps where missing. Exposes a small hover API for map <-> chart cross-hover.
//
// Single mode (geojsonB omitted): unchanged wind+pressure rendering, always
// absolute time -- the relative-time toggle below is a no-op here.
// Compare mode: two wind datasets (A solid, B dashed), pressure omitted
// (four curves on one chart is unreadable) and the pressure axis hidden.
// Compare mode also supports an x-axis time mode toggle (see setRelativeTime):
// absolute (default, real calendar time -- storms from different years land
// in separated clumps) or relative (hours since each storm's OWN first track
// point, so shapes of storms years apart become directly comparable). This
// is a pure axis transform of the same observed values -- no new data, no
// comparative claim.

import { windColor, trendArrow } from './scale.js';

let chartInstance = null;
let hoverCallback = null;
let currentPoints = { A: [], B: [] };
let compareMode = false;
// 'absolute' | 'relative'. Only has any visible effect when compareMode is
// true (checked at every read site below) -- single mode always renders as
// if this were 'absolute', matching the pre-existing behavior exactly.
let timeMode = 'absolute';
let currentOrigins = { A: 0, B: 0 }; // each storm's own first-point ms (relative-time zero point)
let currentNames = {};

function fmtTick(ms) {
  const d = new Date(ms);
  const mm = String(d.getUTCMonth() + 1).padStart(2, '0');
  const dd = String(d.getUTCDate()).padStart(2, '0');
  const hh = String(d.getUTCHours()).padStart(2, '0');
  return `${d.getUTCFullYear()}-${mm}-${dd} ${hh}h`;
}

/** Absolute ms -> chart x-value under the given time mode. */
function computeX(tMs, mode, originMs) {
  return mode === 'relative' ? (tMs - originMs) / 3_600_000 : tMs;
}

function windTooltipLines(p) {
  const lines = [`Wind: ${p.wind_kt != null ? p.wind_kt + ' kt' : '—'}`];
  if (p.dv24_kt != null) {
    const sign = p.dv24_kt > 0 ? '+' : '';
    lines.push(`dv24: ${sign}${p.dv24_kt} kt ${trendArrow(p.trend)}`);
  }
  if (p.ri_candidate) lines.push('RI candidate');
  return lines;
}

export function initChart(canvasId = 'intensity-chart') {
  const ctx = document.getElementById(canvasId).getContext('2d');

  chartInstance = new Chart(ctx, {
    type: 'line',
    data: { datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: 'nearest', intersect: true, axis: 'x' },
      plugins: {
        legend: {
          labels: { color: '#c9d1d9', font: { size: 11 }, boxWidth: 14 }
        },
        tooltip: {
          backgroundColor: 'rgba(13,17,23,0.95)',
          titleColor: '#e6edf3',
          bodyColor: '#c9d1d9',
          borderColor: '#30363d',
          borderWidth: 1,
          callbacks: {
            title: (items) => {
              if (!items.length) return '';
              const item = items[0];
              const slot = compareMode && item.datasetIndex === 1 ? 'B' : 'A';
              const p = currentPoints[slot][item.dataIndex];
              if (!p) return '';
              const absolute = fmtTick(new Date(p.t).getTime());
              if (compareMode && timeMode === 'relative') {
                const hours = Math.round(item.parsed.x);
                const sign = hours >= 0 ? '+' : '';
                return `${sign}${hours}h — ${absolute}`;
              }
              return absolute;
            },
            label: (item) => {
              if (item.dataset.yAxisID === 'yWind') {
                const slot = compareMode && item.datasetIndex === 1 ? 'B' : 'A';
                const p = currentPoints[slot][item.dataIndex];
                if (!p) return '';
                return windTooltipLines(p);
              }
              // Pressure dataset only exists in single mode.
              const p = currentPoints.A[item.dataIndex];
              if (!p) return '';
              return `Pressure: ${p.pressure_mb != null ? p.pressure_mb + ' mb' : '—'}`;
            }
          }
        }
      },
      scales: {
        x: {
          type: 'linear',
          title: { display: false, text: 'Hours since track start', color: '#c9d1d9' },
          ticks: {
            color: '#8b949e',
            maxRotation: 0,
            autoSkip: true,
            callback: (v) => (compareMode && timeMode === 'relative' ? `${Math.round(v)}` : fmtTick(v))
          },
          grid: { color: 'rgba(139,148,158,0.12)' }
        },
        yWind: {
          type: 'linear',
          position: 'left',
          title: { display: true, text: 'Wind (kt)', color: '#c9d1d9' },
          ticks: { color: '#8b949e' },
          grid: { color: 'rgba(139,148,158,0.12)' }
        },
        yPressure: {
          type: 'linear',
          position: 'right',
          display: true,
          title: { display: true, text: 'Pressure (mb)', color: '#8b949e' },
          ticks: { color: '#8b949e' },
          grid: { display: false }
        }
      },
      onHover: (_evt, elements) => {
        if (!hoverCallback) return;
        if (!elements.length) {
          hoverCallback('A', null);
          return;
        }
        const el = elements[0];
        const slot = compareMode && el.datasetIndex === 1 ? 'B' : 'A';
        hoverCallback(slot, el.index);
      }
    }
  });

  return chartInstance;
}

export function clearChart() {
  if (!chartInstance) return;
  chartInstance.data.datasets = [];
  currentPoints = { A: [], B: [] };
  compareMode = false;
  timeMode = 'absolute';
  currentOrigins = { A: 0, B: 0 };
  currentNames = {};
  chartInstance.options.scales.yPressure.display = true;
  chartInstance.options.scales.x.title.display = false;
  chartInstance.update();
}

function buildWindDataset(points, opts, mode = 'absolute', originMs = 0) {
  const windData = points.map((p) => ({ x: computeX(new Date(p.t).getTime(), mode, originMs), y: p.wind_kt }));
  const pointColors = points.map((p) => windColor(p.wind_kt));
  const pointBorderColors = points.map((p) => (p.ri_candidate ? '#ffffff' : windColor(p.wind_kt)));
  const pointBorderWidths = points.map((p) => (p.ri_candidate ? 2 : 1));
  const pointRadii = points.map((p) => (p.ri_candidate ? 6 : 3));
  const pointStyles = points.map((p) => (p.ri_candidate ? 'rectRot' : 'circle'));

  return {
    label: opts.label,
    data: windData,
    yAxisID: 'yWind',
    borderColor: opts.borderColor,
    borderDash: opts.borderDash,
    borderWidth: 2,
    backgroundColor: 'transparent',
    pointBackgroundColor: pointColors,
    pointBorderColor: pointBorderColors,
    pointBorderWidth: pointBorderWidths,
    pointRadius: pointRadii,
    pointHoverRadius: pointRadii.map((r) => r + 3),
    pointStyle: pointStyles,
    spanGaps: false,
    tension: 0.15
  };
}

function buildPressureDataset(points) {
  const pressureData = points.map((p) => ({
    x: new Date(p.t).getTime(),
    y: p.pressure_mb === null || p.pressure_mb === undefined ? null : p.pressure_mb
  }));

  return {
    label: 'Pressure (mb)',
    data: pressureData,
    yAxisID: 'yPressure',
    borderColor: 'rgba(91,163,207,0.6)',
    borderWidth: 1,
    backgroundColor: 'transparent',
    pointRadius: 0,
    pointHoverRadius: 3,
    spanGaps: false,
    tension: 0.15
  };
}

/**
 * @param {object} geojsonA
 * @param {object} definitions unused today (kept for API parity with the map/app layer)
 * @param {object|null} [geojsonB] presence of this arg switches the chart to compare mode
 * @param {{a?: string, b?: string}} [names] storm names for the dataset legend labels
 */
export function renderChart(geojsonA, definitions, geojsonB = null, names = {}) {
  if (!chartInstance) return;

  compareMode = !!geojsonB;
  // Every fresh render (new event, toggling compare) starts back at
  // absolute time; the relative-time checkbox is reset to match by the UI
  // layer (js/app.js) alongside this call.
  timeMode = 'absolute';
  currentNames = names;

  const pointsA = geojsonA.features
    .filter((f) => f.geometry.type === 'Point')
    .map((f) => f.properties);
  currentPoints.A = pointsA;
  currentOrigins.A = pointsA.length ? new Date(pointsA[0].t).getTime() : 0;

  if (!compareMode) {
    currentPoints.B = [];
    currentOrigins.B = 0;
    chartInstance.data.datasets = [
      buildWindDataset(pointsA, { label: 'Wind (kt)', borderColor: 'rgba(230,237,243,0.8)' }),
      buildPressureDataset(pointsA)
    ];
    chartInstance.options.scales.yPressure.display = true;
  } else {
    const pointsB = geojsonB.features
      .filter((f) => f.geometry.type === 'Point')
      .map((f) => f.properties);
    currentPoints.B = pointsB;
    currentOrigins.B = pointsB.length ? new Date(pointsB[0].t).getTime() : 0;

    chartInstance.data.datasets = [
      buildWindDataset(pointsA, { label: names.a || 'Event A', borderColor: 'rgba(230,237,243,0.8)' }),
      buildWindDataset(pointsB, {
        label: names.b || 'Event B',
        borderColor: 'rgba(240,136,62,0.85)',
        borderDash: [6, 4]
      })
    ];
    chartInstance.options.scales.yPressure.display = false;
  }

  chartInstance.options.scales.x.title.display = false;
  chartInstance.update();
}

/**
 * Toggle the compare-mode x-axis between absolute calendar time (default)
 * and hours-since-each-storm's-own-track-start. No-op on the underlying
 * data when not in compare mode (single mode always stays absolute) --
 * callers don't need to guard the call themselves.
 *
 * @param {boolean} enabled true = relative time, false = absolute
 */
export function setRelativeTime(enabled) {
  if (!chartInstance) return;
  timeMode = enabled ? 'relative' : 'absolute';

  if (compareMode) {
    chartInstance.data.datasets[0] = buildWindDataset(
      currentPoints.A,
      { label: currentNames.a || 'Event A', borderColor: 'rgba(230,237,243,0.8)' },
      timeMode,
      currentOrigins.A
    );
    chartInstance.data.datasets[1] = buildWindDataset(
      currentPoints.B,
      { label: currentNames.b || 'Event B', borderColor: 'rgba(240,136,62,0.85)', borderDash: [6, 4] },
      timeMode,
      currentOrigins.B
    );
  }

  chartInstance.options.scales.x.title.display = compareMode && timeMode === 'relative';
  chartInstance.update();
}

/** Whether the chart is currently showing two events (compare mode). */
export function isCompareModeActive() {
  return compareMode;
}

/** Register a callback(slot, index|null) fired when the user hovers a chart point directly. */
export function onPointHover(cb) {
  hoverCallback = cb;
}

/**
 * Programmatically highlight (or clear, with null) the wind-dataset point at
 * `index` in `slot`. In single mode slot is always effectively 'A'.
 */
export function highlightIndex(slot, index) {
  if (!chartInstance) return;
  if (index === null || index === undefined) {
    chartInstance.setActiveElements([]);
    chartInstance.tooltip.setActiveElements([], { x: 0, y: 0 });
    chartInstance.update();
    return;
  }
  const datasetIndex = compareMode && slot === 'B' ? 1 : 0;
  chartInstance.setActiveElements([{ datasetIndex, index }]);
  chartInstance.tooltip.setActiveElements([{ datasetIndex, index }], { x: 0, y: 0 });
  chartInstance.update();
}
