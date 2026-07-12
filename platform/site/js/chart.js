// Chart.js intensity chart: wind (kt, left axis) colored per-point by the
// Saffir-Simpson scale, pressure (mb, right axis) as a thinner line with
// gaps where missing. Exposes a small hover API for map <-> chart cross-hover.

import { windColor, trendArrow } from './scale.js';

let chartInstance = null;
let hoverCallback = null;
let currentPoints = [];

function fmtTick(ms) {
  const d = new Date(ms);
  const mm = String(d.getUTCMonth() + 1).padStart(2, '0');
  const dd = String(d.getUTCDate()).padStart(2, '0');
  const hh = String(d.getUTCHours()).padStart(2, '0');
  return `${d.getUTCFullYear()}-${mm}-${dd} ${hh}h`;
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
            title: (items) => (items.length ? fmtTick(items[0].parsed.x) : ''),
            label: (item) => {
              const p = currentPoints[item.dataIndex];
              if (!p) return '';
              if (item.dataset.yAxisID === 'yWind') {
                const lines = [`Wind: ${p.wind_kt != null ? p.wind_kt + ' kt' : '—'}`];
                if (p.dv24_kt != null) {
                  const sign = p.dv24_kt > 0 ? '+' : '';
                  lines.push(`dv24: ${sign}${p.dv24_kt} kt ${trendArrow(p.trend)}`);
                }
                if (p.ri_candidate) lines.push('RI candidate');
                return lines;
              }
              return `Pressure: ${p.pressure_mb != null ? p.pressure_mb + ' mb' : '—'}`;
            }
          }
        }
      },
      scales: {
        x: {
          type: 'linear',
          ticks: { color: '#8b949e', maxRotation: 0, autoSkip: true, callback: (v) => fmtTick(v) },
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
          title: { display: true, text: 'Pressure (mb)', color: '#8b949e' },
          ticks: { color: '#8b949e' },
          grid: { display: false }
        }
      },
      onHover: (_evt, elements) => {
        if (!hoverCallback) return;
        hoverCallback(elements.length ? elements[0].index : null);
      }
    }
  });

  return chartInstance;
}

export function clearChart() {
  if (!chartInstance) return;
  chartInstance.data.datasets = [];
  currentPoints = [];
  chartInstance.update();
}

export function renderChart(geojson) {
  if (!chartInstance) return;

  const points = geojson.features
    .filter((f) => f.geometry.type === 'Point')
    .map((f) => f.properties);
  currentPoints = points;

  const windData = points.map((p) => ({ x: new Date(p.t).getTime(), y: p.wind_kt }));
  const pressureData = points.map((p) => ({
    x: new Date(p.t).getTime(),
    y: p.pressure_mb === null || p.pressure_mb === undefined ? null : p.pressure_mb
  }));

  const pointColors = points.map((p) => windColor(p.wind_kt));
  const pointBorderColors = points.map((p) => (p.ri_candidate ? '#ffffff' : windColor(p.wind_kt)));
  const pointBorderWidths = points.map((p) => (p.ri_candidate ? 2 : 1));
  const pointRadii = points.map((p) => (p.ri_candidate ? 6 : 3));
  const pointStyles = points.map((p) => (p.ri_candidate ? 'rectRot' : 'circle'));

  chartInstance.data.datasets = [
    {
      label: 'Wind (kt)',
      data: windData,
      yAxisID: 'yWind',
      borderColor: 'rgba(230,237,243,0.8)',
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
    },
    {
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
    }
  ];

  chartInstance.update();
}

/** Register a callback(index|null) fired when the user hovers a chart point directly. */
export function onPointHover(cb) {
  hoverCallback = cb;
}

/** Programmatically highlight (or clear, with null) the wind-dataset point at `index`. */
export function highlightIndex(index) {
  if (!chartInstance) return;
  if (index === null || index === undefined) {
    chartInstance.setActiveElements([]);
    chartInstance.tooltip.setActiveElements([], { x: 0, y: 0 });
  } else {
    chartInstance.setActiveElements([{ datasetIndex: 0, index }]);
    chartInstance.tooltip.setActiveElements([{ datasetIndex: 0, index }], { x: 0, y: 0 });
  }
  chartInstance.update();
}
