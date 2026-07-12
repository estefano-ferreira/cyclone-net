// Leaflet map: track polyline + one circleMarker per best-track fix,
// colored by the Saffir-Simpson-derived wind scale (js/scale.js).
// Exposes a small hover API used to drive the map <-> chart cross-hover.

import { windColor, trendArrow } from './scale.js';

let markers = [];  // { marker, index }
let trackLayer = null;
let hoverCallback = null;
let activeIndex = null;

const BASE_RADIUS = 5;
const HOVER_RADIUS = 9;

export function initMap(containerId = 'map') {
  const map = L.map(containerId, { zoomControl: true, worldCopyJump: true, minZoom: 2 });
  map.setView([20, -50], 3);

  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors ' +
      '&copy; <a href="https://carto.com/attributions">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 18
  }).addTo(map);

  return map;
}

export function clearTrack(map) {
  for (const { marker } of markers) map.removeLayer(marker);
  markers = [];
  if (trackLayer) {
    map.removeLayer(trackLayer);
    trackLayer = null;
  }
  activeIndex = null;
}

function fmtTime(t) {
  if (!t) return '—';
  return t.replace('T', ' ').replace('Z', ' UTC');
}

function fmtNum(v, unit) {
  return v !== null && v !== undefined ? `${v}${unit}` : '—';
}

export function renderTrack(map, geojson) {
  clearTrack(map);

  const lineFeature = geojson.features.find((f) => f.geometry.type === 'LineString');
  const pointFeatures = geojson.features.filter((f) => f.geometry.type === 'Point');

  if (lineFeature) {
    trackLayer = L.geoJSON(lineFeature, {
      style: { color: '#8b949e', weight: 1.5, opacity: 0.65, dashArray: '2,4' }
    }).addTo(map);
  }

  const bounds = [];

  pointFeatures.forEach((f, i) => {
    const [lon, lat] = f.geometry.coordinates;
    const p = f.properties;
    bounds.push([lat, lon]);

    const color = windColor(p.wind_kt);

    const marker = L.circleMarker([lat, lon], {
      radius: BASE_RADIUS,
      color: p.ri_candidate ? '#ffffff' : color,
      weight: p.ri_candidate ? 2 : 1,
      fillColor: color,
      fillOpacity: 0.9
    });

    const dvSign = p.dv24_kt !== null && p.dv24_kt !== undefined && p.dv24_kt > 0 ? '+' : '';
    const tooltipHtml =
      `<div class="cn-tooltip">` +
      `<div class="cn-tooltip-time">${fmtTime(p.t)}</div>` +
      `<div>Wind: ${fmtNum(p.wind_kt, ' kt')}</div>` +
      `<div>Pressure: ${fmtNum(p.pressure_mb, ' mb')}</div>` +
      `<div>dv24: ${p.dv24_kt !== null && p.dv24_kt !== undefined ? dvSign + p.dv24_kt + ' kt' : '—'} ${trendArrow(p.trend)}</div>` +
      (p.ri_candidate ? `<div class="cn-tooltip-ri">RI candidate</div>` : '') +
      `</div>`;

    marker.bindTooltip(tooltipHtml, { direction: 'top', offset: [0, -6], className: 'cn-leaflet-tooltip' });

    marker.on('mouseover', () => {
      setActive(i);
      if (hoverCallback) hoverCallback(i);
    });
    marker.on('mouseout', () => {
      setActive(null);
      if (hoverCallback) hoverCallback(null);
    });

    marker.addTo(map);
    markers.push({ marker, index: i });
  });

  if (bounds.length) {
    map.fitBounds(bounds, { padding: [24, 24] });
  }
}

function setActive(index) {
  if (activeIndex === index) return;

  if (activeIndex !== null && markers[activeIndex]) {
    const prev = markers[activeIndex].marker;
    prev.setStyle({ radius: BASE_RADIUS });
    prev.closeTooltip();
  }

  activeIndex = index;

  if (index !== null && markers[index]) {
    const m = markers[index].marker;
    m.setStyle({ radius: HOVER_RADIUS });
    m.bringToFront();
    m.openTooltip();
  }
}

/** Register a callback(index|null) fired when the user hovers a marker directly. */
export function onMarkerHover(cb) {
  hoverCallback = cb;
}

/** Programmatically highlight (or clear, with null) the marker at `index`. */
export function highlightIndex(index) {
  setActive(index);
}
