// Leaflet map: track polyline + one circleMarker per best-track fix,
// colored by the Saffir-Simpson-derived wind scale (js/scale.js).
// Exposes a small hover API used to drive the map <-> chart cross-hover.
//
// State is kept per comparison slot ('A' | 'B') so slot A (today's single
// track, dashed grey) and slot B (compare mode, solid orange) can be
// rendered/cleared independently without disturbing each other.

import { windColor, trendArrow } from './scale.js';

const state = {
  A: { markers: [], trackLayer: null, bounds: [] },
  B: { markers: [], trackLayer: null, bounds: [] }
};

let activeKey = null; // { slot, index } | null
let hoverCallback = null;

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

/** Clear a single slot's markers/track. Does not touch the other slot. */
export function clearTrack(map, slot = 'A') {
  const s = state[slot];
  for (const { marker } of s.markers) map.removeLayer(marker);
  s.markers = [];
  if (s.trackLayer) {
    map.removeLayer(s.trackLayer);
    s.trackLayer = null;
  }
  s.bounds = [];
  if (activeKey && activeKey.slot === slot) activeKey = null;
  fitAllBounds(map);
}

function fmtTime(t) {
  if (!t) return '—';
  return t.replace('T', ' ').replace('Z', ' UTC');
}

function fmtNum(v, unit) {
  return v !== null && v !== undefined ? `${v}${unit}` : '—';
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

function fitAllBounds(map) {
  const all = [...state.A.bounds, ...state.B.bounds];
  if (all.length) map.fitBounds(all, { padding: [24, 24] });
}

/**
 * @param {L.Map} map
 * @param {object} geojson
 * @param {'A'|'B'} [slot]
 * @param {{name?: string}} [options] name: storm name for the tooltip header.
 *   Slot B always gets one in practice; slot A only when compare is active
 *   (pass undefined otherwise to preserve the single-mode tooltip exactly).
 */
export function renderTrack(map, geojson, slot = 'A', options = {}) {
  clearTrack(map, slot);
  const s = state[slot];
  const name = options.name;

  const lineFeature = geojson.features.find((f) => f.geometry.type === 'LineString');
  const pointFeatures = geojson.features.filter((f) => f.geometry.type === 'Point');

  const trackStyle = slot === 'B'
    ? { color: '#f0883e', weight: 1.5, opacity: 0.8 }
    : { color: '#8b949e', weight: 1.5, opacity: 0.65, dashArray: '2,4' };

  if (lineFeature) {
    s.trackLayer = L.geoJSON(lineFeature, { style: trackStyle }).addTo(map);
  }

  pointFeatures.forEach((f, i) => {
    const [lon, lat] = f.geometry.coordinates;
    const p = f.properties;
    s.bounds.push([lat, lon]);

    const color = windColor(p.wind_kt);

    // Tri-state ri_candidate: true / false / null (undefined label).
    const riUndefined = p.ri_candidate === null;
    const marker = L.circleMarker([lat, lon], {
      radius: BASE_RADIUS,
      color: p.ri_candidate ? '#ffffff' : riUndefined ? '#9aa0a6' : color,
      weight: p.ri_candidate || riUndefined ? 2 : 1,
      fillColor: color,
      fillOpacity: riUndefined ? 0.6 : 0.9
    });

    const dvSign = p.dv24_kt !== null && p.dv24_kt !== undefined && p.dv24_kt > 0 ? '+' : '';
    const nameHeader = name ? `<div class="cn-tooltip-name">${escapeHtml(name)}</div>` : '';
    const tooltipHtml =
      `<div class="cn-tooltip">` +
      nameHeader +
      `<div class="cn-tooltip-time">${fmtTime(p.t)}</div>` +
      `<div>Wind: ${fmtNum(p.wind_kt, ' kt')}</div>` +
      `<div>Pressure: ${fmtNum(p.pressure_mb, ' mb')}</div>` +
      `<div>dv24: ${p.dv24_kt !== null && p.dv24_kt !== undefined ? dvSign + p.dv24_kt + ' kt' : '—'} ${trendArrow(p.trend)}</div>` +
      (p.ri_candidate ? `<div class="cn-tooltip-ri">RI candidate</div>` :
        riUndefined ? `<div class="cn-tooltip-ri">RI undefined (no 24 h track partner)</div>` : '') +
      `</div>`;

    marker.bindTooltip(tooltipHtml, { direction: 'top', offset: [0, -6], className: 'cn-leaflet-tooltip' });

    marker.on('mouseover', () => {
      setActive(slot, i);
      if (hoverCallback) hoverCallback(slot, i);
    });
    marker.on('mouseout', () => {
      // Only clear if this exact marker is still the active one — guards
      // against a late mouseout landing after the pointer already entered
      // a marker in the other slot (which would otherwise wrongly clobber
      // that slot's highlight).
      if (activeKey && activeKey.slot === slot && activeKey.index === i) {
        setActive(null, null);
      }
      if (hoverCallback) hoverCallback(slot, null);
    });

    marker.addTo(map);
    s.markers.push({ marker, index: i });
  });

  fitAllBounds(map);
}

function setActive(slot, index) {
  const isSame = activeKey && activeKey.slot === slot && activeKey.index === index;
  if (isSame) return;

  if (activeKey) {
    const prevEntry = state[activeKey.slot].markers[activeKey.index];
    if (prevEntry) {
      prevEntry.marker.setStyle({ radius: BASE_RADIUS });
      prevEntry.marker.closeTooltip();
    }
  }

  if (slot === null || index === null || index === undefined) {
    activeKey = null;
    return;
  }

  activeKey = { slot, index };
  const entry = state[slot].markers[index];
  if (entry) {
    entry.marker.setStyle({ radius: HOVER_RADIUS });
    entry.marker.bringToFront();
    entry.marker.openTooltip();
  }
}

/** Register a callback(slot, index|null) fired when the user hovers a marker directly. */
export function onMarkerHover(cb) {
  hoverCallback = cb;
}

/**
 * Programmatically highlight (or clear, with null) the marker at `index` in
 * `slot`. Single-active-at-a-time: highlighting one slot clears the other.
 */
export function highlightIndex(slot, index) {
  if (index === null || index === undefined) {
    setActive(null, null);
    return;
  }
  setActive(slot, index);
}
