// All DOM rendering for chrome that isn't the map or the chart: banners,
// event selector, metadata panel, legend, footer provenance, error states.

import { WIND_CATEGORIES } from './scale.js';

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------
// Banners / fatal states
// ---------------------------------------------------------------------

export function renderFileProtocolNotice() {
  document.body.innerHTML = `
    <div class="fatal-notice">
      <h1>This page needs to be served over HTTP</h1>
      <p>
        Your browser blocks <code>fetch()</code> of local JSON data when a page is opened
        directly from disk (the <code>file://</code> protocol). This is a browser security
        restriction, not a bug in this page.
      </p>
      <p>Serve this folder over HTTP instead, for example:</p>
      <pre>python -m http.server 8000 --directory platform/site</pre>
      <p>then open <code>http://localhost:8000/</code>, or publish it via GitHub Pages.</p>
    </div>`;
}

export function renderFatalError(message) {
  const area = $('banner-area');
  area.innerHTML = `
    <div class="banner banner-fatal">
      <strong>Could not load application data.</strong>
      <div>${escapeHtml(message)}</div>
    </div>`;
}

export function renderSchemaBanner(actualVersion, supportedVersion) {
  const area = $('banner-area');
  area.innerHTML = `
    <div class="banner banner-warning">
      Data is newer than this page understands (schema ${actualVersion} &gt; ${supportedVersion}) —
      update the page.
    </div>`;
}

export function renderIntegrityError(err, opts = {}) {
  const html = `
    <div class="banner banner-integrity">
      <strong>Data corrupted or tampered — refusing to display.</strong>
      <div>Artifact: <code>${escapeHtml(err.relPath || '')}</code></div>
    </div>`;
  if (opts.inline) {
    $('event-error').innerHTML = html;
  } else {
    $('banner-area').innerHTML = html;
  }
}

export function renderEventError(message) {
  $('event-error').innerHTML = `
    <div class="banner banner-fatal">
      <strong>Could not load event.</strong>
      <div>${escapeHtml(message)}</div>
    </div>`;
}

export function clearEventError() {
  $('event-error').innerHTML = '';
}

export function setBuildVersion(buildVersion) {
  const chip = $('build-chip');
  chip.textContent = buildVersion;
  chip.title = 'data build version — full provenance in footer';
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

// ---------------------------------------------------------------------
// Event selector
// ---------------------------------------------------------------------

let allEvents = [];
let selectHandler = null;
let selectedSidA = null;
let selectedSidB = null;
let compareMode = false;

/**
 * @param {Array} events
 * @param {(sel: {sidA: string|null, sidB: string|null}) => void} onSelectionChange
 *   Fired on every selection change (click, or toggling compare off). Replaces
 *   the old single-sid onSelect(sid) contract.
 */
export function setupEventSelector(events, onSelectionChange) {
  allEvents = [...events].sort((a, b) => (a.start < b.start ? 1 : a.start > b.start ? -1 : 0));
  selectHandler = onSelectionChange;

  const searchInput = $('event-search');
  searchInput.addEventListener('input', () => renderEventList(searchInput.value));

  const compareToggle = $('compare-toggle');
  compareToggle.addEventListener('change', () => {
    compareMode = compareToggle.checked;
    if (!compareMode) {
      // Compare OFF: keep A, clear B, fall back to single-event behavior.
      selectedSidB = null;
    }
    emitSelection(searchInput.value);
  });

  renderEventList('');
}

function emitSelection(query) {
  renderEventList(query);
  if (selectHandler) selectHandler({ sidA: selectedSidA, sidB: selectedSidB });
}

function handleItemClick(sid) {
  if (!compareMode) {
    selectedSidA = sid;
    selectedSidB = null;
  } else if (selectedSidA === sid) {
    // Deselecting the anchor clears the whole comparison — a lone B with no
    // A would be a meaningless state.
    selectedSidA = null;
    selectedSidB = null;
  } else if (selectedSidB === sid) {
    selectedSidB = null;
  } else if (selectedSidA === null) {
    selectedSidA = sid;
  } else {
    // A is anchored; a new event always lands in (or replaces) B.
    selectedSidB = sid;
  }
}

function renderEventList(query) {
  const list = $('event-list');
  const q = query.trim().toLowerCase();

  const filtered = q
    ? allEvents.filter((e) => {
        const year = e.start ? e.start.slice(0, 4) : '';
        return e.name.toLowerCase().includes(q) || year.includes(q);
      })
    : allEvents;

  list.innerHTML = '';
  if (!filtered.length) {
    list.innerHTML = '<li class="event-list-empty">No matching events.</li>';
    return;
  }

  const frag = document.createDocumentFragment();
  for (const e of filtered) {
    const year = e.start ? e.start.slice(0, 4) : '—';
    const isA = e.sid === selectedSidA;
    const isB = e.sid === selectedSidB;
    const li = document.createElement('li');
    li.className = 'event-item' + (isA ? ' selected' : '') + (isB ? ' selected-b' : '');
    li.dataset.sid = e.sid;
    li.innerHTML = `
      <span class="event-item-name">${escapeHtml(e.name)} (${year})</span>
      <span class="event-item-meta">max ${e.max_wind_kt != null ? Math.round(e.max_wind_kt) + 'kt' : '—'}</span>
      ${isB ? '<span class="compare-b-chip">B</span>' : ''}
      ${e.has_ri ? '<span class="ri-badge" title="Rapid intensification candidate present">RI</span>' : ''}
    `;
    li.addEventListener('click', () => {
      handleItemClick(e.sid);
      emitSelection($('event-search').value);
    });
    frag.appendChild(li);
  }
  list.appendChild(frag);
}

// ---------------------------------------------------------------------
// Metadata panel
// ---------------------------------------------------------------------

// SID -> basin code (e.g. "NA") -> display name, from definitions.basin_names.
// Populated once at boot via setBasinNames(); see js/app.js.
let basinNames = {};

/** Register the basin code -> display name map (definitions.basin_names). */
export function setBasinNames(map) {
  basinNames = map || {};
}

/** "North Atlantic (NA)"; "—" for an empty or unrecognized code. */
function fmtBasin(code) {
  if (!code) return '—';
  const name = basinNames[code];
  if (!name) return '—';
  return `${escapeHtml(name)} (${escapeHtml(code)})`;
}

function fmtPeriodDate(iso) {
  if (!iso) return '—';
  return iso.replace('T', ' ').replace('Z', ' UTC');
}

function fmtDuration(startIso, endIso) {
  if (!startIso || !endIso) return '—';
  const ms = new Date(endIso).getTime() - new Date(startIso).getTime();
  if (!Number.isFinite(ms) || ms < 0) return '—';
  const totalHours = Math.round(ms / 3_600_000);
  const days = Math.floor(totalHours / 24);
  const hours = totalHours % 24;
  if (days > 0) return `${days}d ${hours}h`;
  return `${hours}h`;
}

function metaRowsHtml(meta) {
  return `
      <dt>SID</dt><dd class="mono">${escapeHtml(meta.sid)}</dd>
      <dt>Basin</dt><dd>${fmtBasin(meta.basin)}</dd>
      <dt>Period</dt><dd>${fmtPeriodDate(meta.start)} &ndash; ${fmtPeriodDate(meta.end)}</dd>
      <dt>Duration</dt><dd>${fmtDuration(meta.start, meta.end)}</dd>
      <dt>Max wind</dt><dd>${meta.max_wind_kt != null ? meta.max_wind_kt + ' kt' : '—'}</dd>
      <dt>Min pressure</dt><dd>${meta.min_pressure_mb != null ? meta.min_pressure_mb + ' mb' : '—'}</dd>
      <dt>Points</dt><dd>${meta.n_points}</dd>
      <dt>RI observed</dt><dd>${meta.has_ri ? '<span class="ri-badge">RI</span>' : 'no'}</dd>`;
}

export function renderMetadata(meta) {
  const panel = $('metadata-panel');
  if (!meta) {
    panel.innerHTML = '<p class="placeholder">Select an event to inspect its track.</p>';
    return;
  }
  panel.innerHTML = `
    <dl class="meta-list">
      <dt>Name</dt><dd>${escapeHtml(meta.name)}</dd>
      ${metaRowsHtml(meta)}
    </dl>`;
}

/**
 * Two-column side-by-side metadata for compare mode. Same fields as
 * renderMetadata minus the Name row (carried by the column header instead).
 * Observed best-track facts only — no derived/comparative text (see
 * project honesty guardrail).
 */
export function renderMetadataCompare(metaA, metaB) {
  const panel = $('metadata-panel');
  if (!metaA || !metaB) return;
  const yearA = metaA.start ? metaA.start.slice(0, 4) : '';
  const yearB = metaB.start ? metaB.start.slice(0, 4) : '';
  panel.innerHTML = `
    <div class="meta-compare">
      <div class="meta-compare-col">
        <h3 class="meta-compare-header">${escapeHtml(metaA.name)}${yearA ? ' (' + yearA + ')' : ''}</h3>
        <dl class="meta-list">${metaRowsHtml(metaA)}</dl>
      </div>
      <div class="meta-compare-col">
        <h3 class="meta-compare-header meta-compare-header-b">${escapeHtml(metaB.name)}${yearB ? ' (' + yearB + ')' : ''}</h3>
        <dl class="meta-list">${metaRowsHtml(metaB)}</dl>
      </div>
    </div>`;
}

// ---------------------------------------------------------------------
// Relative-time toggle (compare-mode intensity chart)
//
// Only meaningful/visible when the chart is showing two events: absolute
// calendar time separates storms from different years into disconnected
// clumps, so this switches the chart to hours-since-each-storm's-own-track-
// start instead. Single mode never shows this control.
// ---------------------------------------------------------------------

/** Wire the toggle's change event to onToggle(checked). Call once at boot. */
export function setupRelativeTimeToggle(onToggle) {
  const checkbox = $('relative-time-toggle');
  if (!checkbox) return;
  checkbox.addEventListener('change', () => onToggle(checkbox.checked));
}

/** Show/hide the toggle row; hiding also force-unchecks it (visual reset —
 * the chart's own time mode is reset independently by chart.js::renderChart). */
export function setRelativeTimeToggleVisible(visible) {
  const row = $('chart-time-toggle-row');
  if (row) row.hidden = !visible;
  if (!visible) {
    const checkbox = $('relative-time-toggle');
    if (checkbox) checkbox.checked = false;
  }
}

// ---------------------------------------------------------------------
// Environmental conditions panel
//
// Shows facts only (pressure + reanalysis-derived env_* values) for the
// currently hovered track point, "sticky" to the last-hovered point once
// the mouse leaves (never resets to a blank/placeholder state just because
// the cursor moved off a marker). Slot-A only by design: compare mode's
// slot B keeps its own map/chart cross-highlight working untouched, this
// panel just doesn't track it (see build task notes — avoids overbuilding
// a per-slot env UI for a feature nobody asked to compare side by side).
// Backward compatible: geojsons built without --with-env simply lack the
// env_* keys on every point, which this panel detects and reports plainly.
// ---------------------------------------------------------------------

const ENV_KEYS = ['env_sst_c', 'env_shear_mps', 'env_rh_pct'];

let envPointsA = [];
let envLastIndex = null; // sticky last-hovered/last-selected index into envPointsA

function eventHasEnvKeys(points) {
  return points.some((p) => ENV_KEYS.some((k) => k in p));
}

function envValueHtml(value, unit) {
  if (value === null || value === undefined) {
    return `<span class="env-na" title="not available for this point (no processed cube or channel)">&mdash;</span>`;
  }
  return `${escapeHtml(String(value))}${unit}`;
}

/**
 * Register the point-properties arrays backing the env panel. Call this
 * whenever the selected event(s) change (single mode: pointsB omitted).
 * Resets the sticky index to the storm's most recent point so the panel
 * shows something meaningful before the user hovers anything.
 */
export function setEnvPointsSource(pointsA, _pointsB = []) {
  envPointsA = pointsA || [];
  envLastIndex = envPointsA.length ? envPointsA.length - 1 : null;
  renderEnvPanel();
}

/** Clear the panel back to its pre-selection placeholder (no event chosen). */
export function clearEnvPanel() {
  envPointsA = [];
  envLastIndex = null;
  renderEnvPanel();
}

/**
 * Feed of the existing map/chart cross-hover plumbing. Slot 'B' hovers are
 * intentionally ignored here (see module note above); a null index (mouse
 * left the marker/point) is ALSO ignored so the panel stays on the last
 * real point instead of collapsing to a placeholder mid-inspection.
 */
export function updateEnvPanel(slot, index) {
  if (slot !== 'A') return;
  if (index === null || index === undefined) return;
  envLastIndex = index;
  renderEnvPanel();
}

function renderEnvPanel() {
  const container = $('env-panel');
  if (!container) return;

  if (!envPointsA.length) {
    container.innerHTML = '<p class="placeholder">Select an event to inspect its environment.</p>';
    return;
  }

  if (!eventHasEnvKeys(envPointsA)) {
    container.innerHTML = '<p class="env-quiet">Environmental values not included in this data build.</p>';
    return;
  }

  const p = envLastIndex !== null ? envPointsA[envLastIndex] : undefined;
  if (!p) {
    container.innerHTML = '<p class="placeholder">Hover a track point to inspect.</p>';
    return;
  }

  container.innerHTML = `
    <dl class="env-list">
      <dt>Pressure</dt><dd>${envValueHtml(p.pressure_mb, ' mb')}</dd>
      <dt>SST</dt><dd>${envValueHtml(p.env_sst_c, ' &deg;C')}</dd>
      <dt>Deep-layer shear (850&ndash;200 hPa)</dt><dd>${envValueHtml(p.env_shear_mps, ' m/s')}</dd>
      <dt>Mid-level RH</dt><dd>${envValueHtml(p.env_rh_pct, '%')}</dd>
    </dl>`;
}

// ---------------------------------------------------------------------
// Legend
// ---------------------------------------------------------------------

export function renderLegend(definitions) {
  const container = $('legend-content');

  const categoryRows = WIND_CATEGORIES.map(
    (c) => `
      <div class="legend-row">
        <span class="legend-swatch" style="background:${c.color}"></span>
        <span class="legend-label">${c.label}</span>
        <span class="legend-range">${c.range}</span>
      </div>`
  ).join('');

  const ri = definitions.ri || {};
  const dvConvention = definitions.dv_convention || '';

  container.innerHTML = `
    <div class="legend-categories">${categoryRows}</div>
    <div class="legend-note">
      <span class="legend-halo"></span>
      White halo = RI (rapid intensification) candidate.<br>
      Criterion: <code>${escapeHtml(ri.criterion || 'n/a')}</code><br>
      Reference: ${escapeHtml(ri.reference || 'n/a')}
    </div>
    <div class="legend-note">
      Trend arrows: <span class="trend-up">&#9650;</span> strengthening
      / <span class="trend-down">&#9660;</span> weakening
      / &ndash; steady.<br>
      Convention: ${escapeHtml(dvConvention)}
    </div>`;
}

// ---------------------------------------------------------------------
// Footer provenance
// ---------------------------------------------------------------------

export function renderFooter(definitions) {
  const footer = $('app-footer');
  const src = definitions.source || {};
  const accessedDate = src.accessed ? src.accessed.slice(0, 10) : 'unknown date';

  footer.innerHTML = `
    <p>
      Historical best-track observations (${escapeHtml(src.dataset || 'unknown dataset')},
      DOI <a href="https://doi.org/${encodeURIComponent(src.doi || '')}" target="_blank" rel="noopener">${escapeHtml(src.doi || 'n/a')}</a>,
      accessed ${accessedDate}) &mdash; ${escapeHtml(definitions.note || 'observed data, not predictions.')}
    </p>
    <p>${escapeHtml(definitions.temporal_resolution || '')}</p>
    <p>Source: <a href="${escapeHtml(src.url || '#')}" target="_blank" rel="noopener">${escapeHtml(src.url || 'n/a')}</a></p>
  `;
}
