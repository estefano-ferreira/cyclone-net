// Saffir-Simpson-derived color scale shared by the map, the chart and the
// sidebar legend, so the three stay in lockstep by construction.

export const WIND_CATEGORIES = [
  { key: 'td',   label: 'TD',   range: '< 34 kt',      min: -Infinity, max: 34,        color: '#5ba3cf' },
  { key: 'ts',   label: 'TS',   range: '34-63 kt',     min: 34,        max: 64,        color: '#4fc47f' },
  { key: 'cat1', label: 'Cat 1', range: '64-82 kt',    min: 64,        max: 83,        color: '#f7d154' },
  { key: 'cat2', label: 'Cat 2', range: '83-95 kt',    min: 83,        max: 96,        color: '#f5a623' },
  { key: 'cat3', label: 'Cat 3', range: '96-112 kt',   min: 96,        max: 113,       color: '#f16a3e' },
  { key: 'cat4', label: 'Cat 4', range: '113-136 kt',  min: 113,       max: 137,       color: '#e0301e' },
  { key: 'cat5', label: 'Cat 5', range: '>= 137 kt',   min: 137,       max: Infinity,  color: '#b81365' }
];

export const UNKNOWN_COLOR = '#8b949e';

/** Return the hex color for a given wind speed in knots (or UNKNOWN_COLOR if null/undefined). */
export function windColor(windKt) {
  if (windKt === null || windKt === undefined || Number.isNaN(windKt)) return UNKNOWN_COLOR;
  for (const cat of WIND_CATEGORIES) {
    if (windKt >= cat.min && windKt < cat.max) return cat.color;
  }
  return UNKNOWN_COLOR;
}

/** Trend arrow glyph, per definitions.json dv_convention (forward deltas). */
export function trendArrow(trend) {
  if (trend === 'strengthening') return '▲'; // ▲
  if (trend === 'weakening') return '▼';     // ▼
  return '–';                                 // – (steady / unknown)
}
