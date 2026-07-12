// Boot sequence for the CycloneNet Event Explorer.
// See platform/site/js/loader.js for the integrity-checked fetch pipeline.

import { createLoader, IntegrityError, SUPPORTED_SCHEMA_VERSION } from './loader.js';
import * as mapApi from './map.js';
import * as chartApi from './chart.js';
import * as ui from './ui.js';

async function boot() {
  let manifest;
  try {
    const res = await fetch('data/manifest.json', { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    manifest = await res.json();
  } catch (err) {
    if (location.protocol === 'file:') {
      ui.renderFileProtocolNotice();
    } else {
      ui.renderFatalError(`Could not load data/manifest.json: ${err.message}`);
    }
    return;
  }

  if (manifest.schema_version > SUPPORTED_SCHEMA_VERSION) {
    ui.renderSchemaBanner(manifest.schema_version, SUPPORTED_SCHEMA_VERSION);
    return; // Stop loading data; do not attempt to interpret a newer schema.
  }

  ui.setBuildVersion(manifest.build_version);

  const load = createLoader(manifest);

  let eventsIndex;
  let definitions;
  try {
    [eventsIndex, definitions] = await Promise.all([
      load('events_index.json'),
      load('definitions.json')
    ]);
  } catch (err) {
    if (err instanceof IntegrityError) {
      ui.renderIntegrityError(err);
    } else {
      ui.renderFatalError(err.message);
    }
    return;
  }

  const availableSids = new Set(manifest.events);
  const events = eventsIndex.filter((e) => availableSids.has(e.sid));

  ui.renderLegend(definitions);
  ui.renderFooter(definitions);

  const map = mapApi.initMap('map');
  const chart = chartApi.initChart('intensity-chart');

  // Cross-hover wiring: hovering a marker highlights the matching chart
  // point and vice versa. Neither call re-enters the other (map hover
  // only fires on real mouse events over a marker DOM node; chart onHover
  // only fires on real canvas mouse events), so this cannot loop. Slot-aware
  // now: (slot, index|null) identifies which of A/B was hovered.
  mapApi.onMarkerHover((slot, index) => chartApi.highlightIndex(slot, index));
  chartApi.onPointHover((slot, index) => mapApi.highlightIndex(slot, index));

  // geojsons are integrity-verified by the loader; caching the parsed
  // result (not re-fetching) is safe and avoids refetching on every
  // compare toggle / re-selection of an already-seen event.
  const geojsonCache = new Map();
  async function loadGeojson(sid) {
    if (geojsonCache.has(sid)) return geojsonCache.get(sid);
    const geojson = await load(`events/${sid}.geojson`);
    geojsonCache.set(sid, geojson);
    return geojson;
  }

  ui.setupEventSelector(events, async ({ sidA, sidB }) => {
    ui.clearEventError();

    if (!sidA) {
      mapApi.clearTrack(map, 'A');
      mapApi.clearTrack(map, 'B');
      chartApi.clearChart();
      ui.renderMetadata(null);
      return;
    }

    let geojsonA;
    try {
      geojsonA = await loadGeojson(sidA);
    } catch (err) {
      if (err instanceof IntegrityError) {
        ui.renderIntegrityError(err, { inline: true });
      } else {
        ui.renderEventError(err.message);
      }
      return;
    }
    const metaA = events.find((e) => e.sid === sidA);

    if (!sidB) {
      mapApi.clearTrack(map, 'B');
      mapApi.renderTrack(map, geojsonA, 'A');
      chartApi.renderChart(geojsonA, definitions);
      ui.renderMetadata(metaA);
      return;
    }

    let geojsonB;
    try {
      geojsonB = await loadGeojson(sidB);
    } catch (err) {
      // Keep A rendered as a single-event view; surface the inline error
      // for B without losing A.
      mapApi.clearTrack(map, 'B');
      mapApi.renderTrack(map, geojsonA, 'A');
      chartApi.renderChart(geojsonA, definitions);
      ui.renderMetadata(metaA);
      if (err instanceof IntegrityError) {
        ui.renderIntegrityError(err, { inline: true });
      } else {
        ui.renderEventError(err.message);
      }
      return;
    }
    const metaB = events.find((e) => e.sid === sidB);

    mapApi.renderTrack(map, geojsonA, 'A', { name: metaA.name });
    mapApi.renderTrack(map, geojsonB, 'B', { name: metaB.name });
    chartApi.renderChart(geojsonA, definitions, geojsonB, { a: metaA.name, b: metaB.name });
    ui.renderMetadataCompare(metaA, metaB);
  });
}

boot();
