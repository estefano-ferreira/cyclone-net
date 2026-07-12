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
  // only fires on real canvas mouse events), so this cannot loop.
  mapApi.onMarkerHover((index) => chartApi.highlightIndex(index));
  chartApi.onPointHover((index) => mapApi.highlightIndex(index));

  ui.setupEventSelector(events, async (sid) => {
    ui.clearEventError();
    let geojson;
    try {
      geojson = await load(`events/${sid}.geojson`);
    } catch (err) {
      if (err instanceof IntegrityError) {
        ui.renderIntegrityError(err, { inline: true });
      } else {
        ui.renderEventError(err.message);
      }
      return;
    }

    const meta = events.find((e) => e.sid === sid);
    ui.renderMetadata(meta);
    mapApi.renderTrack(map, geojson, definitions);
    chartApi.renderChart(geojson, definitions);
  });
}

boot();
