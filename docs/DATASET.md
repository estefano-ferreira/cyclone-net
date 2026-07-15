# CycloneNet Dataset Specification

This document describes the dataset used in CycloneNet, including data sources, preprocessing steps, variables, and quality control.

## Data Sources

### 1. ERA5 Reanalysis
- **Provider**: Copernicus Climate Data Store (CDS)
- **Variables**:
  - Sea Surface Temperature (SST) [K]
  - Mean Sea Level Pressure (MSLP) [Pa]
  - 10m u-component of wind (u10) [m/s]
  - 10m v-component of wind (v10) [m/s]
- **Spatial resolution**: 0.25° × 0.25°
- **Temporal resolution**: 6-hourly (00,06,12,18 UTC)
- **Period**: 1979–present (subset to 1989–2024 for this study)
- **Download**: Monthly NetCDF files via CDS API

### 2. IBTrACS Best-Track Data
- **Provider**: NOAA National Centers for Environmental Information
- **Version**: v04r00
- **Variables**:
  - Storm ID (SID)
  - Storm name
  - Basin
  - Timestamp (ISO time)
  - Latitude/Longitude
  - Maximum sustained wind (kt)
  - Minimum central pressure (mb, when available)
- **Temporal resolution**: 6-hourly
- **Period**: 1989–2024
- **Access**: [IBTrACS website](https://www.ncei.noaa.gov/products/international-best-track-archive)

### 3. Tropical Cyclone Heat Potential (TCHP) – Validation Only
- **Provider**: NOAA/AOML
- **Source**: ERDDAP server (https://cwcgom.aoml.noaa.gov/erddap/griddap/aomlTCHP)
- **Variables**:
  - Tropical Cyclone Heat Potential (TCHP) [kJ/cm²]
  - Depth of 26°C isotherm (D26) [m]
- **Spatial resolution**: 0.25° × 0.25°
- **Temporal resolution**: Daily
- **Period**: 1993–present
- **Note**: Used only for validation; never as model input.

## Event Definition

An event is defined as a 6-hourly observation along a storm track. Each event includes:
- The storm center position (lat, lon)
- Environmental fields from ERA5 extracted in a 40×40 grid point window (~10°×10°) centered on the storm.
- Five time steps: t0 (current), t-6h, t-12h, t-18h, t-24h.
- Rapid Intensification (RI) label: 1 if wind speed increases by ≥30 kt over the next 24h, else 0.
- Continuous targets: 12h and 24h wind changes (dv12, dv24) in knots.

## Preprocessing Steps

1. **IBTrACS Processing** (`src/processors/ibtracs.py`):
   - Optional basin filter (disabled in the released configuration — the
     archive spans two basins, East Pacific + North Atlantic; geographic
     selection is done by the bounding box).
   - Remove rows with missing wind or position.
   - Compute RI labels and deltas using 6-hour steps.

2. **ERA5 Extraction** (`src/processors/preprocess_scientific.py`):
   - For each event, locate the corresponding ERA5 monthly file.
   - Extract 40×40 pixel windows around the storm center for each of the 5 time steps.
   - Compute derived channels:
     - Wind speed = sqrt(u10² + v10²)
     - Vorticity = dv/dx - du/dy
     - Divergence = du/dx + dv/dy
     - MSLP gradient magnitude
     - SST anomaly (local mean removal)
   - Apply quality control: reject if center is outside window, excessive NaNs, or temporal collapse.
   - Save cube (H,W,T,C), latitude/longitude grids, and metadata (JSON).

3. **Fuel Potential Prior** (optional):
   - Compute P = relu(SST_anom) * wind_speed * (1 + relu(-divergence)) as a weak supervision signal.
   - Normalized per timestep and saved as `*_fuel_potential.npy`.

4. **TCHP Preprocessing** (`src/processors/preprocess_tchp.py`):
   - For each event, find the corresponding TCHP file.
   - Extract a 5°×5° window around the storm center.
   - Locate the local maximum of TCHP (smoothed) and store coordinates in metadata.

## Variables and Channels

The final data cube has dimensions (H=40, W=40, T=5, C). The channels (C) are:

| Channel Name | Description | Units | Source |
|--------------|-------------|-------|--------|
| sst_K | Sea Surface Temperature | K | ERA5 |
| mslp_Pa | Mean Sea Level Pressure | Pa | ERA5 |
| u10_mps | 10m u-wind | m/s | ERA5 |
| v10_mps | 10m v-wind | m/s | ERA5 |
| wind_mps | Wind speed (calculated) | m/s | Derived |
| vort_1ps | Relative vorticity | s⁻¹ | Derived |
| div_1ps | Divergence | s⁻¹ | Derived |
| grad_mslp_Pa_per_m | MSLP gradient magnitude | Pa/m | Derived |
| sst_anom_K | SST anomaly | K | Derived |

*Note: Heat flux channels (latent, sensible, total) are computed but not used as model inputs to prevent leakage. They are stored for potential loss functions.*

## Data Splits

- Split by storm ID (SID) to prevent leakage.
- Ratios: 70% train, 15% validation, 15% test.
- Splits are fixed and stored in `data/normalized/splits.csv`.

## Quality Control

- Physical ranges: SST ∈ [240,330] K, MSLP ∈ [80000,110000] Pa, wind ≤ 80 m/s.
- Maximum allowed NaN fraction per channel: 50%.
- Temporal integrity: all 5 timesteps must correspond to distinct ERA5 records.
- Geospatial integrity: storm center must lie within the extracted patch.

## Normalization

- Mean and standard deviation computed **only on the training split** for the input channels.
- Statistics saved in `data/normalized/normalization_stats.json`.
- Applied during dataset loading.

## File Structure
