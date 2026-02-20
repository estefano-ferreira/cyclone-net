#!/usr/bin/env python3
"""Validate the energy proxy against latent/sensible heat fluxes.

Run after preprocessing to generate correlation statistics and plots.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

INTERIM_DIR = Path("data/interim")
FIGURES_DIR = Path("outputs/figures/proxy_validation")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_fields(event_id):
    energy = np.load(INTERIM_DIR / f"{event_id}_energy.npy")
    latent = np.load(INTERIM_DIR / f"{event_id}_latent.npy")
    sensible = np.load(INTERIM_DIR / f"{event_id}_sensible.npy")
    return energy, latent, sensible


def compute_spatial_correlation(field1, field2):
    mask = (~np.isnan(field1)) & (~np.isnan(field2))
    if np.sum(mask) < 10:
        return np.nan
    f1 = field1[mask].flatten()
    f2 = field2[mask].flatten()
    return pearsonr(f1, f2)[0]


def main():
    # Find all events with energy and flux fields
    event_ids = [p.stem.replace("_energy", "")
                 for p in INTERIM_DIR.glob("*_energy.npy")]
    results = []
    for eid in event_ids:
        meta_path = INTERIM_DIR / f"{eid}.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get('ri_label', 0) != 1:
            continue  # only positive RI events
        try:
            energy, latent, sensible = load_fields(eid)
        except FileNotFoundError:
            continue
        corr_energy_latent = compute_spatial_correlation(energy, latent)
        corr_energy_sensible = compute_spatial_correlation(energy, sensible)
        results.append({
            'event_id': eid,
            'storm_name': meta.get('storm_name', ''),
            'corr_energy_latent': corr_energy_latent,
            'corr_energy_sensible': corr_energy_sensible,
        })

    # Summary statistics
    corr_l = [r['corr_energy_latent']
              for r in results if not np.isnan(r['corr_energy_latent'])]
    corr_s = [r['corr_energy_sensible']
              for r in results if not np.isnan(r['corr_energy_sensible'])]
    print(f"Number of RI events with flux data: {len(results)}")
    print(
        f"Correlation with latent heat flux: mean={np.mean(corr_l):.3f}, std={np.std(corr_l):.3f}")
    print(
        f"Correlation with sensible heat flux: mean={np.mean(corr_s):.3f}, std={np.std(corr_s):.3f}")

    # Plot example for one event
    if results:
        eid = results[0]['event_id']
        energy, latent, sensible = load_fields(eid)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        im0 = axes[0].imshow(energy, cmap='hot')
        axes[0].set_title('Energy Proxy')
        plt.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(latent, cmap='viridis')
        axes[1].set_title('Latent Heat Flux')
        plt.colorbar(im1, ax=axes[1])
        im2 = axes[2].imshow(sensible, cmap='viridis')
        axes[2].set_title('Sensible Heat Flux')
        plt.colorbar(im2, ax=axes[2])
        plt.suptitle(f'Event {eid}')
        plt.savefig(FIGURES_DIR / f'example_{eid}.png', dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
