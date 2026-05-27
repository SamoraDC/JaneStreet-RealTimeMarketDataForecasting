# Real Jane Street Project Visuals

This directory contains visualization code and generated chart artifacts derived from real project evidence:

- `data/raw/jane-street-real-time-market-data-forecasting/train.parquet`
- real CSV and JSON outputs under `reports/`
- preserved candidate artifacts under `best-candidates/`

The charts do not use mock data. Some visualizations use deterministic samples or cached aggregates to keep rendering time and readability practical; those samples are still derived from real local parquet rows or real validation artifacts.

## Contents

- `figures/`: static PNG figures for project-level exploratory analysis.
- `animations/`: project-level GIF animations generated with `matplotlib.animation.FuncAnimation`.
- `best-candidates/`: static and animated visual analytics for the three preserved reference candidates.
- `intermidiate-data/`: local cached aggregates derived from real data and ignored by Git.
- `manifest.json`: source paths, parameters, and generated project-level visual files.
- `generate_best_candidate_visuals.py`: generator for the preserved-candidate visual gallery.
- `gerar_graficos.py`: legacy generator for project-level figures and animations.

## Regeneration

Generate the project-level visual package:

```bash
uv run python charts/gerar_graficos.py
```

Generate the preserved-candidate visual gallery:

```bash
uv run python charts/generate_best_candidate_visuals.py
```

Use `--refresh-cache` with the project-level generator to recompute real-data aggregates instead of reusing cached intermediate files.
