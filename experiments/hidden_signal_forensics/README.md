# Hidden Signal Forensics

Investigation sandbox for the hypothesis that a highly noisy financial dataset
may contain planted, low-amplitude, non-obvious predictive structure.

This directory is intentionally separate from `src/janestreet/` and from the
submission pipeline. Results here are exploratory until a rule is frozen and
validated out of sample.

## Layout

- `forensics.py`: numerical utilities for random-matrix, spectral, tail,
  digit, correlation, rank, and simple transformation screens.
- `run_forensic_screen.py`: main executable screen over real Jane Street data.
- `tests/`: local tests for this experiment code.
- `reports/`: generated reports and CSVs.
- `samples/`: optional sampled parquet extracts if a future run needs cached
  samples.

## Default Run

```bash
uv run python experiments/hidden_signal_forensics/run_forensic_screen.py \
  --sample-stride 40 \
  --max-rows 300000 \
  --permutations 10 \
  --output-dir experiments/hidden_signal_forensics/reports/smoke_real_sample
```

The screen uses temporal train/validation separation inside the sampled data.
Permutation results are a discovery-bias null: if the best real rule is not
clearly above the best permuted-target rules, it is not evidence of a hidden
signal.

## Evidence Boundary

Mock data is used only for tests. Any claim about artificiality, planted signal,
or alpha must come from real Kaggle rows and must state the split, sample size,
and null controls used.
