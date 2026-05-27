"""Slice diagnostics for frozen strong-family bridge predictions."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl

from multimodels.metrics import TARGET
from multimodels.strong_oof_diagnostics import add_diagnostic_buckets, compare_pair_by_slice


@dataclass(frozen=True)
class BridgeDiagnosticConfig:
    """Configuration for bridge prediction diagnostics."""

    experiment_name: str
    bridge_prediction_path: Path
    output_dir: Path
    candidate: str
    baseline: str
    time_bucket_size: int = 100


def run_bridge_diagnostics(config: BridgeDiagnosticConfig) -> dict[str, Any]:
    """Compare one frozen bridge candidate against a frozen baseline by slices."""

    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pl.read_parquet(config.bridge_prediction_path)
    _require_columns(frame, ["fold", "date_id", "time_id", "symbol_id", "weight", TARGET, config.baseline, config.candidate])
    frame = _ensure_prediction_disagreement(frame)
    frame = add_diagnostic_buckets(frame, baseline=config.baseline, candidate=config.candidate, time_bucket_size=config.time_bucket_size)

    safe_candidate = _safe_name(config.candidate)
    slice_specs = {
        "fold": ("fold",),
        "time_bucket": ("time_bucket",),
        "weight_bucket": ("weight_bucket",),
        "baseline_abs_bucket": ("baseline_abs_bucket",),
        "candidate_abs_bucket": ("candidate_abs_bucket",),
        "disagreement_bucket": ("disagreement_bucket",),
        "fold_weight_bucket": ("fold", "weight_bucket"),
        "symbol_id": ("symbol_id",),
    }
    for name, groups in slice_specs.items():
        compare_pair_by_slice(frame, group_columns=groups, baseline=config.baseline, candidate=config.candidate).write_csv(
            config.output_dir / f"diagnostic_delta_{name}_{safe_candidate}.csv"
        )
    overall = compare_pair_by_slice(frame, group_columns=("__all__",), baseline=config.baseline, candidate=config.candidate)
    summary = _summary_payload(config, overall)
    (config.output_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(config.output_dir / "DIAGNOSTIC_REPORT.md", summary)
    return {"summary": summary, "output_dir": config.output_dir}


def _summary_payload(config: BridgeDiagnosticConfig, overall: pl.DataFrame) -> dict[str, Any]:
    row = overall.row(0, named=True)
    raw = asdict(config)
    raw["bridge_prediction_path"] = str(raw["bridge_prediction_path"])
    raw["output_dir"] = str(raw["output_dir"])
    return {
        "experiment_name": config.experiment_name,
        "config": raw,
        "candidate": config.candidate,
        "baseline": config.baseline,
        "rows": int(row["rows"]),
        "weight_sum": float(row["weight_sum"]),
        "candidate_r2": float(row["candidate_r2"]),
        "baseline_r2": float(row["baseline_r2"]),
        "candidate_delta_r2": float(row["candidate_delta_r2"]),
        "causality_status": "diagnostic only; candidate and baseline are frozen bridge predictions",
        "selection_status": "diagnostic only; no candidate is refit or selected inside this diagnostic",
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# Strong Family Bridge Diagnostics: {summary['experiment_name']}",
        "",
        "## Overall",
        "",
        f"- Candidate: `{summary['candidate']}`.",
        f"- Baseline: `{summary['baseline']}`.",
        f"- Candidate R2: `{summary['candidate_r2']:.9f}`.",
        f"- Baseline R2: `{summary['baseline_r2']:.9f}`.",
        f"- Delta R2: `{summary['candidate_delta_r2']:.9f}`.",
        f"- Rows: `{summary['rows']}`.",
        "",
        "## Audit",
        "",
        f"- Causality status: `{summary['causality_status']}`.",
        f"- Selection status: `{summary['selection_status']}`.",
        "- Positive delta means the bridge reduces weighted squared error versus the baseline in that slice.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _ensure_prediction_disagreement(frame: pl.DataFrame) -> pl.DataFrame:
    if "prediction_disagreement" in frame.columns:
        return frame
    if {"tabm_prediction", "tree_prediction"}.issubset(frame.columns):
        return frame.with_columns((pl.col("tabm_prediction") - pl.col("tree_prediction")).abs().alias("prediction_disagreement"))
    return frame.with_columns(pl.lit(0.0).alias("prediction_disagreement"))


def _require_columns(frame: pl.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _validate_config(config: BridgeDiagnosticConfig) -> None:
    if not config.bridge_prediction_path.exists():
        raise FileNotFoundError(config.bridge_prediction_path)
    if not config.candidate:
        raise ValueError("candidate must not be empty")
    if not config.baseline:
        raise ValueError("baseline must not be empty")
    if config.time_bucket_size <= 0:
        raise ValueError("time_bucket_size must be positive")


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
