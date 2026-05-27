"""Diagnostics for strong OOF candidate comparisons."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import polars as pl

from multimodels.strong_oof import (
    KEYS,
    TARGET,
    add_gateway_rls_predictions,
    add_prediction_context,
    load_joined_predictions,
)


@dataclass(frozen=True)
class DiagnosticConfig:
    """Configuration for strong OOF slice diagnostics."""

    experiment_name: str
    tabm_prediction_dir: Path
    tree_prediction_dir: Path
    output_dir: Path
    candidate: str = "gateway_risk_conservative_rls_abs_pred_s100_prediction"
    baseline: str = "conservative_rls_prediction"
    gateway_risk_strengths: tuple[float, ...] = (25.0, 100.0)
    gateway_risk_profiles: tuple[str, ...] = ("abs_pred",)
    time_bucket_size: int = 100
    sample_stride: int = 1
    max_rows_per_fold: int | None = None


def run_diagnostics(config: DiagnosticConfig) -> dict[str, Any]:
    """Run slice diagnostics for one candidate against one baseline."""

    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_joined_predictions(config.tabm_prediction_dir, config.tree_prediction_dir)
    frame = _apply_sampling(frame, sample_stride=config.sample_stride, max_rows_per_fold=config.max_rows_per_fold)
    frame = add_prediction_context(frame)
    frame, gateway_audits = add_gateway_rls_predictions(
        frame,
        include_risk_shrink=True,
        risk_strengths=config.gateway_risk_strengths,
        risk_profiles=config.gateway_risk_profiles,
    )
    frame = add_prediction_context(frame)
    _require_columns(frame, [config.candidate, config.baseline])
    frame = add_diagnostic_buckets(
        frame,
        baseline=config.baseline,
        candidate=config.candidate,
        time_bucket_size=config.time_bucket_size,
    )

    slice_specs = {
        "fold": ("fold",),
        "time_bucket": ("time_bucket",),
        "weight_bucket": ("weight_bucket",),
        "baseline_abs_bucket": ("baseline_abs_bucket",),
        "candidate_abs_bucket": ("candidate_abs_bucket",),
        "disagreement_bucket": ("disagreement_bucket",),
        "fold_weight_bucket": ("fold", "weight_bucket"),
        "fold_time_bucket": ("fold", "time_bucket"),
        "symbol_id": ("symbol_id",),
    }
    for name, groups in slice_specs.items():
        compare_pair_by_slice(frame, group_columns=groups, baseline=config.baseline, candidate=config.candidate).write_csv(
            config.output_dir / f"delta_vs_baseline_{name}.csv"
        )
    score_overall = compare_pair_by_slice(frame, group_columns=("__all__",), baseline=config.baseline, candidate=config.candidate)
    score_overall.write_csv(config.output_dir / "delta_vs_baseline_overall.csv")

    audit_frame = pl.concat(gateway_audits, how="diagonal") if gateway_audits else pl.DataFrame()
    if not audit_frame.is_empty():
        audit_frame.write_csv(config.output_dir / "gateway_daily_audit.csv")
    summary = _summary_payload(config, frame, score_overall, audit_frame)
    (config.output_dir / "diagnostic_report.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown_summary(config.output_dir / "REPORT.md", summary)
    return {"summary": summary, "output_dir": config.output_dir}


def add_diagnostic_buckets(
    frame: pl.DataFrame,
    *,
    baseline: str,
    candidate: str,
    time_bucket_size: int,
) -> pl.DataFrame:
    """Add target-free diagnostic buckets."""

    if time_bucket_size <= 0:
        raise ValueError("time_bucket_size must be positive")
    thresholds = {
        "weight": _quantiles(frame, "weight", [0.50, 0.90, 0.99]),
        "baseline_abs": _quantiles_expr(frame, pl.col(baseline).abs(), [0.50, 0.90, 0.99]),
        "candidate_abs": _quantiles_expr(frame, pl.col(candidate).abs(), [0.50, 0.90, 0.99]),
        "disagreement": _quantiles(frame, "prediction_disagreement", [0.50, 0.90, 0.99]),
    }
    return frame.with_columns(
        pl.lit("__all__").alias("__all__"),
        (pl.col("time_id") // time_bucket_size).cast(pl.Int16).alias("time_bucket"),
        _bucket_expr(pl.col("weight"), thresholds["weight"]).alias("weight_bucket"),
        _bucket_expr(pl.col(baseline).abs(), thresholds["baseline_abs"]).alias("baseline_abs_bucket"),
        _bucket_expr(pl.col(candidate).abs(), thresholds["candidate_abs"]).alias("candidate_abs_bucket"),
        _bucket_expr(pl.col("prediction_disagreement"), thresholds["disagreement"]).alias("disagreement_bucket"),
    )


def compare_pair_by_slice(
    frame: pl.DataFrame,
    *,
    group_columns: Sequence[str],
    baseline: str,
    candidate: str,
) -> pl.DataFrame:
    """Compare candidate and baseline R2 contributions by group."""

    groups = list(group_columns)
    return (
        frame.group_by(groups)
        .agg(
            pl.len().alias("rows"),
            pl.col("weight").sum().alias("weight_sum"),
            (pl.col("weight") * pl.col(TARGET).pow(2)).sum().alias("denominator"),
            (pl.col("weight") * (pl.col(TARGET) - pl.col(baseline)).pow(2)).sum().alias("baseline_numerator"),
            (pl.col("weight") * (pl.col(TARGET) - pl.col(candidate)).pow(2)).sum().alias("candidate_numerator"),
        )
        .with_columns(
            (1.0 - pl.col("baseline_numerator") / pl.col("denominator")).alias("baseline_r2"),
            (1.0 - pl.col("candidate_numerator") / pl.col("denominator")).alias("candidate_r2"),
            ((pl.col("baseline_numerator") - pl.col("candidate_numerator")) / pl.col("denominator")).alias("candidate_delta_r2"),
        )
        .sort(["candidate_delta_r2", "weight_sum"], descending=[True, True])
    )


def write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    """Write a compact diagnostic report."""

    lines = [
        f"# Strong OOF Diagnostics: {summary['experiment_name']}",
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
        f"- Gateway bad updates: `{summary['gateway_bad_updates']}`.",
        f"- Selection status: `{summary['selection_status']}`.",
        f"- Slice files: `delta_vs_baseline_*.csv`.",
        "",
        "## Interpretation",
        "",
        "- Positive `candidate_delta_r2` means the candidate reduces weighted squared error versus the baseline in that slice.",
        "- Buckets are diagnostic only; they are not used to fit the candidate.",
        "- This report explains where the frozen candidate wins or loses; it is not a new model search by itself.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _summary_payload(config: DiagnosticConfig, frame: pl.DataFrame, overall: pl.DataFrame, audit_frame: pl.DataFrame) -> dict[str, Any]:
    row = overall.row(0, named=True)
    bad_updates = 0
    if not audit_frame.is_empty() and "update_is_strictly_past" in audit_frame.columns:
        bad_updates = int(audit_frame.filter(pl.col("update_is_strictly_past") == False).height)
    return {
        "experiment_name": config.experiment_name,
        "config": _serializable_config(config),
        "candidate": config.candidate,
        "baseline": config.baseline,
        "rows": int(row["rows"]),
        "weight_sum": float(row["weight_sum"]),
        "candidate_r2": float(row["candidate_r2"]),
        "baseline_r2": float(row["baseline_r2"]),
        "candidate_delta_r2": float(row["candidate_delta_r2"]),
        "gateway_bad_updates": bad_updates,
        "selection_status": "diagnostic only; candidate and baseline are fixed before slicing",
    }


def _apply_sampling(frame: pl.DataFrame, *, sample_stride: int, max_rows_per_fold: int | None) -> pl.DataFrame:
    if sample_stride <= 0:
        raise ValueError("sample_stride must be positive")
    lazy = frame.lazy()
    if sample_stride > 1:
        lazy = lazy.with_row_index("__sample_row").filter((pl.col("__sample_row") % sample_stride) == 0).drop("__sample_row")
    sampled = lazy.collect()
    if max_rows_per_fold is not None and max_rows_per_fold > 0:
        sampled = pl.concat([fold_frame.head(max_rows_per_fold) for fold_frame in sampled.partition_by("fold", maintain_order=True)])
    return sampled.sort(list(KEYS))


def _bucket_expr(expr: pl.Expr, thresholds: tuple[float, float, float]) -> pl.Expr:
    return (
        pl.when(expr <= thresholds[0])
        .then(pl.lit("q00_q50"))
        .when(expr <= thresholds[1])
        .then(pl.lit("q50_q90"))
        .when(expr <= thresholds[2])
        .then(pl.lit("q90_q99"))
        .otherwise(pl.lit("q99_q100"))
    )


def _quantiles(frame: pl.DataFrame, column: str, quantiles: Sequence[float]) -> tuple[float, ...]:
    return _quantiles_expr(frame, pl.col(column), quantiles)


def _quantiles_expr(frame: pl.DataFrame, expr: pl.Expr, quantiles: Sequence[float]) -> tuple[float, ...]:
    values = frame.select([expr.quantile(q).alias(str(idx)) for idx, q in enumerate(quantiles)]).row(0)
    return tuple(float(value) for value in values)


def _require_columns(frame: pl.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _validate_config(config: DiagnosticConfig) -> None:
    if config.sample_stride <= 0:
        raise ValueError("sample_stride must be positive")
    if config.time_bucket_size <= 0:
        raise ValueError("time_bucket_size must be positive")
    if not config.candidate or not config.baseline:
        raise ValueError("candidate and baseline must not be empty")


def _serializable_config(config: DiagnosticConfig) -> dict[str, Any]:
    raw = asdict(config)
    for key, value in list(raw.items()):
        if isinstance(value, Path):
            raw[key] = str(value)
    return raw
