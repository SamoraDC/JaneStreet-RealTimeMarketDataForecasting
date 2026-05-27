"""Nested selection over saved fold-level sufficient statistics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import polars as pl

from multimodels.metrics import summarize_scores


REQUIRED_COLUMNS = {
    "fold",
    "candidate",
    "family",
    "rows",
    "weight_sum",
    "numerator",
    "denominator",
    "weighted_zero_mean_r2",
}


@dataclass(frozen=True)
class FoldScoreSource:
    """One named source of fold-level candidate scores."""

    name: str
    path: Path


@dataclass(frozen=True)
class FoldScoreSelectionConfig:
    """Configuration for fold-level nested candidate selection."""

    output_dir: Path
    sources: tuple[FoldScoreSource, ...]
    candidate_ids: tuple[str, ...]
    first_candidate_id: str
    min_history_folds: int = 1
    selection_metric: str = "global_r2"


def run_fold_score_selection(config: FoldScoreSelectionConfig) -> dict[str, Any]:
    """Select candidates fold-by-fold using only earlier fold scores."""

    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    scores = load_fold_scores(config.sources)
    if config.candidate_ids:
        scores = scores.filter(pl.col("candidate_id").is_in(config.candidate_ids))
    if scores.height == 0:
        raise ValueError("candidate pool is empty after filtering")

    pool_summary = summarize_candidate_pool(scores)
    selected = select_nested_by_previous_folds(
        scores,
        candidate_ids=config.candidate_ids,
        first_candidate_id=config.first_candidate_id,
        min_history_folds=config.min_history_folds,
        selection_metric=config.selection_metric,
    )
    nested_scores = selected.select(
        "fold",
        pl.lit("nested_previous_fold_selector").alias("candidate"),
        pl.lit("nested_selector").alias("family"),
        "rows",
        "weight_sum",
        "numerator",
        "denominator",
        "weighted_zero_mean_r2",
    )
    nested_summary = summarize_scores(nested_scores)

    pool_summary.write_csv(config.output_dir / "candidate_pool_summary.csv")
    selected.write_csv(config.output_dir / "nested_selection_by_fold.csv")
    nested_summary.write_csv(config.output_dir / "nested_selection_summary.csv")

    audit = {
        "sources": [{"name": source.name, "path": str(source.path)} for source in config.sources],
        "candidate_ids": list(config.candidate_ids),
        "first_candidate_id": config.first_candidate_id,
        "min_history_folds": config.min_history_folds,
        "selection_metric": config.selection_metric,
        "folds": selected["fold"].to_list(),
        "selected_candidate_ids": selected["selected_candidate_id"].to_list(),
        "target_leakage_check": "passed: selector uses only saved fold-level sufficient statistics",
        "fold_causality_check": "passed: each selected fold uses only earlier folds except the predeclared first-fold candidate",
        "row_level_prediction_check": "not applicable: this diagnostic selects whole-fold candidate statistics and does not construct row-level predictions",
        "promotion_check": "diagnostic only unless reimplemented as a deployable predeclared time policy",
    }
    (config.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    _write_report(config.output_dir / "REPORT.md", config=config, pool_summary=pool_summary, nested_summary=nested_summary, selected=selected)

    return {
        "scores": scores,
        "pool_summary": pool_summary,
        "selected": selected,
        "nested_summary": nested_summary,
        "audit": audit,
    }


def load_fold_scores(sources: Sequence[FoldScoreSource]) -> pl.DataFrame:
    """Load and namespace fold score CSV files."""

    frames: list[pl.DataFrame] = []
    for source in sources:
        frame = pl.read_csv(source.path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{source.path} is missing required columns: {sorted(missing)}")
        frames.append(
            frame.select(
                "fold",
                "candidate",
                "family",
                "rows",
                "weight_sum",
                "numerator",
                "denominator",
                "weighted_zero_mean_r2",
            ).with_columns(
                pl.lit(source.name).alias("source"),
                (pl.lit(source.name + "/") + pl.col("candidate")).alias("candidate_id"),
                (pl.lit(source.name + "/") + pl.col("family")).alias("source_family"),
            )
        )
    if not frames:
        raise ValueError("at least one source is required")
    return pl.concat(frames, how="vertical")


def summarize_candidate_pool(scores: pl.DataFrame) -> pl.DataFrame:
    """Summarize candidate IDs across folds."""

    return (
        scores.group_by(["candidate_id", "source", "candidate", "family"])
        .agg(
            pl.col("rows").sum().alias("rows"),
            pl.col("weight_sum").sum().alias("weight_sum"),
            pl.col("numerator").sum().alias("numerator"),
            pl.col("denominator").sum().alias("denominator"),
            pl.col("weighted_zero_mean_r2").mean().alias("mean_fold_r2"),
            pl.col("weighted_zero_mean_r2").min().alias("min_fold_r2"),
            pl.col("weighted_zero_mean_r2").std().fill_null(0.0).alias("std_fold_r2"),
        )
        .with_columns((1.0 - pl.col("numerator") / pl.col("denominator")).alias("global_r2"))
        .sort(["global_r2", "min_fold_r2"], descending=[True, True])
    )


def select_nested_by_previous_folds(
    scores: pl.DataFrame,
    *,
    candidate_ids: Sequence[str],
    first_candidate_id: str,
    min_history_folds: int,
    selection_metric: str,
) -> pl.DataFrame:
    """Return selected fold rows using only previous folds for selection."""

    folds = sorted(scores["fold"].unique().to_list())
    if not folds:
        raise ValueError("scores contain no folds")
    candidate_filter = tuple(candidate_ids) if candidate_ids else tuple(sorted(scores["candidate_id"].unique().to_list()))
    selected_rows: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(folds):
        current = scores.filter((pl.col("fold") == fold) & pl.col("candidate_id").is_in(candidate_filter))
        if current.height == 0:
            raise ValueError(f"no candidates available for fold {fold}")
        if fold_index < min_history_folds:
            selected_id = first_candidate_id
            selection_score = None
            history_folds = 0
        else:
            previous_folds = folds[:fold_index]
            history = scores.filter((pl.col("fold").is_in(previous_folds)) & pl.col("candidate_id").is_in(candidate_filter))
            ranking = _rank_candidates_from_history(history, selection_metric=selection_metric, min_history_folds=min_history_folds)
            if ranking.height == 0:
                selected_id = first_candidate_id
                selection_score = None
                history_folds = 0
            else:
                best = ranking.row(0, named=True)
                selected_id = str(best["candidate_id"])
                selection_score = float(best["selection_score"])
                history_folds = int(best["history_folds"])
        selected_current = current.filter(pl.col("candidate_id") == selected_id)
        if selected_current.height != 1:
            raise ValueError(f"selected candidate {selected_id!r} is not uniquely available for fold {fold}")
        row = selected_current.row(0, named=True)
        row["selected_candidate_id"] = selected_id
        row["selection_score"] = selection_score
        row["history_folds"] = history_folds
        row["uses_current_fold_for_selection"] = False
        selected_rows.append(row)
    return pl.DataFrame(selected_rows)


def _rank_candidates_from_history(history: pl.DataFrame, *, selection_metric: str, min_history_folds: int) -> pl.DataFrame:
    grouped = (
        history.group_by("candidate_id")
        .agg(
            pl.len().alias("history_folds"),
            pl.col("numerator").sum().alias("history_numerator"),
            pl.col("denominator").sum().alias("history_denominator"),
            pl.col("weighted_zero_mean_r2").mean().alias("history_mean_r2"),
            pl.col("weighted_zero_mean_r2").min().alias("history_min_r2"),
        )
        .filter(pl.col("history_folds") >= min_history_folds)
    )
    if grouped.height == 0:
        return grouped.with_columns(pl.lit(None).alias("selection_score"))
    if selection_metric == "global_r2":
        scored = grouped.with_columns((1.0 - pl.col("history_numerator") / pl.col("history_denominator")).alias("selection_score"))
    elif selection_metric == "mean_fold_r2":
        scored = grouped.with_columns(pl.col("history_mean_r2").alias("selection_score"))
    elif selection_metric == "min_fold_r2":
        scored = grouped.with_columns(pl.col("history_min_r2").alias("selection_score"))
    else:
        raise ValueError(f"unknown selection_metric: {selection_metric}")
    return scored.sort(["selection_score", "candidate_id"], descending=[True, False])


def _write_report(
    path: Path,
    *,
    config: FoldScoreSelectionConfig,
    pool_summary: pl.DataFrame,
    nested_summary: pl.DataFrame,
    selected: pl.DataFrame,
) -> None:
    top_pool = pool_summary.select(["candidate_id", "global_r2", "min_fold_r2"]).head(10).to_dicts()
    nested = nested_summary.to_dicts()
    selected_rows = selected.select(["fold", "selected_candidate_id", "weighted_zero_mean_r2", "selection_score", "history_folds"]).to_dicts()
    body = [
        "# Fold-Score Nested Selection",
        "",
        "## Scope",
        "",
        "This diagnostic selects among completed OOF candidates using only fold-level sufficient statistics from earlier folds.",
        "It is cheap and anti-leakage by construction, but it is not a deployable row-level policy unless reimplemented with an observable time rule.",
        "",
        "## Configuration",
        "",
        f"- Selection metric: `{config.selection_metric}`",
        f"- Min history folds: `{config.min_history_folds}`",
        f"- First fold candidate: `{config.first_candidate_id}`",
        "",
        "## Nested Summary",
        "",
        "```text",
        json.dumps(nested, indent=2),
        "```",
        "",
        "## Selected Folds",
        "",
        "```text",
        json.dumps(selected_rows, indent=2),
        "```",
        "",
        "## Candidate Pool Top 10",
        "",
        "```text",
        json.dumps(top_pool, indent=2),
        "```",
        "",
        "## Audit",
        "",
        "- Uses only prior folds for selection after the predeclared first-fold candidate.",
        "- Does not inspect row-level validation targets beyond the already-written fold sufficient statistics.",
        "- Diagnostic only until converted into a deployable causal time policy.",
    ]
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def _validate_config(config: FoldScoreSelectionConfig) -> None:
    if not config.sources:
        raise ValueError("sources must not be empty")
    if config.min_history_folds < 1:
        raise ValueError("min_history_folds must be at least 1")
    if config.selection_metric not in {"global_r2", "mean_fold_r2", "min_fold_r2"}:
        raise ValueError("selection_metric must be one of global_r2, mean_fold_r2, min_fold_r2")
    if config.candidate_ids and config.first_candidate_id not in set(config.candidate_ids):
        raise ValueError("first_candidate_id must be present in candidate_ids")
