"""Reusable high-weight tail-control utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import polars as pl


DEFAULT_TAIL_BUCKETS: tuple[str, ...] = ("q90_q99", "q99_q100")


@dataclass(frozen=True)
class TailSwitchPolicy:
    """Piecewise policy that uses a candidate prediction only in tail buckets."""

    tail_buckets: tuple[str, ...] = DEFAULT_TAIL_BUCKETS
    base_prediction: str = "ensemble_prediction"
    candidate_prediction: str = "clock_simplex_prediction"
    output: str = "tail_control_prediction"

    def apply(self, frame: pl.DataFrame) -> pl.DataFrame:
        return add_tail_switch_prediction(
            frame,
            base_prediction=self.base_prediction,
            candidate_prediction=self.candidate_prediction,
            tail_buckets=self.tail_buckets,
            output=self.output,
        )


@dataclass(frozen=True)
class GroupedTailAdvantagePolicy:
    """Use the candidate only in tail groups where calibration loss improved."""

    group_columns: tuple[str, ...]
    parameters: pl.DataFrame
    fallback_use_candidate: bool
    tail_buckets: tuple[str, ...] = DEFAULT_TAIL_BUCKETS
    base_prediction: str = "ensemble_prediction"
    candidate_prediction: str = "clock_simplex_prediction"
    output: str = "tail_advantage_prediction"

    def apply(self, frame: pl.DataFrame) -> pl.DataFrame:
        _require_prediction_columns(frame, self.base_prediction, self.candidate_prediction)
        if self.group_columns and self.parameters.height:
            scored = frame.join(self.parameters, on=list(self.group_columns), how="left")
            use_candidate = pl.coalesce(pl.col("_tail_use_candidate"), pl.lit(self.fallback_use_candidate))
            drop_columns = [
                column
                for column in (
                    "_tail_rows",
                    "_tail_base_numerator",
                    "_tail_candidate_numerator",
                    "_tail_use_candidate",
                )
                if column in scored.columns
            ]
        else:
            scored = frame
            use_candidate = pl.lit(self.fallback_use_candidate)
            drop_columns = []
        return (
            scored.with_columns(
                pl.when(pl.col("weight_bucket").is_in(list(self.tail_buckets)) & use_candidate)
                .then(pl.col(self.candidate_prediction))
                .otherwise(pl.col(self.base_prediction))
                .alias(self.output)
            )
            .drop(drop_columns)
        )


def add_tail_switch_prediction(
    frame: pl.DataFrame,
    *,
    base_prediction: str,
    candidate_prediction: str,
    tail_buckets: Sequence[str],
    output: str,
) -> pl.DataFrame:
    """Use `candidate_prediction` only for rows whose `weight_bucket` is in tail."""

    buckets = tuple(tail_buckets)
    if not buckets:
        raise ValueError("tail_buckets must not be empty")
    _require_prediction_columns(frame, base_prediction, candidate_prediction)
    return frame.with_columns(
        pl.when(pl.col("weight_bucket").is_in(list(buckets)))
        .then(pl.col(candidate_prediction))
        .otherwise(pl.col(base_prediction))
        .alias(output)
    )


def fit_grouped_tail_advantage_policy(
    frame: pl.DataFrame,
    *,
    group_columns: Sequence[str],
    base_prediction: str,
    candidate_prediction: str,
    tail_buckets: Sequence[str] = DEFAULT_TAIL_BUCKETS,
    target: str = "responder_6",
    weight: str = "weight",
    min_group_rows: int = 2_000,
    output: str = "tail_advantage_prediction",
) -> GroupedTailAdvantagePolicy:
    """Fit a causal calibration-time selector for tail-control activation."""

    if min_group_rows <= 0:
        raise ValueError("min_group_rows must be positive")
    groups = tuple(group_columns)
    buckets = tuple(tail_buckets)
    if not buckets:
        raise ValueError("tail_buckets must not be empty")
    required = {target, weight, "weight_bucket", base_prediction, candidate_prediction, *groups}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")
    tail = frame.filter(pl.col("weight_bucket").is_in(list(buckets)))
    if tail.is_empty():
        return GroupedTailAdvantagePolicy(
            group_columns=groups,
            parameters=pl.DataFrame(),
            fallback_use_candidate=False,
            tail_buckets=buckets,
            base_prediction=base_prediction,
            candidate_prediction=candidate_prediction,
            output=output,
        )
    with_losses = tail.with_columns(
        [
            (pl.col(weight) * (pl.col(target) - pl.col(base_prediction)).pow(2)).alias("_tail_base_loss"),
            (pl.col(weight) * (pl.col(target) - pl.col(candidate_prediction)).pow(2)).alias("_tail_candidate_loss"),
        ]
    )
    aggregate = with_losses.select(
        pl.col("_tail_base_loss").sum().alias("_tail_base_numerator"),
        pl.col("_tail_candidate_loss").sum().alias("_tail_candidate_numerator"),
    ).row(0, named=True)
    fallback_use_candidate = float(aggregate["_tail_candidate_numerator"]) < float(aggregate["_tail_base_numerator"])
    if not groups:
        return GroupedTailAdvantagePolicy(
            group_columns=(),
            parameters=pl.DataFrame(),
            fallback_use_candidate=fallback_use_candidate,
            tail_buckets=buckets,
            base_prediction=base_prediction,
            candidate_prediction=candidate_prediction,
            output=output,
        )
    parameters = (
        with_losses.group_by(list(groups))
        .agg(
            pl.len().alias("_tail_rows"),
            pl.col("_tail_base_loss").sum().alias("_tail_base_numerator"),
            pl.col("_tail_candidate_loss").sum().alias("_tail_candidate_numerator"),
        )
        .filter(pl.col("_tail_rows") >= min_group_rows)
        .with_columns((pl.col("_tail_candidate_numerator") < pl.col("_tail_base_numerator")).alias("_tail_use_candidate"))
        .select(list(groups) + ["_tail_rows", "_tail_base_numerator", "_tail_candidate_numerator", "_tail_use_candidate"])
    )
    return GroupedTailAdvantagePolicy(
        group_columns=groups,
        parameters=parameters,
        fallback_use_candidate=fallback_use_candidate,
        tail_buckets=buckets,
        base_prediction=base_prediction,
        candidate_prediction=candidate_prediction,
        output=output,
    )


def with_batch_missing_fraction(
    frame: pl.DataFrame,
    *,
    source_columns: Sequence[str],
    output: str = "batch_missing_frac",
) -> pl.DataFrame:
    """Compute gateway-observable feature missingness per `(date_id,time_id)` batch."""

    columns = tuple(source_columns)
    if not columns:
        raise ValueError("source_columns must not be empty")
    required = {"date_id", "time_id", *columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")
    row_missing = sum(pl.col(name).is_null().cast(pl.Float32) for name in columns) / float(len(columns))
    temp = "__row_missing_frac"
    return (
        frame.with_columns(row_missing.alias(temp))
        .with_columns(pl.col(temp).mean().over(["date_id", "time_id"]).cast(pl.Float32).alias(output))
        .drop(temp)
    )


def _require_prediction_columns(frame: pl.DataFrame, base_prediction: str, candidate_prediction: str) -> None:
    missing = {base_prediction, candidate_prediction, "weight_bucket"} - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")
