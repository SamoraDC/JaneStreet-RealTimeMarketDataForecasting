"""Evaluate tail-only switches from clock tournament weight-bucket aggregates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-clock", default="none")
    parser.add_argument("--base-strategy", default="base_ensemble")
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--tail-buckets", default="q99_q100")
    args = parser.parse_args()
    _validate_args(args)

    frame = pl.read_csv(args.input)
    tail_buckets = tuple(part.strip() for part in args.tail_buckets.split(",") if part.strip())
    candidates = tuple(_parse_candidate(raw) for raw in args.candidate)
    outputs = []
    for candidate_clock, candidate_strategy in candidates:
        outputs.append(
            evaluate_tail_switch(
                frame,
                base_clock=args.base_clock,
                base_strategy=args.base_strategy,
                candidate_clock=candidate_clock,
                candidate_strategy=candidate_strategy,
                tail_buckets=tail_buckets,
            )
        )
    summary = pl.DataFrame(outputs).sort(["global_r2", "min_fold_r2"], descending=[True, True])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.write_csv(args.output_dir / "clock_tail_switch_summary.csv")
    report = {
        "experiment": "clock_tail_switch",
        "input": str(args.input),
        "base_clock": args.base_clock,
        "base_strategy": args.base_strategy,
        "tail_buckets": tail_buckets,
        "candidates": candidates,
        "best_strategy": summary.row(0, named=True) if summary.height else None,
    }
    (args.output_dir / "clock_tail_switch_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary)
    print(f"Wrote {args.output_dir}")


def evaluate_tail_switch(
    frame: pl.DataFrame,
    *,
    base_clock: str,
    base_strategy: str,
    candidate_clock: str,
    candidate_strategy: str,
    tail_buckets: tuple[str, ...],
) -> dict[str, float | int | str]:
    if not tail_buckets:
        raise ValueError("tail_buckets must not be empty")
    base = frame.filter((pl.col("clock") == base_clock) & (pl.col("strategy") == base_strategy))
    candidate = frame.filter((pl.col("clock") == candidate_clock) & (pl.col("strategy") == candidate_strategy))
    if base.is_empty():
        raise ValueError("base slice is empty")
    if candidate.is_empty():
        raise ValueError("candidate slice is empty")
    selected = (
        base.join(
            candidate.select(["fold", "weight_bucket", "numerator"]).rename({"numerator": "candidate_numerator"}),
            on=["fold", "weight_bucket"],
            how="left",
        )
        .with_columns(
            pl.when(pl.col("weight_bucket").is_in(tail_buckets))
            .then(pl.coalesce(pl.col("candidate_numerator"), pl.col("numerator")))
            .otherwise(pl.col("numerator"))
            .alias("switch_numerator")
        )
    )
    by_fold = (
        selected.group_by("fold")
        .agg(
            pl.col("rows").sum().alias("rows"),
            pl.col("switch_numerator").sum().alias("numerator"),
            pl.col("denominator").sum().alias("denominator"),
        )
        .with_columns((1.0 - pl.col("numerator") / pl.col("denominator")).alias("weighted_zero_mean_r2"))
    )
    aggregate = by_fold.select(
        pl.len().alias("folds"),
        pl.col("rows").sum().alias("validation_rows"),
        pl.col("weighted_zero_mean_r2").mean().alias("mean_r2"),
        pl.col("weighted_zero_mean_r2").std().alias("std_r2"),
        pl.col("weighted_zero_mean_r2").min().alias("min_fold_r2"),
        pl.col("weighted_zero_mean_r2").max().alias("max_fold_r2"),
        (1.0 - pl.col("numerator").sum() / pl.col("denominator").sum()).alias("global_r2"),
    ).row(0, named=True)
    return {
        "candidate_clock": candidate_clock,
        "candidate_strategy": candidate_strategy,
        "tail_buckets": ",".join(tail_buckets),
        "folds": int(aggregate["folds"]),
        "validation_rows": int(aggregate["validation_rows"]),
        "mean_r2": float(aggregate["mean_r2"]),
        "std_r2": _float_or_zero(aggregate["std_r2"]),
        "min_fold_r2": float(aggregate["min_fold_r2"]),
        "max_fold_r2": float(aggregate["max_fold_r2"]),
        "global_r2": float(aggregate["global_r2"]),
    }


def _parse_candidate(raw: str) -> tuple[str, str]:
    parts = tuple(part.strip() for part in raw.split(":") if part.strip())
    if len(parts) != 2:
        raise ValueError("--candidate must have format clock:strategy")
    return parts


def _float_or_zero(value: object) -> float:
    if value is None:
        return 0.0
    return float(value)


def _validate_args(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if not args.candidate:
        raise ValueError("provide at least one --candidate clock:strategy")
    if not args.tail_buckets.strip():
        raise ValueError("--tail-buckets must not be empty")


if __name__ == "__main__":
    main()
