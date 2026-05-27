"""Evaluate fixed tail-switch variants from online validation bucket aggregates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-strategy", default="base_ensemble")
    parser.add_argument("--candidate-strategy", default="batch_missing_clock_simplex")
    parser.add_argument(
        "--variants",
        default="q99_only=q99_q100,q90_q99_only=q90_q99,q90_q100=q90_q99+q99_q100",
    )
    args = parser.parse_args()
    _validate_args(args)

    frame = pl.read_csv(args.input)
    variants = _parse_variants(args.variants)
    summary_rows = [
        evaluate_tail_variant(
            frame,
            base_strategy=args.base_strategy,
            candidate_strategy=args.candidate_strategy,
            variant_name=name,
            tail_buckets=buckets,
        )
        for name, buckets in variants
    ]
    by_fold = pl.concat(
        [
            evaluate_tail_variant_by_fold(
                frame,
                base_strategy=args.base_strategy,
                candidate_strategy=args.candidate_strategy,
                variant_name=name,
                tail_buckets=buckets,
            )
            for name, buckets in variants
        ]
    )
    summary = pl.DataFrame(summary_rows).sort(["global_r2", "min_r2"], descending=[True, True])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.write_csv(args.output_dir / "online_tail_variant_summary.csv")
    by_fold.write_csv(args.output_dir / "online_tail_variant_by_fold.csv")
    report = {
        "experiment": "online_tail_variant_analysis",
        "input": str(args.input),
        "base_strategy": args.base_strategy,
        "candidate_strategy": args.candidate_strategy,
        "variants": [{"name": name, "tail_buckets": buckets} for name, buckets in variants],
        "best_variant": summary.row(0, named=True) if not summary.is_empty() else None,
        "leakage_note": "Uses only already-scored validation bucket aggregates; no refit or target-derived feature is created.",
    }
    (args.output_dir / "online_tail_variant_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary)
    print(f"Wrote {args.output_dir}")


def evaluate_tail_variant(
    frame: pl.DataFrame,
    *,
    base_strategy: str,
    candidate_strategy: str,
    variant_name: str,
    tail_buckets: tuple[str, ...],
) -> dict[str, float | int | str]:
    by_fold = evaluate_tail_variant_by_fold(
        frame,
        base_strategy=base_strategy,
        candidate_strategy=candidate_strategy,
        variant_name=variant_name,
        tail_buckets=tail_buckets,
    )
    row = by_fold.select(
        [
            pl.len().alias("folds"),
            pl.col("rows").sum().alias("validation_rows"),
            pl.col("weighted_zero_mean_r2").mean().alias("mean_r2"),
            pl.col("weighted_zero_mean_r2").std().alias("std_r2"),
            pl.col("weighted_zero_mean_r2").min().alias("min_r2"),
            pl.col("weighted_zero_mean_r2").max().alias("max_r2"),
            (1.0 - pl.col("numerator").sum() / pl.col("denominator").sum()).alias("global_r2"),
            pl.col("delta_r2").min().alias("min_delta_r2"),
            pl.col("delta_r2").max().alias("max_delta_r2"),
            pl.col("numerator_improvement").sum().alias("numerator_improvement"),
        ]
    ).row(0, named=True)
    return {
        "variant": variant_name,
        "tail_buckets": ",".join(tail_buckets),
        "folds": int(row["folds"]),
        "validation_rows": int(row["validation_rows"]),
        "mean_r2": float(row["mean_r2"]),
        "std_r2": _float_or_zero(row["std_r2"]),
        "min_r2": float(row["min_r2"]),
        "max_r2": float(row["max_r2"]),
        "global_r2": float(row["global_r2"]),
        "min_delta_r2": float(row["min_delta_r2"]),
        "max_delta_r2": float(row["max_delta_r2"]),
        "numerator_improvement": float(row["numerator_improvement"]),
    }


def evaluate_tail_variant_by_fold(
    frame: pl.DataFrame,
    *,
    base_strategy: str,
    candidate_strategy: str,
    variant_name: str,
    tail_buckets: tuple[str, ...],
) -> pl.DataFrame:
    if not tail_buckets:
        raise ValueError("tail_buckets must not be empty")
    base = frame.filter(pl.col("strategy") == base_strategy)
    candidate = frame.filter(pl.col("strategy") == candidate_strategy)
    if base.is_empty():
        raise ValueError("base strategy slice is empty")
    if candidate.is_empty():
        raise ValueError("candidate strategy slice is empty")
    selected = (
        base.join(
            candidate.select(["fold", "weight_bucket", "numerator"]).rename({"numerator": "candidate_numerator"}),
            on=["fold", "weight_bucket"],
            how="left",
        )
        .with_columns(
            pl.when(pl.col("weight_bucket").is_in(list(tail_buckets)))
            .then(pl.coalesce(pl.col("candidate_numerator"), pl.col("numerator")))
            .otherwise(pl.col("numerator"))
            .alias("selected_numerator")
        )
    )
    base_by_fold = (
        base.group_by("fold")
        .agg(
            [
                pl.col("numerator").sum().alias("base_numerator"),
                pl.col("denominator").sum().alias("base_denominator"),
            ]
        )
        .with_columns((1.0 - pl.col("base_numerator") / pl.col("base_denominator")).alias("base_r2"))
    )
    return (
        selected.group_by("fold")
        .agg(
            [
                pl.col("rows").sum().alias("rows"),
                pl.col("selected_numerator").sum().alias("numerator"),
                pl.col("denominator").sum().alias("denominator"),
            ]
        )
        .with_columns((1.0 - pl.col("numerator") / pl.col("denominator")).alias("weighted_zero_mean_r2"))
        .join(base_by_fold, on="fold")
        .with_columns(
            [
                pl.lit(variant_name).alias("variant"),
                pl.lit(",".join(tail_buckets)).alias("tail_buckets"),
                (pl.col("weighted_zero_mean_r2") - pl.col("base_r2")).alias("delta_r2"),
                (pl.col("base_numerator") - pl.col("numerator")).alias("numerator_improvement"),
            ]
        )
        .select(
            [
                "variant",
                "tail_buckets",
                "fold",
                "rows",
                "numerator",
                "denominator",
                "weighted_zero_mean_r2",
                "base_r2",
                "delta_r2",
                "numerator_improvement",
            ]
        )
        .sort(["variant", "fold"])
    )


def _parse_variants(raw: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    variants: list[tuple[str, tuple[str, ...]]] = []
    for part in raw.split(","):
        if not part.strip():
            continue
        name, sep, buckets_raw = part.partition("=")
        if not sep or not name.strip() or not buckets_raw.strip():
            raise ValueError("--variants entries must have format name=bucket+bucket")
        buckets = tuple(bucket.strip() for bucket in buckets_raw.split("+") if bucket.strip())
        if not buckets:
            raise ValueError("variant tail buckets must not be empty")
        variants.append((name.strip(), buckets))
    if not variants:
        raise ValueError("--variants must not be empty")
    return tuple(variants)


def _float_or_zero(value: object) -> float:
    if value is None:
        return 0.0
    return float(value)


def _validate_args(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if not args.base_strategy.strip():
        raise ValueError("--base-strategy must not be empty")
    if not args.candidate_strategy.strip():
        raise ValueError("--candidate-strategy must not be empty")
    _parse_variants(args.variants)


if __name__ == "__main__":
    main()
