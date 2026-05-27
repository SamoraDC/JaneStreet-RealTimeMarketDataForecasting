"""Evaluate row-level blends between saved validation prediction files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import polars as pl

from janestreet.blending import (
    add_convex_blend_prediction,
    fit_convex_blend_weight,
    fit_grouped_convex_blend_weights,
    fit_simplex_blend_weights,
    add_simplex_blend_prediction,
)


TARGET = "responder_6"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabm-prediction-dir", type=Path, required=True)
    parser.add_argument("--tree-prediction-dir", type=Path, required=True)
    parser.add_argument("--time-bucket-size", type=int, default=100)
    parser.add_argument("--min-group-rows", type=int, default=20_000)
    parser.add_argument("--initial-tabm-weight", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _validate_args(args)

    frame = _load_joined_predictions(args.tabm_prediction_dir, args.tree_prediction_dir)
    rows = []
    parameter_rows = []
    strategy_columns = {
        "tabm": "tabm_prediction",
        "tree_ensemble": "tree_prediction",
    }
    for optional in ["ridge_calibrated_prediction", "xgboost_prediction", "lightgbm_prediction", "catboost_prediction"]:
        if optional in frame.columns:
            strategy_columns[optional.removesuffix("_prediction")] = optional

    for strategy, prediction in strategy_columns.items():
        rows.append({"strategy": strategy, "fit_mode": "base", **_score_frame(frame, prediction)})

    global_weight = fit_convex_blend_weight(
        frame,
        left_prediction="tabm_prediction",
        right_prediction="tree_prediction",
    )
    parameter_rows.append({"strategy": "tabm_tree_convex", "fit_mode": "insample", "group": "__global__", "tabm_weight": global_weight, "rows": frame.height})
    insample = add_convex_blend_prediction(
        frame,
        blend_weight=global_weight,
        left_prediction="tabm_prediction",
        right_prediction="tree_prediction",
        output="tabm_tree_convex_insample_prediction",
    )
    rows.append({"strategy": "tabm_tree_convex", "fit_mode": "insample_diagnostic", **_score_frame(insample, "tabm_tree_convex_insample_prediction")})

    simplex_columns = tuple(column for column in ["tabm_prediction", "tree_prediction", "xgboost_prediction", "lightgbm_prediction", "ridge_calibrated_prediction"] if column in frame.columns)
    simplex_weights = fit_simplex_blend_weights(frame, prediction_columns=simplex_columns)
    parameter_rows.extend(
        {"strategy": "simplex_all", "fit_mode": "insample", "group": prediction, "tabm_weight": weight if prediction == "tabm_prediction" else None, "rows": frame.height, "weight": weight}
        for prediction, weight in simplex_weights.items()
    )
    simplex = add_simplex_blend_prediction(frame, weights=simplex_weights, output="simplex_all_insample_prediction")
    rows.append({"strategy": "simplex_all", "fit_mode": "insample_diagnostic", **_score_frame(simplex, "simplex_all_insample_prediction")})

    walk_global, walk_global_params = _walk_forward_global(frame, initial_tabm_weight=args.initial_tabm_weight)
    parameter_rows.extend(walk_global_params)
    rows.append({"strategy": "tabm_tree_convex", "fit_mode": "walk_forward_previous_folds", **_score_frame(walk_global, "walk_forward_global_prediction")})

    for group_columns in [("weight_bucket",), ("time_bucket",), ("weight_bucket", "time_bucket"), ("disagreement_bucket",), ("weight_bucket", "disagreement_bucket")]:
        walk_grouped, grouped_params = _walk_forward_grouped(
            frame,
            group_columns=group_columns,
            min_group_rows=args.min_group_rows,
            initial_tabm_weight=args.initial_tabm_weight,
            time_bucket_size=args.time_bucket_size,
        )
        strategy = "tabm_tree_grouped_" + "_".join(group_columns)
        parameter_rows.extend(
            {
                "strategy": strategy,
                **row,
            }
            for row in grouped_params
        )
        rows.append({"strategy": strategy, "fit_mode": "walk_forward_previous_folds", **_score_frame(walk_grouped, "walk_forward_grouped_prediction")})

    oracle = _oracle_best_by_fold(frame)
    rows.append({"strategy": "oracle_best_by_fold", "fit_mode": "leaky_upper_bound", **_score_frame(oracle, "oracle_prediction")})

    result = pl.DataFrame(rows).sort(["global_r2", "min_fold_r2"], descending=[True, True])
    parameters = pl.DataFrame(parameter_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.write_csv(args.output_dir / "prediction_blend_summary.csv")
    if not parameters.is_empty():
        parameters.write_csv(args.output_dir / "prediction_blend_parameters.csv")
    report = {
        "experiment": "prediction_blend",
        "tabm_prediction_dir": str(args.tabm_prediction_dir),
        "tree_prediction_dir": str(args.tree_prediction_dir),
        "rows": frame.height,
        "folds": frame["fold"].n_unique(),
        "caveat": "insample and oracle rows are diagnostic only; walk_forward rows use only earlier folds for blend fitting.",
        "best": result.row(0, named=True),
    }
    (args.output_dir / "prediction_blend_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(result)
    print(f"Wrote {args.output_dir}")


def _validate_args(args: argparse.Namespace) -> None:
    if not args.tabm_prediction_dir.exists():
        raise FileNotFoundError(args.tabm_prediction_dir)
    if not args.tree_prediction_dir.exists():
        raise FileNotFoundError(args.tree_prediction_dir)
    if args.time_bucket_size <= 0 or args.min_group_rows <= 0:
        raise ValueError("bucket size and min group rows must be positive")
    if not 0.0 <= args.initial_tabm_weight <= 1.0:
        raise ValueError("--initial-tabm-weight must be in [0, 1]")


def _load_joined_predictions(tabm_dir: Path, tree_dir: Path) -> pl.DataFrame:
    keys = ["fold", "date_id", "time_id", "symbol_id"]
    tabm = pl.scan_parquet(str(tabm_dir / "*.parquet")).select(keys + ["weight", TARGET, "tabm_prediction"])
    tree_columns = ["ensemble_prediction", "ridge_calibrated_prediction", "xgboost_prediction", "lightgbm_prediction", "catboost_prediction"]
    tree_schema = pl.scan_parquet(str(tree_dir / "*.parquet")).collect_schema()
    selected_tree_columns = [column for column in tree_columns if column in tree_schema.names()]
    tree = (
        pl.scan_parquet(str(tree_dir / "*.parquet"))
        .select(keys + selected_tree_columns)
        .rename({"ensemble_prediction": "tree_prediction"})
    )
    joined = tabm.join(tree, on=keys, how="inner").collect()
    if joined.height == 0:
        raise ValueError("prediction join is empty")
    return joined.sort(keys)


def _score_frame(frame: pl.DataFrame, prediction: str) -> dict[str, float | int]:
    by_fold = (
        frame.group_by("fold")
        .agg(
            pl.len().alias("rows"),
            pl.col("weight").sum().alias("weight_sum"),
            (pl.col("weight") * (pl.col(TARGET) - pl.col(prediction)).pow(2)).sum().alias("numerator"),
            (pl.col("weight") * pl.col(TARGET).pow(2)).sum().alias("denominator"),
        )
        .with_columns((1.0 - pl.col("numerator") / pl.col("denominator")).alias("weighted_zero_mean_r2"))
    )
    totals = by_fold.select(
        pl.col("rows").sum().alias("rows"),
        pl.col("weight_sum").sum().alias("weight_sum"),
        pl.col("numerator").sum().alias("numerator"),
        pl.col("denominator").sum().alias("denominator"),
        pl.col("weighted_zero_mean_r2").mean().alias("mean_fold_r2"),
        pl.col("weighted_zero_mean_r2").std().alias("std_fold_r2"),
        pl.col("weighted_zero_mean_r2").min().alias("min_fold_r2"),
        pl.col("weighted_zero_mean_r2").max().alias("max_fold_r2"),
    ).row(0, named=True)
    return {
        "folds": int(by_fold.height),
        "rows": int(totals["rows"]),
        "weight_sum": float(totals["weight_sum"]),
        "numerator": float(totals["numerator"]),
        "denominator": float(totals["denominator"]),
        "global_r2": 1.0 - float(totals["numerator"]) / float(totals["denominator"]),
        "mean_fold_r2": float(totals["mean_fold_r2"]),
        "std_fold_r2": 0.0 if totals["std_fold_r2"] is None else float(totals["std_fold_r2"]),
        "min_fold_r2": float(totals["min_fold_r2"]),
        "max_fold_r2": float(totals["max_fold_r2"]),
    }


def _walk_forward_global(frame: pl.DataFrame, *, initial_tabm_weight: float) -> tuple[pl.DataFrame, list[dict[str, float | int | str | None]]]:
    frames = []
    rows = []
    folds = frame.select("fold").unique().sort("fold")["fold"].to_list()
    for idx, fold in enumerate(folds):
        current = frame.filter(pl.col("fold") == fold)
        if idx == 0:
            blend_weight = initial_tabm_weight
            fit_rows = 0
        else:
            calibration = frame.filter(pl.col("fold").is_in(folds[:idx]))
            blend_weight = fit_convex_blend_weight(calibration, left_prediction="tabm_prediction", right_prediction="tree_prediction")
            fit_rows = calibration.height
        rows.append({"strategy": "tabm_tree_convex", "fit_mode": "walk_forward_previous_folds", "fold": fold, "group": "__global__", "tabm_weight": blend_weight, "rows": fit_rows})
        frames.append(
            add_convex_blend_prediction(
                current,
                blend_weight=blend_weight,
                left_prediction="tabm_prediction",
                right_prediction="tree_prediction",
                output="walk_forward_global_prediction",
            )
        )
    return pl.concat(frames), rows


def _walk_forward_grouped(
    frame: pl.DataFrame,
    *,
    group_columns: Sequence[str],
    min_group_rows: int,
    initial_tabm_weight: float,
    time_bucket_size: int,
) -> tuple[pl.DataFrame, list[dict[str, float | int | str | None]]]:
    frames = []
    parameter_rows = []
    folds = frame.select("fold").unique().sort("fold")["fold"].to_list()
    for idx, fold in enumerate(folds):
        current_raw = frame.filter(pl.col("fold") == fold)
        if idx == 0:
            current = _add_fixed_buckets(current_raw, time_bucket_size=time_bucket_size)
            blended = add_convex_blend_prediction(
                current,
                blend_weight=initial_tabm_weight,
                left_prediction="tabm_prediction",
                right_prediction="tree_prediction",
                output="walk_forward_grouped_prediction",
            )
            parameter_rows.append({"fit_mode": "walk_forward_previous_folds", "fold": fold, "group": "__global__", "tabm_weight": initial_tabm_weight, "rows": 0})
            frames.append(blended)
            continue
        calibration_raw = frame.filter(pl.col("fold").is_in(folds[:idx]))
        thresholds = _fit_thresholds(calibration_raw)
        calibration = _add_dynamic_buckets(calibration_raw, thresholds=thresholds, time_bucket_size=time_bucket_size)
        current = _add_dynamic_buckets(current_raw, thresholds=thresholds, time_bucket_size=time_bucket_size)
        grouped = fit_grouped_convex_blend_weights(
            calibration,
            group_columns=group_columns,
            left_prediction="tabm_prediction",
            right_prediction="tree_prediction",
            min_group_rows=min_group_rows,
        )
        parameter_rows.append({"fit_mode": "walk_forward_previous_folds", "fold": fold, "group": "__fallback__", "tabm_weight": grouped.fallback_weight, "rows": calibration.height})
        for row in grouped.parameters.iter_rows(named=True):
            group = "|".join(str(row[column]) for column in group_columns)
            parameter_rows.append({"fit_mode": "walk_forward_previous_folds", "fold": fold, "group": group, "tabm_weight": float(row["_blend_weight"]), "rows": int(row["_blend_rows"])})
        frames.append(
            grouped.apply(
                current,
                left_prediction="tabm_prediction",
                right_prediction="tree_prediction",
                output="walk_forward_grouped_prediction",
            )
        )
    return pl.concat(frames), parameter_rows


def _fit_thresholds(frame: pl.DataFrame) -> dict[str, float]:
    return {
        "weight_q50": float(frame["weight"].quantile(0.50)),
        "weight_q90": float(frame["weight"].quantile(0.90)),
        "weight_q99": float(frame["weight"].quantile(0.99)),
        "disagreement_q50": float((frame["tabm_prediction"] - frame["tree_prediction"]).abs().quantile(0.50)),
        "disagreement_q90": float((frame["tabm_prediction"] - frame["tree_prediction"]).abs().quantile(0.90)),
        "disagreement_q99": float((frame["tabm_prediction"] - frame["tree_prediction"]).abs().quantile(0.99)),
    }


def _add_fixed_buckets(frame: pl.DataFrame, *, time_bucket_size: int) -> pl.DataFrame:
    thresholds = _fit_thresholds(frame)
    return _add_dynamic_buckets(frame, thresholds=thresholds, time_bucket_size=time_bucket_size)


def _add_dynamic_buckets(frame: pl.DataFrame, *, thresholds: dict[str, float], time_bucket_size: int) -> pl.DataFrame:
    disagreement = (pl.col("tabm_prediction") - pl.col("tree_prediction")).abs()
    return frame.with_columns(
        [
            (pl.col("time_id") // time_bucket_size).cast(pl.Int16).alias("time_bucket"),
            (
                pl.when(pl.col("weight") <= thresholds["weight_q50"])
                .then(pl.lit("q00_q50"))
                .when(pl.col("weight") <= thresholds["weight_q90"])
                .then(pl.lit("q50_q90"))
                .when(pl.col("weight") <= thresholds["weight_q99"])
                .then(pl.lit("q90_q99"))
                .otherwise(pl.lit("q99_q100"))
                .alias("weight_bucket")
            ),
            (
                pl.when(disagreement <= thresholds["disagreement_q50"])
                .then(pl.lit("q00_q50"))
                .when(disagreement <= thresholds["disagreement_q90"])
                .then(pl.lit("q50_q90"))
                .when(disagreement <= thresholds["disagreement_q99"])
                .then(pl.lit("q90_q99"))
                .otherwise(pl.lit("q99_q100"))
                .alias("disagreement_bucket")
            ),
        ]
    )


def _oracle_best_by_fold(frame: pl.DataFrame) -> pl.DataFrame:
    by_fold = []
    for fold in frame.select("fold").unique().sort("fold")["fold"].to_list():
        current = frame.filter(pl.col("fold") == fold)
        tabm_score = _score_frame(current, "tabm_prediction")["global_r2"]
        tree_score = _score_frame(current, "tree_prediction")["global_r2"]
        prediction = "tabm_prediction" if tabm_score >= tree_score else "tree_prediction"
        by_fold.append(current.with_columns(pl.col(prediction).cast(pl.Float64).alias("oracle_prediction")))
    return pl.concat(by_fold)


if __name__ == "__main__":
    main()
