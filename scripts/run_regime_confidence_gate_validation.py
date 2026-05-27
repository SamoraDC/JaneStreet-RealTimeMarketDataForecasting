"""Nested walk-forward validation for regime/confidence gates over the active ensemble."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import polars as pl

from janestreet.blending import GroupedBlendWeights, fit_grouped_convex_blend_weights
from janestreet.calibration import (
    ShrinkageCalibrator,
    add_abs_prediction_bucket,
    fit_abs_prediction_thresholds,
    fit_shrinkage_calibrator,
)
from janestreet.linear import feature_columns_from_schema
from janestreet.paths import PROJECT_ROOT, TRAIN_PARQUET_DIR


DEFAULT_GATE_CANDIDATES = (
    "global_blend",
    "clock_blend",
    "clock_weight_blend",
    "clock_weight_abs_blend",
    "clock_weight_disagreement_blend",
    "clock_weight_delta_blend",
    "clock_weight_shrink",
    "clock_weight_abs_shrink",
    "clock_weight_disagreement_shrink",
)

BLEND_GROUPS: dict[str, tuple[str, ...]] = {
    "global_blend": (),
    "clock_blend": ("clock_bucket",),
    "clock_weight_blend": ("clock_bucket", "weight_bucket"),
    "clock_weight_abs_blend": ("clock_bucket", "weight_bucket", "ensemble_abs_bucket"),
    "clock_weight_disagreement_blend": ("clock_bucket", "weight_bucket", "disagreement_bucket"),
    "clock_weight_delta_blend": ("clock_bucket", "weight_bucket", "candidate_delta_bucket"),
}

SHRINK_GROUPS: dict[str, tuple[str, ...]] = {
    "clock_weight_shrink": ("clock_bucket", "weight_bucket"),
    "clock_weight_abs_shrink": ("clock_bucket", "weight_bucket", "ensemble_abs_bucket"),
    "clock_weight_disagreement_shrink": ("clock_bucket", "weight_bucket", "disagreement_bucket"),
}


@dataclass(frozen=True)
class GatePolicy:
    name: str
    kind: str
    group_columns: tuple[str, ...]
    model: GroupedBlendWeights | ShrinkageCalibrator
    output: str

    def apply(self, frame: pl.DataFrame) -> pl.DataFrame:
        if self.kind == "blend":
            assert isinstance(self.model, GroupedBlendWeights)
            return self.model.apply(
                frame,
                left_prediction="clock_simplex_prediction",
                right_prediction="ensemble_prediction",
                output=self.output,
            )
        if self.kind == "shrink":
            assert isinstance(self.model, ShrinkageCalibrator)
            return self.model.apply(
                frame,
                prediction="ensemble_prediction",
                output=self.output,
            )
        raise ValueError(f"unknown gate policy kind: {self.kind}")

    def metadata(self) -> dict[str, float | int | str | None]:
        if isinstance(self.model, GroupedBlendWeights):
            return {
                "fallback_blend_weight": float(self.model.fallback_weight),
                "fallback_alpha": None,
                "parameter_rows": int(self.model.parameters.height),
            }
        return {
            "fallback_blend_weight": None,
            "fallback_alpha": float(self.model.fallback_alpha),
            "parameter_rows": int(self.model.parameters.height),
        }


@dataclass(frozen=True)
class GroupSelectionPolicy:
    """Selection-window gate that activates a fitted candidate only in improving groups."""

    name: str
    group_columns: tuple[str, ...]
    parameters: pl.DataFrame
    base_prediction: str
    candidate_prediction: str
    output: str

    def apply(self, frame: pl.DataFrame) -> pl.DataFrame:
        if not self.group_columns or self.parameters.is_empty():
            return frame.with_columns(pl.col(self.base_prediction).alias(self.output))
        joined = frame.join(self.parameters, on=list(self.group_columns), how="left")
        use_candidate = pl.coalesce(pl.col("_group_gate_use_candidate"), pl.lit(False))
        drop_columns = [
            column
            for column in (
                "_group_gate_rows",
                "_group_gate_base_numerator",
                "_group_gate_candidate_numerator",
                "_group_gate_denominator",
                "_group_gate_delta_r2",
                "_group_gate_use_candidate",
            )
            if column in joined.columns
        ]
        return (
            joined.with_columns(
                pl.when(use_candidate).then(pl.col(self.candidate_prediction)).otherwise(pl.col(self.base_prediction)).alias(
                    self.output
                )
            )
            .drop(drop_columns)
        )

    def metadata(self) -> dict[str, int | float]:
        if self.parameters.is_empty():
            return {
                "group_gate_parameter_rows": 0,
                "group_gate_active_groups": 0,
                "group_gate_selection_rows": 0,
                "group_gate_selection_active_rows": 0,
                "group_gate_mean_delta_r2": 0.0,
            }
        row = self.parameters.select(
            [
                pl.len().alias("parameter_rows"),
                pl.col("_group_gate_use_candidate").sum().alias("active_groups"),
                pl.col("_group_gate_rows").sum().alias("selection_rows"),
                pl.when(pl.col("_group_gate_use_candidate"))
                .then(pl.col("_group_gate_rows"))
                .otherwise(0)
                .sum()
                .alias("active_rows"),
                pl.col("_group_gate_delta_r2").mean().alias("mean_delta_r2"),
            ]
        ).row(0, named=True)
        return {
            "group_gate_parameter_rows": int(row["parameter_rows"]),
            "group_gate_active_groups": int(row["active_groups"]),
            "group_gate_selection_rows": int(row["selection_rows"]),
            "group_gate_selection_active_rows": int(row["active_rows"]),
            "group_gate_mean_delta_r2": float(row["mean_delta_r2"]),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=2)
    parser.add_argument("--train-window", type=int, default=120)
    parser.add_argument("--selection-window", type=int, default=20)
    parser.add_argument("--valid-window", type=int, default=20)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--min-date-id", type=int, default=None)
    parser.add_argument("--max-date-id", type=int, default=None)
    parser.add_argument("--inner-oof-folds", type=int, default=3)
    parser.add_argument("--inner-valid-window", type=int, default=20)
    parser.add_argument("--engines", default="xgboost,lightgbm")
    parser.add_argument("--train-sample-frac", type=float, default=0.10)
    parser.add_argument("--gbdt-seeds", default="17,23,37")
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument("--lightgbm-min-child-samples", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--chunk-days", type=int, default=10)
    parser.add_argument("--time-bucket-size", type=int, default=100)
    parser.add_argument("--clock-name", default="batch_missing")
    parser.add_argument("--clock-bucket-count", type=int, default=20)
    parser.add_argument("--selection-min-delta", type=float, default=0.0)
    parser.add_argument("--n-operational-source-features", type=int, default=32)
    parser.add_argument("--operational-windows", default="16,64")
    parser.add_argument("--min-group-rows", type=int, default=2_000)
    parser.add_argument("--clip-target-abs-quantile", type=float, default=0.999)
    parser.add_argument("--gate-candidates", default=",".join(DEFAULT_GATE_CANDIDATES))
    parser.add_argument("--group-selection-min-rows", type=int, default=1_000)
    parser.add_argument("--group-selection-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/time_geometry/reports/regime_confidence_gate_walk_forward"),
    )
    args = parser.parse_args()
    _validate_args(args)

    clock = _load_clock_tournament_module()
    online = _load_online_tail_module()
    runner = clock._load_tree_ensemble_runner()
    engines = clock._parse_engines(args.engines)
    windows = clock._parse_windows(args.operational_windows)
    gate_candidates = _parse_gate_candidates(args.gate_candidates)

    train = pl.scan_parquet(str(TRAIN_PARQUET_DIR))
    schema = train.collect_schema()
    base_features = feature_columns_from_schema(schema)
    source_features = base_features[: args.n_operational_source_features]
    model_features = tuple(dict.fromkeys([*base_features, "time_id", "symbol_id"]))
    bounds = train.select(
        [
            pl.min("date_id").alias("min_date_id"),
            pl.max("date_id").alias("max_date_id"),
            pl.max("time_id").alias("max_time_id"),
        ]
    ).collect()
    dataset_min_date_id = int(bounds["min_date_id"][0])
    dataset_max_date_id = int(bounds["max_date_id"][0])
    fold_min_date_id = dataset_min_date_id if args.min_date_id is None else args.min_date_id
    fold_max_date_id = dataset_max_date_id if args.max_date_id is None else args.max_date_id
    online._validate_date_bounds(
        dataset_min_date_id=dataset_min_date_id,
        dataset_max_date_id=dataset_max_date_id,
        fold_min_date_id=fold_min_date_id,
        fold_max_date_id=fold_max_date_id,
    )
    max_time_id = int(bounds["max_time_id"][0])
    folds = online.make_online_tail_folds(
        min_date_id=fold_min_date_id,
        max_date_id=fold_max_date_id,
        n_folds=args.n_folds,
        train_window=args.train_window,
        selection_window=args.selection_window,
        valid_window=args.valid_window,
        gap=args.gap,
    )
    clock_data = clock._with_clock_inputs(
        train,
        source_features=source_features,
        windows=windows,
        max_time_id=max_time_id,
    )
    prediction_columns = ("ridge_calibrated_prediction",) + tuple(f"{engine}_prediction" for engine in engines)

    score_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    observability_rows: list[dict[str, Any]] = []
    weight_bucket_rows: list[pl.DataFrame] = []
    simplex_parameter_rows: list[pl.DataFrame] = []

    for fold in folds:
        calibration = clock._collect_base_predictions(
            runner=runner,
            train=train,
            base_features=base_features,
            model_features=model_features,
            engines=engines,
            fold=fold.selection_model_fold(),
            args=args,
            inner=True,
        )
        calibration = clock._add_base_ensemble(
            calibration,
            runner=runner,
            engines=engines,
            args=args,
            fold=fold.selection_model_fold(),
        )
        selection = _collect_scored_period(
            clock=clock,
            runner=runner,
            train=train,
            base_features=base_features,
            model_features=model_features,
            engines=engines,
            args=args,
            model_fold=fold.selection_model_fold(),
            calibration=calibration,
        )
        validation = _collect_scored_period(
            clock=clock,
            runner=runner,
            train=train,
            base_features=base_features,
            model_features=model_features,
            engines=engines,
            args=args,
            model_fold=fold.validation_model_fold(),
            calibration=calibration,
        )

        calibration = clock._join_clock_values(clock_data, calibration, fold.train_start, fold.train_end)
        selection = clock._join_clock_values(clock_data, selection, fold.selection_start, fold.selection_end)
        validation = clock._join_clock_values(clock_data, validation, fold.valid_start, fold.valid_end)

        for phase, frame, start, end in (
            ("selection", selection, fold.selection_start, fold.selection_end),
            ("validation", validation, fold.valid_start, fold.valid_end),
        ):
            observability_rows.append(
                {
                    **_fold_metadata(fold),
                    "phase": phase,
                    **online.audit_batch_missing_observability(
                        train,
                        frame,
                        source_features=source_features,
                        start=start,
                        end=end,
                    ),
                }
            )

        binner = clock.fit_clock_binner(
            calibration,
            clock_name=args.clock_name,
            bucket_count=args.clock_bucket_count,
            max_time_id=max_time_id,
        )
        calibration = clock.apply_clock_binner(calibration, binner)
        selection = clock.apply_clock_binner(selection, binner)
        validation = clock.apply_clock_binner(validation, binner)

        grouped_simplex = clock.fit_grouped_simplex_weights(
            calibration,
            group_columns=("clock_bucket",),
            prediction_columns=prediction_columns,
            min_group_rows=args.min_group_rows,
        )
        simplex_parameter_rows.append(
            grouped_simplex.parameters.with_columns(
                [
                    pl.lit(fold.name).alias("fold"),
                    pl.lit(args.clock_name).alias("clock"),
                ]
            )
        )
        calibration = grouped_simplex.apply(calibration, output="clock_simplex_prediction")
        selection = grouped_simplex.apply(selection, output="clock_simplex_prediction")
        validation = grouped_simplex.apply(validation, output="clock_simplex_prediction")

        calibration, selection, validation = add_gate_meta_features(
            calibration,
            selection,
            validation,
            prediction_columns=prediction_columns,
        )
        policies = fit_gate_policies(
            calibration,
            candidates=gate_candidates,
            min_group_rows=args.min_group_rows,
        )
        selection = apply_gate_policies(selection, policies)
        validation = apply_gate_policies(validation, policies)
        group_selection_policies = fit_group_selection_policies(
            selection,
            policies=policies,
            min_group_rows=args.group_selection_min_rows,
            min_delta=args.group_selection_min_delta,
        )
        selection = apply_group_selection_policies(selection, group_selection_policies)
        validation = apply_group_selection_policies(validation, group_selection_policies)

        strategy_predictions = {
            "base_ensemble": "ensemble_prediction",
            f"{args.clock_name}_clock_simplex": "clock_simplex_prediction",
            **{policy.name: policy.output for policy in policies},
            **{policy.name: policy.output for policy in group_selection_policies},
        }
        selection_scores = {
            strategy: _score(clock, selection, prediction)
            for strategy, prediction in strategy_predictions.items()
        }
        selected_strategy = select_gate_strategy(
            selection_scores=selection_scores,
            base_strategy="base_ensemble",
            eligible_strategies=tuple(policy.name for policy in (*policies, *group_selection_policies)),
            min_delta=args.selection_min_delta,
        )
        validation = add_selected_gate_prediction(
            validation,
            selected_strategy=selected_strategy,
            strategy_predictions=strategy_predictions,
            output="selection_chosen_gate_prediction",
        )
        strategy_predictions["selection_chosen_gate"] = "selection_chosen_gate_prediction"

        for phase, frame in (("selection", selection), ("validation", validation)):
            phase_predictions = dict(strategy_predictions)
            if phase == "selection":
                phase_predictions.pop("selection_chosen_gate", None)
            for strategy, prediction in phase_predictions.items():
                score_rows.append(
                    {
                        **_fold_metadata(fold),
                        "phase": phase,
                        "clock": args.clock_name,
                        "strategy": strategy,
                        "selected_by_selection": strategy == selected_strategy,
                        **_score(clock, frame, prediction),
                    }
                )
                if phase == "validation":
                    weight_bucket_rows.append(
                        clock._weight_bucket_slice(frame, prediction=prediction)
                        .with_columns(
                            [
                                pl.lit(fold.name).alias("fold"),
                                pl.lit(args.clock_name).alias("clock"),
                                pl.lit(strategy).alias("strategy"),
                            ]
                        )
                        .select(
                            [
                                "fold",
                                "clock",
                                "strategy",
                                "weight_bucket",
                                "rows",
                                "weight_sum",
                                "numerator",
                                "denominator",
                                "weighted_zero_mean_r2",
                            ]
                        )
                    )

        base_selection_r2 = float(selection_scores["base_ensemble"]["weighted_zero_mean_r2"])
        selected_selection_r2 = float(selection_scores[selected_strategy]["weighted_zero_mean_r2"])
        for policy in policies:
            policy_rows.append(
                {
                    **_fold_metadata(fold),
                    "clock": args.clock_name,
                    "strategy": policy.name,
                    "kind": policy.kind,
                    "group_columns": ",".join(policy.group_columns),
                    "selected_strategy": selected_strategy,
                    "selected_by_selection": policy.name == selected_strategy,
                    "selection_base_r2": base_selection_r2,
                    "selection_strategy_r2": float(selection_scores[policy.name]["weighted_zero_mean_r2"]),
                    "selection_strategy_delta_r2": float(
                        selection_scores[policy.name]["weighted_zero_mean_r2"]
                    )
                    - base_selection_r2,
                    "selection_selected_delta_r2": selected_selection_r2 - base_selection_r2,
                    "selection_min_delta": args.selection_min_delta,
                    **policy.metadata(),
                    "group_gate_parameter_rows": None,
                    "group_gate_active_groups": None,
                    "group_gate_selection_rows": None,
                    "group_gate_selection_active_rows": None,
                    "group_gate_mean_delta_r2": None,
                }
            )
        for policy in group_selection_policies:
            selected_score = selection_scores.get(policy.name)
            policy_rows.append(
                {
                    **_fold_metadata(fold),
                    "clock": args.clock_name,
                    "strategy": policy.name,
                    "kind": "group_selection",
                    "group_columns": ",".join(policy.group_columns),
                    "selected_strategy": selected_strategy,
                    "selected_by_selection": policy.name == selected_strategy,
                    "selection_base_r2": base_selection_r2,
                    "selection_strategy_r2": (
                        float(selected_score["weighted_zero_mean_r2"]) if selected_score is not None else None
                    ),
                    "selection_strategy_delta_r2": (
                        float(selected_score["weighted_zero_mean_r2"]) - base_selection_r2
                        if selected_score is not None
                        else None
                    ),
                    "selection_selected_delta_r2": selected_selection_r2 - base_selection_r2,
                    "selection_min_delta": args.selection_min_delta,
                    "fallback_blend_weight": None,
                    "fallback_alpha": None,
                    "parameter_rows": int(policy.parameters.height),
                    **policy.metadata(),
                }
            )

        del calibration, selection, validation
        gc.collect()

    results = pl.DataFrame(score_rows)
    validation_summary = _summary(results.filter(pl.col("phase") == "validation"))
    selection_summary = _summary(results.filter(pl.col("phase") == "selection"))
    policy_frame = pl.DataFrame(policy_rows)
    observability_frame = pl.DataFrame(observability_rows)
    weight_bucket_by_fold = pl.concat(weight_bucket_rows) if weight_bucket_rows else pl.DataFrame()
    weight_bucket_summary = _combine_weight_bucket_slices(weight_bucket_by_fold) if weight_bucket_rows else pl.DataFrame()
    simplex_parameters = pl.concat(simplex_parameter_rows, how="diagonal") if simplex_parameter_rows else pl.DataFrame()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.write_csv(args.output_dir / "regime_confidence_gate_by_fold.csv")
    validation_summary.write_csv(args.output_dir / "regime_confidence_gate_validation_summary.csv")
    selection_summary.write_csv(args.output_dir / "regime_confidence_gate_selection_summary.csv")
    policy_frame.write_csv(args.output_dir / "regime_confidence_gate_policy_by_fold.csv")
    observability_frame.write_csv(args.output_dir / "gateway_observability_by_fold.csv")
    simplex_parameters.write_csv(args.output_dir / "clock_simplex_parameters.csv")
    if not weight_bucket_by_fold.is_empty():
        weight_bucket_by_fold.write_csv(args.output_dir / "regime_confidence_gate_weight_bucket_by_fold.csv")
        weight_bucket_summary.write_csv(args.output_dir / "regime_confidence_gate_weight_bucket_summary.csv")

    report = {
        "experiment": "regime_confidence_gate_walk_forward",
        "hypothesis": (
            "Clock simplex is more useful as a causal regime/confidence signal over the active ensemble "
            "than as a fixed tail switch."
        ),
        "folds": [_fold_metadata(fold) for fold in folds],
        "train_window": args.train_window,
        "selection_window": args.selection_window,
        "valid_window": args.valid_window,
        "dataset_date_bounds": {
            "min_date_id": dataset_min_date_id,
            "max_date_id": dataset_max_date_id,
        },
        "fold_date_bounds": {
            "min_date_id": fold_min_date_id,
            "max_date_id": fold_max_date_id,
        },
        "engines": engines,
        "train_sample_frac": args.train_sample_frac,
        "gbdt_seeds": args.gbdt_seeds,
        "max_iter": args.max_iter,
        "clock": args.clock_name,
        "gate_candidates": gate_candidates,
        "group_selection_min_rows": args.group_selection_min_rows,
        "group_selection_min_delta": args.group_selection_min_delta,
        "causality_contract": {
            "model_training_dates": "strictly before selection and validation",
            "gate_policy_family": "pre-specified by command-line candidate names before validation scoring",
            "gate_fit": "fit on inner OOF predictions from the train window",
            "selection_usage": "selection chooses among fitted policies; validation targets are not used for selection",
            "group_selection_usage": "selection may activate fitted candidates by pre-specified risk groups; validation targets are not used",
            "batch_missing_frac": "computed from the current (date_id,time_id) gateway batch only",
            "validation_targets_used_for_fitting": False,
            "official_lags_used": False,
        },
        "anti_leakage_audit": {
            "train_selection_validation_disjoint": all(online._fold_is_ordered(fold) for fold in folds),
            "selection_used_to_choose_gate": True,
            "validation_used_to_choose_gate": False,
            "direct_clock_simplex_eligible_for_selection": False,
            "gateway_observability_max_abs_diff": (
                float(observability_frame["max_abs_diff"].max()) if not observability_frame.is_empty() else None
            ),
        },
        "best_validation_strategy": validation_summary.row(0, named=True) if not validation_summary.is_empty() else None,
    }
    (args.output_dir / "regime_confidence_gate_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(validation_summary)
    print(policy_frame)
    print(f"Wrote {args.output_dir}")


def add_gate_meta_features(
    calibration: pl.DataFrame,
    selection: pl.DataFrame,
    validation: pl.DataFrame,
    *,
    prediction_columns: Sequence[str],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    frames = tuple(add_model_disagreement(frame, prediction_columns=prediction_columns) for frame in (calibration, selection, validation))
    calibration = frames[0].with_columns(
        (pl.col("clock_simplex_prediction") - pl.col("ensemble_prediction")).abs().alias("candidate_delta_abs")
    )
    selection = frames[1].with_columns(
        (pl.col("clock_simplex_prediction") - pl.col("ensemble_prediction")).abs().alias("candidate_delta_abs")
    )
    validation = frames[2].with_columns(
        (pl.col("clock_simplex_prediction") - pl.col("ensemble_prediction")).abs().alias("candidate_delta_abs")
    )

    ensemble_thresholds = fit_abs_prediction_thresholds(calibration, prediction="ensemble_prediction")
    disagreement_thresholds = fit_abs_prediction_thresholds(calibration, prediction="model_disagreement")
    delta_thresholds = fit_abs_prediction_thresholds(calibration, prediction="candidate_delta_abs")
    return tuple(
        add_abs_prediction_bucket(
            add_abs_prediction_bucket(
                add_abs_prediction_bucket(
                    frame,
                    ensemble_thresholds,
                    prediction="ensemble_prediction",
                    output="ensemble_abs_bucket",
                ),
                disagreement_thresholds,
                prediction="model_disagreement",
                output="disagreement_bucket",
            ),
            delta_thresholds,
            prediction="candidate_delta_abs",
            output="candidate_delta_bucket",
        )
        for frame in (calibration, selection, validation)
    )


def add_model_disagreement(
    frame: pl.DataFrame,
    *,
    prediction_columns: Sequence[str],
    output: str = "model_disagreement",
) -> pl.DataFrame:
    columns = tuple(prediction_columns)
    if len(columns) < 2:
        raise ValueError("prediction_columns must contain at least two columns")
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"missing prediction columns: {', '.join(sorted(missing))}")
    expressions = [pl.col(column) for column in columns]
    return frame.with_columns((pl.max_horizontal(expressions) - pl.min_horizontal(expressions)).alias(output))


def fit_gate_policies(
    calibration: pl.DataFrame,
    *,
    candidates: Sequence[str],
    min_group_rows: int,
) -> tuple[GatePolicy, ...]:
    policies: list[GatePolicy] = []
    for name in candidates:
        if name in BLEND_GROUPS:
            model = fit_grouped_convex_blend_weights(
                calibration,
                group_columns=BLEND_GROUPS[name],
                left_prediction="clock_simplex_prediction",
                right_prediction="ensemble_prediction",
                min_group_rows=min_group_rows,
            )
            policies.append(GatePolicy(name, "blend", BLEND_GROUPS[name], model, f"{name}_prediction"))
        elif name in SHRINK_GROUPS:
            model = fit_shrinkage_calibrator(
                calibration,
                name=name,
                group_columns=SHRINK_GROUPS[name],
                prediction="ensemble_prediction",
                min_group_rows=min_group_rows,
                clip_abs=None,
            )
            policies.append(GatePolicy(name, "shrink", SHRINK_GROUPS[name], model, f"{name}_prediction"))
        else:
            raise ValueError(f"unknown gate candidate: {name}")
    return tuple(policies)


def apply_gate_policies(frame: pl.DataFrame, policies: Sequence[GatePolicy]) -> pl.DataFrame:
    scored = frame
    for policy in policies:
        scored = policy.apply(scored)
    return scored


def fit_group_selection_policies(
    selection: pl.DataFrame,
    *,
    policies: Sequence[GatePolicy],
    min_group_rows: int,
    min_delta: float,
) -> tuple[GroupSelectionPolicy, ...]:
    if min_group_rows <= 0:
        raise ValueError("min_group_rows must be positive")
    if min_delta < 0.0:
        raise ValueError("min_delta must be non-negative")
    selected: list[GroupSelectionPolicy] = []
    for policy in policies:
        if not policy.group_columns:
            continue
        selected.append(
            fit_group_selection_policy(
                selection,
                name=f"group_select_{policy.name}",
                group_columns=policy.group_columns,
                base_prediction="ensemble_prediction",
                candidate_prediction=policy.output,
                min_group_rows=min_group_rows,
                min_delta=min_delta,
            )
        )
    return tuple(selected)


def fit_group_selection_policy(
    selection: pl.DataFrame,
    *,
    name: str,
    group_columns: Sequence[str],
    base_prediction: str,
    candidate_prediction: str,
    min_group_rows: int,
    min_delta: float = 0.0,
    target: str = "responder_6",
    weight: str = "weight",
) -> GroupSelectionPolicy:
    groups = tuple(group_columns)
    if not groups:
        raise ValueError("group_columns must not be empty")
    if min_group_rows <= 0:
        raise ValueError("min_group_rows must be positive")
    if min_delta < 0.0:
        raise ValueError("min_delta must be non-negative")
    required = {target, weight, base_prediction, candidate_prediction, *groups}
    missing = required - set(selection.columns)
    if missing:
        raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")
    parameters = (
        selection.with_columns(
            [
                (pl.col(weight) * (pl.col(target) - pl.col(base_prediction)).pow(2)).alias("_group_gate_base_loss"),
                (pl.col(weight) * (pl.col(target) - pl.col(candidate_prediction)).pow(2)).alias(
                    "_group_gate_candidate_loss"
                ),
                (pl.col(weight) * pl.col(target).pow(2)).alias("_group_gate_target_energy"),
            ]
        )
        .group_by(list(groups))
        .agg(
            [
                pl.len().alias("_group_gate_rows"),
                pl.col("_group_gate_base_loss").sum().alias("_group_gate_base_numerator"),
                pl.col("_group_gate_candidate_loss").sum().alias("_group_gate_candidate_numerator"),
                pl.col("_group_gate_target_energy").sum().alias("_group_gate_denominator"),
            ]
        )
        .with_columns(
            (
                (pl.col("_group_gate_base_numerator") - pl.col("_group_gate_candidate_numerator"))
                / pl.col("_group_gate_denominator")
            )
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias("_group_gate_delta_r2")
        )
        .with_columns(
            (
                (pl.col("_group_gate_rows") >= min_group_rows)
                & (pl.col("_group_gate_denominator") > 1e-12)
                & (pl.col("_group_gate_delta_r2") > min_delta)
            ).alias("_group_gate_use_candidate")
        )
        .select(
            list(groups)
            + [
                "_group_gate_rows",
                "_group_gate_base_numerator",
                "_group_gate_candidate_numerator",
                "_group_gate_denominator",
                "_group_gate_delta_r2",
                "_group_gate_use_candidate",
            ]
        )
    )
    return GroupSelectionPolicy(
        name=name,
        group_columns=groups,
        parameters=parameters,
        base_prediction=base_prediction,
        candidate_prediction=candidate_prediction,
        output=f"{name}_prediction",
    )


def apply_group_selection_policies(frame: pl.DataFrame, policies: Sequence[GroupSelectionPolicy]) -> pl.DataFrame:
    scored = frame
    for policy in policies:
        scored = policy.apply(scored)
    return scored


def select_gate_strategy(
    *,
    selection_scores: dict[str, dict[str, float | int]],
    base_strategy: str,
    eligible_strategies: Sequence[str],
    min_delta: float,
) -> str:
    if base_strategy not in selection_scores:
        raise ValueError(f"base strategy {base_strategy!r} is missing from selection_scores")
    if min_delta < 0.0:
        raise ValueError("min_delta must be non-negative")
    base_r2 = float(selection_scores[base_strategy]["weighted_zero_mean_r2"])
    best_strategy = base_strategy
    best_r2 = base_r2
    for strategy in eligible_strategies:
        if strategy not in selection_scores:
            raise ValueError(f"eligible strategy {strategy!r} is missing from selection_scores")
        r2 = float(selection_scores[strategy]["weighted_zero_mean_r2"])
        if r2 > best_r2:
            best_strategy = strategy
            best_r2 = r2
    if best_strategy == base_strategy or best_r2 <= base_r2 + min_delta:
        return base_strategy
    return best_strategy


def add_selected_gate_prediction(
    frame: pl.DataFrame,
    *,
    selected_strategy: str,
    strategy_predictions: dict[str, str],
    output: str,
) -> pl.DataFrame:
    if selected_strategy not in strategy_predictions:
        raise ValueError(f"unknown selected strategy: {selected_strategy}")
    return frame.with_columns(pl.col(strategy_predictions[selected_strategy]).alias(output))


def _collect_scored_period(
    *,
    clock: Any,
    runner: Any,
    train: pl.LazyFrame,
    base_features: tuple[str, ...],
    model_features: tuple[str, ...],
    engines: tuple[str, ...],
    args: argparse.Namespace,
    model_fold: Any,
    calibration: pl.DataFrame,
) -> pl.DataFrame:
    frame = clock._collect_base_predictions(
        runner=runner,
        train=train,
        base_features=base_features,
        model_features=model_features,
        engines=engines,
        fold=model_fold,
        args=args,
        inner=False,
    )
    return clock._add_base_ensemble(
        frame,
        runner=runner,
        engines=engines,
        args=args,
        fold=model_fold,
        calibration=calibration,
    )


def _score(clock: Any, frame: pl.DataFrame, prediction: str) -> dict[str, float | int]:
    return clock._score(calibration=frame, prediction=prediction)


def _summary(results: pl.DataFrame) -> pl.DataFrame:
    if results.is_empty():
        return pl.DataFrame()
    return (
        results.group_by("strategy")
        .agg(
            [
                pl.len().alias("folds"),
                pl.col("weighted_zero_mean_r2").mean().alias("mean_r2"),
                pl.col("weighted_zero_mean_r2").std().alias("std_r2"),
                pl.col("weighted_zero_mean_r2").min().alias("min_r2"),
                pl.col("weighted_zero_mean_r2").max().alias("max_r2"),
                (1.0 - pl.col("numerator").sum() / pl.col("denominator").sum()).alias("global_r2"),
                pl.col("rows").sum().alias("validation_rows"),
                pl.col("selected_by_selection").sum().alias("selected_folds"),
            ]
        )
        .sort(["global_r2", "min_r2"], descending=[True, True])
    )


def _combine_weight_bucket_slices(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame()
    return (
        frame.group_by(["clock", "strategy", "weight_bucket"])
        .agg(
            [
                pl.col("rows").sum(),
                pl.col("weight_sum").sum(),
                pl.col("numerator").sum(),
                pl.col("denominator").sum(),
            ]
        )
        .with_columns((1.0 - pl.col("numerator") / pl.col("denominator")).alias("weighted_zero_mean_r2"))
        .sort(["clock", "strategy", "weight_bucket"])
    )


def _fold_metadata(fold: Any) -> dict[str, int | str]:
    return {
        "fold": fold.name,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "selection_start": fold.selection_start,
        "selection_end": fold.selection_end,
        "valid_start": fold.valid_start,
        "valid_end": fold.valid_end,
    }


def _parse_gate_candidates(raw: str) -> tuple[str, ...]:
    candidates = tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
    allowed = set(BLEND_GROUPS) | set(SHRINK_GROUPS)
    unknown = set(candidates) - allowed
    if unknown:
        raise ValueError(f"unknown gate candidates: {', '.join(sorted(unknown))}")
    if not candidates:
        raise ValueError("--gate-candidates must not be empty")
    return candidates


def _load_clock_tournament_module() -> Any:
    path = PROJECT_ROOT / "scripts" / "run_clock_tournament.py"
    spec = importlib.util.spec_from_file_location("run_clock_tournament", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_online_tail_module() -> Any:
    path = PROJECT_ROOT / "scripts" / "run_online_tail_control_validation.py"
    spec = importlib.util.spec_from_file_location("run_online_tail_control_validation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("n_folds", "train_window", "selection_window", "valid_window", "inner_oof_folds", "inner_valid_window"):
        _require_positive(name, getattr(args, name))
    _require_non_negative("gap", args.gap)
    if not 0.0 < args.train_sample_frac <= 1.0:
        raise ValueError("--train-sample-frac must be in (0, 1]")
    for name in ("max_iter", "max_leaf_nodes", "n_jobs", "chunk_days", "clock_bucket_count", "min_group_rows"):
        _require_positive(name, getattr(args, name))
    if args.clock_bucket_count <= 1:
        raise ValueError("--clock-bucket-count must be greater than 1")
    if args.n_operational_source_features <= 0:
        raise ValueError("--n-operational-source-features must be positive")
    _parse_gate_candidates(args.gate_candidates)
    if args.selection_min_delta < 0.0:
        raise ValueError("--selection-min-delta must be non-negative")


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


if __name__ == "__main__":
    main()
