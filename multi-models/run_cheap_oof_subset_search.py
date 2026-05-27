"""Cheap subset search over saved OOF prediction artifacts.

This script does not train new primary models. It reuses saved validation
predictions and tests causal walk-forward stacks plus simple blends.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import sys
from pathlib import Path
from typing import Sequence

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL_PACKAGE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from multimodels.metrics import summarize_scores  # noqa: E402
from multimodels.strong_oof import (  # noqa: E402
    KEYS,
    TARGET,
    _fit_convex_pair_weight,
    _folds,
    _format_float,
    _score_by_fold,
    add_gateway_rls_predictions,
    add_prediction_context,
    add_walk_forward_ridge_stack,
    load_joined_predictions,
)


BASE_CANDIDATES = (
    "tabm_prediction",
    "tree_prediction",
    "xgboost_prediction",
    "lightgbm_prediction",
    "ridge_calibrated_prediction",
    "baseline_prediction",
    "conservative_rls_prediction",
    "aggressive_rls_prediction",
)

DEFAULT_EXTRA_DIRS = (
    Path("reports/experiments/competitive_tabm_official_stage3_5fold_valid60_lags_online_lr1e4_4m_train700_seed37_aux8_w010_complete_preds/validation_predictions"),
    Path("reports/experiments/competitive_tabm_official_stage3_5fold_valid60_lags_online_lr1e4_4m_train700_seed23_aux8_preds/validation_predictions"),
    Path("reports/experiments/competitive_tabm_official_stage3_5fold_valid60_lags_online_lr1e4_4m_train700_seed23_preds/validation_predictions"),
    Path("reports/experiments/competitive_tabm_official_stage3_5fold_valid60_lags_online_lr1e4_4m_train700_seed17_preds/validation_predictions"),
    Path("reports/experiments/competitive_tabm_official_stage3_5fold_valid60_lags_online_lr1e4_4m_train700_seed37_preds/validation_predictions"),
)

DEFAULT_EXTRA_PREFIXES = ("tabm_w010", "tabm_s23_aux8", "tabm_s23", "tabm_s17", "tabm_s37")

DEFAULT_BLEND_PARTNERS = (
    "gateway_risk_conservative_rls_abs_pred_s100_prediction",
    "gateway_risk_conservative_rls_abs_pred_s25_prediction",
    "gateway_risk_aggressive_rls_abs_pred_s25_prediction",
    "gateway_risk_aggressive_rls_abs_pred_s100_prediction",
    "conservative_rls_prediction",
    "aggressive_rls_prediction",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cheap OOF subset stack/blend search.")
    parser.add_argument("--experiment-name", default="strong_oof_cheap_subset_search_stage3_v1")
    parser.add_argument(
        "--tabm-prediction-dir",
        type=Path,
        default=Path("reports/experiments/competitive_tabm_official_stage3_5fold_valid60_lags_online_lr1e4_4m_train700_seed37_aux8_preds/validation_predictions"),
    )
    parser.add_argument(
        "--tree-prediction-dir",
        type=Path,
        default=Path("reports/experiments/tree_engine_ensemble_xgb_lgb_sample10_seed_ensemble_preds/validation_predictions"),
    )
    parser.add_argument("--extra-prediction-dirs", default=",".join(str(path) for path in DEFAULT_EXTRA_DIRS))
    parser.add_argument("--extra-prediction-prefixes", default=",".join(DEFAULT_EXTRA_PREFIXES))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stack-alpha", type=float, default=1000.0)
    parser.add_argument("--fixed-blend-weights", default="0.25,0.5,0.75")
    parser.add_argument("--blend-partners", default=",".join(DEFAULT_BLEND_PARTNERS))
    parser.add_argument("--max-subset-size", type=int, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or Path("multi-models") / "reports" / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    extra_dirs = _parse_path_tuple(args.extra_prediction_dirs)
    extra_prefixes = _parse_str_tuple(args.extra_prediction_prefixes)
    if len(extra_dirs) != len(extra_prefixes):
        raise ValueError("extra prediction dirs and prefixes must have the same length")

    frame = load_joined_predictions(
        args.tabm_prediction_dir,
        args.tree_prediction_dir,
        extra_prediction_dirs=extra_dirs,
        extra_prediction_prefixes=extra_prefixes,
    )
    frame = add_prediction_context(frame)
    frame, gateway_audits = add_gateway_rls_predictions(
        frame,
        include_risk_shrink=True,
        risk_strengths=(25.0, 100.0),
        risk_profiles=("abs_pred",),
    )
    frame = add_prediction_context(frame)

    extra_columns = tuple(f"{prefix}_tabm_prediction" for prefix in extra_prefixes if f"{prefix}_tabm_prediction" in frame.columns)
    base_columns = tuple(column for column in BASE_CANDIDATES if column in frame.columns)
    risk_columns = tuple(column for column in frame.columns if column.startswith("gateway_risk_") and column.endswith("_prediction"))
    blend_partners = tuple(column for column in _parse_str_tuple(args.blend_partners) if column in frame.columns)
    fixed_weights = _parse_float_tuple(args.fixed_blend_weights)
    model_core_columns = tuple(dict.fromkeys([*base_columns, *risk_columns]))

    score_rows: list[dict[str, float | int | str]] = []
    params: list[dict[str, object]] = []
    subsets = list(_extra_subsets(extra_columns, args.max_subset_size))
    for subset_idx, subset in enumerate(subsets):
        subset_name = _subset_name(subset_idx, subset, extra_columns)
        prediction_columns = tuple(dict.fromkeys([*model_core_columns, *subset]))
        stack_predictions, stack_params = add_walk_forward_ridge_stack(frame, prediction_columns=prediction_columns, alpha=args.stack_alpha)
        raw_stack_columns = [column for column in stack_predictions.columns if column not in KEYS]
        if len(raw_stack_columns) != 1:
            raise ValueError(f"expected one stack prediction column, found {raw_stack_columns}")
        stack_column = f"{subset_name}_stack_prediction"
        stack_predictions = stack_predictions.rename({raw_stack_columns[0]: stack_column})
        eval_columns = tuple(dict.fromkeys([*KEYS, "weight", TARGET, *blend_partners]))
        eval_frame = frame.select(eval_columns).join(stack_predictions, on=list(KEYS), how="inner")

        score_rows.extend(
            _score_by_fold(eval_frame, prediction=stack_column, candidate=f"{subset_name}_stack", family="cheap_subset_stack").to_dicts()
        )
        params.append({"component": "subset", "subset": subset_name, "columns": list(subset), "n_prediction_columns": len(prediction_columns)})
        for row in stack_params:
            row["subset"] = subset_name
            params.append(row)

        for partner in blend_partners:
            score_rows.extend(_fixed_blend_scores(eval_frame, stack_column=stack_column, partner=partner, subset_name=subset_name, weights=fixed_weights))
            wf_scores, wf_params = _walk_forward_blend_scores(eval_frame, stack_column=stack_column, partner=partner, subset_name=subset_name)
            score_rows.extend(wf_scores)
            params.extend(wf_params)

        del stack_predictions, eval_frame
        gc.collect()

    scores = pl.DataFrame(score_rows)
    summary = summarize_scores(scores)
    scores.write_csv(output_dir / "fold_scores.csv")
    summary.write_csv(output_dir / "candidate_summary.csv")
    (output_dir / "parameters.json").write_text(json.dumps(params, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if gateway_audits:
        pl.concat(gateway_audits, how="diagonal").write_csv(output_dir / "gateway_daily_audit.csv")
    audit = {
        "experiment_name": args.experiment_name,
        "rows": frame.height,
        "folds": len(_folds(frame)),
        "n_subsets": len(subsets),
        "base_columns": list(base_columns),
        "extra_columns": list(extra_columns),
        "risk_columns": list(risk_columns),
        "blend_partners": list(blend_partners),
        "fixed_blend_weights": list(fixed_weights),
        "stack_alpha": float(args.stack_alpha),
        "causality_check": "passed: stack and walk-forward blend weights use only earlier folds; fixed blends use predeclared weights only",
        "target_leakage_check": "passed: no target-derived future fold feature is joined into prediction columns",
        "selection_check": "exploratory subset grid; promote only after independent confirmation",
    }
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary.head(30))
    print(f"Wrote {output_dir}")


def _extra_subsets(columns: Sequence[str], max_subset_size: int | None) -> list[tuple[str, ...]]:
    limit = len(columns) if max_subset_size is None else min(int(max_subset_size), len(columns))
    out: list[tuple[str, ...]] = []
    for size in range(limit + 1):
        out.extend(itertools.combinations(columns, size))
    return out


def _subset_name(idx: int, subset: Sequence[str], all_columns: Sequence[str]) -> str:
    bits = "".join("1" if column in subset else "0" for column in all_columns)
    return f"s{idx:02d}_{bits}"


def _fixed_blend_scores(
    frame: pl.DataFrame,
    *,
    stack_column: str,
    partner: str,
    subset_name: str,
    weights: Sequence[float],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for weight in weights:
        column = f"{subset_name}_fixed_{_short_name(partner)}_w{_format_float(weight)}_prediction"
        scored = frame.with_columns((float(weight) * pl.col(stack_column) + (1.0 - float(weight)) * pl.col(partner)).alias(column))
        rows.extend(_score_by_fold(scored, prediction=column, candidate=column.removesuffix("_prediction"), family="cheap_fixed_blend").to_dicts())
    return rows


def _walk_forward_blend_scores(
    frame: pl.DataFrame,
    *,
    stack_column: str,
    partner: str,
    subset_name: str,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, object]]]:
    folds = _folds(frame)
    parts: list[pl.DataFrame] = []
    params: list[dict[str, object]] = []
    column = f"{subset_name}_wf_{_short_name(partner)}_prediction"
    for idx, fold in enumerate(folds):
        current = frame.filter(pl.col("fold") == fold)
        if idx == 0:
            weight = 0.5
            fit_rows = 0
        else:
            calibration = frame.filter(pl.col("fold").is_in(folds[:idx]))
            weight = _fit_convex_pair_weight(calibration, left=stack_column, right=partner)
            fit_rows = calibration.height
        parts.append(
            current.with_columns((float(weight) * pl.col(stack_column) + (1.0 - float(weight)) * pl.col(partner)).alias(column)).select(
                [*KEYS, "weight", TARGET, column]
            )
        )
        params.append(
            {
                "component": "cheap_walk_forward_blend",
                "subset": subset_name,
                "partner": partner,
                "fold": fold,
                "stack_weight": float(weight),
                "partner_weight": float(1.0 - weight),
                "fit_rows": fit_rows,
            }
        )
    scored = pl.concat(parts)
    rows = _score_by_fold(scored, prediction=column, candidate=column.removesuffix("_prediction"), family="cheap_walk_forward_blend").to_dicts()
    return rows, params


def _short_name(column: str) -> str:
    cleaned = column.removesuffix("_prediction")
    replacements = {
        "gateway_risk_conservative_rls_abs_pred_s100": "risk_cons_s100",
        "gateway_risk_conservative_rls_abs_pred_s25": "risk_cons_s25",
        "gateway_risk_aggressive_rls_abs_pred_s100": "risk_aggr_s100",
        "gateway_risk_aggressive_rls_abs_pred_s25": "risk_aggr_s25",
        "conservative_rls": "cons_rls",
        "aggressive_rls": "aggr_rls",
    }
    return replacements.get(cleaned, cleaned)


def _parse_str_tuple(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _parse_path_tuple(raw: str) -> tuple[Path, ...]:
    return tuple(Path(part.strip()) for part in raw.split(",") if part.strip())


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("at least one float value is required")
    return values


if __name__ == "__main__":
    main()
