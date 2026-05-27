"""End-to-end modular validation pipeline."""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl

from janestreet.folds import DateFold, make_rolling_folds
from janestreet.paths import TRAIN_PARQUET_DIR, TRAIN_WITH_RESPONDER_LAGS_PARQUET

from multimodels.features import (
    BASE_KEYS,
    TARGET,
    WEIGHT,
    add_context_features,
    add_raw_preprocessing_features,
    feature_columns_from_schema,
    lag_columns_from_schema,
    scan_parquet_dir,
)
from multimodels.metrics import score_arrays, summarize_scores
from multimodels.models import (
    fit_grouped_scale_calibrator,
    fit_linear_stacker,
    fit_pls_model,
    fit_residual_rule,
    fit_ridge_model,
    MicrostructureRegimeBinner,
    risk_shrink,
)
from multimodels.reporting import write_json, write_markdown_report


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str = "multi_model_lab"
    output_dir: Path = Path("multi-models/reports/multi_model_lab")
    train_parquet_dir: Path = TRAIN_PARQUET_DIR
    use_processed_lags: bool = False
    n_folds: int = 5
    train_window: int = 120
    valid_window: int = 60
    gap: int = 0
    sample_stride: int = 1
    max_train_rows: int | None = None
    max_valid_rows: int | None = None
    max_features: int | None = None
    rank_bins: int = 255
    ridge_alpha: float = 10000.0
    pls_components: int = 8
    stack_alpha: float = 1000.0
    risk_alpha: float = 1000.0
    risk_auxiliary_targets: tuple[str, ...] = ("abs_responder6", "sq_responder6", "abs_error", "sq_error", "high_error")
    high_error_quantile: float = 0.90
    risk_strengths: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1, 0.2)
    residual_features: tuple[str, ...] = ("feature_47", "feature_59", "feature_04")
    auxiliary_base_candidates: tuple[str, ...] = ("ridge_rank_alpha10000", "pls_rank_k8")
    include_context_features: bool = True
    cross_sectional_features: tuple[str, ...] = ("feature_04", "feature_47", "feature_59")
    time_bucket_size: int = 100
    raw_preprocess_features: tuple[str, ...] = ()
    raw_preprocess_modes: tuple[str, ...] = ()
    min_regime_rows: int = 2000
    regime_prior_strength: float = 1000.0
    regime_symbol_mod: int = 8
    write_predictions: bool = False
    seed: int = 37


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    """Run the configured multi-model validation experiment."""

    _validate_config(config)
    np.random.default_rng(config.seed)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = TRAIN_WITH_RESPONDER_LAGS_PARQUET if config.use_processed_lags else config.train_parquet_dir
    lazy = scan_parquet_dir(train_path)
    schema = lazy.collect_schema()
    base_features = feature_columns_from_schema(schema, max_features=config.max_features)
    lag_features = lag_columns_from_schema(schema) if config.use_processed_lags else ()
    raw_preprocessed_features: tuple[str, ...] = ()
    if config.include_context_features:
        lazy, context_features = add_context_features(
            lazy,
            base_features=base_features,
            lag_features=lag_features,
            cross_sectional_features=config.cross_sectional_features,
            time_bucket_size=config.time_bucket_size,
        )
    else:
        context_features = ()
    if config.raw_preprocess_features and config.raw_preprocess_modes:
        available_raw = tuple(name for name in config.raw_preprocess_features if name in base_features)
        lazy, raw_preprocessed_features = add_raw_preprocessing_features(
            lazy,
            raw_feature_columns=available_raw,
            modes=config.raw_preprocess_modes,
        )
    model_features = tuple(dict.fromkeys([*base_features, *lag_features, *context_features, *raw_preprocessed_features]))
    folds = _make_folds(lazy, config)

    score_rows: list[dict[str, float | int | str]] = []
    fold_audit_rows: list[dict[str, float | int | str | bool]] = []
    auxiliary_score_rows: list[dict[str, float | int | str]] = []
    prediction_frames: list[pl.DataFrame] = []

    for fold in folds:
        train = _collect_fold_frame(lazy, fold=fold, split="train", config=config, columns=model_features)
        valid = _collect_fold_frame(lazy, fold=fold, split="valid", config=config, columns=model_features)
        if train.height == 0 or valid.height == 0:
            continue
        fold_scores, fold_auxiliary_scores, fold_audit, predictions = _run_fold(train, valid, fold=fold, config=config, model_features=model_features)
        score_rows.extend(fold_scores)
        auxiliary_score_rows.extend(fold_auxiliary_scores)
        fold_audit_rows.append(fold_audit)
        if config.write_predictions:
            prediction_frames.append(predictions)
        del train, valid, fold_scores, fold_audit, predictions
        gc.collect()

    scores = pl.DataFrame(score_rows)
    summary = summarize_scores(scores)
    scores.write_csv(output_dir / "fold_scores.csv")
    summary.write_csv(output_dir / "candidate_summary.csv")
    auxiliary_scores = pl.DataFrame(auxiliary_score_rows) if auxiliary_score_rows else pl.DataFrame()
    if not auxiliary_scores.is_empty():
        auxiliary_scores.write_csv(output_dir / "auxiliary_scores.csv")
    if config.write_predictions and prediction_frames:
        pl.concat(prediction_frames, how="diagonal").write_parquet(output_dir / "validation_predictions.parquet")

    audit = {
        "experiment_name": config.experiment_name,
        "config": _serializable_config(config),
        "n_folds": len(folds),
        "folds": [asdict(fold) for fold in folds],
        "fold_audit": fold_audit_rows,
        "n_base_features": len(base_features),
        "n_lag_features": len(lag_features),
        "n_context_features": len(context_features),
        "n_raw_preprocessed_features": len(raw_preprocessed_features),
        "n_model_features": len(model_features),
        "uses_processed_lags": bool(config.use_processed_lags),
        "uses_context_features": bool(config.include_context_features),
        "raw_preprocess_features": list(config.raw_preprocess_features),
        "raw_preprocess_modes": list(config.raw_preprocess_modes),
        "raw_preprocessed_features": list(raw_preprocessed_features),
        "target_leakage_check": _target_leakage_check(model_features),
        "fold_causality_check": _fold_causality_check(folds, gap=config.gap),
        "selection_check": "passed: risk strengths and auxiliary bases are fixed by config, not selected inside validation",
        "residual_features": [name for name in config.residual_features if name in model_features],
        "risk_auxiliary_targets": list(config.risk_auxiliary_targets),
        "artifact_manifest": artifact_manifest(config),
        "known_reference_points": {
            "conservative_stage3_local": 0.013836465,
            "historical_max1398": 0.015425344,
        },
    }
    write_json(output_dir / "audit.json", audit)
    write_markdown_report(path=output_dir / "REPORT.md", experiment_name=config.experiment_name, summary=summary, audit=audit)
    return {"scores": scores, "auxiliary_scores": auxiliary_scores, "summary": summary, "audit": audit, "output_dir": output_dir}


def _run_fold(
    train: pl.DataFrame,
    valid: pl.DataFrame,
    *,
    fold: DateFold,
    config: ExperimentConfig,
    model_features: tuple[str, ...],
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]], dict[str, float | int | str | bool], pl.DataFrame]:
    y_train = train[TARGET].to_numpy().astype(np.float64, copy=False)
    y_valid = valid[TARGET].to_numpy().astype(np.float64, copy=False)
    w_train = train[WEIGHT].to_numpy().astype(np.float64, copy=False)
    w_valid = valid[WEIGHT].to_numpy().astype(np.float64, copy=False)

    ridge_name = _ridge_artifact_name(config)
    pls_name = _pls_artifact_name(config)
    ridge = fit_ridge_model(
        name=ridge_name,
        family="alpha_latent_global",
        train=train,
        feature_columns=model_features,
        target=y_train,
        weight=w_train,
        alpha=config.ridge_alpha,
        encoder_kind="rank",
        rank_bins=config.rank_bins,
    )
    pls = fit_pls_model(
        name=pls_name,
        family="alpha_latent_recent",
        train=train,
        feature_columns=model_features,
        target=y_train,
        components=config.pls_components,
        rank_bins=config.rank_bins,
    )
    models = (ridge, pls)
    train_model_preds = np.column_stack([model.predict(train) for model in models])
    valid_model_preds = np.column_stack([model.predict(valid) for model in models])

    score_rows: list[dict[str, float | int | str]] = []
    auxiliary_score_rows: list[dict[str, float | int | str]] = []
    prediction_columns: dict[str, np.ndarray] = {}
    train_prediction_columns: dict[str, np.ndarray] = {}
    for idx, model in enumerate(models):
        train_prediction_columns[model.name] = train_model_preds[:, idx]
        pred = valid_model_preds[:, idx]
        prediction_columns[model.name] = pred
        score_rows.append(
            score_arrays(
                fold=fold.name,
                candidate=model.name,
                family=model.family,
                y_true=y_valid,
                y_pred=pred,
                weight=w_valid,
            )
        )

    stacker = fit_linear_stacker(
        name="latent_alpha_linear_stack",
        model_names=tuple(model.name for model in models),
        predictions=train_model_preds,
        target=y_train,
        weight=w_train,
        alpha=config.stack_alpha,
    )
    train_stack = stacker.predict(train_model_preds)
    valid_stack = stacker.predict(valid_model_preds)
    train_prediction_columns[stacker.name] = train_stack
    prediction_columns[stacker.name] = valid_stack
    score_rows.append(
        score_arrays(
            fold=fold.name,
            candidate=stacker.name,
            family="alpha_latent_stack",
            y_true=y_valid,
            y_pred=valid_stack,
            weight=w_valid,
        )
    )

    auxiliary_audits = []
    for base_name in config.auxiliary_base_candidates:
        if base_name not in train_prediction_columns or base_name not in prediction_columns:
            continue
        aux_scores, aux_target_scores, aux_predictions, aux_audit = _apply_auxiliary_layers(
            base_name=base_name,
            train_base=train_prediction_columns[base_name],
            valid_base=prediction_columns[base_name],
            train=train,
            valid=valid,
            y_train=y_train,
            y_valid=y_valid,
            w_train=w_train,
            w_valid=w_valid,
            fold=fold,
            config=config,
            model_features=model_features,
        )
        score_rows.extend(aux_scores)
        auxiliary_score_rows.extend(aux_target_scores)
        prediction_columns.update(aux_predictions)
        auxiliary_audits.append(aux_audit)

    predictions = valid.select(list(BASE_KEYS) + [WEIGHT, TARGET]).with_columns(pl.lit(fold.name).alias("fold"))
    if "row_id" in valid.columns:
        predictions = valid.select(["row_id"] + list(BASE_KEYS) + [WEIGHT, TARGET]).with_columns(pl.lit(fold.name).alias("fold"))
    for name, values in prediction_columns.items():
        predictions = predictions.with_columns(pl.Series(name, values))

    audit = {
        "fold": fold.name,
        "train_rows": train.height,
        "valid_rows": valid.height,
        f"stack_coeff_{ridge_name}": float(stacker.coefficients[0]),
        f"stack_coeff_{pls_name}": float(stacker.coefficients[1]),
        "risk_strengths_are_fixed_grid": True,
        "risk_auxiliary_targets": ",".join(config.risk_auxiliary_targets),
        "auxiliary_bases_evaluated": ",".join(audit["base_name"] for audit in auxiliary_audits),
        "auxiliary_audit_count": len(auxiliary_audits),
    }
    return score_rows, auxiliary_score_rows, audit, predictions


def _apply_auxiliary_layers(
    *,
    base_name: str,
    train_base: np.ndarray,
    valid_base: np.ndarray,
    train: pl.DataFrame,
    valid: pl.DataFrame,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    w_train: np.ndarray,
    w_valid: np.ndarray,
    fold: DateFold,
    config: ExperimentConfig,
    model_features: tuple[str, ...],
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]], dict[str, np.ndarray], dict[str, float | int | str | bool]]:
    """Apply residual, risk and regime layers to one fixed base prediction."""

    score_rows: list[dict[str, float | int | str]] = []
    auxiliary_score_rows: list[dict[str, float | int | str]] = []
    prediction_columns: dict[str, np.ndarray] = {}
    train_residual = y_train - train_base
    fitted_residual_rules = []
    for feature in config.residual_features:
        if feature not in train.columns:
            continue
        rule = fit_residual_rule(
            train=train,
            feature=feature,
            residual=train_residual,
            weight=w_train,
            name=f"{base_name}__residual_{feature}_z",
        )
        fitted_residual_rules.append(rule)
        individual_name = f"{base_name}__{feature}_z_residual"
        individual_pred = valid_base + rule.predict(valid)
        prediction_columns[individual_name] = individual_pred
        score_rows.append(
            score_arrays(
                fold=fold.name,
                candidate=individual_name,
                family="residual_correction",
                y_true=y_valid,
                y_pred=individual_pred,
                weight=w_valid,
            )
        )
    valid_residual_add = np.zeros(valid.height, dtype=np.float64)
    for rule in fitted_residual_rules:
        valid_residual_add += rule.predict(valid)
    train_residual_add = np.zeros(train.height, dtype=np.float64)
    for rule in fitted_residual_rules:
        train_residual_add += rule.predict(train)
    train_residual_stack = train_base + train_residual_add
    valid_residual_stack = valid_base + valid_residual_add
    residual_name = f"{base_name}__plus_residual_rules"
    prediction_columns[residual_name] = valid_residual_stack
    score_rows.append(
        score_arrays(
            fold=fold.name,
            candidate=residual_name,
            family="residual_correction",
            y_true=y_valid,
            y_pred=valid_residual_stack,
            weight=w_valid,
        )
    )

    regime_group_counts: dict[str, int] = {}
    regime_default_scales: dict[str, float] = {}
    risk_targets, high_error_threshold = _fit_risk_targets(
        y_train=y_train,
        train_base=train_residual_stack,
        high_error_quantile=config.high_error_quantile,
        requested=config.risk_auxiliary_targets,
    )
    valid_risk_targets = _apply_risk_targets(
        y=y_valid,
        base=valid_residual_stack,
        high_error_threshold=high_error_threshold,
        requested=tuple(risk_targets.keys()),
    )
    primary_micro_risk: tuple[np.ndarray, np.ndarray, float] | None = None
    for risk_name, risk_target in risk_targets.items():
        risk_model_name = f"{base_name}__risk_{risk_name}_ridge_rank"
        risk_model = fit_ridge_model(
            name=risk_model_name,
            family="risk_volatility_uncertainty",
            train=train,
            feature_columns=model_features,
            target=risk_target,
            weight=w_train,
            alpha=config.risk_alpha,
            encoder_kind="rank",
            rank_bins=config.rank_bins,
        )
        train_risk = np.maximum(risk_model.predict(train), 0.0)
        valid_risk = np.maximum(risk_model.predict(valid), 0.0)
        auxiliary_row = score_arrays(
            fold=fold.name,
            candidate=f"{risk_model_name}_score",
            family=f"risk_auxiliary_{risk_name}",
            y_true=valid_risk_targets[risk_name],
            y_pred=valid_risk,
            weight=w_valid,
        )
        auxiliary_row["target_name"] = risk_name
        auxiliary_score_rows.append(auxiliary_row)
        train_risk_mean = float(np.average(train_risk, weights=w_train)) if float(np.sum(w_train)) > 0.0 else float(np.mean(train_risk))
        if config.write_predictions:
            prediction_columns[f"{risk_model_name}_score"] = valid_risk
        if risk_name == "abs_error":
            primary_micro_risk = (train_risk, valid_risk, train_risk_mean)
        for strength in config.risk_strengths:
            suffix = _format_float(strength)
            candidate_name = f"{base_name}__risk_{risk_name}_shrink_s{suffix}"
            train_shrunk = risk_shrink(train_residual_stack, train_risk, train_risk_mean=train_risk_mean, strength=strength)
            valid_shrunk = risk_shrink(valid_residual_stack, valid_risk, train_risk_mean=train_risk_mean, strength=strength)
            if config.write_predictions:
                prediction_columns[candidate_name] = valid_shrunk
            score_rows.append(
                score_arrays(
                    fold=fold.name,
                    candidate=candidate_name,
                    family="risk_shrinkage",
                    y_true=y_valid,
                    y_pred=valid_shrunk,
                    weight=w_valid,
                )
            )

    if primary_micro_risk is None:
        primary_micro_risk = (np.abs(train_residual_stack), np.abs(valid_residual_stack), float(np.mean(np.abs(train_residual_stack))))
    train_risk, valid_risk, train_risk_mean = primary_micro_risk
    for strength in config.risk_strengths:
        suffix = _format_float(strength)
        train_shrunk = risk_shrink(train_residual_stack, train_risk, train_risk_mean=train_risk_mean, strength=strength)
        valid_shrunk = risk_shrink(valid_residual_stack, valid_risk, train_risk_mean=train_risk_mean, strength=strength)
        binner = MicrostructureRegimeBinner.fit(
            frame=train,
            base_pred=train_shrunk,
            risk=train_risk,
            time_bucket_size=config.time_bucket_size,
            symbol_mod=config.regime_symbol_mod,
        )
        train_groups = binner.transform(train, base_pred=train_shrunk, risk=train_risk)
        valid_groups = binner.transform(valid, base_pred=valid_shrunk, risk=valid_risk)
        calibrator = fit_grouped_scale_calibrator(
            group_codes=train_groups,
            prediction=train_shrunk,
            target=y_train,
            weight=w_train,
            min_rows=config.min_regime_rows,
            prior_strength=config.regime_prior_strength,
        )
        final_name = f"{base_name}__risk_abs_error_s{suffix}_micro_regime_scaled"
        final_pred = calibrator.apply(valid_groups, valid_shrunk)
        if config.write_predictions:
            prediction_columns[final_name] = final_pred
        regime_group_counts[final_name] = len(calibrator.group_scales)
        regime_default_scales[final_name] = calibrator.default_scale
        score_rows.append(
            score_arrays(
                fold=fold.name,
                candidate=final_name,
                family="regime_microstructure",
                y_true=y_valid,
                y_pred=final_pred,
                weight=w_valid,
            )
        )

    audit = {
        "base_name": base_name,
        "residual_rule_count": len(fitted_residual_rules),
        "risk_model_train_mean": train_risk_mean,
        "regime_group_count_max": max(regime_group_counts.values()) if regime_group_counts else 0,
        "regime_default_scale_mean": float(np.mean(list(regime_default_scales.values()))) if regime_default_scales else 0.0,
        "risk_auxiliary_targets": ",".join(risk_targets.keys()),
    }
    return score_rows, auxiliary_score_rows, prediction_columns, audit


def _make_folds(lazy: pl.LazyFrame, config: ExperimentConfig) -> list[DateFold]:
    bounds = lazy.select(pl.col("date_id").min().alias("min_date_id"), pl.col("date_id").max().alias("max_date_id")).collect()
    return make_rolling_folds(
        min_date_id=int(bounds["min_date_id"][0]),
        max_date_id=int(bounds["max_date_id"][0]),
        n_folds=config.n_folds,
        train_window=config.train_window,
        valid_window=config.valid_window,
        gap=config.gap,
    )


def artifact_manifest(config: ExperimentConfig) -> list[dict[str, str]]:
    """Return the concrete model-family artifacts implemented by this pipeline."""

    ridge = _ridge_artifact_name(config)
    pls = _pls_artifact_name(config)
    return [
        {"artifact": ridge, "family": "alpha_latent_global", "role": "Ridge on train-fitted rank encodings"},
        {"artifact": pls, "family": "alpha_latent_recent", "role": "PLS rank compression"},
        {
            "artifact": "gateway_risk_conservative_rls_abs_pred_s100_prediction",
            "family": "alpha_strong_existing",
            "role": "best confirmed strong OOF candidate from gateway RLS adapter",
        },
        {"artifact": f"{ridge}__feature_47_z_residual", "family": "residual_correction", "role": "global residual bridge"},
        {"artifact": f"{pls}__feature_59_z_residual", "family": "residual_correction", "role": "recent residual bridge"},
        {"artifact": f"{ridge}__risk_abs_responder6_ridge_rank_score", "family": "risk_volatility_uncertainty", "role": "absolute responder risk target"},
        {"artifact": f"{ridge}__risk_sq_responder6_ridge_rank_score", "family": "risk_volatility_uncertainty", "role": "squared responder risk target"},
        {"artifact": f"{ridge}__risk_abs_error_ridge_rank_score", "family": "risk_volatility_uncertainty", "role": "absolute model error target"},
        {"artifact": f"{ridge}__risk_abs_error_s0p05_micro_regime_scaled", "family": "regime_microstructure", "role": "observable microstructure scale gate"},
    ]


def _ridge_artifact_name(config: ExperimentConfig) -> str:
    return f"ridge_rank_alpha{_format_float(config.ridge_alpha)}"


def _pls_artifact_name(config: ExperimentConfig) -> str:
    return f"pls_rank_k{int(config.pls_components)}"


def _fit_risk_targets(
    *,
    y_train: np.ndarray,
    train_base: np.ndarray,
    high_error_quantile: float,
    requested: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], float]:
    residual = np.asarray(y_train, dtype=np.float64) - np.asarray(train_base, dtype=np.float64)
    abs_error = np.abs(residual)
    threshold = float(np.quantile(abs_error, high_error_quantile)) if abs_error.size else 0.0
    return _apply_risk_targets(y=y_train, base=train_base, high_error_threshold=threshold, requested=requested), threshold


def _apply_risk_targets(
    *,
    y: np.ndarray,
    base: np.ndarray,
    high_error_threshold: float,
    requested: tuple[str, ...],
) -> dict[str, np.ndarray]:
    residual = np.asarray(y, dtype=np.float64) - np.asarray(base, dtype=np.float64)
    abs_error = np.abs(residual)
    all_targets = {
        "abs_responder6": np.abs(y),
        "sq_responder6": np.square(y),
        "abs_error": abs_error,
        "sq_error": np.square(residual),
        "high_error": (abs_error >= float(high_error_threshold)).astype(np.float64),
    }
    return {name: all_targets[name] for name in requested if name in all_targets}


def _collect_fold_frame(
    lazy: pl.LazyFrame,
    *,
    fold: DateFold,
    split: str,
    config: ExperimentConfig,
    columns: tuple[str, ...],
) -> pl.DataFrame:
    if split == "train":
        filtered = lazy.filter(fold.train_filter())
        max_rows = config.max_train_rows
    elif split == "valid":
        filtered = lazy.filter(fold.valid_filter())
        max_rows = config.max_valid_rows
    else:
        raise ValueError(f"unknown split: {split}")
    if config.sample_stride > 1:
        filtered = filtered.with_row_index("__sample_row").filter((pl.col("__sample_row") % config.sample_stride) == 0).drop("__sample_row")
    if max_rows is not None and max_rows > 0:
        filtered = filtered.limit(max_rows)
    select_columns = [name for name in [*BASE_KEYS, WEIGHT, TARGET, *columns] if name in filtered.collect_schema().names()]
    if "row_id" in filtered.collect_schema().names():
        select_columns = ["row_id", *select_columns]
    return filtered.select([pl.col(name) for name in dict.fromkeys(select_columns)]).collect()


def _validate_config(config: ExperimentConfig) -> None:
    if config.n_folds <= 0:
        raise ValueError("n_folds must be positive")
    if config.train_window <= 0 or config.valid_window <= 0:
        raise ValueError("train_window and valid_window must be positive")
    if config.sample_stride <= 0:
        raise ValueError("sample_stride must be positive")
    if config.rank_bins < 4:
        raise ValueError("rank_bins must be at least 4")
    if config.time_bucket_size <= 0:
        raise ValueError("time_bucket_size must be positive")
    if config.regime_symbol_mod <= 0:
        raise ValueError("regime_symbol_mod must be positive")
    if not 0.0 < config.high_error_quantile < 1.0:
        raise ValueError("high_error_quantile must be between 0 and 1")
    allowed_raw_modes = {
        "batch_rank",
        "batch_demean",
        "batch_zscore",
        "batch_abs_zscore",
        "batch_top_bottom",
        "row_missing_count",
        "row_abs_mean",
        "row_l2_energy",
    }
    unknown_raw_modes = sorted(set(config.raw_preprocess_modes) - allowed_raw_modes)
    if unknown_raw_modes:
        raise ValueError(f"unknown raw_preprocess_modes: {unknown_raw_modes}")
    unsafe_raw_features = [name for name in config.raw_preprocess_features if name == TARGET or name.startswith("responder_")]
    if unsafe_raw_features:
        raise ValueError(f"raw_preprocess_features cannot include target/responder columns: {unsafe_raw_features}")
    if config.raw_preprocess_modes and not config.raw_preprocess_features:
        raise ValueError("raw_preprocess_modes requires raw_preprocess_features")
    allowed_risk_targets = {"abs_responder6", "sq_responder6", "abs_error", "sq_error", "high_error"}
    unknown = [name for name in config.risk_auxiliary_targets if name not in allowed_risk_targets]
    if unknown:
        raise ValueError(f"unknown risk auxiliary targets: {unknown}")


def _target_leakage_check(features: tuple[str, ...]) -> str:
    forbidden = [name for name in features if name == TARGET or (name.startswith("responder_") and not name.endswith("_lag_1"))]
    if forbidden:
        return f"FAILED: forbidden target-like columns in model features: {forbidden}"
    return "passed"


def _fold_causality_check(folds: list[DateFold], *, gap: int) -> str:
    for fold in folds:
        if fold.train_end + gap >= fold.valid_start:
            return f"FAILED: {fold.name} train_end overlaps validation start"
    return "passed"


def _serializable_config(config: ExperimentConfig) -> dict[str, object]:
    raw = asdict(config)
    for key in ("output_dir", "train_parquet_dir"):
        raw[key] = str(raw[key])
    return raw


def _format_float(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")
