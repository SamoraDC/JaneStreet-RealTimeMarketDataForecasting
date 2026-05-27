"""Deep hidden-signal forensic suite with frozen rules, nulls, interactions and residual mining."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import polars as pl
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_forensic_screen as screen  # noqa: E402
from forensics import (  # noqa: E402
    optimal_univariate_fit,
    weighted_centered_corr,
    weighted_zero_mean_r2_arrays,
)
from janestreet.paths import TRAIN_PARQUET_DIR  # noqa: E402


TARGET = "responder_6"
FROZEN_RULES = (("feature_04", "z"), ("feature_16", "square_z"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parquet-dir", type=Path, default=TRAIN_PARQUET_DIR)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "reports" / "deep_forensic_suite")
    parser.add_argument("--sample-stride", type=int, default=80)
    parser.add_argument("--max-rows", type=int, default=250_000)
    parser.add_argument("--date-start", type=int, default=900)
    parser.add_argument("--date-end", type=int, default=None)
    parser.add_argument("--valid-day-fraction", type=float, default=0.25)
    parser.add_argument("--top-features", type=int, default=14)
    parser.add_argument("--null-runs", type=int, default=8)
    parser.add_argument("--iaaft-runs", type=int, default=4)
    parser.add_argument("--iaaft-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()

    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = screen.load_sample(
        args.train_parquet_dir,
        sample_stride=args.sample_stride,
        max_rows=args.max_rows,
        date_start=args.date_start,
        date_end=args.date_end,
    )
    feature_columns = [column for column in frame.columns if column.startswith("feature_")]
    split_date = screen.temporal_split_date(frame, args.valid_day_fraction)
    train = frame.filter(pl.col("date_id") < split_date)
    valid = frame.filter(pl.col("date_id") >= split_date)
    rng = np.random.default_rng(args.seed)

    univariate = screen.univariate_transform_screen(train, valid, feature_columns)
    univariate.write_csv(args.output_dir / "deep_univariate_screen.csv")
    top_feature_names = select_top_features(univariate, args.top_features)

    frozen = frozen_rule_walk_forward(frame, FROZEN_RULES, min_train_windows=1)
    frozen.write_csv(args.output_dir / "frozen_rule_walk_forward.csv")

    frozen_nulls = frozen_rule_nulls(train, valid, FROZEN_RULES, null_runs=args.null_runs, rng=rng)
    frozen_nulls.write_csv(args.output_dir / "frozen_rule_nulls.csv")

    interactions = interaction_screen(train, valid, top_feature_names)
    interactions.write_csv(args.output_dir / "interaction_screen.csv")

    interaction_nulls = interaction_search_nulls(train, valid, top_feature_names, null_runs=args.null_runs, rng=rng)
    interaction_nulls.write_csv(args.output_dir / "interaction_search_nulls.csv")

    cross_sectional = cross_sectional_screen(train, valid, top_feature_names)
    cross_sectional.write_csv(args.output_dir / "cross_sectional_screen.csv")

    latent_models, latent_predictions = latent_factor_models(train, valid, feature_columns)
    latent_models.write_csv(args.output_dir / "latent_factor_models.csv")

    residual = residual_mining(train, valid, feature_columns, latent_predictions)
    residual.write_csv(args.output_dir / "residual_mining.csv")

    iaaft = iaaft_surrogate_search_nulls(
        train,
        valid,
        feature_columns,
        runs=args.iaaft_runs,
        iterations=args.iaaft_iterations,
        rng=rng,
    )
    iaaft.write_csv(args.output_dir / "iaaft_surrogate_search_nulls.csv")

    summary = summarize_deep_suite(
        frame=frame,
        train=train,
        valid=valid,
        split_date=split_date,
        top_feature_names=top_feature_names,
        univariate=univariate,
        frozen=frozen,
        frozen_nulls=frozen_nulls,
        interactions=interactions,
        interaction_nulls=interaction_nulls,
        cross_sectional=cross_sectional,
        latent_models=latent_models,
        residual=residual,
        iaaft=iaaft,
        args=args,
    )
    (args.output_dir / "deep_suite_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_deep_markdown(summary, args.output_dir / "DEEP_SYNTHESIS.md")
    print(json.dumps(summary["headline"], indent=2))
    print(f"Wrote {args.output_dir}")


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_stride <= 0:
        raise ValueError("--sample-stride must be positive")
    if args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")
    if args.top_features < 2:
        raise ValueError("--top-features must be at least 2")
    if args.null_runs < 1:
        raise ValueError("--null-runs must be positive")
    if args.iaaft_runs < 0:
        raise ValueError("--iaaft-runs must be non-negative")
    if args.iaaft_iterations < 1:
        raise ValueError("--iaaft-iterations must be positive")


def select_top_features(univariate: pl.DataFrame, top_n: int) -> tuple[str, ...]:
    selected: list[str] = []
    for row in univariate.iter_rows(named=True):
        feature = str(row["feature"])
        if feature not in selected:
            selected.append(feature)
        if len(selected) >= top_n:
            break
    return tuple(selected)


def frozen_rule_walk_forward(frame: pl.DataFrame, rules: Sequence[tuple[str, str]], *, min_train_windows: int = 1, windows: int = 5) -> pl.DataFrame:
    dates = frame.select("date_id").unique().sort("date_id")["date_id"].to_numpy()
    if dates.size < windows + 1:
        raise ValueError("not enough dates for frozen walk-forward")
    date_blocks = np.array_split(dates, windows)
    rows = []
    for idx in range(min_train_windows, len(date_blocks)):
        train_dates = np.concatenate(date_blocks[:idx])
        valid_dates = date_blocks[idx]
        train = frame.filter(pl.col("date_id").is_in(train_dates.tolist()))
        valid = frame.filter(pl.col("date_id").is_in(valid_dates.tolist()))
        for feature, transform in rules:
            train_phi, valid_phi = materialize_transform(train, valid, feature, transform)
            alpha = optimal_univariate_fit(train_phi, train[TARGET].to_numpy(), train["weight"].to_numpy())
            valid_pred = alpha * valid_phi
            rows.append(
                {
                    "rule": f"{feature}::{transform}",
                    "window_index": idx,
                    "train_start": int(train_dates.min()),
                    "train_end": int(train_dates.max()),
                    "valid_start": int(valid_dates.min()),
                    "valid_end": int(valid_dates.max()),
                    "train_rows": train.height,
                    "valid_rows": valid.height,
                    "alpha": alpha,
                    "valid_r2": weighted_zero_mean_r2_arrays(valid[TARGET].to_numpy(), valid_pred, valid["weight"].to_numpy()),
                    "valid_corr": weighted_centered_corr(valid_phi, valid[TARGET].to_numpy(), valid["weight"].to_numpy()),
                }
            )
    return pl.DataFrame(rows).sort(["rule", "window_index"])


def materialize_transform(train: pl.DataFrame, valid: pl.DataFrame, feature: str, transform: str) -> tuple[np.ndarray, np.ndarray]:
    for name, train_phi, valid_phi in screen.transformed_feature_views(train[feature].to_numpy(), valid[feature].to_numpy()):
        if name == transform:
            return train_phi, valid_phi
    raise ValueError(f"unknown transform {transform!r}")


def frozen_rule_nulls(
    train: pl.DataFrame,
    valid: pl.DataFrame,
    rules: Sequence[tuple[str, str]],
    *,
    null_runs: int,
    rng: np.random.Generator,
) -> pl.DataFrame:
    rows = []
    for feature, transform in rules:
        train_phi, valid_phi = materialize_transform(train, valid, feature, transform)
        for null_kind in ("iid", "block", "date", "circular"):
            for run in range(null_runs):
                y_train = null_target(train, kind=null_kind, rng=rng)
                y_valid = null_target(valid, kind=null_kind, rng=rng)
                alpha = optimal_univariate_fit(train_phi, y_train, train["weight"].to_numpy())
                r2 = weighted_zero_mean_r2_arrays(y_valid, alpha * valid_phi, valid["weight"].to_numpy())
                rows.append({"rule": f"{feature}::{transform}", "null_kind": null_kind, "run": run, "valid_r2": r2})
    return pl.DataFrame(rows)


def null_target(frame: pl.DataFrame, *, kind: str, rng: np.random.Generator, block_size: int = 2048) -> np.ndarray:
    y = frame[TARGET].to_numpy()
    if kind == "iid":
        return rng.permutation(y)
    if kind == "circular":
        shift = int(rng.integers(1, max(2, y.size)))
        return np.roll(y, shift)
    if kind == "block":
        blocks = [y[start : start + block_size] for start in range(0, y.size, block_size)]
        order = rng.permutation(len(blocks))
        return np.concatenate([blocks[int(idx)] for idx in order])
    if kind == "date":
        parts = [part[TARGET].to_numpy() for part in frame.partition_by("date_id", maintain_order=True)]
        order = rng.permutation(len(parts))
        return np.concatenate([parts[int(idx)] for idx in order])
    raise ValueError(f"unknown null kind: {kind}")


def interaction_screen(train: pl.DataFrame, valid: pl.DataFrame, features: Sequence[str]) -> pl.DataFrame:
    prepared = prepare_base_views(train, valid, features)
    rows = []
    y_train = train[TARGET].to_numpy()
    y_valid = valid[TARGET].to_numpy()
    w_train = train["weight"].to_numpy()
    w_valid = valid["weight"].to_numpy()
    for i, left in enumerate(features):
        for right in features[i + 1 :]:
            for name, train_phi, valid_phi in interaction_views(prepared[left], prepared[right]):
                alpha = optimal_univariate_fit(train_phi, y_train, w_train)
                rows.append(
                    {
                        "left": left,
                        "right": right,
                        "interaction": name,
                        "alpha": alpha,
                        "valid_r2": weighted_zero_mean_r2_arrays(y_valid, alpha * valid_phi, w_valid),
                        "valid_corr": weighted_centered_corr(valid_phi, y_valid, w_valid),
                    }
                )
    return pl.DataFrame(rows).sort(["valid_r2", "valid_corr"], descending=[True, True])


def interaction_search_nulls(
    train: pl.DataFrame,
    valid: pl.DataFrame,
    features: Sequence[str],
    *,
    null_runs: int,
    rng: np.random.Generator,
) -> pl.DataFrame:
    prepared = prepare_base_views(train, valid, features)
    rows = []
    w_train = train["weight"].to_numpy()
    w_valid = valid["weight"].to_numpy()
    for null_kind in ("iid", "block", "date", "circular"):
        for run in range(null_runs):
            y_train = null_target(train, kind=null_kind, rng=rng)
            y_valid = null_target(valid, kind=null_kind, rng=rng)
            best = {"valid_r2": -np.inf, "left": "", "right": "", "interaction": ""}
            for i, left in enumerate(features):
                for right in features[i + 1 :]:
                    for name, train_phi, valid_phi in interaction_views(prepared[left], prepared[right]):
                        alpha = optimal_univariate_fit(train_phi, y_train, w_train)
                        r2 = weighted_zero_mean_r2_arrays(y_valid, alpha * valid_phi, w_valid)
                        if r2 > float(best["valid_r2"]):
                            best = {"valid_r2": r2, "left": left, "right": right, "interaction": name}
            rows.append({"null_kind": null_kind, "run": run, **best})
    return pl.DataFrame(rows)


def prepare_base_views(train: pl.DataFrame, valid: pl.DataFrame, features: Sequence[str]) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for feature in features:
        train_filled, valid_filled = screen.fill_by_train_mean(train[feature].to_numpy(), valid[feature].to_numpy())
        train_mean = float(np.mean(train_filled))
        train_std = float(np.std(train_filled)) or 1.0
        train_z = (train_filled - train_mean) / train_std
        valid_z = (valid_filled - train_mean) / train_std
        result[feature] = {
            "train_z": train_z,
            "valid_z": valid_z,
            "train_rank": screen.centered_train_rank(train_filled),
            "valid_rank": screen.centered_reference_rank(valid_filled, train_filled),
            "train_sign": np.sign(train_filled),
            "valid_sign": np.sign(valid_filled),
        }
    return result


def interaction_views(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> Iterable[tuple[str, np.ndarray, np.ndarray]]:
    yield "sign_product", left["train_sign"] * right["train_sign"], left["valid_sign"] * right["valid_sign"]
    yield "rank_diff", left["train_rank"] - right["train_rank"], left["valid_rank"] - right["valid_rank"]
    yield "z_diff", left["train_z"] - right["train_z"], left["valid_z"] - right["valid_z"]
    yield "abs_z_diff", np.abs(left["train_z"] - right["train_z"]), np.abs(left["valid_z"] - right["valid_z"])
    yield "z_product", left["train_z"] * right["train_z"], left["valid_z"] * right["valid_z"]
    yield "rank_product", left["train_rank"] * right["train_rank"], left["valid_rank"] * right["valid_rank"]


def cross_sectional_screen(train: pl.DataFrame, valid: pl.DataFrame, features: Sequence[str]) -> pl.DataFrame:
    rows = []
    for group_name, group_columns in {
        "date": ["date_id"],
        "date_time": ["date_id", "time_id"],
    }.items():
        train_cs = add_cross_sectional_columns(train, features, group_columns)
        valid_cs = add_cross_sectional_columns(valid, features, group_columns)
        for feature in features:
            for suffix in ("cs_z", "cs_rank"):
                column = f"{feature}_{suffix}_{group_name}"
                alpha = optimal_univariate_fit(train_cs[column].to_numpy(), train_cs[TARGET].to_numpy(), train_cs["weight"].to_numpy())
                pred = alpha * valid_cs[column].to_numpy()
                rows.append(
                    {
                        "feature": feature,
                        "group": group_name,
                        "transform": suffix,
                        "valid_r2": weighted_zero_mean_r2_arrays(valid_cs[TARGET].to_numpy(), pred, valid_cs["weight"].to_numpy()),
                        "valid_corr": weighted_centered_corr(valid_cs[column].to_numpy(), valid_cs[TARGET].to_numpy(), valid_cs["weight"].to_numpy()),
                    }
                )
    return pl.DataFrame(rows).sort(["valid_r2", "valid_corr"], descending=[True, True])


def add_cross_sectional_columns(frame: pl.DataFrame, features: Sequence[str], group_columns: Sequence[str]) -> pl.DataFrame:
    expressions = []
    group_name = "date_time" if len(group_columns) > 1 else "date"
    for feature in features:
        mean = pl.col(feature).mean().over(group_columns)
        std = pl.col(feature).std().over(group_columns)
        count = pl.len().over(group_columns)
        rank = pl.col(feature).rank(method="average").over(group_columns)
        expressions.extend(
            [
                ((pl.col(feature) - mean) / pl.when(std > 0).then(std).otherwise(1.0)).fill_null(0.0).alias(f"{feature}_cs_z_{group_name}"),
                ((rank - 1.0) / pl.when(count > 1).then(count - 1).otherwise(1) - 0.5).fill_null(0.0).alias(f"{feature}_cs_rank_{group_name}"),
            ]
        )
    return frame.with_columns(expressions)


def latent_factor_models(train: pl.DataFrame, valid: pl.DataFrame, features: Sequence[str]) -> tuple[pl.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    x_train, x_valid = rank_matrix(train, valid, features)
    y_train = train[TARGET].to_numpy()
    y_valid = valid[TARGET].to_numpy()
    w_train = train["weight"].to_numpy()
    w_valid = valid["weight"].to_numpy()
    rows = []
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for alpha in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        model = Ridge(alpha=alpha)
        model.fit(x_train, y_train, sample_weight=w_train)
        train_pred = model.predict(x_train)
        valid_pred = model.predict(x_valid)
        name = f"ridge_rank_alpha{alpha:g}"
        predictions[name] = (train_pred, valid_pred)
        rows.append({"model": name, "family": "ridge_rank", "valid_r2": weighted_zero_mean_r2_arrays(y_valid, valid_pred, w_valid)})
    for components in (1, 2, 4, 8):
        if components > min(x_train.shape):
            continue
        model = PLSRegression(n_components=components, scale=False)
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train).reshape(-1)
        valid_pred = model.predict(x_valid).reshape(-1)
        name = f"pls_rank_k{components}"
        predictions[name] = (train_pred, valid_pred)
        rows.append({"model": name, "family": "pls_rank_unweighted_fit", "valid_r2": weighted_zero_mean_r2_arrays(y_valid, valid_pred, w_valid)})
    return pl.DataFrame(rows).sort("valid_r2", descending=True), predictions


def rank_matrix(train: pl.DataFrame, valid: pl.DataFrame, features: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    train_columns = []
    valid_columns = []
    for feature in features:
        train_x, valid_x = screen.fill_by_train_mean(train[feature].to_numpy(), valid[feature].to_numpy())
        train_columns.append(screen.centered_train_rank(train_x))
        valid_columns.append(screen.centered_reference_rank(valid_x, train_x))
    return np.column_stack(train_columns).astype(np.float64), np.column_stack(valid_columns).astype(np.float64)


def residual_mining(train: pl.DataFrame, valid: pl.DataFrame, features: Sequence[str], baseline_predictions: dict[str, tuple[np.ndarray, np.ndarray]]) -> pl.DataFrame:
    if not baseline_predictions:
        return pl.DataFrame()
    model_name, (train_baseline, valid_baseline) = max(
        baseline_predictions.items(),
        key=lambda item: weighted_zero_mean_r2_arrays(valid[TARGET].to_numpy(), item[1][1], valid["weight"].to_numpy()),
    )
    y_train = train[TARGET].to_numpy()
    w_train = train["weight"].to_numpy()
    residual_train_y = y_train - train_baseline
    y_valid = valid[TARGET].to_numpy()
    w_valid = valid["weight"].to_numpy()
    rows = []
    for feature in features:
        train_x = train[feature].to_numpy()
        valid_x = valid[feature].to_numpy()
        for transform_name, train_phi, valid_phi in screen.transformed_feature_views(train_x, valid_x):
            alpha = optimal_univariate_fit(train_phi, residual_train_y, w_train)
            final_pred = valid_baseline + alpha * valid_phi
            rows.append(
                {
                    "baseline_model": model_name,
                    "residual_feature": feature,
                    "residual_transform": transform_name,
                    "alpha": alpha,
                    "baseline_valid_r2": weighted_zero_mean_r2_arrays(y_valid, valid_baseline, w_valid),
                    "final_valid_r2": weighted_zero_mean_r2_arrays(y_valid, final_pred, w_valid),
                }
            )
    return pl.DataFrame(rows).with_columns((pl.col("final_valid_r2") - pl.col("baseline_valid_r2")).alias("delta_r2")).sort("final_valid_r2", descending=True)


def iaaft_surrogate_search_nulls(
    train: pl.DataFrame,
    valid: pl.DataFrame,
    features: Sequence[str],
    *,
    runs: int,
    iterations: int,
    rng: np.random.Generator,
) -> pl.DataFrame:
    if runs <= 0:
        return pl.DataFrame({"run": [], "best_valid_r2": []})
    y_full = np.concatenate([train[TARGET].to_numpy(), valid[TARGET].to_numpy()])
    rows = []
    for run in range(runs):
        surrogate = iaaft(y_full, iterations=iterations, rng=rng)
        train_y = surrogate[: train.height]
        valid_y = surrogate[train.height :]
        best = screen.best_univariate_score(train, valid, features, train_y=train_y, valid_y=valid_y)
        rows.append({"run": run, "iterations": iterations, **best})
    return pl.DataFrame(rows).sort("best_valid_r2", descending=True)


def iaaft(values: np.ndarray, *, iterations: int, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    sorted_x = np.sort(x)
    target_amplitudes = np.abs(np.fft.rfft(x - np.mean(x)))
    y = rng.permutation(x)
    for _ in range(iterations):
        spectrum = np.fft.rfft(y - np.mean(y))
        phases = np.exp(1j * np.angle(spectrum))
        y = np.fft.irfft(target_amplitudes * phases, n=x.size)
        order = np.argsort(y, kind="mergesort")
        adjusted = np.empty_like(y)
        adjusted[order] = sorted_x
        y = adjusted
    return y


def summarize_deep_suite(
    *,
    frame: pl.DataFrame,
    train: pl.DataFrame,
    valid: pl.DataFrame,
    split_date: int,
    top_feature_names: Sequence[str],
    univariate: pl.DataFrame,
    frozen: pl.DataFrame,
    frozen_nulls: pl.DataFrame,
    interactions: pl.DataFrame,
    interaction_nulls: pl.DataFrame,
    cross_sectional: pl.DataFrame,
    latent_models: pl.DataFrame,
    residual: pl.DataFrame,
    iaaft: pl.DataFrame,
    args: argparse.Namespace,
) -> dict[str, object]:
    best_uni = univariate.row(0, named=True)
    best_interaction = interactions.row(0, named=True)
    best_cross = cross_sectional.row(0, named=True)
    best_latent = latent_models.row(0, named=True)
    best_residual = residual.row(0, named=True) if not residual.is_empty() else {}
    return {
        "headline": {
            "rows": frame.height,
            "train_rows": train.height,
            "valid_rows": valid.height,
            "date_min": int(frame["date_id"].min()),
            "date_max": int(frame["date_id"].max()),
            "split_date": split_date,
            "top_features": list(top_feature_names),
            "best_univariate": f"{best_uni['feature']}::{best_uni['transform']}",
            "best_univariate_valid_r2": float(best_uni["valid_r2"]),
            "best_interaction": f"{best_interaction['left']}:{best_interaction['right']}::{best_interaction['interaction']}",
            "best_interaction_valid_r2": float(best_interaction["valid_r2"]),
            "best_cross_sectional": f"{best_cross['feature']}::{best_cross['transform']}::{best_cross['group']}",
            "best_cross_sectional_valid_r2": float(best_cross["valid_r2"]),
            "best_latent_model": str(best_latent["model"]),
            "best_latent_valid_r2": float(best_latent["valid_r2"]),
            "best_residual_final_r2": float(best_residual.get("final_valid_r2", 0.0)),
            "best_residual_delta_r2": float(best_residual.get("delta_r2", 0.0)),
            "iaaft_best_null_r2": float(iaaft["best_valid_r2"].max()) if not iaaft.is_empty() else None,
            "interaction_null_max_r2": float(interaction_nulls["valid_r2"].max()) if not interaction_nulls.is_empty() else None,
        },
        "args": vars(args) | {"train_parquet_dir": str(args.train_parquet_dir), "output_dir": str(args.output_dir)},
        "audit": {
            "leakage_status": "All model-selection screens fit parameters on train and report validation R2; cross-sectional transforms use same-row feature groups only and are diagnostic.",
            "selection_bias_status": "Univariate/interactions are adaptive screens. Promotion requires frozen validation on separate windows.",
            "nulls": ["iid", "block", "date", "circular", "iaaft"],
        },
    }


def write_deep_markdown(summary: dict[str, object], path: Path) -> None:
    h = summary["headline"]
    content = f"""# Deep Hidden Signal Forensic Suite

## Headline

- Rows: `{h["rows"]}` (`{h["train_rows"]}` train, `{h["valid_rows"]}` validation).
- Dates: `{h["date_min"]}` to `{h["date_max"]}`; split date `{h["split_date"]}`.
- Best univariate: `{h["best_univariate"]}` with R2 `{h["best_univariate_valid_r2"]}`.
- Best interaction: `{h["best_interaction"]}` with R2 `{h["best_interaction_valid_r2"]}`.
- Best cross-sectional transform: `{h["best_cross_sectional"]}` with R2 `{h["best_cross_sectional_valid_r2"]}`.
- Best latent model: `{h["best_latent_model"]}` with R2 `{h["best_latent_valid_r2"]}`.
- Best residual correction final R2: `{h["best_residual_final_r2"]}`; delta `{h["best_residual_delta_r2"]}`.
- IAAFT best null R2: `{h["iaaft_best_null_r2"]}`.
- Interaction-search null max R2: `{h["interaction_null_max_r2"]}`.

## Audit Boundary

This suite is exhaustive for the finite hypothesis families implemented here:
frozen simple rules, strong nulls, low-order interactions, cross-sectional
normalization, rank latent models, residual mining and IAAFT surrogates. It is
not proof that no other hidden signal exists.

Any candidate found here is still adaptive unless it is explicitly frozen and
validated on a later temporal window.
"""
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
