"""Run a controlled hidden-signal forensic screen on Jane Street rows."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import polars as pl


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forensics import (  # noqa: E402
    autocorrelation,
    average_rank,
    binned_mutual_information,
    correlation_eigenvalues,
    final_digit_counts,
    hill_tail_index_abs,
    marchenko_pastur_bounds,
    optimal_univariate_fit,
    tail_mean_spread,
    top_periodogram_peaks,
    weighted_centered_corr,
    weighted_mean,
    weighted_zero_mean_r2_arrays,
)
from janestreet.paths import TRAIN_PARQUET_DIR  # noqa: E402


TARGET = "responder_6"
KEY_COLUMNS = ["date_id", "time_id", "symbol_id", "weight", TARGET]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parquet-dir", type=Path, default=TRAIN_PARQUET_DIR)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "reports" / "forensic_screen")
    parser.add_argument("--sample-stride", type=int, default=40)
    parser.add_argument("--max-rows", type=int, default=300_000)
    parser.add_argument("--date-start", type=int, default=None)
    parser.add_argument("--date-end", type=int, default=None)
    parser.add_argument("--valid-day-fraction", type=float, default=0.25)
    parser.add_argument("--permutations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--periodogram-top-k", type=int, default=20)
    args = parser.parse_args()

    validate_args(args)
    frame = load_sample(
        args.train_parquet_dir,
        sample_stride=args.sample_stride,
        max_rows=args.max_rows,
        date_start=args.date_start,
        date_end=args.date_end,
    )
    feature_columns = [column for column in frame.columns if column.startswith("feature_")]
    split_date = temporal_split_date(frame, args.valid_day_fraction)
    train = frame.filter(pl.col("date_id") < split_date)
    valid = frame.filter(pl.col("date_id") >= split_date)
    if train.is_empty() or valid.is_empty():
        raise ValueError("temporal split produced an empty train or validation frame")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    feature_audit = feature_forensics(frame, feature_columns)
    feature_audit.write_csv(args.output_dir / "feature_forensics.csv")

    mp_report, eigenvalues = random_matrix_report(frame, feature_columns)
    (args.output_dir / "marchenko_pastur_report.json").write_text(json.dumps(mp_report, indent=2) + "\n", encoding="utf-8")
    pl.DataFrame({"eigenvalue": eigenvalues}).write_csv(args.output_dir / "correlation_eigenvalues.csv")

    univariate = univariate_transform_screen(train, valid, feature_columns)
    univariate.write_csv(args.output_dir / "univariate_transform_screen.csv")

    permutation_null = permutation_null_screen(
        train,
        valid,
        feature_columns,
        permutations=args.permutations,
        seed=args.seed,
    )
    permutation_null.write_csv(args.output_dir / "permutation_null.csv")

    modulo = modulo_periodicity_screen(train, valid, modulo_values=default_modulo_values())
    modulo.write_csv(args.output_dir / "row_index_modulo_screen.csv")

    spectral = target_spectral_report(frame, top_k=args.periodogram_top_k)
    (args.output_dir / "target_spectral_report.json").write_text(json.dumps(spectral, indent=2) + "\n", encoding="utf-8")

    digit = digit_forensics(frame, feature_columns)
    digit.write_csv(args.output_dir / "digit_forensics.csv")

    summary = summarize_screen(
        frame=frame,
        train=train,
        valid=valid,
        feature_columns=feature_columns,
        split_date=split_date,
        univariate=univariate,
        permutation_null=permutation_null,
        modulo=modulo,
        mp_report=mp_report,
        spectral=spectral,
        args=args,
    )
    (args.output_dir / "screen_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown_summary(summary, args.output_dir / "SUMMARY.md")
    print(json.dumps(summary["headline"], indent=2))
    print(f"Wrote {args.output_dir}")


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_stride <= 0:
        raise ValueError("--sample-stride must be positive")
    if args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")
    if not (0.05 <= args.valid_day_fraction <= 0.50):
        raise ValueError("--valid-day-fraction must be between 0.05 and 0.50")
    if args.permutations < 0:
        raise ValueError("--permutations must be non-negative")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")


def load_sample(
    train_parquet_dir: Path,
    *,
    sample_stride: int,
    max_rows: int,
    date_start: int | None,
    date_end: int | None,
) -> pl.DataFrame:
    feature_columns = [f"feature_{idx:02d}" for idx in range(79)]
    lazy = pl.scan_parquet(str(train_parquet_dir / "**" / "*.parquet")).select(KEY_COLUMNS + feature_columns)
    if date_start is not None:
        lazy = lazy.filter(pl.col("date_id") >= date_start)
    if date_end is not None:
        lazy = lazy.filter(pl.col("date_id") <= date_end)
    lazy = lazy.with_row_index("row_index")
    if sample_stride > 1:
        lazy = lazy.filter((pl.col("row_index") % sample_stride) == 0)
    return lazy.head(max_rows).collect().sort(["date_id", "time_id", "symbol_id"]).with_row_index("sample_index")


def temporal_split_date(frame: pl.DataFrame, valid_day_fraction: float) -> int:
    dates = frame.select("date_id").unique().sort("date_id")["date_id"].to_numpy()
    if dates.size < 4:
        raise ValueError("at least four dates are required for temporal split")
    split_idx = max(1, int(np.floor(dates.size * (1.0 - valid_day_fraction))))
    split_idx = min(split_idx, dates.size - 1)
    return int(dates[split_idx])


def feature_forensics(frame: pl.DataFrame, feature_columns: Sequence[str]) -> pl.DataFrame:
    rows = []
    total_rows = frame.height
    for column in feature_columns:
        values = frame[column].to_numpy()
        finite = np.isfinite(values)
        finite_values = values[finite]
        if finite_values.size:
            mean = float(np.mean(finite_values))
            std = float(np.std(finite_values))
            zero_frac = float(np.mean(finite_values == 0.0))
            unique_count = int(np.unique(finite_values).size)
        else:
            mean = 0.0
            std = 0.0
            zero_frac = 0.0
            unique_count = 0
        rows.append(
            {
                "feature": column,
                "rows": total_rows,
                "missing_fraction": float(1.0 - finite.mean()) if values.size else 1.0,
                "unique_count_sample": unique_count,
                "unique_fraction_sample": unique_count / max(1, finite_values.size),
                "zero_fraction_finite": zero_frac,
                "mean": mean,
                "std": std,
            }
        )
    return pl.DataFrame(rows).sort(["missing_fraction", "unique_fraction_sample"], descending=[True, False])


def random_matrix_report(frame: pl.DataFrame, feature_columns: Sequence[str]) -> tuple[dict[str, object], np.ndarray]:
    values = frame.select(feature_columns).to_numpy()
    eigenvalues = correlation_eigenvalues(values)
    bounds = marchenko_pastur_bounds(n_samples=values.shape[0], n_features=values.shape[1])
    upper_outliers = eigenvalues[eigenvalues > bounds.lambda_plus]
    lower_outliers = eigenvalues[eigenvalues < bounds.lambda_minus]
    report = {
        "bounds": asdict(bounds),
        "eigenvalue_min": float(eigenvalues.min(initial=0.0)),
        "eigenvalue_max": float(eigenvalues.max(initial=0.0)),
        "upper_outlier_count": int(upper_outliers.size),
        "lower_outlier_count": int(lower_outliers.size),
        "upper_outlier_values": [float(value) for value in upper_outliers[-20:]],
        "interpretation": (
            "Eigenvalues above lambda_plus indicate correlation structure beyond iid standardized noise. "
            "In real financial features this can be ordinary latent factor structure, not proof of synthetic generation."
        ),
    }
    return report, eigenvalues


def univariate_transform_screen(train: pl.DataFrame, valid: pl.DataFrame, feature_columns: Sequence[str]) -> pl.DataFrame:
    y_train = train[TARGET].to_numpy()
    y_valid = valid[TARGET].to_numpy()
    w_train = train["weight"].to_numpy()
    w_valid = valid["weight"].to_numpy()
    rows = []
    for column in feature_columns:
        train_x = train[column].to_numpy()
        valid_x = valid[column].to_numpy()
        for transform_name, train_phi, valid_phi in transformed_feature_views(train_x, valid_x):
            alpha = optimal_univariate_fit(train_phi, y_train, w_train)
            train_pred = alpha * train_phi
            valid_pred = alpha * valid_phi
            low_train, high_train, spread_train = tail_mean_spread(train_phi, y_train, w_train)
            low_valid, high_valid, spread_valid = tail_mean_spread(valid_phi, y_valid, w_valid)
            rows.append(
                {
                    "feature": column,
                    "transform": transform_name,
                    "alpha": alpha,
                    "train_r2": weighted_zero_mean_r2_arrays(y_train, train_pred, w_train),
                    "valid_r2": weighted_zero_mean_r2_arrays(y_valid, valid_pred, w_valid),
                    "train_corr": weighted_centered_corr(train_phi, y_train, w_train),
                    "valid_corr": weighted_centered_corr(valid_phi, y_valid, w_valid),
                    "train_tail_low_mean": low_train,
                    "train_tail_high_mean": high_train,
                    "train_tail_spread": spread_train,
                    "valid_tail_low_mean": low_valid,
                    "valid_tail_high_mean": high_valid,
                    "valid_tail_spread": spread_valid,
                    "train_binned_mi": binned_mutual_information(train_phi, y_train),
                    "valid_binned_mi": binned_mutual_information(valid_phi, y_valid),
                }
            )
    return pl.DataFrame(rows).sort(["valid_r2", "valid_corr"], descending=[True, True])


def transformed_feature_views(train_x: np.ndarray, valid_x: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    train_filled, valid_filled = fill_by_train_mean(train_x, valid_x)
    train_mean = float(np.mean(train_filled))
    train_std = float(np.std(train_filled))
    if train_std <= 0.0:
        train_std = 1.0
    train_z = (train_filled - train_mean) / train_std
    valid_z = (valid_filled - train_mean) / train_std
    q05 = float(np.quantile(train_filled, 0.05))
    q95 = float(np.quantile(train_filled, 0.95))
    rank_train = centered_train_rank(train_filled)
    rank_valid = centered_reference_rank(valid_filled, train_filled)
    return [
        ("z", train_z, valid_z),
        ("rank", rank_train, rank_valid),
        ("sign", np.sign(train_filled), np.sign(valid_filled)),
        ("abs_z", np.abs(train_z), np.abs(valid_z)),
        ("square_z", train_z * train_z, valid_z * valid_z),
        ("positive_indicator", (train_filled > 0.0).astype(np.float64), (valid_filled > 0.0).astype(np.float64)),
        ("top05_indicator", (train_filled >= q95).astype(np.float64), (valid_filled >= q95).astype(np.float64)),
        ("bottom05_indicator", (train_filled <= q05).astype(np.float64), (valid_filled <= q05).astype(np.float64)),
        (
            "signed_tail05",
            (train_filled >= q95).astype(np.float64) - (train_filled <= q05).astype(np.float64),
            (valid_filled >= q95).astype(np.float64) - (valid_filled <= q05).astype(np.float64),
        ),
    ]


def fill_by_train_mean(train_x: np.ndarray, valid_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train_x, dtype=np.float64)
    valid = np.asarray(valid_x, dtype=np.float64)
    finite_train = np.isfinite(train)
    fill_value = float(np.mean(train[finite_train])) if finite_train.any() else 0.0
    return np.where(np.isfinite(train), train, fill_value), np.where(np.isfinite(valid), valid, fill_value)


def centered_train_rank(values: np.ndarray) -> np.ndarray:
    if values.size <= 1:
        return np.zeros(values.size, dtype=np.float64)
    return average_rank(values) / float(values.size - 1) - 0.5


def centered_reference_rank(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if reference.size <= 1:
        return np.zeros(values.size, dtype=np.float64)
    sorted_reference = np.sort(reference)
    return np.searchsorted(sorted_reference, values, side="right") / float(reference.size) - 0.5


def permutation_null_screen(
    train: pl.DataFrame,
    valid: pl.DataFrame,
    feature_columns: Sequence[str],
    *,
    permutations: int,
    seed: int,
) -> pl.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    train_y = train[TARGET].to_numpy()
    valid_y = valid[TARGET].to_numpy()
    for permutation_id in range(permutations):
        permuted_train = rng.permutation(train_y)
        permuted_valid = rng.permutation(valid_y)
        best = best_univariate_score(train, valid, feature_columns, train_y=permuted_train, valid_y=permuted_valid)
        rows.append({"permutation_id": permutation_id, **best})
    return pl.DataFrame(rows) if rows else pl.DataFrame({"permutation_id": [], "best_valid_r2": []})


def best_univariate_score(
    train: pl.DataFrame,
    valid: pl.DataFrame,
    feature_columns: Sequence[str],
    *,
    train_y: np.ndarray,
    valid_y: np.ndarray,
) -> dict[str, float | str]:
    w_train = train["weight"].to_numpy()
    w_valid = valid["weight"].to_numpy()
    best: dict[str, float | str] = {"best_feature": "", "best_transform": "", "best_valid_r2": -np.inf}
    for column in feature_columns:
        train_x = train[column].to_numpy()
        valid_x = valid[column].to_numpy()
        for transform_name, train_phi, valid_phi in transformed_feature_views(train_x, valid_x):
            alpha = optimal_univariate_fit(train_phi, train_y, w_train)
            valid_r2 = weighted_zero_mean_r2_arrays(valid_y, alpha * valid_phi, w_valid)
            if valid_r2 > float(best["best_valid_r2"]):
                best = {"best_feature": column, "best_transform": transform_name, "best_valid_r2": valid_r2}
    return best


def default_modulo_values() -> tuple[int, ...]:
    return tuple(list(range(2, 65)) + [89, 97, 127])


def modulo_periodicity_screen(train: pl.DataFrame, valid: pl.DataFrame, modulo_values: Iterable[int], *, index_column: str = "sample_index") -> pl.DataFrame:
    rows = []
    if index_column not in train.columns or index_column not in valid.columns:
        index_column = "row_index"
    train_row = train[index_column].to_numpy()
    valid_row = valid[index_column].to_numpy()
    train_y = train[TARGET].to_numpy()
    valid_y = valid[TARGET].to_numpy()
    train_w = train["weight"].to_numpy()
    valid_w = valid["weight"].to_numpy()
    global_mean = weighted_mean(train_y, train_w)
    for modulo in modulo_values:
        train_code = train_row % modulo
        valid_code = valid_row % modulo
        means = np.full(modulo, global_mean, dtype=np.float64)
        counts = np.zeros(modulo, dtype=np.int64)
        for code in range(modulo):
            mask = train_code == code
            counts[code] = int(mask.sum())
            if counts[code] > 0:
                means[code] = weighted_mean(train_y[mask], train_w[mask])
        pred = means[valid_code]
        rows.append(
            {
                "modulo": modulo,
                "index_column": index_column,
                "min_train_bucket_rows": int(counts.min()) if counts.size else 0,
                "max_abs_bucket_mean": float(np.max(np.abs(means))) if means.size else 0.0,
                "valid_r2": weighted_zero_mean_r2_arrays(valid_y, pred, valid_w),
            }
        )
    return pl.DataFrame(rows).sort("valid_r2", descending=True)


def target_spectral_report(frame: pl.DataFrame, *, top_k: int) -> dict[str, object]:
    ordered = frame.sort(["date_id", "time_id", "symbol_id"])
    y = ordered[TARGET].to_numpy()
    y2 = y * y
    hill_k = max(10, min(5000, int(np.sqrt(y.size))))
    return {
        "rows": int(y.size),
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y)),
        "target_hill_tail_index_abs": hill_tail_index_abs(y, hill_k),
        "acf_target": autocorrelation(y, [1, 2, 5, 10, 20, 50, 100]),
        "acf_target_squared": autocorrelation(y2, [1, 2, 5, 10, 20, 50, 100]),
        "top_periodogram_peaks": top_periodogram_peaks(y, top_k=top_k),
    }


def digit_forensics(frame: pl.DataFrame, feature_columns: Sequence[str]) -> pl.DataFrame:
    rows = []
    expected = None
    for column in feature_columns:
        counts = final_digit_counts(frame[column].to_numpy(), decimals=6)
        if expected is None:
            expected = counts.sum() / 10.0
        total = int(counts.sum())
        chi_square = float(np.sum((counts - total / 10.0) ** 2 / max(total / 10.0, 1.0))) if total else 0.0
        row = {
            "feature": column,
            "total_digits": total,
            "chi_square_vs_uniform": chi_square,
            "max_digit_share": float(counts.max(initial=0) / total) if total else 0.0,
        }
        row.update({f"digit_{idx}": int(value) for idx, value in enumerate(counts)})
        rows.append(row)
    return pl.DataFrame(rows).sort("chi_square_vs_uniform", descending=True)


def summarize_screen(
    *,
    frame: pl.DataFrame,
    train: pl.DataFrame,
    valid: pl.DataFrame,
    feature_columns: Sequence[str],
    split_date: int,
    univariate: pl.DataFrame,
    permutation_null: pl.DataFrame,
    modulo: pl.DataFrame,
    mp_report: dict[str, object],
    spectral: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    best_uni = univariate.row(0, named=True)
    best_modulo = modulo.row(0, named=True)
    null_values = permutation_null["best_valid_r2"].to_numpy() if "best_valid_r2" in permutation_null.columns else np.array([])
    null_p90 = float(np.quantile(null_values, 0.90)) if null_values.size else None
    null_p95 = float(np.quantile(null_values, 0.95)) if null_values.size else None
    null_max = float(np.max(null_values)) if null_values.size else None
    return {
        "headline": {
            "rows": frame.height,
            "train_rows": train.height,
            "valid_rows": valid.height,
            "date_min": int(frame["date_id"].min()),
            "date_max": int(frame["date_id"].max()),
            "split_date": split_date,
            "feature_count": len(feature_columns),
            "best_univariate_valid_r2": float(best_uni["valid_r2"]),
            "best_univariate": f"{best_uni['feature']}::{best_uni['transform']}",
            "best_modulo_valid_r2": float(best_modulo["valid_r2"]),
            "best_modulo": int(best_modulo["modulo"]),
            "permutation_best_valid_r2_p95": null_p95,
            "permutation_best_valid_r2_max": null_max,
            "mp_upper_outlier_count": mp_report["upper_outlier_count"],
        },
        "args": vars(args) | {"train_parquet_dir": str(args.train_parquet_dir), "output_dir": str(args.output_dir)},
        "marchenko_pastur": mp_report,
        "spectral": spectral,
        "permutation_null": {
            "p90": null_p90,
            "p95": null_p95,
            "max": null_max,
            "permutations": int(null_values.size),
            "interpretation": "Real discovery must exceed the null created by repeating the same search over permuted targets.",
        },
        "methodological_caveat": (
            "This is an exploratory screen on a sampled subset. It can falsify easy hidden-pattern hypotheses, "
            "but it cannot promote an alpha without a frozen out-of-sample validation."
        ),
    }


def write_markdown_summary(summary: dict[str, object], path: Path) -> None:
    headline = summary["headline"]
    null = summary["permutation_null"]
    content = f"""# Hidden Signal Forensic Screen

## Headline

- Rows: `{headline["rows"]}` (`{headline["train_rows"]}` train, `{headline["valid_rows"]}` validation).
- Dates: `{headline["date_min"]}` to `{headline["date_max"]}`; split date `{headline["split_date"]}`.
- Best univariate rule: `{headline["best_univariate"]}` with validation R2 `{headline["best_univariate_valid_r2"]}`.
- Best row-index modulo rule: `mod {headline["best_modulo"]}` with validation R2 `{headline["best_modulo_valid_r2"]}`.
- Permutation null p95/max: `{null["p95"]}` / `{null["max"]}`.
- Marchenko-Pastur upper outliers: `{headline["mp_upper_outlier_count"]}`.

## Interpretation Boundary

This screen is exploratory. A positive simple rule is only interesting if it is
materially above the permutation-search null and remains stable in a frozen
temporal validation. A low score is not evidence that no market signal exists;
it only weakens the specific hidden-pattern families tested here.
"""
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
