"""Build per-fold averaged validation predictions from compatible prediction dirs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


TARGET = "responder_6"
KEYS = ["fold", "date_id", "time_id", "symbol_id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--prediction-column", default="tabm_prediction")
    parser.add_argument("--output-column", default="tabm_prediction")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _validate_args(args)

    prediction_dir = args.output_dir / "validation_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    fold_files = _fold_files(args.prediction_dirs)
    rows = []
    for fold_name, paths in fold_files.items():
        averaged = _build_fold_average(
            paths,
            prediction_column=args.prediction_column,
            output_column=args.output_column,
        )
        output_path = prediction_dir / fold_name
        averaged.write_parquet(output_path)
        rows.append({"fold_file": fold_name, **_score_frame(averaged, args.output_column)})

    summary = pl.DataFrame(rows).sort("fold_file")
    summary.write_csv(args.output_dir / "prediction_average_by_fold.csv")
    report = {
        "experiment": "prediction_average",
        "prediction_dirs": [str(path) for path in args.prediction_dirs],
        "prediction_column": args.prediction_column,
        "output_column": args.output_column,
        "folds": len(rows),
        "rows": int(summary["rows"].sum()),
        "global_r2": _global_score_from_fold_rows(summary),
    }
    (args.output_dir / "prediction_average_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(summary)
    print(json.dumps(report, indent=2))


def _validate_args(args: argparse.Namespace) -> None:
    if len(args.prediction_dirs) < 2:
        raise ValueError("at least two prediction dirs are required")
    for path in args.prediction_dirs:
        if not path.exists():
            raise FileNotFoundError(path)


def _fold_files(prediction_dirs: list[Path]) -> dict[str, list[Path]]:
    file_sets = []
    for directory in prediction_dirs:
        files = {path.name: path for path in sorted(directory.glob("*.parquet"))}
        if not files:
            raise FileNotFoundError(f"no parquet files in {directory}")
        file_sets.append(files)
    common_names = set(file_sets[0])
    for files in file_sets[1:]:
        common_names &= set(files)
    if not common_names:
        raise ValueError("prediction dirs do not share fold parquet names")
    if any(set(files) != common_names for files in file_sets):
        missing = [sorted(common_names ^ set(files)) for files in file_sets]
        raise ValueError(f"prediction dirs must contain the same fold files; mismatches={missing}")
    return {name: [files[name] for files in file_sets] for name in sorted(common_names)}


def _build_fold_average(
    paths: list[Path],
    *,
    prediction_column: str,
    output_column: str,
) -> pl.DataFrame:
    base_alias = "_prediction_00"
    lazy = pl.scan_parquet(paths[0]).select(KEYS + ["weight", TARGET, prediction_column]).rename({prediction_column: base_alias})
    aliases = [base_alias]
    for idx, path in enumerate(paths[1:], start=1):
        alias = f"_prediction_{idx:02d}"
        aliases.append(alias)
        other = pl.scan_parquet(path).select(KEYS + [prediction_column]).rename({prediction_column: alias})
        lazy = lazy.join(other, on=KEYS, how="inner")
    average_expr = sum(pl.col(alias).cast(pl.Float64) for alias in aliases) / len(aliases)
    averaged = lazy.with_columns(average_expr.alias(output_column)).select(KEYS + ["weight", TARGET, output_column]).collect()
    expected = pl.scan_parquet(paths[0]).select(pl.len()).collect().item()
    if averaged.height != expected:
        raise ValueError(f"join lost rows for {paths[0].name}: expected={expected}, got={averaged.height}")
    return averaged.sort(KEYS)


def _score_frame(frame: pl.DataFrame, prediction: str) -> dict[str, float | int]:
    row = frame.select(
        pl.len().alias("rows"),
        (pl.col("weight") * (pl.col(TARGET) - pl.col(prediction)).pow(2)).sum().alias("numerator"),
        (pl.col("weight") * pl.col(TARGET).pow(2)).sum().alias("denominator"),
    ).row(0, named=True)
    denominator = float(row["denominator"])
    if denominator <= 0.0:
        raise ValueError("weighted target energy must be positive")
    return {
        "rows": int(row["rows"]),
        "numerator": float(row["numerator"]),
        "denominator": denominator,
        "weighted_zero_mean_r2": 1.0 - float(row["numerator"]) / denominator,
    }


def _global_score_from_fold_rows(summary: pl.DataFrame) -> float:
    numerator = float(summary["numerator"].sum())
    denominator = float(summary["denominator"].sum())
    if denominator <= 0.0:
        raise ValueError("weighted target energy must be positive")
    return 1.0 - numerator / denominator


if __name__ == "__main__":
    main()
