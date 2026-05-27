"""Simple baseline evaluators."""

from collections.abc import Sequence

import polars as pl

from janestreet.folds import DateFold


def evaluate_constant_prediction_by_fold(
    data: pl.LazyFrame,
    folds: Sequence[DateFold],
    *,
    prediction_value: float,
    target: str = "responder_6",
    weight: str = "weight",
) -> pl.DataFrame:
    """Evaluate a constant prediction over each validation fold."""

    rows: list[dict[str, float | int | str]] = []
    for fold in folds:
        result = (
            data.filter(fold.valid_filter())
            .select(
                pl.len().alias("rows"),
                pl.n_unique("date_id").alias("valid_days_present"),
                pl.min("date_id").alias("observed_valid_start"),
                pl.max("date_id").alias("observed_valid_end"),
                (pl.col(weight) * (pl.col(target) - prediction_value).pow(2)).sum().alias("numerator"),
                (pl.col(weight) * pl.col(target).pow(2)).sum().alias("denominator"),
                pl.sum(weight).alias("weight_sum"),
            )
            .collect()
            .row(0, named=True)
        )
        denominator = float(result["denominator"])
        if denominator <= 0.0:
            raise ValueError(f"{fold.name} has non-positive weighted target energy")

        numerator = float(result["numerator"])
        rows.append(
            {
                "fold": fold.name,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "valid_start": fold.valid_start,
                "valid_end": fold.valid_end,
                "train_days": fold.train_days,
                "valid_days": fold.valid_days,
                "valid_days_present": int(result["valid_days_present"]),
                "observed_valid_start": int(result["observed_valid_start"]),
                "observed_valid_end": int(result["observed_valid_end"]),
                "rows": int(result["rows"]),
                "weight_sum": float(result["weight_sum"]),
                "numerator": numerator,
                "denominator": denominator,
                "weighted_zero_mean_r2": 1.0 - numerator / denominator,
            }
        )

    return pl.DataFrame(rows)

