"""Temporal validation folds."""

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class DateFold:
    """Inclusive date-id ranges for one temporal validation fold."""

    name: str
    train_start: int
    train_end: int
    valid_start: int
    valid_end: int

    @property
    def train_days(self) -> int:
        return self.train_end - self.train_start + 1

    @property
    def valid_days(self) -> int:
        return self.valid_end - self.valid_start + 1

    def train_filter(self) -> pl.Expr:
        return pl.col("date_id").is_between(self.train_start, self.train_end)

    def valid_filter(self) -> pl.Expr:
        return pl.col("date_id").is_between(self.valid_start, self.valid_end)


def make_expanding_folds(
    *,
    min_date_id: int,
    max_date_id: int,
    n_folds: int,
    valid_window: int,
    gap: int = 0,
    min_train_window: int = 1,
) -> list[DateFold]:
    """Create expanding walk-forward folds over integer `date_id`.

    Validation blocks occupy the most recent `n_folds * valid_window` dates.
    Each fold trains on all dates before validation, optionally leaving a
    temporal gap between train and validation.
    """

    _require_positive("n_folds", n_folds)
    _require_positive("valid_window", valid_window)
    _require_non_negative("gap", gap)
    _require_positive("min_train_window", min_train_window)
    if min_date_id > max_date_id:
        raise ValueError("min_date_id must be <= max_date_id")

    first_valid_start = max_date_id - n_folds * valid_window + 1
    if first_valid_start <= min_date_id:
        raise ValueError("not enough dates for requested validation windows")

    folds: list[DateFold] = []
    for idx in range(n_folds):
        valid_start = first_valid_start + idx * valid_window
        valid_end = valid_start + valid_window - 1
        train_start = min_date_id
        train_end = valid_start - gap - 1
        train_days = train_end - train_start + 1
        if train_days < min_train_window:
            raise ValueError(
                f"fold {idx + 1} has {train_days} train days; "
                f"minimum is {min_train_window}"
            )

        folds.append(
            DateFold(
                name=f"wf_{idx + 1:02d}",
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
            )
        )

    return folds


def make_rolling_folds(
    *,
    min_date_id: int,
    max_date_id: int,
    n_folds: int,
    train_window: int,
    valid_window: int,
    gap: int = 0,
) -> list[DateFold]:
    """Create rolling recent-window folds over integer `date_id`."""

    _require_positive("n_folds", n_folds)
    _require_positive("train_window", train_window)
    _require_positive("valid_window", valid_window)
    _require_non_negative("gap", gap)
    if min_date_id > max_date_id:
        raise ValueError("min_date_id must be <= max_date_id")

    first_valid_start = max_date_id - n_folds * valid_window + 1
    first_train_start = first_valid_start - gap - train_window
    if first_train_start < min_date_id:
        raise ValueError("not enough dates for requested rolling folds")

    folds: list[DateFold] = []
    for idx in range(n_folds):
        valid_start = first_valid_start + idx * valid_window
        valid_end = valid_start + valid_window - 1
        train_end = valid_start - gap - 1
        train_start = train_end - train_window + 1
        folds.append(
            DateFold(
                name=f"rw_{idx + 1:02d}",
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
            )
        )

    return folds


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")

