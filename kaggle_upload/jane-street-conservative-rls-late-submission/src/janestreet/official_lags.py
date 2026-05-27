"""Official responder lag features and gateway-style lag state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import polars as pl


RESPONDER_COLUMNS: tuple[str, ...] = tuple(f"responder_{idx}" for idx in range(9))
LAG_JOIN_KEYS: tuple[str, ...] = ("date_id", "time_id", "symbol_id")
DAILY_LAG_JOIN_KEYS: tuple[str, ...] = ("date_id", "symbol_id")


def responder_lag_columns(
    responder_columns: Sequence[str] = RESPONDER_COLUMNS,
    *,
    date_lag: int = 1,
) -> tuple[str, ...]:
    """Return official lag feature names for responder columns."""

    _require_positive_lag(date_lag)
    return tuple(f"{column}_lag_{date_lag}" for column in responder_columns)


def daily_last_responder_lag_columns(
    responder_columns: Sequence[str] = RESPONDER_COLUMNS,
    *,
    date_lag: int = 1,
) -> tuple[str, ...]:
    """Return previous-date daily-last lag feature names."""

    _require_positive_lag(date_lag)
    return tuple(f"{column}_daily_last_lag_{date_lag}" for column in responder_columns)


def with_responder_lag_features(
    data: pl.LazyFrame,
    *,
    responder_columns: Sequence[str] = RESPONDER_COLUMNS,
    date_lag: int = 1,
) -> pl.LazyFrame:
    """Add responder lags reconstructed as previous-date same time/symbol values.

    The Kaggle gateway serves `lags` at the first time step of a date. Those lags
    are keyed by `date_id`, `time_id`, and `symbol_id`, so the train-time
    reconstruction shifts historical responders forward by one `date_id` while
    preserving intraday time and symbol identity.
    """

    _require_positive_lag(date_lag)
    lag_columns = responder_lag_columns(responder_columns, date_lag=date_lag)
    lag_frame = data.select(
        [
            (pl.col("date_id") + date_lag).alias("date_id"),
            pl.col("time_id"),
            pl.col("symbol_id"),
        ]
        + [pl.col(source).alias(target) for source, target in zip(responder_columns, lag_columns, strict=True)]
    )
    return data.join(lag_frame, on=list(LAG_JOIN_KEYS), how="left")


def gateway_lags_from_training_date(
    data: pl.LazyFrame,
    *,
    current_date_id: int,
    responder_columns: Sequence[str] = RESPONDER_COLUMNS,
    date_lag: int = 1,
) -> pl.DataFrame:
    """Build the lag frame that would be served at `time_id == 0`.

    This is a local validation utility. For `current_date_id=d`, it exposes
    responders from `d - date_lag`, with the lag frame's `date_id` set to `d`.
    """

    _require_positive_lag(date_lag)
    source_date_id = current_date_id - date_lag
    lag_columns = responder_lag_columns(responder_columns, date_lag=date_lag)
    return (
        data.filter(pl.col("date_id") == source_date_id)
        .select(
            [
                pl.lit(current_date_id).alias("date_id"),
                pl.col("time_id"),
                pl.col("symbol_id"),
            ]
            + [
                pl.col(source).alias(target)
                for source, target in zip(responder_columns, lag_columns, strict=True)
            ]
        )
        .collect()
    )


def daily_last_responder_lags(
    data: pl.LazyFrame,
    *,
    responder_columns: Sequence[str] = RESPONDER_COLUMNS,
    date_lag: int = 1,
) -> pl.LazyFrame:
    """Build previous-date last-observation responder lags by symbol.

    These features use the final `time_id` available for each `(date_id,
    symbol_id)`, shift it to the succeeding date, and join by `(date_id,
    symbol_id)`. They are compatible with the gateway because all previous-day
    responders are available at the first time step of the next date.
    """

    _require_positive_lag(date_lag)
    lag_columns = daily_last_responder_lag_columns(responder_columns, date_lag=date_lag)
    return (
        data.select(["date_id", "time_id", "symbol_id", *responder_columns])
        .sort(["date_id", "symbol_id", "time_id"])
        .group_by(["date_id", "symbol_id"], maintain_order=True)
        .agg(
            [
                pl.col(column).last().alias(lag_column)
                for column, lag_column in zip(responder_columns, lag_columns, strict=True)
            ]
        )
        .with_columns((pl.col("date_id") + date_lag).cast(pl.Int16).alias("date_id"))
    )


@dataclass
class GatewayResponderLagCache:
    """Stateful helper matching the competition lag delivery pattern."""

    responder_columns: tuple[str, ...] = RESPONDER_COLUMNS
    date_lag: int = 1
    _lags: pl.DataFrame | None = None

    @property
    def lag_columns(self) -> tuple[str, ...]:
        return responder_lag_columns(self.responder_columns, date_lag=self.date_lag)

    def add_to_batch(self, test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame:
        """Attach cached official lags to one test batch.

        Pass the gateway-provided `lags` only when it is non-null. The cache then
        reuses that frame for later `time_id` batches in the same date.
        """

        if lags is not None:
            self._lags = lags.select(list(LAG_JOIN_KEYS) + list(self.lag_columns))
        if self._lags is None:
            return test.with_columns([pl.lit(None, dtype=pl.Float32).alias(name) for name in self.lag_columns])
        return test.join(self._lags, on=list(LAG_JOIN_KEYS), how="left")


def _require_positive_lag(date_lag: int) -> None:
    if date_lag <= 0:
        raise ValueError("date_lag must be positive")
