"""Causal multiscale/wavelet-lite feature transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import polars as pl


@dataclass(frozen=True)
class MultiscaleSpec:
    """Source columns and EWM spans for wavelet-lite bands."""

    columns: tuple[str, ...]
    spans: tuple[int, ...] = (4, 16, 64)

    @property
    def output_columns(self) -> tuple[str, ...]:
        names: list[str] = []
        for column in self.columns:
            for fast, slow in zip(self.spans, self.spans[1:]):
                names.append(f"{column}_ms_band_{fast}_{slow}")
                names.append(f"{column}_ms_absband_{fast}_{slow}")
        return tuple(names)


def parse_multiscale_columns(raw: str) -> tuple[str, ...]:
    """Parse comma-separated multiscale source columns."""

    return tuple(part.strip() for part in raw.split(",") if part.strip())


def parse_multiscale_spans(raw: str) -> tuple[int, ...]:
    """Parse comma-separated positive EWM spans."""

    spans = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if len(spans) < 2:
        raise ValueError("multiscale spans must contain at least two values")
    if any(span <= 0 for span in spans):
        raise ValueError("multiscale spans must be positive")
    if tuple(sorted(spans)) != spans:
        raise ValueError("multiscale spans must be sorted ascending")
    return spans


def require_multiscale_columns(columns: Sequence[str], available: Sequence[str]) -> None:
    """Validate multiscale source columns."""

    available_set = set(available)
    missing = [column for column in columns if column not in available_set]
    if missing:
        raise ValueError(f"unknown multiscale columns: {', '.join(missing)}")


def with_multiscale_features(data: pl.LazyFrame, spec: MultiscaleSpec) -> pl.LazyFrame:
    """Add causal EWM band-pass features per `date_id, symbol_id`."""

    _validate_spec(spec)
    partition = ["date_id", "symbol_id"]
    order_by = "time_id"
    mean_names: list[str] = []
    mean_exprs: list[pl.Expr] = []
    for column in spec.columns:
        source = pl.col(column).fill_null(0.0).cast(pl.Float64)
        for span in spec.spans:
            name = f"__ms_{column}_{span}"
            mean_names.append(name)
            mean_exprs.append(
                source.ewm_mean(span=span, adjust=False, min_samples=1)
                .over(partition, order_by=order_by)
                .alias(name)
            )

    with_means = data.with_columns(mean_exprs)
    feature_exprs: list[pl.Expr] = []
    for column in spec.columns:
        for fast, slow in zip(spec.spans, spec.spans[1:]):
            band = pl.col(f"__ms_{column}_{fast}") - pl.col(f"__ms_{column}_{slow}")
            feature_exprs.extend(
                [
                    band.cast(pl.Float32).alias(f"{column}_ms_band_{fast}_{slow}"),
                    band.abs().cast(pl.Float32).alias(f"{column}_ms_absband_{fast}_{slow}"),
                ]
            )
    return with_means.with_columns(feature_exprs).drop(mean_names)


def _validate_spec(spec: MultiscaleSpec) -> None:
    if not spec.columns:
        raise ValueError("MultiscaleSpec.columns must not be empty")
    if len(spec.spans) < 2:
        raise ValueError("MultiscaleSpec.spans must contain at least two values")
    if any(span <= 0 for span in spec.spans):
        raise ValueError("MultiscaleSpec.spans must be positive")
    if tuple(sorted(spec.spans)) != spec.spans:
        raise ValueError("MultiscaleSpec.spans must be sorted ascending")
