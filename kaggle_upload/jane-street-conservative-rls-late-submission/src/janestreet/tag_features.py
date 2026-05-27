"""Feature-tag market-state transforms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import polars as pl


@dataclass(frozen=True)
class FeatureTagSpec:
    """Mapping from official feature tags to feature columns."""

    tag_to_features: dict[str, tuple[str, ...]]

    @property
    def tag_names(self) -> tuple[str, ...]:
        return tuple(self.tag_to_features)

    @property
    def market_columns(self) -> tuple[str, ...]:
        return tuple(f"tag_{tag}_market_loo" for tag in self.tag_names)

    @property
    def deviation_columns(self) -> tuple[str, ...]:
        return tuple(f"tag_{tag}_deviation" for tag in self.tag_names)

    @property
    def output_columns(self) -> tuple[str, ...]:
        return self.market_columns + self.deviation_columns


def load_feature_tag_spec(
    features_csv: Path,
    feature_columns: Sequence[str],
    *,
    min_features_per_tag: int = 1,
) -> FeatureTagSpec:
    """Load official feature tags for known feature columns."""

    if min_features_per_tag <= 0:
        raise ValueError("min_features_per_tag must be positive")
    known_features = set(feature_columns)
    frame = pl.read_csv(features_csv)
    tag_columns = [name for name in frame.columns if name.startswith("tag_")]
    tag_to_features: dict[str, tuple[str, ...]] = {}
    for tag in tag_columns:
        features = tuple(
            frame.filter((pl.col(tag) == True) & pl.col("feature").is_in(known_features))["feature"].to_list()
        )
        if len(features) >= min_features_per_tag:
            tag_to_features[tag.replace("tag_", "")] = features
    if not tag_to_features:
        raise ValueError("no usable feature tags found")
    return FeatureTagSpec(tag_to_features=tag_to_features)


def with_feature_tag_market_state(
    data: pl.LazyFrame,
    spec: FeatureTagSpec,
    *,
    group_columns: Sequence[str] = ("date_id", "time_id"),
) -> pl.LazyFrame:
    """Add tag-level leave-one-out market state and deviations."""

    groups = list(group_columns)
    if not groups:
        raise ValueError("group_columns must not be empty")
    internal_columns = {tag: f"__tag_{tag}_factor" for tag in spec.tag_names}
    with_factors = data.with_columns(
        [
            pl.mean_horizontal(
                [pl.col(feature).fill_null(0.0).cast(pl.Float64) for feature in spec.tag_to_features[tag]]
            ).alias(internal_columns[tag])
            for tag in spec.tag_names
        ]
    )
    with_market = with_factors.with_columns(
        [
            _leave_one_out_mean_expr(internal_columns[tag], groups).alias(f"tag_{tag}_market_loo")
            for tag in spec.tag_names
        ]
    )
    return with_market.with_columns(
        [
            (pl.col(internal_columns[tag]) - pl.col(f"tag_{tag}_market_loo")).alias(
                f"tag_{tag}_deviation"
            )
            for tag in spec.tag_names
        ]
    )


def _leave_one_out_mean_expr(column: str, groups: list[str]) -> pl.Expr:
    count = pl.len().over(groups)
    total = pl.col(column).sum().over(groups)
    return pl.when(count > 1).then((total - pl.col(column)) / (count - 1)).otherwise(pl.col(column))
