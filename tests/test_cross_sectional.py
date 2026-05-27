import polars as pl
import pytest

from janestreet.cross_sectional import (
    make_random_projection_spec,
    with_cross_sectional_random_projections,
)


def test_cross_sectional_projection_adds_market_and_deviation_columns():
    frame = pl.DataFrame(
        {
            "date_id": [1, 1, 1, 1],
            "time_id": [0, 0, 1, 1],
            "feature_00": [1.0, 3.0, 10.0, 14.0],
            "feature_01": [2.0, 4.0, 20.0, 24.0],
        }
    )
    spec = make_random_projection_spec(["feature_00", "feature_01"], n_projections=2, seed=7)

    result = with_cross_sectional_random_projections(frame.lazy(), spec).collect()

    for column in spec.output_columns:
        assert column in result.columns


def test_cross_sectional_deviation_sums_to_zero_within_timestamp():
    frame = pl.DataFrame(
        {
            "date_id": [1, 1, 1, 1],
            "time_id": [0, 0, 1, 1],
            "feature_00": [1.0, 3.0, 10.0, 14.0],
            "feature_01": [2.0, 4.0, 20.0, 24.0],
        }
    )
    spec = make_random_projection_spec(["feature_00", "feature_01"], n_projections=1, seed=11)

    result = with_cross_sectional_random_projections(frame.lazy(), spec).collect()
    grouped = result.group_by(["date_id", "time_id"]).agg(pl.col(spec.deviation_columns[0]).sum().alias("sum_dev"))

    assert grouped["sum_dev"].to_list() == pytest.approx([0.0, 0.0])
