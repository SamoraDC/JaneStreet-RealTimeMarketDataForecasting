import polars as pl
import pytest

from janestreet.diagnostics import aggregate_weighted_r2_by_slice, combine_slice_aggregates


def test_aggregate_weighted_r2_by_slice_scores_zero_prediction():
    frame = pl.DataFrame(
        {
            "slice": ["a", "a", "b"],
            "responder_6": [1.0, -2.0, 3.0],
            "prediction": [0.0, 0.0, 0.0],
            "weight": [1.0, 2.0, 3.0],
        }
    )

    result = aggregate_weighted_r2_by_slice(frame, "slice")

    assert result["weighted_zero_mean_r2"].to_list() == pytest.approx([0.0, 0.0])


def test_combine_slice_aggregates_sums_components():
    left = pl.DataFrame(
        {
            "slice": ["a"],
            "rows": [2],
            "weight_sum": [3.0],
            "numerator": [2.0],
            "denominator": [4.0],
        }
    )
    right = pl.DataFrame(
        {
            "slice": ["a"],
            "rows": [1],
            "weight_sum": [2.0],
            "numerator": [1.0],
            "denominator": [6.0],
        }
    )

    result = combine_slice_aggregates([left, right], "slice")

    assert result["rows"][0] == 3
    assert result["weighted_zero_mean_r2"][0] == pytest.approx(0.7)

