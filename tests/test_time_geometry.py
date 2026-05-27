import polars as pl
import pytest

from janestreet.time_geometry import (
    OperationalTimeSpec,
    parse_operational_time_windows,
    with_operational_time_features,
)


def test_operational_time_features_are_causal_per_symbol_day() -> None:
    frame = pl.DataFrame(
        {
            "date_id": [1, 1, 1, 1, 2],
            "time_id": [0, 1, 0, 1, 0],
            "symbol_id": [0, 0, 1, 1, 0],
            "weight": [2.0, 3.0, 5.0, 7.0, 11.0],
            "feature_00": [1.0, None, 2.0, 4.0, 8.0],
            "feature_01": [3.0, 5.0, None, 9.0, 10.0],
        }
    )
    spec = OperationalTimeSpec(source_columns=("feature_00", "feature_01"), windows=(2,), max_time_id=1)

    result = with_operational_time_features(frame.lazy(), spec).collect().sort(["date_id", "symbol_id", "time_id"])

    assert result["ot_symbol_tick_index"].to_list() == pytest.approx([1, 2, 1, 2, 1])
    assert result["ot_symbol_weight_cum"].to_list() == pytest.approx([2.0, 5.0, 5.0, 12.0, 11.0])
    assert result["ot_missing_count"].to_list() == pytest.approx([0.0, 1.0, 1.0, 0.0, 0.0])
    assert "ot_symbol_weight_ewm_2" in result.columns


def test_parse_operational_time_windows_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        parse_operational_time_windows("16,0")
