import polars as pl
import pytest

from janestreet.temporal_geometry import (
    TemporalGeometrySpec,
    parse_temporal_geometry_columns,
    parse_temporal_geometry_windows,
    require_temporal_geometry_columns,
    with_temporal_geometry_features,
)


def test_temporal_geometry_features_are_intraday_and_causal():
    frame = pl.DataFrame(
        {
            "date_id": [0, 0, 0, 0, 1],
            "time_id": [0, 1, 2, 3, 0],
            "symbol_id": [7, 7, 7, 7, 7],
            "feature_00": [1.0, 3.0, 2.0, None, 10.0],
        }
    )

    result = with_temporal_geometry_features(
        frame.lazy(),
        TemporalGeometrySpec(columns=("feature_00",), windows=(2,)),
    ).collect()

    assert result["feature_00_tg_diff1"].to_list() == pytest.approx([0.0, 2.0, -1.0, -2.0, 0.0])
    assert result["feature_00_tg_tv_2"].to_list() == pytest.approx([0.0, 2.0, 3.0, 3.0, 0.0])
    assert result["feature_00_tg_qv_2"].to_list() == pytest.approx([0.0, 4.0, 5.0, 5.0, 0.0])


def test_temporal_geometry_spec_exposes_output_columns():
    spec = TemporalGeometrySpec(columns=("feature_01",), windows=(3, 5))

    assert spec.output_columns == (
        "feature_01_tg_diff1",
        "feature_01_tg_tv_3",
        "feature_01_tg_qv_3",
        "feature_01_tg_rough_3",
        "feature_01_tg_tv_5",
        "feature_01_tg_qv_5",
        "feature_01_tg_rough_5",
    )


def test_parse_temporal_geometry_inputs():
    assert parse_temporal_geometry_columns("feature_01, feature_02") == ("feature_01", "feature_02")
    assert parse_temporal_geometry_windows("5, 20") == (5, 20)

    with pytest.raises(ValueError, match="positive"):
        parse_temporal_geometry_windows("5, 0")


def test_require_temporal_geometry_columns_rejects_missing_columns():
    with pytest.raises(ValueError, match="unknown"):
        require_temporal_geometry_columns(["feature_02"], ["feature_01"])
