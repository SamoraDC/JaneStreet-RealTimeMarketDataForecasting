import polars as pl
import pytest

from janestreet.multiscale_features import (
    MultiscaleSpec,
    parse_multiscale_columns,
    parse_multiscale_spans,
    require_multiscale_columns,
    with_multiscale_features,
)


def test_multiscale_features_are_causal_and_reset_by_day():
    frame = pl.DataFrame(
        {
            "date_id": [0, 0, 0, 1],
            "time_id": [0, 1, 2, 0],
            "symbol_id": [3, 3, 3, 3],
            "feature_00": [0.0, 1.0, 2.0, 10.0],
        }
    )

    result = with_multiscale_features(
        frame.lazy(),
        MultiscaleSpec(columns=("feature_00",), spans=(2, 4)),
    ).collect()

    assert result["feature_00_ms_band_2_4"].to_list() == pytest.approx(
        [0.0, 0.2666666667, 0.5155555556, 0.0]
    )
    assert result["feature_00_ms_absband_2_4"].to_list() == pytest.approx(
        [0.0, 0.2666666667, 0.5155555556, 0.0]
    )


def test_multiscale_spec_exposes_output_columns():
    spec = MultiscaleSpec(columns=("feature_01",), spans=(4, 16, 64))

    assert spec.output_columns == (
        "feature_01_ms_band_4_16",
        "feature_01_ms_absband_4_16",
        "feature_01_ms_band_16_64",
        "feature_01_ms_absband_16_64",
    )


def test_multiscale_parsing_and_validation():
    assert parse_multiscale_columns("feature_00, feature_01") == ("feature_00", "feature_01")
    assert parse_multiscale_spans("4,16,64") == (4, 16, 64)

    with pytest.raises(ValueError, match="at least two"):
        parse_multiscale_spans("4")
    with pytest.raises(ValueError, match="ascending"):
        parse_multiscale_spans("16,4")
    with pytest.raises(ValueError, match="unknown"):
        require_multiscale_columns(["feature_02"], ["feature_00"])
