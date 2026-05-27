import polars as pl
import pytest

from janestreet.koopman_features import (
    KoopmanSpec,
    parse_koopman_columns,
    require_koopman_columns,
    with_koopman_features,
)


def test_koopman_features_are_causal_and_reset_by_day():
    frame = pl.DataFrame(
        {
            "date_id": [0, 0, 0, 1],
            "time_id": [0, 1, 2, 0],
            "symbol_id": [2, 2, 2, 2],
            "feature_00": [1.0, 3.0, 2.0, 10.0],
        }
    )

    result = with_koopman_features(
        frame.lazy(),
        KoopmanSpec(columns=("feature_00",), span=2),
    ).collect()

    assert result["feature_00_kp_square"].to_list() == pytest.approx([1.0, 9.0, 4.0, 100.0])
    assert result["feature_00_kp_delta"].to_list() == pytest.approx([0.0, 2.0, -1.0, 0.0])
    assert result["feature_00_kp_value_delta"].to_list() == pytest.approx([0.0, 6.0, -2.0, 0.0])
    assert result["feature_00_kp_value_ewm_2"][3] == pytest.approx(100.0)


def test_koopman_spec_exposes_output_columns():
    spec = KoopmanSpec(columns=("feature_01",), span=8)

    assert spec.output_columns == (
        "feature_01_kp_square",
        "feature_01_kp_delta",
        "feature_01_kp_value_delta",
        "feature_01_kp_value_ewm_8",
    )


def test_koopman_parsing_and_validation():
    assert parse_koopman_columns("feature_00, feature_01") == ("feature_00", "feature_01")

    with pytest.raises(ValueError, match="unknown"):
        require_koopman_columns(["feature_02"], ["feature_00"])

    with pytest.raises(ValueError, match="positive"):
        with_koopman_features(
            pl.DataFrame({"date_id": [], "time_id": [], "symbol_id": [], "feature_00": []}).lazy(),
            KoopmanSpec(columns=("feature_00",), span=0),
        )
