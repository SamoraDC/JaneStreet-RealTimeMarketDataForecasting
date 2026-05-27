import numpy as np
import polars as pl
import pytest

from janestreet.reservoir_features import (
    make_reservoir_spec,
    parse_reservoir_columns,
    parse_reservoir_spans,
    require_reservoir_columns,
    with_reservoir_features,
)


def test_reservoir_features_are_causal_within_symbol_day():
    frame = pl.DataFrame(
        {
            "date_id": [0, 0, 0, 1],
            "time_id": [0, 1, 2, 0],
            "symbol_id": [1, 1, 1, 1],
            "feature_00": [0.0, 1.0, 2.0, 10.0],
        }
    )
    spec = make_reservoir_spec(("feature_00",), n_states=1, spans=(2,), seed=1)

    result = with_reservoir_features(frame.lazy(), spec).collect()

    output = result["reservoir_s00_ewm_2"].to_list()
    assert output[0] == pytest.approx(0.0)
    assert output[1] != pytest.approx(output[0])
    assert output[3] == pytest.approx(float(np.tanh(spec.weights[0, 0] * 10.0)))


def test_reservoir_parsing_and_validation():
    assert parse_reservoir_columns("feature_00, feature_01") == ("feature_00", "feature_01")
    assert parse_reservoir_spans("5, 20") == (5, 20)

    with pytest.raises(ValueError, match="positive"):
        parse_reservoir_spans("5,0")

    with pytest.raises(ValueError, match="unknown"):
        require_reservoir_columns(["feature_02"], ["feature_00"])
