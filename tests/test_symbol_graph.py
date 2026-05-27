import polars as pl
import pytest

from janestreet.symbol_graph import (
    SymbolGraphSpec,
    add_symbol_graph_features,
    parse_symbol_graph_columns,
    require_symbol_graph_columns,
)


def test_add_symbol_graph_features_uses_same_timestamp_neighbors():
    frame = pl.DataFrame(
        {
            "date_id": [0, 0, 0, 0],
            "time_id": [0, 0, 1, 1],
            "symbol_id": [1, 2, 1, 2],
            "feature_00": [10.0, 14.0, 20.0, 23.0],
        }
    )
    spec = SymbolGraphSpec(columns=("feature_00",), neighbors={1: (2,), 2: (1,)})

    result = add_symbol_graph_features(frame, spec)

    assert result["symbol_graph_neighbor_count"].to_list() == [1, 1, 1, 1]
    assert result["feature_00_sg_neighbor_mean"].to_list() == pytest.approx([14.0, 10.0, 23.0, 20.0])
    assert result["feature_00_sg_deviation"].to_list() == pytest.approx([-4.0, 4.0, -3.0, 3.0])


def test_add_symbol_graph_features_fills_missing_neighbors():
    frame = pl.DataFrame(
        {
            "date_id": [0],
            "time_id": [0],
            "symbol_id": [1],
            "feature_00": [10.0],
        }
    )
    spec = SymbolGraphSpec(columns=("feature_00",), neighbors={1: (2,)})

    result = add_symbol_graph_features(frame, spec)

    assert result["symbol_graph_neighbor_count"][0] == 1
    assert result["feature_00_sg_neighbor_mean"][0] == pytest.approx(0.0)
    assert result["feature_00_sg_deviation"][0] == pytest.approx(10.0)


def test_symbol_graph_parsing_and_validation():
    assert parse_symbol_graph_columns("feature_00, feature_01") == ("feature_00", "feature_01")

    with pytest.raises(ValueError, match="unknown"):
        require_symbol_graph_columns(["feature_02"], ["feature_00"])
