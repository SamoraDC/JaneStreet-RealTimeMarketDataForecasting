from pathlib import Path

import polars as pl
import pytest

from janestreet.tag_features import FeatureTagSpec, load_feature_tag_spec, with_feature_tag_market_state


def test_load_feature_tag_spec_filters_known_features(tmp_path: Path):
    features_csv = tmp_path / "features.csv"
    features_csv.write_text(
        "feature,tag_0,tag_1\n"
        "feature_00,true,false\n"
        "feature_01,true,true\n"
        "feature_99,false,true\n",
        encoding="utf-8",
    )

    spec = load_feature_tag_spec(features_csv, ["feature_00", "feature_01"])

    assert spec.tag_to_features["0"] == ("feature_00", "feature_01")
    assert spec.tag_to_features["1"] == ("feature_01",)


def test_tag_market_state_adds_leave_one_out_features():
    frame = pl.DataFrame(
        {
            "date_id": [1, 1],
            "time_id": [0, 0],
            "feature_00": [1.0, 3.0],
            "feature_01": [2.0, 4.0],
        }
    )
    spec = FeatureTagSpec({"0": ("feature_00", "feature_01")})

    result = with_feature_tag_market_state(frame.lazy(), spec).collect()

    assert result["tag_0_market_loo"].to_list() == pytest.approx([3.5, 1.5])
    assert result["tag_0_deviation"].to_list() == pytest.approx([-2.0, 2.0])
