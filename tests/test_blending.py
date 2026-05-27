import polars as pl
import pytest

from janestreet.blending import (
    add_convex_blend_prediction,
    add_simplex_blend_prediction,
    fit_convex_blend_weight,
    fit_grouped_convex_blend_weights,
    fit_simplex_blend_weights,
)


def test_fit_convex_blend_weight_prefers_perfect_left_prediction():
    frame = pl.DataFrame(
        {
            "responder_6": [1.0, 2.0, 3.0],
            "left": [1.0, 2.0, 3.0],
            "right": [0.0, 0.0, 0.0],
            "weight": [1.0, 1.0, 1.0],
        }
    )

    assert fit_convex_blend_weight(frame, left_prediction="left", right_prediction="right") == pytest.approx(1.0)


def test_add_convex_blend_prediction_combines_columns():
    frame = pl.DataFrame({"left": [1.0], "right": [3.0]})

    result = add_convex_blend_prediction(
        frame,
        blend_weight=0.25,
        left_prediction="left",
        right_prediction="right",
    )

    assert result["blend_prediction"][0] == pytest.approx(2.5)


def test_add_convex_blend_prediction_rejects_invalid_weight():
    frame = pl.DataFrame({"left": [1.0], "right": [3.0]})

    with pytest.raises(ValueError, match="blend_weight"):
        add_convex_blend_prediction(
            frame,
            blend_weight=1.1,
            left_prediction="left",
            right_prediction="right",
        )


def test_fit_grouped_convex_blend_weights_uses_group_overrides():
    frame = pl.DataFrame(
        {
            "bucket": ["a", "a", "b", "b"],
            "responder_6": [1.0, 2.0, 3.0, 4.0],
            "left": [1.0, 2.0, 0.0, 0.0],
            "right": [0.0, 0.0, 3.0, 4.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )

    grouped = fit_grouped_convex_blend_weights(
        frame,
        group_columns=["bucket"],
        left_prediction="left",
        right_prediction="right",
        min_group_rows=1,
    )
    result = grouped.apply(
        frame,
        left_prediction="left",
        right_prediction="right",
        output="grouped_blend",
    )

    assert grouped.parameters.sort("bucket")["_blend_weight"].to_list() == pytest.approx([1.0, 0.0])
    assert result["grouped_blend"].to_list() == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_fit_grouped_convex_blend_weights_falls_back_for_small_groups():
    frame = pl.DataFrame(
        {
            "bucket": ["a", "b"],
            "responder_6": [1.0, 1.0],
            "left": [1.0, 1.0],
            "right": [0.0, 0.0],
            "weight": [1.0, 1.0],
        }
    )

    grouped = fit_grouped_convex_blend_weights(
        frame,
        group_columns=["bucket"],
        left_prediction="left",
        right_prediction="right",
        min_group_rows=3,
    )

    assert grouped.fallback_weight == pytest.approx(1.0)
    assert grouped.parameters["_blend_weight"].to_list() == pytest.approx([1.0, 1.0])


def test_fit_simplex_blend_weights_selects_best_convex_mix():
    frame = pl.DataFrame(
        {
            "responder_6": [0.0, 1.0, 2.0, 3.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
            "bad": [3.0, 3.0, 3.0, 3.0],
            "good": [0.0, 1.0, 2.0, 3.0],
            "also_bad": [-1.0, -1.0, -1.0, -1.0],
        }
    )

    weights = fit_simplex_blend_weights(
        frame,
        prediction_columns=("bad", "good", "also_bad"),
    )
    result = add_simplex_blend_prediction(frame, weights=weights)

    assert weights["good"] == pytest.approx(1.0)
    assert result["ensemble_prediction"].to_list() == pytest.approx([0.0, 1.0, 2.0, 3.0])
