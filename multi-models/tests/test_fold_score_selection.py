from __future__ import annotations

import polars as pl
import pytest

from multimodels.fold_score_selection import (
    FoldScoreSelectionConfig,
    FoldScoreSource,
    load_fold_scores,
    run_fold_score_selection,
    select_nested_by_previous_folds,
)


def _scores() -> pl.DataFrame:
    rows = []
    for fold, denom, a_r2, b_r2 in (
        ("rw_01", 100.0, 0.10, 0.05),
        ("rw_02", 200.0, 0.05, 0.20),
        ("rw_03", 300.0, 0.01, 0.30),
    ):
        for candidate, r2 in (("a", a_r2), ("b", b_r2)):
            rows.append(
                {
                    "fold": fold,
                    "candidate": candidate,
                    "family": "mock",
                    "rows": 10,
                    "weight_sum": 10.0,
                    "numerator": denom * (1.0 - r2),
                    "denominator": denom,
                    "weighted_zero_mean_r2": r2,
                    "source": "mock",
                    "candidate_id": f"mock/{candidate}",
                    "source_family": "mock/mock",
                }
            )
    return pl.DataFrame(rows)


def test_nested_selector_uses_only_previous_folds() -> None:
    selected = select_nested_by_previous_folds(
        _scores(),
        candidate_ids=("mock/a", "mock/b"),
        first_candidate_id="mock/a",
        min_history_folds=1,
        selection_metric="global_r2",
    )

    assert selected["selected_candidate_id"].to_list() == ["mock/a", "mock/a", "mock/b"]
    assert selected["uses_current_fold_for_selection"].to_list() == [False, False, False]
    assert selected.filter(pl.col("fold") == "rw_03")["history_folds"].item() == 2


def test_run_fold_score_selection_writes_summary(tmp_path) -> None:
    path = tmp_path / "fold_scores.csv"
    _scores().drop("source", "candidate_id", "source_family").write_csv(path)

    result = run_fold_score_selection(
        FoldScoreSelectionConfig(
            output_dir=tmp_path / "out",
            sources=(FoldScoreSource("mock", path),),
            candidate_ids=("mock/a", "mock/b"),
            first_candidate_id="mock/a",
            min_history_folds=1,
            selection_metric="global_r2",
        )
    )

    assert (tmp_path / "out" / "nested_selection_summary.csv").exists()
    summary = result["nested_summary"].row(0, named=True)
    expected_numerator = 100.0 * 0.90 + 200.0 * 0.95 + 300.0 * 0.70
    expected_denominator = 600.0
    assert summary["global_r2"] == pytest.approx(1.0 - expected_numerator / expected_denominator)


def test_load_fold_scores_requires_sufficient_statistics(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    pl.DataFrame({"fold": ["rw_01"]}).write_csv(path)

    with pytest.raises(ValueError, match="missing required columns"):
        load_fold_scores((FoldScoreSource("bad", path),))
