from __future__ import annotations

import polars as pl
import pytest

from multimodels.strong_oof_diagnostics import add_diagnostic_buckets, compare_pair_by_slice


def test_compare_pair_by_slice_positive_delta_means_candidate_better() -> None:
    frame = pl.DataFrame(
        {
            "fold": ["rw_01", "rw_01"],
            "date_id": [1, 1],
            "time_id": [0, 1],
            "symbol_id": [0, 0],
            "weight": [1.0, 1.0],
            "responder_6": [1.0, -1.0],
            "prediction_disagreement": [0.1, 0.2],
            "baseline": [0.0, 0.0],
            "candidate": [1.0, -1.0],
        }
    )
    frame = add_diagnostic_buckets(frame, baseline="baseline", candidate="candidate", time_bucket_size=100)

    result = compare_pair_by_slice(frame, group_columns=("__all__",), baseline="baseline", candidate="candidate")

    row = result.row(0, named=True)
    assert row["baseline_r2"] == pytest.approx(0.0)
    assert row["candidate_r2"] == pytest.approx(1.0)
    assert row["candidate_delta_r2"] == pytest.approx(1.0)


def test_diagnostic_buckets_are_target_free_and_present() -> None:
    frame = pl.DataFrame(
        {
            "fold": ["rw_01", "rw_01", "rw_01", "rw_01"],
            "date_id": [1, 1, 1, 1],
            "time_id": [0, 1, 2, 3],
            "symbol_id": [0, 1, 2, 3],
            "weight": [1.0, 2.0, 3.0, 4.0],
            "responder_6": [0.0, 0.0, 0.0, 0.0],
            "prediction_disagreement": [0.0, 0.1, 0.2, 0.3],
            "baseline": [0.0, 0.1, 0.2, 0.3],
            "candidate": [0.0, 0.2, 0.4, 0.6],
        }
    )

    out = add_diagnostic_buckets(frame, baseline="baseline", candidate="candidate", time_bucket_size=2)

    for column in ("time_bucket", "weight_bucket", "baseline_abs_bucket", "candidate_abs_bucket", "disagreement_bucket"):
        assert column in out.columns
