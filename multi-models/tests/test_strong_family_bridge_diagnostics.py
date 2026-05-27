from __future__ import annotations

import polars as pl
import pytest

from multimodels.strong_family_bridge_diagnostics import BridgeDiagnosticConfig, run_bridge_diagnostics


def test_bridge_diagnostics_reports_positive_delta_for_better_candidate(tmp_path) -> None:
    frame = pl.DataFrame(
        {
            "fold": ["rw_01", "rw_01", "rw_02", "rw_02"],
            "date_id": [1, 1, 2, 2],
            "time_id": [0, 1, 0, 1],
            "symbol_id": [0, 1, 0, 1],
            "weight": [1.0, 1.0, 1.0, 1.0],
            "responder_6": [1.0, -1.0, 1.0, -1.0],
            "baseline": [0.0, 0.0, 0.0, 0.0],
            "candidate": [0.8, -0.8, 0.8, -0.8],
        }
    )
    path = tmp_path / "bridge_predictions.parquet"
    frame.write_parquet(path)

    result = run_bridge_diagnostics(
        BridgeDiagnosticConfig(
            experiment_name="unit",
            bridge_prediction_path=path,
            output_dir=tmp_path / "report",
            candidate="candidate",
            baseline="baseline",
        )
    )

    summary = result["summary"]
    assert summary["candidate_delta_r2"] == pytest.approx(0.96)
    assert (tmp_path / "report" / "DIAGNOSTIC_REPORT.md").exists()
    assert (tmp_path / "report" / "diagnostic_delta_fold_candidate.csv").exists()
