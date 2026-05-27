from __future__ import annotations

import polars as pl

from multimodels.report_scoreboard import (
    ReportScoreboardConfig,
    classify_report,
    load_candidate_summaries,
    run_report_scoreboard,
)


def _write_summary(path, *, rows: int, global_r2: float, min_fold_r2: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "candidate": ["candidate_a"],
            "family": ["family"],
            "rows": [rows],
            "weight_sum": [100.0],
            "numerator": [10.0],
            "denominator": [100.0],
            "mean_fold_r2": [global_r2],
            "min_fold_r2": [min_fold_r2],
            "std_fold_r2": [0.0],
            "global_r2": [global_r2],
        }
    ).write_csv(path)


def test_classify_report_names() -> None:
    assert classify_report("strong_oof_hist_max1398_gateway") == "historical"
    assert classify_report("strong_oof_stage3_gateway") == "stage3"
    assert classify_report("smoke_stride1000") == "smoke"
    assert classify_report("family_artifacts_5fold_lags_stride100") == "sampled"


def test_load_candidate_summaries_and_skip_invalid(tmp_path) -> None:
    root = tmp_path / "reports"
    _write_summary(root / "strong_oof_stage3_a" / "candidate_summary.csv", rows=100, global_r2=0.2, min_fold_r2=0.1)
    (root / "invalid" / "candidate_summary.csv").parent.mkdir(parents=True)
    pl.DataFrame({"candidate": ["bad"]}).write_csv(root / "invalid" / "candidate_summary.csv")

    frame, skipped = load_candidate_summaries(root)

    assert frame.height == 1
    assert frame.select("report").item() == "strong_oof_stage3_a"
    assert frame.select("category").item() == "stage3"
    assert skipped[0]["reason"].startswith("missing_columns:")


def test_run_report_scoreboard_filters_and_writes(tmp_path) -> None:
    root = tmp_path / "reports"
    _write_summary(root / "strong_oof_stage3_small" / "candidate_summary.csv", rows=10, global_r2=0.9, min_fold_r2=0.1)
    _write_summary(root / "strong_oof_stage3_full" / "candidate_summary.csv", rows=1000, global_r2=0.2, min_fold_r2=0.1)
    _write_summary(root / "strong_oof_hist_full" / "candidate_summary.csv", rows=1000, global_r2=0.3, min_fold_r2=0.2)

    output = tmp_path / "scoreboard"
    result = run_report_scoreboard(
        ReportScoreboardConfig(
            reports_root=root,
            output_dir=output,
            min_rows=100,
            categories=("stage3",),
            top_k=5,
        )
    )

    assert result["filtered"].height == 1
    assert result["filtered"].select("report").item() == "strong_oof_stage3_full"
    assert (output / "top_candidates.csv").exists()
    assert (output / "REPORT.md").exists()
