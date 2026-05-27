"""CLI for aggregating saved multi-model report scoreboards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL_PACKAGE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from multimodels.report_scoreboard import ReportScoreboardConfig, run_report_scoreboard  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate saved candidate_summary.csv files into a scoreboard.")
    parser.add_argument("--reports-root", type=Path, default=Path("multi-models/reports"))
    parser.add_argument("--output-dir", type=Path, default=Path("multi-models/reports/report_scoreboard"))
    parser.add_argument("--min-rows", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--categories", default="")
    parser.add_argument("--include-reports", default="")
    parser.add_argument("--exclude-reports", default="")
    args = parser.parse_args()

    result = run_report_scoreboard(
        ReportScoreboardConfig(
            reports_root=args.reports_root,
            output_dir=args.output_dir,
            min_rows=args.min_rows,
            top_k=args.top_k,
            categories=_parse_str_tuple(args.categories),
            include_reports=_parse_str_tuple(args.include_reports),
            exclude_reports=_parse_str_tuple(args.exclude_reports),
        )
    )
    print(result["top"].select(["category", "report", "candidate", "global_r2", "min_fold_r2"]).head(args.top_k))
    print(f"Wrote {args.output_dir}")


def _parse_str_tuple(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


if __name__ == "__main__":
    main()
