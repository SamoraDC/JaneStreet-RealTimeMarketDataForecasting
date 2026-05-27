"""CLI for cheap nested selection over saved fold score CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL_PACKAGE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from multimodels.fold_score_selection import FoldScoreSelectionConfig, FoldScoreSource, run_fold_score_selection  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nested fold-score selection over saved OOF summaries.")
    parser.add_argument("--source", action="append", default=[], help="Named fold_scores source in NAME=PATH format. Repeatable.")
    parser.add_argument("--candidate-id", action="append", default=[], help="Candidate id in SOURCE/CANDIDATE format. Repeatable.")
    parser.add_argument("--first-candidate-id", required=True, help="Predeclared candidate for folds without enough history.")
    parser.add_argument("--min-history-folds", type=int, default=1)
    parser.add_argument("--selection-metric", choices=("global_r2", "mean_fold_r2", "min_fold_r2"), default="global_r2")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = FoldScoreSelectionConfig(
        output_dir=args.output_dir,
        sources=tuple(_parse_source(item) for item in args.source),
        candidate_ids=tuple(args.candidate_id),
        first_candidate_id=args.first_candidate_id,
        min_history_folds=args.min_history_folds,
        selection_metric=args.selection_metric,
    )
    result = run_fold_score_selection(config)
    print(result["nested_summary"])
    print(f"Wrote {args.output_dir}")


def _parse_source(raw: str) -> FoldScoreSource:
    if "=" not in raw:
        raise ValueError(f"source must be NAME=PATH, got {raw!r}")
    name, path = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("source name must not be empty")
    return FoldScoreSource(name=name, path=Path(path.strip()))


if __name__ == "__main__":
    main()
