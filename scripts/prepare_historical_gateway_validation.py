"""Prepare the clean historical validation run for frozen gateway candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_MAX_DATE_ID = 1398
DEFAULT_TABM_DIR = Path("reports/experiments/competitive_tabm_official_stage3_hist_max1398_5fold_valid60_lags_online_lr1e4_4m_train700_seed37_aux8_preds")
DEFAULT_TREE_DIR = Path("reports/experiments/tree_engine_ensemble_hist_max1398_xgb_lgb_sample10_seed_ensemble_preds")
DEFAULT_VALIDATION_DIR = Path("reports/experiments/frozen_gateway_candidate_validation_hist_max1398_stage3_protocol")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-date-id", type=int, default=DEFAULT_MAX_DATE_ID)
    parser.add_argument("--tabm-output-dir", type=Path, default=DEFAULT_TABM_DIR)
    parser.add_argument("--tree-output-dir", type=Path, default=DEFAULT_TREE_DIR)
    parser.add_argument("--validation-output-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiments/historical_gateway_validation_plan"))
    args = parser.parse_args()

    plan = build_plan(
        max_date_id=args.max_date_id,
        tabm_output_dir=args.tabm_output_dir,
        tree_output_dir=args.tree_output_dir,
        validation_output_dir=args.validation_output_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "historical_gateway_validation_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "historical_gateway_validation_plan.md").write_text(render_markdown(plan), encoding="utf-8")
    print(json.dumps(plan["status"], indent=2))
    print(f"Wrote {args.output_dir}")


def build_plan(*, max_date_id: int, tabm_output_dir: Path, tree_output_dir: Path, validation_output_dir: Path) -> dict[str, object]:
    if max_date_id < 0:
        raise ValueError("--max-date-id must be non-negative")
    tabm_prediction_dir = tabm_output_dir / "validation_predictions"
    tree_prediction_dir = tree_output_dir / "validation_predictions"
    artifacts = {
        "tabm_predictions": _prediction_status(tabm_prediction_dir),
        "tree_predictions": _prediction_status(tree_prediction_dir),
    }
    ready = all(status["ready"] for status in artifacts.values())
    commands = {
        "tabm_oof": _tabm_command(max_date_id=max_date_id, output_dir=tabm_output_dir),
        "tree_oof": _tree_command(max_date_id=max_date_id, output_dir=tree_output_dir),
        "frozen_gateway_validation": _validation_command(
            tabm_prediction_dir=tabm_prediction_dir,
            tree_prediction_dir=tree_prediction_dir,
            output_dir=validation_output_dir,
        ),
    }
    missing = [name for name, status in artifacts.items() if not status["ready"]]
    return {
        "experiment": "historical_gateway_validation_plan",
        "purpose": "Generate a clean pre-Stage-3 OOF validation for the two frozen gateway Bayesian candidates.",
        "max_date_id": max_date_id,
        "status": {
            "ready_for_validation": ready,
            "missing_artifacts": missing,
        },
        "artifacts": artifacts,
        "commands": commands,
        "methodological_note": (
            "Do not tune candidates on this historical run. The only valid comparison is the two frozen candidates "
            "already selected on Stage 3 plus the TabM/tree walk-forward baseline."
        ),
    }


def render_markdown(plan: dict[str, object]) -> str:
    status = plan["status"]
    artifacts = plan["artifacts"]
    commands = plan["commands"]
    lines = [
        "# Historical Gateway Validation Plan",
        "",
        f"- `max_date_id`: `{plan['max_date_id']}`",
        f"- Ready for validation: `{status['ready_for_validation']}`",
        f"- Missing artifacts: `{', '.join(status['missing_artifacts']) if status['missing_artifacts'] else 'none'}`",
        "",
        "## Artifact Status",
        "",
    ]
    for name, artifact_status in artifacts.items():
        lines.append(f"- `{name}`: ready=`{artifact_status['ready']}`, parquet_count=`{artifact_status['parquet_count']}`, path=`{artifact_status['path']}`")
    lines.extend(
        [
            "",
            "## Commands",
            "",
            "Run these in order. Do not change candidate hyperparameters after seeing the historical result.",
            "",
        ]
    )
    for name, command in commands.items():
        lines.extend([f"### {name}", "", "```bash", command, "```", ""])
    lines.extend(["## Methodological Note", "", str(plan["methodological_note"]), ""])
    return "\n".join(lines)


def _prediction_status(path: Path) -> dict[str, object]:
    files = sorted(path.glob("*.parquet")) if path.exists() else []
    return {
        "path": str(path),
        "ready": len(files) >= 5,
        "parquet_count": len(files),
        "files": [file.name for file in files],
    }


def _tabm_command(*, max_date_id: int, output_dir: Path) -> str:
    return " ".join(
        [
            "uv run python scripts/run_competitive_tabular_nn.py",
            "--model-type tabm",
            "--n-folds 5",
            "--train-window 700",
            "--valid-window 60",
            f"--max-date-id {max_date_id}",
            "--max-train-rows 4000000",
            "--epochs 4",
            "--batch-size 8192",
            "--hidden-size 512",
            "--depth 3",
            "--dropout 0.25",
            "--ensemble-size 16",
            "--learning-rate 2.5e-4",
            "--weight-decay 8e-4",
            "--aux-targets responder_0,responder_1,responder_2,responder_3,responder_4,responder_5,responder_7,responder_8",
            "--aux-loss-weight 0.25",
            "--online-update",
            "--online-learning-rate 1e-4",
            "--online-epochs 1",
            "--device cuda",
            "--torch-threads 4",
            "--seed 37",
            "--save-predictions",
            f"--output-dir {output_dir}",
        ]
    )


def _tree_command(*, max_date_id: int, output_dir: Path) -> str:
    return " ".join(
        [
            "uv run python scripts/run_tree_engine_ensemble.py",
            "--n-folds 5",
            "--train-window 120",
            "--valid-window 60",
            f"--max-date-id {max_date_id}",
            "--inner-oof-folds 3",
            "--inner-valid-window 20",
            "--engines xgboost,lightgbm",
            "--train-sample-frac 0.10",
            "--gbdt-seeds 17,23,37",
            "--max-iter 80",
            "--learning-rate 0.03",
            "--max-leaf-nodes 31",
            "--n-jobs 4",
            "--chunk-days 10",
            "--save-predictions",
            f"--output-dir {output_dir}",
        ]
    )


def _validation_command(*, tabm_prediction_dir: Path, tree_prediction_dir: Path, output_dir: Path) -> str:
    return " ".join(
        [
            "uv run python scripts/run_frozen_gateway_candidate_validation.py",
            "--experiment-name frozen_gateway_candidate_validation_hist_max1398_stage3_protocol",
            f"--tabm-prediction-dir {tabm_prediction_dir}",
            f"--tree-prediction-dir {tree_prediction_dir}",
            f"--output-dir {output_dir}",
        ]
    )


if __name__ == "__main__":
    main()
