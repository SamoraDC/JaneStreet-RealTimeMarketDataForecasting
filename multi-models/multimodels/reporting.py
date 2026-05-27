"""Report writing for multi-model experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown_report(
    *,
    path: Path,
    experiment_name: str,
    summary: pl.DataFrame,
    audit: dict[str, Any],
) -> None:
    """Write a compact human-readable experiment synthesis."""

    lines = [
        f"# Multi-Model Experiment: {experiment_name}",
        "",
        "## Headline",
        "",
    ]
    if summary.height:
        best = summary.row(0, named=True)
        lines.extend(
            [
                f"- Best candidate: `{best['candidate']}`.",
                f"- Family: `{best['family']}`.",
                f"- Global weighted zero-mean R2: `{best['global_r2']:.9f}`.",
                f"- Mean fold R2: `{best['mean_fold_r2']:.9f}`.",
                f"- Min fold R2: `{best['min_fold_r2']:.9f}`.",
                "",
            ]
        )
    else:
        lines.extend(["- No scores were produced.", ""])
    lines.extend(
        [
            "## Audit",
            "",
            f"- Folds: `{audit['n_folds']}`.",
            f"- Model features: `{audit['n_model_features']}`.",
            f"- Uses processed lags: `{audit['uses_processed_lags']}`.",
            f"- Uses context features: `{audit['uses_context_features']}`.",
            f"- Raw preprocessed features: `{audit.get('n_raw_preprocessed_features', 0)}`.",
            f"- Raw preprocessing modes: `{', '.join(audit.get('raw_preprocess_modes', [])) or 'none'}`.",
            f"- Target leakage check: `{audit['target_leakage_check']}`.",
            f"- Fold causality check: `{audit['fold_causality_check']}`.",
            f"- Selection check: `{audit['selection_check']}`.",
            f"- Residual rules: `{', '.join(audit['residual_features']) or 'none'}`.",
            f"- Risk auxiliary targets: `{', '.join(audit.get('risk_auxiliary_targets', [])) or 'none'}`.",
            "",
            "## Artifact Manifest",
            "",
            *[
                f"- `{row['artifact']}`: {row['family']} - {row['role']}."
                for row in audit.get("artifact_manifest", [])
            ],
            "",
            "## Methodological Status",
            "",
            "- This is an experimental OOF validation stack, not yet a Kaggle runtime artifact.",
            "- Risk and regime models are auxiliary: they modulate the alpha prediction; they do not change the required final output schema.",
            "- Promotion requires comparison against the preserved conservative and historical checkpoints under the same fold protocol.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
