"""Generate English visual analytics for the preserved best candidates.

The gallery is intentionally built from the preserved CSV artifacts under
best-candidates/. It does not use mock data and does not require the raw Kaggle
parquet files. Each candidate gets:

- 5 static charts
- 5 two-dimensional GIF animations
- 5 three-dimensional GIF animations
"""

from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEST_CANDIDATES_DIR = PROJECT_ROOT / "best-candidates"
OUTPUT_ROOT = PROJECT_ROOT / "charts" / "best-candidates"

TARGET_R2 = 0.02
MILLION = 1_000_000.0

COLORS = {
    "selected": "#005BBB",
    "reference": "#D62828",
    "target": "#111111",
    "green": "#1A936F",
    "amber": "#F4A261",
    "purple": "#6A4C93",
    "cyan": "#118AB2",
    "gray": "#6C757D",
    "light_gray": "#E9ECEF",
}

FAMILY_COLORS = {
    "fixed_blend": "#005BBB",
    "strong_base": "#1A936F",
    "strong_stack": "#6A4C93",
    "residual_tail": "#F4A261",
    "dynamic_gateway_rls": "#118AB2",
    "baseline": "#D62828",
    "unknown": "#6C757D",
}

SHORT_NAMES = {
    "fixed_blend_0_w0p75_fixed_blend": "75/25 Fixed Blend",
    "strong_oof_ridge_stack": "Strong OOF Ridge Stack",
    "conservative_rls_prediction": "Conservative RLS",
    "aggressive_rls_prediction": "Aggressive RLS",
    "tabm_prediction": "TabM",
    "tree_prediction": "Tree Ensemble",
    "tabm_tree_convex_walk_forward": "TabM/Tree Walk-Forward",
    "dynamic_gateway_rls_experts_alpha10000_f0p995": "Conservative Dynamic RLS",
    "dynamic_gateway_rls_components_no_tree_ensemble_alpha1000_f0p995": "Aggressive Dynamic RLS",
    "gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p95_residual_tail": "Residual-Tail q0.95",
    "gateway_risk_conservative_rls_abs_pred_s100_prediction": "Risk Conservative RLS",
}


@dataclass(frozen=True)
class ReportSpec:
    label: str
    summary_csv: Path
    fold_csv: Path
    regime: str


@dataclass(frozen=True)
class CandidateSpec:
    slug: str
    display_name: str
    selected_id: str
    reference_ids: tuple[str, ...]
    reports: tuple[ReportSpec, ...]
    thesis: str


@dataclass
class CandidateData:
    spec: CandidateSpec
    summary: pd.DataFrame
    folds: pd.DataFrame
    primary_report: str
    selected_summary: pd.DataFrame
    selected_folds: pd.DataFrame
    primary_summary: pd.DataFrame
    primary_folds: pd.DataFrame
    reference_id: str | None
    reference_folds: pd.DataFrame


@dataclass(frozen=True)
class Artifact:
    candidate: str
    kind: str
    path: Path
    description: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=24, help="Frames per GIF animation.")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for GIF output.")
    parser.add_argument("--top-n", type=int, default=10, help="Candidates shown in leaderboard/frontier charts.")
    args = parser.parse_args()

    if args.frames < 8:
        raise ValueError("--frames must be at least 8 for readable animations")
    if args.fps < 1:
        raise ValueError("--fps must be positive")

    configure_matplotlib()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    artifacts: list[Artifact] = []
    for spec in candidate_specs():
        data = load_candidate_data(spec)
        candidate_dir = OUTPUT_ROOT / spec.slug
        charts_dir = candidate_dir / "charts"
        animations_dir = candidate_dir / "animations"
        animations_3d_dir = candidate_dir / "animations-3d"
        for directory in (charts_dir, animations_dir, animations_3d_dir):
            directory.mkdir(parents=True, exist_ok=True)

        artifacts.extend(generate_static_charts(data, charts_dir, top_n=args.top_n))
        artifacts.extend(generate_2d_animations(data, animations_dir, top_n=args.top_n, frames=args.frames, fps=args.fps))
        artifacts.extend(generate_3d_animations(data, animations_3d_dir, top_n=args.top_n, frames=args.frames, fps=args.fps))

    write_manifest(artifacts, args)
    write_gallery_index(artifacts)

    counts = count_artifacts(artifacts)
    for slug, values in counts.items():
        print(
            f"{slug}: {values['chart']} charts, "
            f"{values['animation']} 2D animations, "
            f"{values['animation-3d']} 3D animations"
        )
    print(f"Wrote {OUTPUT_ROOT.relative_to(PROJECT_ROOT)}")


def configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 135,
            "savefig.dpi": 160,
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.24,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def candidate_specs() -> tuple[CandidateSpec, ...]:
    return (
        CandidateSpec(
            slug="batch-mean-std-fixed-blend",
            display_name="Batch Mean/Std Fixed Blend",
            selected_id="fixed_blend_0_w0p75_fixed_blend",
            reference_ids=("strong_oof_ridge_stack", "conservative_rls_prediction", "tabm_prediction"),
            thesis="Target-free batch mean/std context on prediction space lifted the Stage 3 local score.",
            reports=(
                ReportSpec(
                    label="Stage 3 local",
                    regime="Recent Stage 3",
                    summary_csv=BEST_CANDIDATES_DIR
                    / "batch_mean_std_fixed_blend/artifacts/strong_oof_subset_s23aux8_s17_gateway_batch_mean_std_stage3_narrow_v1/candidate_summary.csv",
                    fold_csv=BEST_CANDIDATES_DIR
                    / "batch_mean_std_fixed_blend/artifacts/strong_oof_subset_s23aux8_s17_gateway_batch_mean_std_stage3_narrow_v1/fold_scores.csv",
                ),
                ReportSpec(
                    label="Historical max1398",
                    regime="Historical max1398",
                    summary_csv=BEST_CANDIDATES_DIR
                    / "batch_mean_std_fixed_blend/artifacts/strong_oof_hist_max1398_s23aux8_s17_gateway_batch_mean_std_exact_v1/candidate_summary.csv",
                    fold_csv=BEST_CANDIDATES_DIR
                    / "batch_mean_std_fixed_blend/artifacts/strong_oof_hist_max1398_s23aux8_s17_gateway_batch_mean_std_exact_v1/fold_scores.csv",
                ),
            ),
        ),
        CandidateSpec(
            slug="conservative-dynamic-gateway-rls",
            display_name="Conservative Dynamic Gateway RLS",
            selected_id="dynamic_gateway_rls_experts_alpha10000_f0p995",
            reference_ids=("tabm_tree_convex_walk_forward", "dynamic_gateway_rls_experts_alpha10000_f1"),
            thesis="A causal daily RLS gateway with forgetting provides a robust conservative reference.",
            reports=(
                ReportSpec(
                    label="Stage 3 local",
                    regime="Recent Stage 3",
                    summary_csv=BEST_CANDIDATES_DIR
                    / "conservative_dynamic_gateway_rls/validation/dynamic_gateway_rls_stage3/dynamic_gateway_rls_summary.csv",
                    fold_csv=BEST_CANDIDATES_DIR
                    / "conservative_dynamic_gateway_rls/validation/dynamic_gateway_rls_stage3/dynamic_gateway_rls_by_fold.csv",
                ),
                ReportSpec(
                    label="Historical max1398",
                    regime="Historical max1398",
                    summary_csv=BEST_CANDIDATES_DIR
                    / "conservative_dynamic_gateway_rls/validation/dynamic_gateway_rls_hist_max1398/dynamic_gateway_rls_summary.csv",
                    fold_csv=BEST_CANDIDATES_DIR
                    / "conservative_dynamic_gateway_rls/validation/dynamic_gateway_rls_hist_max1398/dynamic_gateway_rls_by_fold.csv",
                ),
            ),
        ),
        CandidateSpec(
            slug="historical-residual-tail",
            display_name="Historical Residual-Tail Risk Gate",
            selected_id="gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p95_residual_tail",
            reference_ids=("gateway_risk_conservative_rls_abs_pred_s100_prediction", "conservative_rls_prediction"),
            thesis="Residual-tail conditioning is the strongest preserved full historical reference.",
            reports=(
                ReportSpec(
                    label="Historical max1398",
                    regime="Historical max1398",
                    summary_csv=BEST_CANDIDATES_DIR
                    / "historical_residual_tail/artifacts/strong_oof_hist_max1398_gateway_residual_tail_modes_v1/candidate_summary.csv",
                    fold_csv=BEST_CANDIDATES_DIR
                    / "historical_residual_tail/artifacts/strong_oof_hist_max1398_gateway_residual_tail_modes_v1/fold_scores.csv",
                ),
            ),
        ),
    )


def load_candidate_data(spec: CandidateSpec) -> CandidateData:
    summary_frames = []
    fold_frames = []
    for report_index, report in enumerate(spec.reports):
        summary_frames.append(normalize_summary(report, report_index))
        fold_frames.append(normalize_folds(report, report_index))

    summary = pd.concat(summary_frames, ignore_index=True)
    folds = pd.concat(fold_frames, ignore_index=True)
    primary_report = spec.reports[0].label
    primary_summary = summary[summary["report"] == primary_report].copy()
    primary_folds = folds[folds["report"] == primary_report].copy()

    selected_summary = summary[summary["candidate_id"] == spec.selected_id].copy()
    selected_folds = folds[(folds["candidate_id"] == spec.selected_id) & (folds["report"] == primary_report)].copy()
    if selected_summary.empty:
        raise ValueError(f"Selected candidate is absent from summaries: {spec.selected_id}")
    if selected_folds.empty:
        fallback_report = selected_summary.sort_values("report_order").iloc[0]["report"]
        selected_folds = folds[(folds["candidate_id"] == spec.selected_id) & (folds["report"] == fallback_report)].copy()
        primary_report = str(fallback_report)
        primary_summary = summary[summary["report"] == primary_report].copy()
        primary_folds = folds[folds["report"] == primary_report].copy()
    if selected_folds.empty:
        raise ValueError(f"Selected candidate is absent from fold scores: {spec.selected_id}")

    reference_id = choose_reference_id(spec, primary_summary)
    if reference_id is None:
        reference_folds = pd.DataFrame(columns=primary_folds.columns)
    else:
        reference_folds = primary_folds[primary_folds["candidate_id"] == reference_id].copy()

    return CandidateData(
        spec=spec,
        summary=summary,
        folds=folds,
        primary_report=primary_report,
        selected_summary=selected_summary,
        selected_folds=selected_folds.sort_values("fold_index"),
        primary_summary=primary_summary,
        primary_folds=primary_folds,
        reference_id=reference_id,
        reference_folds=reference_folds.sort_values("fold_index"),
    )


def normalize_summary(report: ReportSpec, report_index: int) -> pd.DataFrame:
    require_file(report.summary_csv)
    frame = pd.read_csv(report.summary_csv)
    id_col = "candidate" if "candidate" in frame.columns else "strategy"
    family_col = "family" if "family" in frame.columns else "method_family"
    required = {id_col, family_col, "rows", "weight_sum", "numerator", "denominator", "global_r2", "mean_fold_r2", "min_fold_r2", "std_fold_r2"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{report.summary_csv} is missing columns: {sorted(missing)}")

    normalized = frame.copy()
    normalized["candidate_id"] = normalized[id_col].astype(str)
    normalized["family"] = normalized[family_col].astype(str)
    normalized["report"] = report.label
    normalized["regime"] = report.regime
    normalized["report_order"] = report_index
    for col in ("rows", "weight_sum", "numerator", "denominator", "global_r2", "mean_fold_r2", "min_fold_r2", "std_fold_r2"):
        normalized[col] = pd.to_numeric(normalized[col], errors="raise")
    normalized["gap_to_target"] = TARGET_R2 - normalized["global_r2"]
    normalized["explained_weighted_error"] = normalized["denominator"] - normalized["numerator"]
    normalized["short_name"] = normalized["candidate_id"].map(short_name)
    return normalized


def normalize_folds(report: ReportSpec, report_index: int) -> pd.DataFrame:
    require_file(report.fold_csv)
    frame = pd.read_csv(report.fold_csv)
    id_col = "candidate" if "candidate" in frame.columns else "strategy"
    family_col = "family" if "family" in frame.columns else "method_family"
    required = {id_col, family_col, "fold", "rows", "weight_sum", "numerator", "denominator", "weighted_zero_mean_r2"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{report.fold_csv} is missing columns: {sorted(missing)}")

    normalized = frame.copy()
    normalized["candidate_id"] = normalized[id_col].astype(str)
    normalized["family"] = normalized[family_col].astype(str)
    normalized["report"] = report.label
    normalized["regime"] = report.regime
    normalized["report_order"] = report_index
    normalized["fold_index"] = normalized["fold"].map(fold_index)
    normalized["r2"] = pd.to_numeric(normalized["weighted_zero_mean_r2"], errors="raise")
    for col in ("rows", "weight_sum", "numerator", "denominator"):
        normalized[col] = pd.to_numeric(normalized[col], errors="raise")
    normalized["explained_weighted_error"] = normalized["denominator"] - normalized["numerator"]
    normalized["short_name"] = normalized["candidate_id"].map(short_name)
    return normalized


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise ValueError(f"Empty required source file: {path}")


def fold_index(value: object) -> int:
    text = str(value)
    match = re.search(r"(\d+)", text)
    if not match:
        return 0
    return int(match.group(1))


def choose_reference_id(spec: CandidateSpec, summary: pd.DataFrame) -> str | None:
    available = set(summary["candidate_id"])
    for candidate_id in spec.reference_ids:
        if candidate_id in available:
            return candidate_id
    non_selected = summary[summary["candidate_id"] != spec.selected_id].sort_values("global_r2", ascending=False)
    if non_selected.empty:
        return None
    return str(non_selected.iloc[0]["candidate_id"])


def short_name(candidate_id: object, max_len: int = 34) -> str:
    text = str(candidate_id)
    if text in SHORT_NAMES:
        return SHORT_NAMES[text]
    cleaned = (
        text.replace("gateway_risk_", "")
        .replace("_prediction", "")
        .replace("_candidate", "")
        .replace("dynamic_gateway_rls_", "dynamic ")
        .replace("_", " ")
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip().title()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rstrip() + "."


def wrapped(text: str, width: int = 24) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def generate_static_charts(data: CandidateData, directory: Path, *, top_n: int) -> list[Artifact]:
    artifacts = [
        plot_fold_r2_term_structure(data, directory / "01_fold_r2_term_structure.png"),
        plot_candidate_leaderboard(data, directory / "02_candidate_r2_leaderboard.png", top_n=top_n),
        plot_risk_return_frontier(data, directory / "03_risk_return_frontier.png", top_n=top_n),
        plot_weighted_error_budget(data, directory / "04_weighted_error_budget.png"),
        plot_fold_dynamics_phase_map(data, directory / "05_fold_dynamics_phase_map.png"),
    ]
    return artifacts


def generate_2d_animations(data: CandidateData, directory: Path, *, top_n: int, frames: int, fps: int) -> list[Artifact]:
    artifacts = [
        animate_fold_r2_progression(data, directory / "01_fold_r2_progression.gif", frames=frames, fps=fps),
        animate_leaderboard_reveal(data, directory / "02_leaderboard_reveal.gif", top_n=top_n, frames=frames, fps=fps),
        animate_gap_to_target(data, directory / "03_gap_to_target_walk.gif", frames=frames, fps=fps),
        animate_stability_frontier(data, directory / "04_stability_frontier_scan.gif", top_n=top_n, frames=frames, fps=fps),
        animate_phase_space(data, directory / "05_phase_space_flow.gif", frames=frames, fps=fps),
    ]
    return artifacts


def generate_3d_animations(data: CandidateData, directory: Path, *, top_n: int, frames: int, fps: int) -> list[Artifact]:
    artifacts = [
        animate_3d_fold_surface(data, directory / "01_3d_fold_surface.gif", top_n=top_n, frames=frames, fps=fps),
        animate_3d_risk_return_frontier(data, directory / "02_3d_risk_return_frontier.gif", top_n=top_n, frames=frames, fps=fps),
        animate_3d_error_energy_waterfall(data, directory / "03_3d_error_energy_waterfall.gif", frames=frames, fps=fps),
        animate_3d_candidate_orbit(data, directory / "04_3d_candidate_orbit.gif", frames=frames, fps=fps),
        animate_3d_nonlinear_stability_manifold(data, directory / "05_3d_nonlinear_stability_manifold.gif", frames=frames, fps=fps),
    ]
    return artifacts


def plot_fold_r2_term_structure(data: CandidateData, path: Path) -> Artifact:
    selected = data.selected_folds
    reference = data.reference_folds
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(selected["fold"], selected["r2"], marker="o", linewidth=2.4, color=COLORS["selected"], label=short_name(data.spec.selected_id))
    if not reference.empty:
        ax.plot(reference["fold"], reference["r2"], marker="s", linewidth=1.8, color=COLORS["reference"], label=short_name(data.reference_id))
    ax.axhline(TARGET_R2, color=COLORS["target"], linestyle="--", linewidth=1.4, label="R2 target = 0.020")
    ax.axhline(0.0, color=COLORS["gray"], linewidth=0.8)
    ax.set_title(f"{data.spec.display_name}: Fold R2 Term Structure")
    ax.set_xlabel("Rolling validation fold")
    ax.set_ylabel("Weighted zero-mean R2")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.3f}")
    ax.legend(loc="best")
    add_footer(fig, data)
    save_static(fig, path)
    return Artifact(data.spec.slug, "chart", path, "Fold-level R2 profile versus the selected reference and the 0.020 target.")


def plot_candidate_leaderboard(data: CandidateData, path: Path, *, top_n: int) -> Artifact:
    frame = top_candidates(data.primary_summary, top_n=top_n).sort_values("global_r2", ascending=True).reset_index(drop=True)
    colors = [COLORS["selected"] if candidate == data.spec.selected_id else FAMILY_COLORS.get(family, COLORS["gray"]) for candidate, family in zip(frame["candidate_id"], frame["family"])]
    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(frame))
    ax.barh(y_pos, frame["global_r2"], color=colors, alpha=0.92)
    ax.set_yticks(y_pos, frame["short_name"].map(lambda name: wrapped(name, 28)))
    ax.axvline(TARGET_R2, color=COLORS["target"], linestyle="--", linewidth=1.3, label="R2 target = 0.020")
    ax.set_title(f"{data.spec.display_name}: Candidate Leaderboard")
    ax.set_xlabel("Global weighted zero-mean R2")
    ax.set_ylabel("Candidate")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.3f}")
    selected_value = float(data.primary_summary.loc[data.primary_summary["candidate_id"] == data.spec.selected_id, "global_r2"].iloc[0])
    selected_positions = frame.index[frame["candidate_id"] == data.spec.selected_id].to_list()
    if selected_positions:
        ax.text(selected_value, selected_positions[0], f"  {selected_value:.6f}", va="center", color=COLORS["selected"], fontweight="bold")
    ax.legend(loc="lower right")
    add_footer(fig, data)
    save_static(fig, path)
    return Artifact(data.spec.slug, "chart", path, "Primary-report global R2 leaderboard with the selected candidate highlighted.")


def plot_risk_return_frontier(data: CandidateData, path: Path, *, top_n: int) -> Artifact:
    frame = top_candidates(data.primary_summary, top_n=top_n)
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    sizes = normalize_to_range(frame["rows"].to_numpy(dtype=float), 80, 360)
    for family, group in frame.groupby("family"):
        ax.scatter(
            group["global_r2"],
            group["min_fold_r2"],
            s=sizes[group.index.to_numpy() - frame.index.min()] if frame.index.is_monotonic_increasing else 180,
            color=FAMILY_COLORS.get(family, COLORS["gray"]),
            alpha=0.78,
            label=family.replace("_", " "),
            edgecolor="white",
            linewidth=0.7,
        )
    selected = data.primary_summary[data.primary_summary["candidate_id"] == data.spec.selected_id].iloc[0]
    ax.scatter([selected["global_r2"]], [selected["min_fold_r2"]], marker="*", s=520, color=COLORS["selected"], edgecolor="black", linewidth=0.7, label="selected candidate")
    ax.axvline(TARGET_R2, color=COLORS["target"], linestyle="--", linewidth=1.2)
    ax.axhline(0.0, color=COLORS["gray"], linewidth=0.8)
    ax.set_title(f"{data.spec.display_name}: Risk/Return Frontier")
    ax.set_xlabel("Global R2")
    ax.set_ylabel("Worst-fold R2")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.3f}")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.3f}")
    ax.legend(loc="best", ncols=2)
    add_footer(fig, data)
    save_static(fig, path)
    return Artifact(data.spec.slug, "chart", path, "Global R2 versus worst-fold R2 for the candidate family.")


def plot_weighted_error_budget(data: CandidateData, path: Path) -> Artifact:
    frame = data.selected_folds.copy()
    x = np.arange(len(frame))
    target_energy = frame["denominator"].to_numpy(dtype=float) / MILLION
    sse = frame["numerator"].to_numpy(dtype=float) / MILLION
    explained = frame["explained_weighted_error"].to_numpy(dtype=float) / MILLION

    fig, ax = plt.subplots(figsize=(9, 5.4))
    width = 0.34
    ax.bar(x - width / 2, target_energy, width=width, color=COLORS["light_gray"], edgecolor=COLORS["gray"], label="Target energy denominator")
    ax.bar(x + width / 2, sse, width=width, color=COLORS["amber"], alpha=0.88, label="Weighted squared error numerator")
    ax2 = ax.twinx()
    ax2.plot(x, frame["r2"], marker="o", color=COLORS["selected"], linewidth=2.2, label="Fold R2")
    ax2.axhline(TARGET_R2, color=COLORS["target"], linestyle="--", linewidth=1.0)
    for i, value in enumerate(explained):
        ax.text(i, max(target_energy[i], sse[i]) * 1.006, f"+{value:.2f}M", ha="center", va="bottom", fontsize=8, color=COLORS["green"])
    ax.set_title(f"{data.spec.display_name}: Weighted Error Budget")
    ax.set_xlabel("Rolling validation fold")
    ax.set_ylabel("Weighted sum, millions")
    ax2.set_ylabel("Fold R2")
    ax.set_xticks(x, frame["fold"])
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    add_footer(fig, data)
    save_static(fig, path)
    return Artifact(data.spec.slug, "chart", path, "Fold-level numerator, denominator, and explained weighted error budget.")


def plot_fold_dynamics_phase_map(data: CandidateData, path: Path) -> Artifact:
    r2 = data.selected_folds["r2"].to_numpy(dtype=float)
    x, y = phase_pairs(r2)
    expansion = lyapunov_style_expansion(r2)
    phase_error = float(np.sqrt(np.mean((y - x) ** 2))) if len(x) else 0.0
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    diagonal = np.linspace(0.0, 1.0, 100)
    ax.plot(diagonal, diagonal, color=COLORS["gray"], linewidth=1.2, alpha=0.7, label="No fold-to-fold change")
    ax.fill_between(diagonal, diagonal, 1.0, color=COLORS["green"], alpha=0.08, label="Improvement region")
    ax.fill_between(diagonal, 0.0, diagonal, color=COLORS["reference"], alpha=0.06, label="Deterioration region")
    ax.scatter(x, y, s=150, c=np.abs(y - x), cmap="viridis", edgecolor="white", linewidth=0.8, zorder=4, label="Observed transitions")
    for i, (x_value, y_value) in enumerate(zip(x, y), start=1):
        ax.annotate(f"rw_{i:02d}->rw_{i + 1:02d}", (x_value, y_value), xytext=(6, 5), textcoords="offset points", fontsize=8)
    ax.plot(x, y, color=COLORS["selected"], linewidth=1.4, alpha=0.8)
    ax.set_title(f"{data.spec.display_name}: Nonlinear Fold-Stability Phase Map")
    ax.set_xlabel("Normalized R2 at fold t")
    ax.set_ylabel("Normalized R2 at fold t+1")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.text(
        0.01,
        0.02,
        f"Fold-state RMSE={phase_error:.3f}; Lyapunov-style expansion proxy={expansion:.3f}.",
        transform=ax.transAxes,
        fontsize=8,
        color=COLORS["gray"],
    )
    ax.legend(loc="upper right")
    add_footer(fig, data)
    save_static(fig, path)
    return Artifact(data.spec.slug, "chart", path, "Nonlinear phase-space diagnostic of fold-to-fold R2 stability.")


def animate_fold_r2_progression(data: CandidateData, path: Path, *, frames: int, fps: int) -> Artifact:
    selected = data.selected_folds
    reference = data.reference_folds
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y_min, y_max = r2_limits(pd.concat([selected, reference], ignore_index=True) if not reference.empty else selected)
    selected_x = np.arange(len(selected), dtype=float)
    selected_y = selected["r2"].to_numpy(dtype=float)
    reference_x = np.arange(len(reference), dtype=float)
    reference_y = reference["r2"].to_numpy(dtype=float) if not reference.empty else np.array([], dtype=float)

    def update(frame_index: int) -> None:
        ax.clear()
        progress = frame_index / max(frames - 1, 1)
        x_vis, y_vis = interpolated_path(selected_x, selected_y, progress)
        ax.plot(x_vis, y_vis, marker="o", color=COLORS["selected"], linewidth=2.5, label=short_name(data.spec.selected_id))
        if not reference.empty:
            rx_vis, ry_vis = interpolated_path(reference_x, reference_y, progress)
            ax.plot(rx_vis, ry_vis, marker="s", color=COLORS["reference"], linewidth=1.8, label=short_name(data.reference_id))
        ax.axhline(TARGET_R2, color=COLORS["target"], linestyle="--", linewidth=1.2, label="R2 target = 0.020")
        ax.axvline(progress * max(len(selected) - 1, 1), color=COLORS["gray"], linewidth=0.8, alpha=0.45)
        ax.set_xticks(selected_x, selected["fold"].to_list())
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(-0.15, max(len(selected) - 1, 1) + 0.15)
        ax.set_title(f"{data.spec.display_name}: Animated Fold R2 Progression")
        ax.set_xlabel("Rolling validation fold")
        ax.set_ylabel("Weighted zero-mean R2")
        ax.legend(loc="best")

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps, repeat=True)
    save_animation(anim, fig, path, fps=fps)
    return Artifact(data.spec.slug, "animation", path, "Animated reveal of fold-level R2 for selected and reference candidates.")


def animate_leaderboard_reveal(data: CandidateData, path: Path, *, top_n: int, frames: int, fps: int) -> Artifact:
    frame = top_candidates(data.primary_summary, top_n=top_n).sort_values("global_r2", ascending=True).reset_index(drop=True)
    final = frame["global_r2"].to_numpy(dtype=float)
    labels = frame["short_name"].map(lambda name: wrapped(name, 26)).to_list()
    colors = [COLORS["selected"] if candidate == data.spec.selected_id else FAMILY_COLORS.get(family, COLORS["gray"]) for candidate, family in zip(frame["candidate_id"], frame["family"])]
    fig, ax = plt.subplots(figsize=(9, 5.8))

    def update(frame_index: int) -> None:
        ax.clear()
        scale = smoothstep(frame_index / max(frames - 1, 1))
        ax.barh(labels, final * scale, color=colors, alpha=0.94)
        ax.axvline(TARGET_R2, color=COLORS["target"], linestyle="--", linewidth=1.2, label="R2 target = 0.020")
        ax.set_xlim(0, max(TARGET_R2 * 1.06, final.max() * 1.15))
        ax.set_title(f"{data.spec.display_name}: Animated R2 Leaderboard")
        ax.set_xlabel("Global weighted zero-mean R2")
        ax.legend(loc="lower right")

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps, repeat=True)
    save_animation(anim, fig, path, fps=fps)
    return Artifact(data.spec.slug, "animation", path, "Animated candidate leaderboard growing from zero to realized global R2.")


def animate_gap_to_target(data: CandidateData, path: Path, *, frames: int, fps: int) -> Artifact:
    points = build_progress_points(data)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x_full = points["step"].to_numpy(dtype=float)
    y_full = points["r2"].to_numpy(dtype=float)

    def update(frame_index: int) -> None:
        ax.clear()
        progress = frame_index / max(frames - 1, 1)
        x_vis, y_vis = interpolated_path(x_full, y_full, progress)
        ax.plot(x_vis, y_vis, marker="o", linewidth=2.2, color=COLORS["selected"], label="Selected candidate")
        ax.fill_between(x_vis, y_vis, TARGET_R2, color=COLORS["selected"], alpha=0.12)
        ax.axvline(progress * max(len(points) - 1, 1), color=COLORS["gray"], linewidth=0.8, alpha=0.45)
        ax.axhline(TARGET_R2, color=COLORS["target"], linestyle="--", linewidth=1.2, label="R2 target = 0.020")
        ax.set_ylim(*r2_limits_from_values(np.r_[points["r2"].to_numpy(dtype=float), TARGET_R2]))
        ax.set_xlim(-0.15, max(len(points) - 1, 1) + 0.15)
        ax.set_title(f"{data.spec.display_name}: Gap to 0.020 R2")
        ax.set_xlabel("Evidence step")
        ax.set_ylabel("R2")
        ax.set_xticks(points["step"], points["label"], rotation=20, ha="right")
        ax.legend(loc="best")

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps, repeat=True)
    save_animation(anim, fig, path, fps=fps)
    return Artifact(data.spec.slug, "animation", path, "Animated path showing the remaining gap from observed R2 to the 0.020 target.")


def animate_stability_frontier(data: CandidateData, path: Path, *, top_n: int, frames: int, fps: int) -> Artifact:
    frame = top_candidates(data.primary_summary, top_n=top_n).sort_values("global_r2", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    x_lim = (max(0, frame["global_r2"].min() - 0.0005), max(TARGET_R2 * 1.03, frame["global_r2"].max() + 0.0008))
    y_lim = (min(0, frame["min_fold_r2"].min() - 0.0005), max(TARGET_R2 * 0.75, frame["min_fold_r2"].max() + 0.0008))
    scan_pad = (x_lim[1] - x_lim[0]) * 0.20

    def update(frame_index: int) -> None:
        ax.clear()
        progress = frame_index / max(frames - 1, 1)
        scan_x = x_lim[0] + progress * (x_lim[1] - x_lim[0])
        for _, row in frame.iterrows():
            activation = np.clip((scan_x - row["global_r2"] + scan_pad) / scan_pad, 0.12, 1.0)
            color = COLORS["selected"] if row["candidate_id"] == data.spec.selected_id else FAMILY_COLORS.get(row["family"], COLORS["gray"])
            ax.scatter(row["global_r2"], row["min_fold_r2"], s=60 + 150 * activation, color=color, edgecolor="white", linewidth=0.8, alpha=activation)
            if activation > 0.78 or row["candidate_id"] == data.spec.selected_id:
                ax.annotate(short_name(row["candidate_id"], max_len=22), (row["global_r2"], row["min_fold_r2"]), xytext=(5, 4), textcoords="offset points", fontsize=8, alpha=activation)
        ax.axvline(scan_x, color=COLORS["purple"], linewidth=1.1, alpha=0.75, label="moving R2 scan")
        ax.axvline(TARGET_R2, color=COLORS["target"], linestyle="--", linewidth=1.1)
        ax.axhline(0.0, color=COLORS["gray"], linewidth=0.8)
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_title(f"{data.spec.display_name}: Animated Stability Frontier")
        ax.set_xlabel("Global R2")
        ax.set_ylabel("Worst-fold R2")

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps, repeat=True)
    save_animation(anim, fig, path, fps=fps)
    return Artifact(data.spec.slug, "animation", path, "Animated scan of global R2 versus worst-fold R2 stability.")


def animate_phase_space(data: CandidateData, path: Path, *, frames: int, fps: int) -> Artifact:
    r2 = data.selected_folds["r2"].to_numpy(dtype=float)
    x, y = phase_pairs(r2)
    diagonal = np.linspace(0.0, 1.0, 100)
    expansion = lyapunov_style_expansion(r2)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    def update(frame_index: int) -> None:
        ax.clear()
        progress = frame_index / max(frames - 1, 1)
        x_vis, y_vis = interpolated_path(x, y, progress)
        ax.plot(diagonal, diagonal, color=COLORS["gray"], alpha=0.7, linewidth=1.2, label="No fold-to-fold change")
        ax.fill_between(diagonal, diagonal, 1.0, color=COLORS["green"], alpha=0.08)
        ax.fill_between(diagonal, 0.0, diagonal, color=COLORS["reference"], alpha=0.06)
        ax.scatter(
            x_vis,
            y_vis,
            s=145,
            c=np.abs(y_vis - x_vis),
            cmap="viridis",
            edgecolor="white",
            linewidth=0.8,
            label="Fold transitions",
        )
        if len(x_vis) > 1:
            ax.plot(x_vis, y_vis, color=COLORS["selected"], linewidth=1.4)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"{data.spec.display_name}: Animated Nonlinear Stability Flow")
        ax.set_xlabel("Normalized R2 at fold t")
        ax.set_ylabel("Normalized R2 at fold t+1")
        ax.legend(loc="upper right")
        ax.text(0.01, 0.02, f"Expansion proxy={expansion:.3f}; color = absolute fold-state jump.", transform=ax.transAxes, fontsize=8, color=COLORS["gray"])

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps, repeat=True)
    save_animation(anim, fig, path, fps=fps)
    return Artifact(data.spec.slug, "animation", path, "Animated nonlinear stability flow of fold-to-fold R2 transitions.")


def animate_3d_fold_surface(data: CandidateData, path: Path, *, top_n: int, frames: int, fps: int) -> Artifact:
    top = top_candidates(data.primary_summary, top_n=top_n)
    candidate_ids = top["candidate_id"].to_list()
    fold_frame = data.primary_folds[data.primary_folds["candidate_id"].isin(candidate_ids)].copy()
    pivot = fold_frame.pivot_table(index="candidate_id", columns="fold_index", values="r2", aggfunc="mean")
    pivot = pivot.reindex(candidate_ids).dropna(how="all")
    z = pivot.to_numpy(dtype=float)
    x_values = np.arange(z.shape[1])
    y_values = np.arange(z.shape[0])
    x, y = np.meshgrid(x_values, y_values)

    fig = plt.figure(figsize=(8.4, 6))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_index: int) -> None:
        ax.clear()
        progress = smoothstep(frame_index / max(frames - 1, 1))
        activation = np.clip(progress * (z.shape[1] + 0.75) - x, 0.0, 1.0)
        z_progress = np.where(activation > 0.02, z * activation, np.nan)
        ax.plot_surface(x, y, z_progress, cmap="viridis", edgecolor="white", linewidth=0.25, alpha=0.93)
        ax.contour(x, y, z, zdir="z", offset=np.nanmin(z) - 0.001, levels=8, cmap="viridis", linewidths=0.65, alpha=0.55)
        ax.set_title(f"{data.spec.display_name}: 3D Fold R2 Surface Completion")
        ax.set_xlabel("Fold index")
        ax.set_ylabel("Candidate")
        ax.set_zlabel("R2")
        ax.set_xticks(x_values, [str(col) for col in pivot.columns])
        ax.set_yticks(y_values, [wrapped(short_name(idx), 18) for idx in pivot.index], fontsize=7)
        ax.text2D(0.02, 0.03, f"Completion: {progress:0.0%} of fold surface revealed left to right.", transform=ax.transAxes, fontsize=8, color=COLORS["gray"])
        ax.view_init(elev=28, azim=42)
        set_3d_r2_limits(ax, z)

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps, repeat=True)
    save_animation(anim, fig, path, fps=fps)
    return Artifact(data.spec.slug, "animation-3d", path, "Fixed-camera 3D surface that progressively completes fold R2 across top candidates.")


def animate_3d_risk_return_frontier(data: CandidateData, path: Path, *, top_n: int, frames: int, fps: int) -> Artifact:
    frame = top_candidates(data.primary_summary, top_n=top_n).reset_index(drop=True)
    fig = plt.figure(figsize=(8, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    colors = [COLORS["selected"] if candidate == data.spec.selected_id else FAMILY_COLORS.get(family, COLORS["gray"]) for candidate, family in zip(frame["candidate_id"], frame["family"])]
    x_values = frame["global_r2"].to_numpy(dtype=float)
    y_values = frame["min_fold_r2"].to_numpy(dtype=float)
    z_values = frame["std_fold_r2"].to_numpy(dtype=float)
    point_order = np.linspace(0.0, 1.0, len(frame))

    def update(frame_index: int) -> None:
        ax.clear()
        progress = smoothstep(frame_index / max(frames - 1, 1))
        activation = np.clip((progress - point_order) * max(len(frame), 1), 0.0, 1.0)
        z_current = z_values * activation
        sizes = 40 + 110 * activation
        ax.scatter(x_values, y_values, z_current, s=sizes, c=colors, edgecolor="white", linewidth=0.6, alpha=0.9)
        for x_val, y_val, z_val, active in zip(x_values, y_values, z_current, activation):
            if active > 0.02:
                ax.plot([x_val, x_val], [y_val, y_val], [0.0, z_val], color=COLORS["gray"], alpha=0.22, linewidth=0.8)
        selected_mask = frame["candidate_id"] == data.spec.selected_id
        if selected_mask.any():
            selected_index = int(np.flatnonzero(selected_mask.to_numpy())[0])
            selected_activation = activation[selected_index]
            if selected_activation > 0.02:
                ax.scatter(
                    [x_values[selected_index]],
                    [y_values[selected_index]],
                    [z_current[selected_index]],
                    s=170 + 220 * selected_activation,
                    c=COLORS["selected"],
                    marker="*",
                    edgecolor="black",
                    linewidth=0.8,
                )
        ax.set_title(f"{data.spec.display_name}: 3D Risk/Return Frontier Completion")
        ax.set_xlabel("Global R2")
        ax.set_ylabel("Worst-fold R2")
        ax.set_zlabel("Fold R2 volatility")
        ax.text2D(0.02, 0.03, f"Completion: {progress:0.0%} of candidate frontier populated.", transform=ax.transAxes, fontsize=8, color=COLORS["gray"])
        ax.view_init(elev=24, azim=36)
        ax.set_zlim(0.0, max(z_values.max() * 1.18, 0.001))

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps, repeat=True)
    save_animation(anim, fig, path, fps=fps)
    return Artifact(data.spec.slug, "animation-3d", path, "Fixed-camera 3D risk-return-volatility frontier populated candidate by candidate.")


def animate_3d_error_energy_waterfall(data: CandidateData, path: Path, *, frames: int, fps: int) -> Artifact:
    frame = data.selected_folds.copy()
    x = np.arange(len(frame))
    y = np.array([0, 1, 2], dtype=float)
    labels = ["Target energy", "Weighted SSE", "Explained gap"]
    values = np.vstack(
        [
            frame["denominator"].to_numpy(dtype=float) / MILLION,
            frame["numerator"].to_numpy(dtype=float) / MILLION,
            frame["explained_weighted_error"].to_numpy(dtype=float) / MILLION,
        ]
    )
    fig = plt.figure(figsize=(8.4, 5.8))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_index: int) -> None:
        ax.clear()
        progress = smoothstep(frame_index / max(frames - 1, 1))
        fold_order = np.linspace(0.0, 1.0, len(frame))
        for j, color in enumerate((COLORS["light_gray"], COLORS["amber"], COLORS["green"])):
            metric_offset = j / (len(labels) + 1)
            activation = np.clip((progress - 0.18 * metric_offset - fold_order * 0.08) * 1.35, 0.0, 1.0)
            ax.bar3d(
                x - 0.22,
                np.full_like(x, y[j]) - 0.22,
                np.zeros_like(x, dtype=float),
                0.44,
                0.44,
                values[j] * activation,
                color=color,
                edgecolor="white",
                linewidth=0.25,
                alpha=0.9,
            )
        ax.set_title(f"{data.spec.display_name}: 3D Weighted Error Budget Completion")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Metric")
        ax.set_zlabel("Weighted sum, millions")
        ax.set_xticks(x, frame["fold"].to_list())
        ax.set_yticks(y, labels)
        ax.text2D(0.02, 0.03, f"Completion: {progress:0.0%}; bars grow by metric and fold.", transform=ax.transAxes, fontsize=8, color=COLORS["gray"])
        ax.view_init(elev=27, azim=44)
        ax.set_zlim(0.0, max(values.max() * 1.15, 0.001))

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps, repeat=True)
    save_animation(anim, fig, path, fps=fps)
    return Artifact(data.spec.slug, "animation-3d", path, "3D waterfall of denominator, numerator, and explained error gap by fold.")


def animate_3d_candidate_orbit(data: CandidateData, path: Path, *, frames: int, fps: int) -> Artifact:
    points = build_orbit_points(data)
    fig = plt.figure(figsize=(8, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    x_values = points["report_order"].to_numpy(dtype=float)
    y_values = points["fold_index"].to_numpy(dtype=float)
    z_values = points["r2"].to_numpy(dtype=float)

    def update(frame_index: int) -> None:
        ax.clear()
        progress = frame_index / max(frames - 1, 1)
        x_vis, y_vis, z_vis = interpolated_path_3d(x_values, y_values, z_values, progress)
        ax.plot(x_vis, y_vis, z_vis, color=COLORS["selected"], linewidth=2.0)
        ax.scatter(x_vis, y_vis, z_vis, s=95, c=np.linspace(0, 1, len(x_vis)), cmap="plasma", edgecolor="white", linewidth=0.5)
        ax.axhline(0, color=COLORS["gray"], linewidth=0.6)
        ax.set_title(f"{data.spec.display_name}: 3D Candidate Path Completion")
        ax.set_xlabel("Report regime")
        ax.set_ylabel("Fold index")
        ax.set_zlabel("Fold R2")
        ax.set_xticks(sorted(points["report_order"].unique()), [wrapped(label, 16) for label in points.drop_duplicates("report_order").sort_values("report_order")["report"]])
        ax.text2D(0.02, 0.03, f"Completion: {smoothstep(progress):0.0%} of the regime/fold trajectory drawn.", transform=ax.transAxes, fontsize=8, color=COLORS["gray"])
        ax.view_init(elev=24, azim=38)
        set_3d_r2_limits(ax, z_values)

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps, repeat=True)
    save_animation(anim, fig, path, fps=fps)
    return Artifact(data.spec.slug, "animation-3d", path, "3D orbit connecting selected-candidate fold scores across available regimes.")


def animate_3d_nonlinear_stability_manifold(data: CandidateData, path: Path, *, frames: int, fps: int) -> Artifact:
    r2 = data.selected_folds["r2"].to_numpy(dtype=float)
    transition_x, transition_y = phase_pairs(r2)
    transition_z = np.abs(transition_y - transition_x)
    grid = np.linspace(0.0, 1.0, 55)
    x_grid, y_grid = np.meshgrid(grid, grid)
    z_grid = np.abs(y_grid - x_grid)
    expansion = lyapunov_style_expansion(r2)

    fig = plt.figure(figsize=(8, 5.8))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_index: int) -> None:
        ax.clear()
        progress = smoothstep(frame_index / max(frames - 1, 1))
        reveal_field = (x_grid + y_grid) / 2.0
        activation = np.clip((progress - reveal_field) * 3.2 + 0.18, 0.0, 1.0)
        z_progress = np.where(activation > 0.02, z_grid * activation, np.nan)
        ax.plot_surface(x_grid, y_grid, z_progress, cmap="magma", alpha=0.58, linewidth=0, antialiased=True)
        x_vis, y_vis, z_vis = interpolated_path_3d(transition_x, transition_y, transition_z, progress)
        ax.scatter(x_vis, y_vis, z_vis, s=130, color=COLORS["cyan"], edgecolor="white", linewidth=0.7, label="Fold transitions")
        if len(x_vis) > 1:
            ax.plot(x_vis, y_vis, z_vis, color=COLORS["cyan"], linewidth=2.0)
        ax.set_title(f"{data.spec.display_name}: 3D Nonlinear Stability Manifold Completion")
        ax.set_xlabel("Normalized state x(t)")
        ax.set_ylabel("Normalized next state x(t+1)")
        ax.set_zlabel("Absolute state jump")
        ax.text2D(0.02, 0.03, f"Completion: {progress:0.0%}; surface=|x(t+1)-x(t)|; expansion proxy={expansion:.3f}.", transform=ax.transAxes, fontsize=8, color=COLORS["gray"])
        ax.view_init(elev=26, azim=38)
        ax.set_zlim(0.0, 1.0)

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps, repeat=True)
    save_animation(anim, fig, path, fps=fps)
    return Artifact(data.spec.slug, "animation-3d", path, "Fixed-camera 3D stability manifold progressively completed with observed fold transitions.")


def top_candidates(summary: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    sorted_frame = summary.sort_values("global_r2", ascending=False).head(top_n).copy()
    return sorted_frame.reset_index(drop=True)


def normalize_to_range(values: np.ndarray, low: float, high: float) -> np.ndarray:
    if len(values) == 0:
        return values
    v_min = float(np.nanmin(values))
    v_max = float(np.nanmax(values))
    if math.isclose(v_min, v_max):
        return np.full_like(values, (low + high) / 2.0, dtype=float)
    return low + (values - v_min) * (high - low) / (v_max - v_min)


def phase_pairs(r2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norm = normalize_series(r2)
    if len(norm) < 2:
        return np.array([norm[0] if len(norm) else 0.5]), np.array([norm[0] if len(norm) else 0.5])
    return norm[:-1], norm[1:]


def normalize_series(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    if math.isclose(v_min, v_max):
        return np.full_like(values, 0.5, dtype=float)
    return np.clip((values - v_min) / (v_max - v_min), 0.02, 0.98)


def lyapunov_style_expansion(r2: np.ndarray) -> float:
    values = np.asarray(r2, dtype=float)
    if values.size < 3:
        return 0.0
    deltas = np.abs(np.diff(values))
    ratios = (deltas[1:] + 1e-9) / (deltas[:-1] + 1e-9)
    return float(np.mean(np.log(ratios)))


def r2_limits(frame: pd.DataFrame) -> tuple[float, float]:
    return r2_limits_from_values(np.r_[frame["r2"].to_numpy(dtype=float), TARGET_R2])


def r2_limits_from_values(values: np.ndarray) -> tuple[float, float]:
    v_min = float(np.nanmin(values))
    v_max = float(np.nanmax(values))
    pad = max(0.001, (v_max - v_min) * 0.25)
    return v_min - pad, v_max + pad


def reveal_count(frame_index: int, frames: int, total: int) -> int:
    if total <= 1:
        return max(total, 1)
    fraction = smoothstep(frame_index / max(frames - 1, 1))
    return int(np.clip(math.ceil(fraction * total), 1, total))


def interpolated_path(x_values: np.ndarray, y_values: np.ndarray, progress: float) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    if len(x_values) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    if len(x_values) == 1:
        return x_values.copy(), y_values.copy()
    position = np.clip(progress, 0.0, 1.0) * (len(x_values) - 1)
    index = int(np.floor(position))
    fraction = float(position - index)
    end = min(index + 1, len(x_values))
    x_visible = list(x_values[:end])
    y_visible = list(y_values[:end])
    if index < len(x_values) - 1 and fraction > 0:
        x_visible.append(x_values[index] + fraction * (x_values[index + 1] - x_values[index]))
        y_visible.append(y_values[index] + fraction * (y_values[index + 1] - y_values[index]))
    return np.asarray(x_visible, dtype=float), np.asarray(y_visible, dtype=float)


def interpolated_path_3d(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    progress: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    z_values = np.asarray(z_values, dtype=float)
    if len(x_values) == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty
    if len(x_values) == 1:
        return x_values.copy(), y_values.copy(), z_values.copy()
    position = np.clip(progress, 0.0, 1.0) * (len(x_values) - 1)
    index = int(np.floor(position))
    fraction = float(position - index)
    end = min(index + 1, len(x_values))
    x_visible = list(x_values[:end])
    y_visible = list(y_values[:end])
    z_visible = list(z_values[:end])
    if index < len(x_values) - 1 and fraction > 0:
        x_visible.append(x_values[index] + fraction * (x_values[index + 1] - x_values[index]))
        y_visible.append(y_values[index] + fraction * (y_values[index + 1] - y_values[index]))
        z_visible.append(z_values[index] + fraction * (z_values[index + 1] - z_values[index]))
    return (
        np.asarray(x_visible, dtype=float),
        np.asarray(y_visible, dtype=float),
        np.asarray(z_visible, dtype=float),
    )


def smoothstep(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def build_progress_points(data: CandidateData) -> pd.DataFrame:
    selected = data.selected_summary.sort_values("report_order")
    if len(selected) > 1:
        points = selected[["report", "global_r2"]].rename(columns={"report": "label", "global_r2": "r2"}).copy()
        points["step"] = np.arange(len(points))
        return points[["step", "label", "r2"]]
    folds = data.selected_folds.sort_values("fold_index").copy()
    points = pd.DataFrame(
        {
            "step": np.arange(len(folds)),
            "label": folds["fold"].to_list(),
            "r2": folds["r2"].to_numpy(dtype=float),
        }
    )
    return points


def build_orbit_points(data: CandidateData) -> pd.DataFrame:
    selected = data.folds[data.folds["candidate_id"] == data.spec.selected_id].copy()
    if selected.empty:
        selected = data.selected_folds.copy()
    selected = selected.sort_values(["report_order", "fold_index"]).reset_index(drop=True)
    return selected


def set_3d_r2_limits(ax: plt.Axes, values: np.ndarray) -> None:
    z_min, z_max = r2_limits_from_values(np.asarray(values, dtype=float))
    ax.set_zlim(z_min, z_max)


def add_footer(fig: plt.Figure, data: CandidateData) -> None:
    best = data.selected_summary.sort_values("report_order").iloc[0]
    footer = (
        f"Source: best-candidates CSV artifacts | Primary regime: {data.primary_report} | "
        f"Selected global R2: {best['global_r2']:.6f} | Target: 0.020"
    )
    fig.text(0.01, 0.012, footer, ha="left", va="bottom", fontsize=7.5, color=COLORS["gray"])


def save_static(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def save_animation(anim: animation.FuncAnimation, fig: plt.Figure, path: Path, *, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    anim.save(path, writer=writer, dpi=95)
    plt.close(fig)


def write_manifest(artifacts: list[Artifact], args: argparse.Namespace) -> None:
    manifest = {
        "description": "English visual analytics gallery for preserved Jane Street best candidates.",
        "source_root": str(BEST_CANDIDATES_DIR.relative_to(PROJECT_ROOT)),
        "target_r2": TARGET_R2,
        "generation_args": vars(args),
        "artifact_count": len(artifacts),
        "artifacts": [
            {
                "candidate": artifact.candidate,
                "kind": artifact.kind,
                "path": str(artifact.path.relative_to(PROJECT_ROOT)),
                "description": artifact.description,
            }
            for artifact in artifacts
        ],
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def write_gallery_index(artifacts: list[Artifact]) -> None:
    lines = [
        "# Best-Candidate Visual Analytics Gallery",
        "",
        "All labels, filenames, and descriptions in this generated gallery are in English.",
        "The charts are generated only from preserved real validation artifacts under `best-candidates/`.",
        "",
    ]
    by_candidate: dict[str, list[Artifact]] = {}
    for artifact in artifacts:
        by_candidate.setdefault(artifact.candidate, []).append(artifact)
    for slug in sorted(by_candidate):
        lines.append(f"## {slug}")
        lines.append("")
        for artifact in by_candidate[slug]:
            rel = artifact.path.relative_to(OUTPUT_ROOT)
            lines.append(f"- `{artifact.kind}`: [{rel}]({rel}) - {artifact.description}")
        lines.append("")
    (OUTPUT_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def count_artifacts(artifacts: list[Artifact]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for artifact in artifacts:
        candidate_counts = counts.setdefault(artifact.candidate, {"chart": 0, "animation": 0, "animation-3d": 0})
        candidate_counts[artifact.kind] += 1
    return counts


if __name__ == "__main__":
    main()
