"""Generate real-data 3D charts and GIF animations for Jane Street research.

All plots are derived from the local Kaggle parquet data and existing report
files under reports/. The script intentionally fails if a required real source
is missing.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import animation
from matplotlib.colors import Normalize, TwoSlopeNorm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PARQUET_DIR = PROJECT_ROOT / "data/raw/jane-street-real-time-market-data-forecasting/train.parquet"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_ROOT = PROJECT_ROOT / "graficos"
FIGURES_DIR = OUTPUT_ROOT / "figuras"
ANIMATIONS_DIR = OUTPUT_ROOT / "animacoes"
CACHE_DIR = OUTPUT_ROOT / "dados_intermediarios"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"

TARGET = "responder_6"
WEIGHT = "weight"


@dataclass(frozen=True)
class PlotArtifact:
    kind: str
    path: Path
    description: str


@dataclass(frozen=True)
class PcaGeometry:
    frame: pl.DataFrame
    explained_variance_ratio: np.ndarray
    feature_count: int


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--date-stride", type=int, default=17)
    parser.add_argument("--time-stride", type=int, default=37)
    parser.add_argument("--max-pca-points", type=int, default=50_000)
    parser.add_argument("--max-animation-points", type=int, default=7_000)
    parser.add_argument("--gif-frames", type=int, default=48)
    parser.add_argument("--date-bucket-size", type=int, default=50)
    parser.add_argument("--time-bucket-size", type=int, default=100)
    args = parser.parse_args()

    configure_matplotlib()
    ensure_output_dirs()
    required_sources = require_real_sources()

    pca_geometry = load_or_build_pca_geometry(
        date_stride=args.date_stride,
        time_stride=args.time_stride,
        max_points=args.max_pca_points,
        refresh_cache=args.refresh_cache,
    )
    intraday = load_or_build_intraday_aggregates(
        date_bucket_size=args.date_bucket_size,
        time_bucket_size=args.time_bucket_size,
        refresh_cache=args.refresh_cache,
    )
    stage_summary = build_stage_summary()
    stage_by_fold = build_stage_by_fold()

    artifacts: list[PlotArtifact] = []
    artifacts.append(plot_pca_geometry(pca_geometry))
    artifacts.append(plot_intraday_geometry(intraday["date_time"], intraday["date_time_symbol"]))
    artifacts.append(plot_model_behavior(stage_summary, stage_by_fold))
    artifacts.append(plot_failure_diagnostics())
    artifacts.append(
        animate_pca_rotation(
            pca_geometry,
            max_points=args.max_animation_points,
            frames=args.gif_frames,
        )
    )
    artifacts.append(animate_intraday_evolution(intraday["date_time_symbol"]))
    artifacts.append(animate_model_progression(stage_by_fold))

    write_manifest(
        artifacts=artifacts,
        required_sources=required_sources,
        args=vars(args),
        pca_geometry=pca_geometry,
    )

    for artifact in artifacts:
        print(f"Wrote {artifact.path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")


def configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 170,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def ensure_output_dirs() -> None:
    for directory in (FIGURES_DIR, ANIMATIONS_DIR, CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def require_real_sources() -> list[Path]:
    sources = [
        TRAIN_PARQUET_DIR,
        REPORTS_DIR / "baselines/zero_baseline.csv",
        REPORTS_DIR / "baselines/ridge_sweep.csv",
        REPORTS_DIR / "experiments/ridge_calibration/ridge_calibration_summary.csv",
        REPORTS_DIR / "experiments/ridge_calibration/ridge_calibration_by_fold.csv",
        REPORTS_DIR / "experiments/ridge_calibration_high_weight_oof3x20/ridge_calibration_summary.csv",
        REPORTS_DIR / "experiments/ridge_calibration_high_weight_oof3x20/ridge_calibration_by_fold.csv",
        REPORTS_DIR / "experiments/sklearn_gbdt_conservative/sklearn_gbdt_summary.csv",
        REPORTS_DIR / "experiments/sklearn_gbdt_conservative/sklearn_gbdt_by_fold.csv",
        REPORTS_DIR / "experiments/ridge_gbdt_blend/ridge_gbdt_blend_summary.csv",
        REPORTS_DIR / "experiments/ridge_gbdt_blend/ridge_gbdt_blend_by_fold.csv",
        REPORTS_DIR / "experiments/ridge_gbdt_blend/time_bucket.csv",
        REPORTS_DIR / "experiments/ridge_gbdt_blend/weight_bucket.csv",
        REPORTS_DIR / "diagnostics/ridge_rw_02/by_date_id_symbol_id.csv",
        REPORTS_DIR / "diagnostics/ridge_rw_02_failure_slice/target_by_time_bucket.csv",
        REPORTS_DIR / "diagnostics/ridge_rw_02_failure_slice/feature_contributions.csv",
        REPORTS_DIR / "diagnostics/ridge_rw_02_failure_slice/summary.json",
    ]
    missing = [path for path in sources if not path.exists()]
    if missing:
        formatted = "\n".join(str(path.relative_to(PROJECT_ROOT)) for path in missing)
        raise FileNotFoundError(f"Missing required real source files:\n{formatted}")
    empty = [path for path in sources if path.is_file() and path.stat().st_size == 0]
    if empty:
        formatted = "\n".join(str(path.relative_to(PROJECT_ROOT)) for path in empty)
        raise ValueError(f"Required source files are empty:\n{formatted}")
    return sources


def read_csv_required(relative_path: str) -> pl.DataFrame:
    path = REPORTS_DIR / relative_path
    if not path.exists():
        raise FileNotFoundError(path)
    return pl.read_csv(path)


def load_or_build_pca_geometry(
    *,
    date_stride: int,
    time_stride: int,
    max_points: int,
    refresh_cache: bool,
) -> PcaGeometry:
    if date_stride <= 0 or time_stride <= 0 or max_points <= 0:
        raise ValueError("date_stride, time_stride and max_points must be positive")

    cache_path = CACHE_DIR / f"pca_geometry_d{date_stride}_t{time_stride}_n{max_points}.parquet"
    metadata_path = CACHE_DIR / f"pca_geometry_d{date_stride}_t{time_stride}_n{max_points}.json"
    if cache_path.exists() and metadata_path.exists() and not refresh_cache:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return PcaGeometry(
            frame=pl.read_parquet(cache_path),
            explained_variance_ratio=np.asarray(metadata["explained_variance_ratio"], dtype=float),
            feature_count=int(metadata["feature_count"]),
        )

    train = pl.scan_parquet(str(TRAIN_PARQUET_DIR))
    schema = train.collect_schema()
    feature_columns = tuple(name for name in schema.names() if name.startswith("feature_"))
    if not feature_columns:
        raise ValueError("No feature_* columns found in train parquet")

    sample = (
        train.filter(
            ((pl.col("date_id").cast(pl.Int32) % date_stride) == 0)
            & ((pl.col("time_id").cast(pl.Int32) % time_stride) == 0)
        )
        .select(
            [
                pl.col("date_id").cast(pl.Int32),
                pl.col("time_id").cast(pl.Int32),
                pl.col("symbol_id").cast(pl.Int32),
                pl.col(WEIGHT).cast(pl.Float64),
                pl.col(TARGET).cast(pl.Float64),
            ]
            + [pl.col(name).fill_null(0.0).cast(pl.Float64).alias(name) for name in feature_columns]
        )
        .collect()
        .sort(["date_id", "time_id", "symbol_id"])
    )
    if sample.height == 0:
        raise ValueError("Deterministic real-data sample is empty; reduce strides")

    sample = deterministic_cap(sample, max_points)
    x = sample.select(feature_columns).to_numpy()
    x_scaled = StandardScaler().fit_transform(x)
    pca = PCA(n_components=3, svd_solver="full")
    coords = pca.fit_transform(x_scaled)

    geometry = sample.select(["date_id", "time_id", "symbol_id", WEIGHT, TARGET]).with_columns(
        [
            pl.Series("pc1", coords[:, 0]),
            pl.Series("pc2", coords[:, 1]),
            pl.Series("pc3", coords[:, 2]),
        ]
    )
    metadata = {
        "source": str(TRAIN_PARQUET_DIR.relative_to(PROJECT_ROOT)),
        "date_stride": date_stride,
        "time_stride": time_stride,
        "max_points": max_points,
        "sample_rows": geometry.height,
        "feature_count": len(feature_columns),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
    geometry.write_parquet(cache_path)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return PcaGeometry(
        frame=geometry,
        explained_variance_ratio=pca.explained_variance_ratio_,
        feature_count=len(feature_columns),
    )


def deterministic_cap(frame: pl.DataFrame, max_points: int) -> pl.DataFrame:
    if frame.height <= max_points:
        return frame
    stride = math.ceil(frame.height / max_points)
    return frame.with_row_index("_row_id").filter((pl.col("_row_id") % stride) == 0).drop("_row_id")


def load_or_build_intraday_aggregates(
    *,
    date_bucket_size: int,
    time_bucket_size: int,
    refresh_cache: bool,
) -> dict[str, pl.DataFrame]:
    if date_bucket_size <= 0 or time_bucket_size <= 0:
        raise ValueError("date_bucket_size and time_bucket_size must be positive")

    date_time_path = CACHE_DIR / f"intraday_date_time_d{date_bucket_size}_t{time_bucket_size}.parquet"
    symbol_path = CACHE_DIR / f"intraday_date_time_symbol_d{date_bucket_size}_t{time_bucket_size}.parquet"
    if date_time_path.exists() and symbol_path.exists() and not refresh_cache:
        return {
            "date_time": pl.read_parquet(date_time_path),
            "date_time_symbol": pl.read_parquet(symbol_path),
        }

    train = pl.scan_parquet(str(TRAIN_PARQUET_DIR)).select(
        [
            pl.col("date_id").cast(pl.Int32),
            pl.col("time_id").cast(pl.Int32),
            pl.col("symbol_id").cast(pl.Int32),
            pl.col(WEIGHT).cast(pl.Float64),
            pl.col(TARGET).cast(pl.Float64),
        ]
    )
    bucketed = train.with_columns(
        [
            ((pl.col("date_id") // date_bucket_size) * date_bucket_size).alias("date_bucket"),
            (pl.col("time_id") // time_bucket_size).alias("time_bucket"),
        ]
    )

    date_time = aggregate_target_behavior(bucketed, ["date_bucket", "time_bucket"])
    date_time_symbol = aggregate_target_behavior(bucketed, ["date_bucket", "time_bucket", "symbol_id"])
    date_time.write_parquet(date_time_path)
    date_time_symbol.write_parquet(symbol_path)
    return {"date_time": date_time, "date_time_symbol": date_time_symbol}


def aggregate_target_behavior(frame: pl.LazyFrame, group_cols: list[str]) -> pl.DataFrame:
    return (
        frame.group_by(group_cols)
        .agg(
            pl.len().alias("rows"),
            pl.col(WEIGHT).sum().alias("weight_sum"),
            pl.col(TARGET).mean().alias("target_mean"),
            pl.col(TARGET).std().alias("target_std"),
            (pl.col(TARGET).pow(2).mean().sqrt()).alias("target_rms"),
            (pl.col(WEIGHT) * pl.col(TARGET)).sum().alias("weighted_target_sum"),
        )
        .with_columns((pl.col("weighted_target_sum") / pl.col("weight_sum")).alias("weighted_target_mean"))
        .drop("weighted_target_sum")
        .sort(group_cols)
        .collect()
    )


def build_stage_summary() -> pl.DataFrame:
    zero = read_csv_required("baselines/zero_baseline.csv")
    ridge = read_csv_required("baselines/ridge_sweep.csv").filter(pl.col("alpha") == 1000.0)
    calibration = read_csv_required("experiments/ridge_calibration/ridge_calibration_summary.csv")
    oof = read_csv_required("experiments/ridge_calibration_high_weight_oof3x20/ridge_calibration_summary.csv")
    gbdt = read_csv_required("experiments/sklearn_gbdt_conservative/sklearn_gbdt_summary.csv")
    blend = read_csv_required("experiments/ridge_gbdt_blend/ridge_gbdt_blend_summary.csv")

    rows = [
        {
            "stage": "Zero baseline",
            "short_stage": "Zero",
            "global_r2": weighted_r2_from_components(zero),
            "mean_r2": float(zero["weighted_zero_mean_r2"].mean()),
            "min_r2": float(zero["weighted_zero_mean_r2"].min()),
            "source": "baselines/zero_baseline.csv",
        },
        {
            "stage": "Raw Ridge a=1000",
            "short_stage": "Ridge",
            "global_r2": weighted_r2_from_components(ridge),
            "mean_r2": float(ridge["weighted_zero_mean_r2"].mean()),
            "min_r2": float(ridge["weighted_zero_mean_r2"].min()),
            "source": "baselines/ridge_sweep.csv",
        },
        row_from_strategy(
            calibration,
            strategy="time_weight",
            stage="Clip + time/weight calibration",
            short_stage="T/W cal",
            source="experiments/ridge_calibration/ridge_calibration_summary.csv",
        ),
        row_from_strategy(
            oof,
            strategy="weight_predabs",
            stage="OOF weight/predabs calibration",
            short_stage="OOF cal",
            source="experiments/ridge_calibration_high_weight_oof3x20/ridge_calibration_summary.csv",
        ),
        {
            "stage": "Conservative sklearn GBDT",
            "short_stage": "GBDT",
            "global_r2": float(gbdt["global_r2"][0]),
            "mean_r2": float(gbdt["mean_r2"][0]),
            "min_r2": float(gbdt["min_r2"][0]),
            "source": "experiments/sklearn_gbdt_conservative/sklearn_gbdt_summary.csv",
        },
        row_from_strategy(
            blend,
            strategy="blend",
            stage="Ridge OOF + GBDT blend",
            short_stage="Blend",
            source="experiments/ridge_gbdt_blend/ridge_gbdt_blend_summary.csv",
        ),
    ]
    return pl.DataFrame(rows)


def weighted_r2_from_components(frame: pl.DataFrame) -> float:
    numerator = float(frame["numerator"].sum())
    denominator = float(frame["denominator"].sum())
    if denominator <= 0.0:
        raise ValueError("denominator must be positive")
    return 1.0 - numerator / denominator


def row_from_strategy(
    frame: pl.DataFrame,
    *,
    strategy: str,
    stage: str,
    short_stage: str,
    source: str,
) -> dict[str, float | str]:
    row = frame.filter(pl.col("strategy") == strategy)
    if row.height != 1:
        raise ValueError(f"Expected one row for strategy={strategy}, got {row.height}")
    return {
        "stage": stage,
        "short_stage": short_stage,
        "global_r2": float(row["global_r2"][0]),
        "mean_r2": float(row["mean_r2"][0]),
        "min_r2": float(row["min_r2"][0]),
        "source": source,
    }


def build_stage_by_fold() -> pl.DataFrame:
    calibration = read_csv_required("experiments/ridge_calibration/ridge_calibration_by_fold.csv")
    oof = read_csv_required("experiments/ridge_calibration_high_weight_oof3x20/ridge_calibration_by_fold.csv")
    gbdt = read_csv_required("experiments/sklearn_gbdt_conservative/sklearn_gbdt_by_fold.csv")
    blend = read_csv_required("experiments/ridge_gbdt_blend/ridge_gbdt_blend_by_fold.csv")

    frames = [
        fold_stage(calibration.filter(pl.col("strategy") == "raw"), "Ridge", 0),
        fold_stage(calibration.filter(pl.col("strategy") == "time_weight"), "T/W cal", 1),
        fold_stage(oof.filter(pl.col("strategy") == "weight_predabs"), "OOF cal", 2),
        fold_stage(gbdt, "GBDT", 3),
        fold_stage(blend.filter(pl.col("strategy") == "blend"), "Blend", 4),
    ]
    return pl.concat(frames).sort(["stage_order", "fold"])


def fold_stage(frame: pl.DataFrame, stage: str, stage_order: int) -> pl.DataFrame:
    if frame.height == 0:
        raise ValueError(f"No fold rows for stage {stage}")
    return frame.select(
        [
            pl.lit(stage).alias("stage"),
            pl.lit(stage_order).alias("stage_order"),
            pl.col("fold"),
            pl.col("valid_start"),
            pl.col("valid_end"),
            pl.col("weighted_zero_mean_r2"),
        ]
    )


def plot_pca_geometry(pca_geometry: PcaGeometry) -> PlotArtifact:
    frame = pca_geometry.frame
    out_path = FIGURES_DIR / "geometria_pca_3d.png"
    pc1 = frame["pc1"].to_numpy()
    pc2 = frame["pc2"].to_numpy()
    pc3 = frame["pc3"].to_numpy()
    target = frame[TARGET].to_numpy()
    weights = np.log1p(frame[WEIGHT].to_numpy())

    target_norm = centered_norm(target, lower_q=1, upper_q=99)
    weight_norm = Normalize(vmin=float(np.nanmin(weights)), vmax=float(np.nanmax(weights)))

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    s1 = ax1.scatter(pc1, pc2, pc3, c=target, cmap="coolwarm", norm=target_norm, s=5, alpha=0.45, linewidths=0)
    style_3d_axis(ax1, "PCA 3D das features reais", "PC1", "PC2", "PC3")
    cb1 = fig.colorbar(s1, ax=ax1, shrink=0.66, pad=0.08)
    cb1.set_label("responder_6 real")

    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    s2 = ax2.scatter(pc1, pc2, pc3, c=weights, cmap="viridis", norm=weight_norm, s=5, alpha=0.45, linewidths=0)
    style_3d_axis(ax2, "A mesma geometria colorida por log1p(weight)", "PC1", "PC2", "PC3")
    cb2 = fig.colorbar(s2, ax=ax2, shrink=0.66, pad=0.08)
    cb2.set_label("log1p(weight)")

    ax3 = fig.add_subplot(2, 2, 3)
    hb = ax3.hexbin(pc1, pc2, C=target, gridsize=65, cmap="coolwarm", reduce_C_function=np.mean, mincnt=1)
    ax3.set_title("Media do alvo no plano PC1/PC2")
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    cb3 = fig.colorbar(hb, ax=ax3)
    cb3.set_label("mean responder_6")

    ax4 = fig.add_subplot(2, 2, 4)
    variance = pca_geometry.explained_variance_ratio * 100.0
    ax4.bar(["PC1", "PC2", "PC3"], variance, color=["#4464ad", "#2a9d8f", "#e76f51"])
    ax4.set_title("Variancia explicada pela projecao 3D")
    ax4.set_ylabel("% da variancia padronizada")
    ax4.text(
        0.02,
        0.95,
        f"{frame.height:,} linhas reais renderizadas\n{pca_geometry.feature_count} feature_* usadas",
        transform=ax4.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#cccccc", "alpha": 0.9},
    )

    fig.suptitle("Geometria das features reais do train.parquet", fontsize=16, fontweight="bold")
    fig.savefig(out_path)
    plt.close(fig)
    return PlotArtifact("figure", out_path, "3D PCA geometry from deterministic real-data rows")


def plot_intraday_geometry(date_time: pl.DataFrame, date_time_symbol: pl.DataFrame) -> PlotArtifact:
    out_path = FIGURES_DIR / "natureza_temporal_intraday_3d.png"
    x_dates, y_times, z_target = matrix_from_frame(date_time, "date_bucket", "time_bucket", "weighted_target_mean")
    _, _, z_rms = matrix_from_frame(date_time, "date_bucket", "time_bucket", "target_rms")

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    grid_x, grid_y = np.meshgrid(x_dates, y_times, indexing="ij")

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    surf = ax1.plot_surface(grid_x, grid_y, z_target, cmap="coolwarm", linewidth=0, antialiased=True, alpha=0.92)
    style_3d_axis(ax1, "Alvo medio ponderado por data/time bucket", "date bucket", "time bucket", "weighted mean")
    fig.colorbar(surf, ax=ax1, shrink=0.64, pad=0.08)

    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    surf2 = ax2.plot_surface(grid_x, grid_y, z_rms, cmap="magma", linewidth=0, antialiased=True, alpha=0.92)
    style_3d_axis(ax2, "Energia intraday do alvo real", "date bucket", "time bucket", "target RMS")
    fig.colorbar(surf2, ax=ax2, shrink=0.64, pad=0.08)

    ax3 = fig.add_subplot(2, 2, 3)
    im = ax3.imshow(
        z_target.T,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        extent=[float(x_dates.min()), float(x_dates.max()), float(y_times.min()), float(y_times.max())],
    )
    ax3.set_title("Mapa do alvo medio ponderado")
    ax3.set_xlabel("date bucket")
    ax3.set_ylabel("time bucket")
    fig.colorbar(im, ax=ax3).set_label("weighted mean responder_6")

    ax4 = fig.add_subplot(2, 2, 4)
    by_time = (
        date_time.group_by("time_bucket")
        .agg(
            pl.col("rows").sum().alias("rows"),
            pl.col("weight_sum").sum().alias("weight_sum"),
            pl.col("target_rms").mean().alias("target_rms"),
        )
        .sort("time_bucket")
    )
    ax4.plot(by_time["time_bucket"], by_time["target_rms"], marker="o", color="#264653", label="target RMS")
    ax4b = ax4.twinx()
    ax4b.bar(
        by_time["time_bucket"],
        by_time["weight_sum"],
        alpha=0.28,
        color="#e76f51",
        label="weight sum",
    )
    ax4.set_title("Perfil medio por bucket intraday")
    ax4.set_xlabel("time bucket")
    ax4.set_ylabel("target RMS")
    ax4b.set_ylabel("weight sum")
    ax4.legend(loc="upper left")
    ax4b.legend(loc="upper right")

    fig.suptitle(
        f"Natureza temporal do alvo real ({date_time_symbol.height:,} grupos date/time/symbol)",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(out_path)
    plt.close(fig)
    return PlotArtifact("figure", out_path, "3D and heatmap intraday target behavior from real parquet aggregates")


def plot_model_behavior(stage_summary: pl.DataFrame, stage_by_fold: pl.DataFrame) -> PlotArtifact:
    out_path = FIGURES_DIR / "comportamento_tecnicas_resultados_3d.png"
    time_bucket = read_csv_required("experiments/ridge_gbdt_blend/time_bucket.csv")
    weight_bucket = read_csv_required("experiments/ridge_gbdt_blend/weight_bucket.csv")

    fig = plt.figure(figsize=(20, 13), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.90, wspace=0.28, hspace=0.35)
    ax1 = fig.add_subplot(2, 2, 1)
    x = np.arange(stage_summary.height)
    global_bps = stage_summary["global_r2"].to_numpy() * 10_000.0
    min_bps = stage_summary["min_r2"].to_numpy() * 10_000.0
    ax1.bar(x, global_bps, color=["#8d99ae", "#d62828", "#f77f00", "#2a9d8f", "#457b9d", "#6a4c93"])
    ax1.plot(x, min_bps, marker="o", color="#222222", linewidth=2, label="min fold")
    ax1.axhline(0.0, color="#222222", linewidth=0.8)
    ax1.set_title("Evolucao real das tecnicas")
    ax1.set_ylabel("weighted zero-mean R2 x 1e4")
    ax1.set_xticks(x, stage_summary["short_stage"].to_list(), rotation=20, ha="right")
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    plot_fold_stage_bars(ax2, stage_by_fold)

    ax3 = fig.add_subplot(2, 2, 3)
    for strategy, color in [("ridge_calibrated", "#2a9d8f"), ("gbdt", "#457b9d"), ("blend", "#6a4c93")]:
        subset = time_bucket.filter(pl.col("strategy") == strategy).sort("time_bucket")
        ax3.plot(
            subset["time_bucket"],
            subset["weighted_zero_mean_r2"].to_numpy() * 10_000.0,
            marker="o",
            linewidth=2,
            label=strategy,
            color=color,
        )
    ax3.axhline(0.0, color="#222222", linewidth=0.8)
    ax3.set_title("Resultado por time bucket no blend report")
    ax3.set_xlabel("time bucket")
    ax3.set_ylabel("R2 x 1e4")
    ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    weight_order = ["q00_q50", "q50_q90", "q90_q99", "q99_q100"]
    width = 0.24
    offsets = {"ridge_calibrated": -width, "gbdt": 0.0, "blend": width}
    colors = {"ridge_calibrated": "#2a9d8f", "gbdt": "#457b9d", "blend": "#6a4c93"}
    base_x = np.arange(len(weight_order))
    for strategy, offset in offsets.items():
        subset = weight_bucket.filter(pl.col("strategy") == strategy)
        values = []
        for bucket in weight_order:
            row = subset.filter(pl.col("weight_bucket") == bucket)
            if row.height != 1:
                raise ValueError(f"Missing {strategy}/{bucket} in weight bucket report")
            values.append(float(row["weighted_zero_mean_r2"][0]) * 10_000.0)
        ax4.bar(base_x + offset, values, width=width, label=strategy, color=colors[strategy], alpha=0.86)
    ax4.axhline(0.0, color="#222222", linewidth=0.8)
    ax4.set_title("Resultado por bucket de peso")
    ax4.set_xlabel("weight bucket")
    ax4.set_ylabel("R2 x 1e4")
    ax4.set_xticks(base_x, weight_order, rotation=20, ha="right")
    ax4.legend()

    fig.suptitle("Comportamento dos resultados reais conforme tecnicas aplicadas", fontsize=16, fontweight="bold")
    fig.savefig(out_path)
    plt.close(fig)
    return PlotArtifact("figure", out_path, "Real report metrics across Ridge, calibration, GBDT and blend")


def plot_failure_diagnostics() -> PlotArtifact:
    out_path = FIGURES_DIR / "diagnostico_falha_rw02_3d.png"
    by_slice = read_csv_required("diagnostics/ridge_rw_02/by_date_id_symbol_id.csv")
    time_slice = read_csv_required("diagnostics/ridge_rw_02_failure_slice/target_by_time_bucket.csv").sort("time_bucket")
    contributions = read_csv_required("diagnostics/ridge_rw_02_failure_slice/feature_contributions.csv")
    summary_path = REPORTS_DIR / "diagnostics/ridge_rw_02_failure_slice/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    target_date = int(summary["target"]["date_id"])
    target_symbol = int(summary["target"]["symbol_id"])
    actual_outlier_r2 = float(summary["group_summaries"][0]["weighted_zero_mean_r2"])

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    r2 = by_slice["weighted_zero_mean_r2"].to_numpy()
    r2_for_plot = np.clip(r2, -0.25, 0.08)
    norm = TwoSlopeNorm(vmin=-0.25, vcenter=0.0, vmax=0.08)
    sc = ax1.scatter(
        by_slice["date_id"].to_numpy(),
        by_slice["symbol_id"].to_numpy(),
        r2_for_plot,
        c=r2_for_plot,
        cmap="coolwarm",
        norm=norm,
        s=np.clip(np.log1p(by_slice["weight_sum"].to_numpy()) * 9.0, 12.0, 80.0),
        alpha=0.75,
        linewidths=0,
    )
    ax1.scatter([target_date], [target_symbol], [-0.25], marker="X", s=150, color="#111111")
    ax1.text(target_date, target_symbol, -0.25, f" actual {actual_outlier_r2:.2f}", color="#111111")
    style_3d_axis(ax1, "rw_02 por date_id/symbol_id (R2 clipped para escala)", "date_id", "symbol_id", "R2 clipped")
    fig.colorbar(sc, ax=ax1, shrink=0.65, pad=0.08).set_label("R2 clipped")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(time_slice["time_bucket"], time_slice["target_mean"], marker="o", label="target_mean", color="#2a9d8f")
    ax2.plot(time_slice["time_bucket"], time_slice["prediction_mean"], marker="o", label="prediction_mean", color="#d62828")
    ax2.set_title(f"Slice dominante date_id={target_date}, symbol_id={target_symbol}")
    ax2.set_xlabel("time bucket")
    ax2.set_ylabel("media")
    ax2.legend()

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.bar(
        time_slice["time_bucket"],
        time_slice["weighted_zero_mean_r2"].to_numpy(),
        color=np.where(time_slice["weighted_zero_mean_r2"].to_numpy() < 0.0, "#d62828", "#2a9d8f"),
    )
    ax3.axhline(0.0, color="#222222", linewidth=0.8)
    ax3.set_title("R2 por time bucket no slice de falha")
    ax3.set_xlabel("time bucket")
    ax3.set_ylabel("weighted zero-mean R2")

    ax4 = fig.add_subplot(2, 2, 4)
    top = contributions.sort("abs_target_linear_contribution", descending=True).head(12).sort(
        "abs_target_linear_contribution"
    )
    colors = np.where(top["target_linear_contribution"].to_numpy() < 0.0, "#d62828", "#2a9d8f")
    ax4.barh(top["feature"], top["target_linear_contribution"], color=colors)
    ax4.axvline(0.0, color="#222222", linewidth=0.8)
    ax4.set_title("Maiores contribuicoes lineares no slice")
    ax4.set_xlabel("contribuicao linear")

    fig.suptitle("Diagnostico real da falha rw_02", fontsize=16, fontweight="bold")
    fig.savefig(out_path)
    plt.close(fig)
    return PlotArtifact("figure", out_path, "3D and slice diagnostics for the real rw_02 failure report")


def animate_pca_rotation(pca_geometry: PcaGeometry, *, max_points: int, frames: int) -> PlotArtifact:
    out_path = ANIMATIONS_DIR / "geometria_pca_3d_rotacao.gif"
    frame = deterministic_cap(pca_geometry.frame.sort(["date_id", "time_id", "symbol_id"]), max_points)
    pc1 = frame["pc1"].to_numpy()
    pc2 = frame["pc2"].to_numpy()
    pc3 = frame["pc3"].to_numpy()
    target = frame[TARGET].to_numpy()

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    sc = ax.scatter(pc1, pc2, pc3, c=target, cmap="coolwarm", norm=centered_norm(target, 1, 99), s=7, alpha=0.55)
    style_3d_axis(ax, "Rotacao da geometria PCA real", "PC1", "PC2", "PC3")
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.08)
    cb.set_label("responder_6 real")

    def update(frame_idx: int) -> tuple:
        ax.view_init(elev=24, azim=360.0 * frame_idx / frames)
        ax.set_title(f"Rotacao da geometria PCA real | frame {frame_idx + 1}/{frames}")
        return (ax,)

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=70, blit=False)
    ani.save(out_path, writer=animation.PillowWriter(fps=12))
    plt.close(fig)
    return PlotArtifact("animation", out_path, "FuncAnimation rotating 3D PCA scatter from real parquet sample")


def animate_intraday_evolution(date_time_symbol: pl.DataFrame) -> PlotArtifact:
    out_path = ANIMATIONS_DIR / "evolucao_intraday_symbol_date_3d.gif"
    time_buckets = date_time_symbol["time_bucket"].unique().sort().to_list()
    values = date_time_symbol["weighted_target_mean"].to_numpy()
    norm = centered_norm(values, lower_q=1, upper_q=99)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    sm = plt.cm.ScalarMappable(norm=norm, cmap="coolwarm")
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.08)
    cb.set_label("weighted mean responder_6")

    z_min = float(np.nanpercentile(values, 1))
    z_max = float(np.nanpercentile(values, 99))

    def update(bucket: int) -> tuple:
        ax.clear()
        subset = date_time_symbol.filter(pl.col("time_bucket") == bucket).sort(["date_bucket", "symbol_id"])
        z = np.clip(subset["weighted_target_mean"].to_numpy(), z_min, z_max)
        ax.scatter(
            subset["date_bucket"].to_numpy(),
            subset["symbol_id"].to_numpy(),
            z,
            c=z,
            cmap="coolwarm",
            norm=norm,
            s=np.clip(np.log1p(subset["rows"].to_numpy()) * 3.0, 12.0, 48.0),
            alpha=0.78,
            linewidths=0,
        )
        ax.set_zlim(z_min, z_max)
        style_3d_axis(
            ax,
            f"Comportamento real por simbolo/data | time bucket {bucket}",
            "date bucket",
            "symbol_id",
            "weighted target mean",
        )
        return (ax,)

    ani = animation.FuncAnimation(fig, update, frames=time_buckets, interval=650, blit=False)
    ani.save(out_path, writer=animation.PillowWriter(fps=2))
    plt.close(fig)
    return PlotArtifact("animation", out_path, "FuncAnimation of real date/symbol target geometry across intraday buckets")


def animate_model_progression(stage_by_fold: pl.DataFrame) -> PlotArtifact:
    out_path = ANIMATIONS_DIR / "progressao_tecnicas_folds_3d.gif"
    stages = (
        stage_by_fold.select(["stage_order", "stage"])
        .unique()
        .sort("stage_order")["stage"]
        .to_list()
    )
    frames = list(range(1, len(stages) + 1))
    z = stage_by_fold["weighted_zero_mean_r2"].to_numpy() * 10_000.0
    z_min = min(float(np.nanmin(z)), 0.0)
    z_max = max(float(np.nanmax(z)), 0.0)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    def update(n_stages: int) -> tuple:
        ax.clear()
        subset = stage_by_fold.filter(pl.col("stage_order") < n_stages)
        plot_fold_stage_bars(ax, subset, zlim=(z_min, z_max))
        ax.set_title(f"Progressao real das tecnicas | {stages[n_stages - 1]}")
        return (ax,)

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=900, blit=False)
    ani.save(out_path, writer=animation.PillowWriter(fps=1.2))
    plt.close(fig)
    return PlotArtifact("animation", out_path, "FuncAnimation of real fold metrics as techniques are added")


def plot_fold_stage_bars(ax, stage_by_fold: pl.DataFrame, zlim: tuple[float, float] | None = None) -> None:
    folds = stage_by_fold["fold"].unique().sort().to_list()
    stages = (
        stage_by_fold.select(["stage_order", "stage"])
        .unique()
        .sort("stage_order")["stage"]
        .to_list()
    )
    stage_to_y = {stage: idx for idx, stage in enumerate(stages)}
    fold_to_x = {fold: idx for idx, fold in enumerate(folds)}

    xs: list[float] = []
    ys: list[float] = []
    zbase: list[float] = []
    heights: list[float] = []
    colors: list[str] = []
    palette = ["#d62828", "#f77f00", "#2a9d8f", "#457b9d", "#6a4c93"]
    for row in stage_by_fold.iter_rows(named=True):
        value = float(row["weighted_zero_mean_r2"]) * 10_000.0
        xs.append(fold_to_x[row["fold"]])
        ys.append(stage_to_y[row["stage"]])
        zbase.append(min(0.0, value))
        heights.append(abs(value))
        colors.append(palette[int(row["stage_order"]) % len(palette)])

    if xs:
        ax.bar3d(xs, ys, zbase, 0.62, 0.62, heights, color=colors, alpha=0.82, shade=True)
    ax.set_xticks(np.arange(len(folds)) + 0.31, folds, rotation=15, ha="right")
    ax.set_yticks(np.arange(len(stages)) + 0.31, stages)
    ax.set_xlabel("fold")
    ax.set_ylabel("tecnica")
    ax.set_zlabel("R2 x 1e4")
    ax.set_title("Walk-forward por fold e tecnica")
    ax.view_init(elev=24, azim=-58)
    if zlim is not None:
        pad = (zlim[1] - zlim[0]) * 0.12 if zlim[1] > zlim[0] else 1.0
        ax.set_zlim(zlim[0] - pad, zlim[1] + pad)


def matrix_from_frame(
    frame: pl.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values = np.asarray(sorted(frame[x_col].unique().to_list()), dtype=float)
    y_values = np.asarray(sorted(frame[y_col].unique().to_list()), dtype=float)
    matrix = np.full((len(x_values), len(y_values)), np.nan, dtype=float)
    x_index = {value: idx for idx, value in enumerate(x_values)}
    y_index = {value: idx for idx, value in enumerate(y_values)}
    for row in frame.select([x_col, y_col, value_col]).iter_rows(named=True):
        matrix[x_index[float(row[x_col])], y_index[float(row[y_col])]] = float(row[value_col])
    return x_values, y_values, matrix


def centered_norm(values: np.ndarray, lower_q: float, upper_q: float) -> TwoSlopeNorm:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("values must contain at least one finite value")
    vmin = float(np.nanpercentile(finite, lower_q))
    vmax = float(np.nanpercentile(finite, upper_q))
    if vmin >= 0.0:
        vmin = -max(abs(vmax) * 0.05, 1e-6)
    if vmax <= 0.0:
        vmax = max(abs(vmin) * 0.05, 1e-6)
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0
    return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)


def style_3d_axis(ax, title: str, xlabel: str, ylabel: str, zlabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.view_init(elev=24, azim=-52)
    ax.xaxis.pane.set_alpha(0.08)
    ax.yaxis.pane.set_alpha(0.08)
    ax.zaxis.pane.set_alpha(0.08)


def write_manifest(
    *,
    artifacts: list[PlotArtifact],
    required_sources: list[Path],
    args: dict[str, object],
    pca_geometry: PcaGeometry,
) -> None:
    manifest = {
        "real_data_only": True,
        "notes": [
            "No mock data was used.",
            "PCA charts use deterministic real parquet rows selected by date/time stride for renderability.",
            "Report charts use existing real validation reports under reports/.",
            "Some color/z scales are clipped only for legibility and are labeled in figure titles.",
        ],
        "parameters": args,
        "pca_rows_rendered": pca_geometry.frame.height,
        "pca_feature_count": pca_geometry.feature_count,
        "pca_explained_variance_ratio": pca_geometry.explained_variance_ratio.tolist(),
        "sources": [str(path.relative_to(PROJECT_ROOT)) for path in required_sources],
        "artifacts": [
            {
                "kind": artifact.kind,
                "path": str(artifact.path.relative_to(PROJECT_ROOT)),
                "description": artifact.description,
            }
            for artifact in artifacts
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
