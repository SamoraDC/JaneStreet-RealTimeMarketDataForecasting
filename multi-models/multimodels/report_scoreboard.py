"""Aggregate saved multi-model experiment summaries into an audit scoreboard."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import polars as pl


REQUIRED_COLUMNS = (
    "candidate",
    "family",
    "rows",
    "weight_sum",
    "numerator",
    "denominator",
    "mean_fold_r2",
    "min_fold_r2",
    "std_fold_r2",
    "global_r2",
)


@dataclass(frozen=True)
class ReportScoreboardConfig:
    """Configuration for saved-report aggregation."""

    reports_root: Path = Path("multi-models/reports")
    output_dir: Path = Path("multi-models/reports/report_scoreboard")
    min_rows: int = 0
    top_k: int = 30
    categories: tuple[str, ...] = ()
    include_reports: tuple[str, ...] = ()
    exclude_reports: tuple[str, ...] = ()


def run_report_scoreboard(config: ReportScoreboardConfig) -> dict[str, Any]:
    """Build a leaderboard from existing candidate_summary.csv files."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    all_candidates, skipped = load_candidate_summaries(config.reports_root)
    filtered = filter_scoreboard(
        all_candidates,
        min_rows=config.min_rows,
        categories=config.categories,
        include_reports=config.include_reports,
        exclude_reports=config.exclude_reports,
    )
    top = top_candidates(filtered, top_k=config.top_k)
    category_top = top_by_category(filtered, top_k=config.top_k)

    all_candidates.write_csv(config.output_dir / "all_candidates.csv")
    filtered.write_csv(config.output_dir / "filtered_candidates.csv")
    top.write_csv(config.output_dir / "top_candidates.csv")
    category_top.write_csv(config.output_dir / "top_by_category.csv")
    audit = {
        "config": _json_ready(asdict(config)),
        "reports_scanned": int(all_candidates["report"].n_unique()) if all_candidates.height else 0,
        "candidate_rows": int(all_candidates.height),
        "filtered_rows": int(filtered.height),
        "skipped": skipped,
        "status": "diagnostic_only",
        "leakage_note": "This aggregates already-computed fold summaries only; it does not fit, select, or deploy a row-level model.",
    }
    (config.output_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_scoreboard_report(config.output_dir / "REPORT.md", config=config, top=top, category_top=category_top, audit=audit)
    return {"all": all_candidates, "filtered": filtered, "top": top, "category_top": category_top, "audit": audit}


def load_candidate_summaries(reports_root: Path) -> tuple[pl.DataFrame, list[dict[str, str]]]:
    """Load every candidate_summary.csv under reports_root."""

    if not reports_root.exists():
        raise FileNotFoundError(reports_root)
    frames: list[pl.DataFrame] = []
    skipped: list[dict[str, str]] = []
    for path in sorted(reports_root.glob("*/candidate_summary.csv")):
        report = path.parent.name
        try:
            frame = pl.read_csv(path)
        except Exception as exc:  # pragma: no cover - defensive audit path
            skipped.append({"report": report, "path": str(path), "reason": f"read_error: {exc}"})
            continue
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            skipped.append({"report": report, "path": str(path), "reason": "missing_columns:" + ",".join(missing)})
            continue
        frames.append(
            frame.select(REQUIRED_COLUMNS).with_columns(
                pl.lit(report).alias("report"),
                pl.lit(classify_report(report)).alias("category"),
            )
        )
    if not frames:
        empty = pl.DataFrame({column: [] for column in (*REQUIRED_COLUMNS, "report", "category")})
        return empty, skipped
    return pl.concat(frames, how="diagonal_relaxed"), skipped


def classify_report(report: str) -> str:
    """Classify reports by naming convention for audit filtering."""

    name = report.lower()
    if "hist" in name or "max1398" in name:
        return "historical"
    if "smoke" in name:
        return "smoke"
    if "stage3" in name:
        return "stage3"
    if "family_artifacts" in name or "stride" in name:
        return "sampled"
    return "other"


def filter_scoreboard(
    frame: pl.DataFrame,
    *,
    min_rows: int,
    categories: Sequence[str],
    include_reports: Sequence[str],
    exclude_reports: Sequence[str],
) -> pl.DataFrame:
    """Filter candidates without changing their original metrics."""

    out = frame
    if min_rows > 0:
        out = out.filter(pl.col("rows") >= int(min_rows))
    cats = tuple(categories)
    if cats:
        out = out.filter(pl.col("category").is_in(cats))
    includes = tuple(include_reports)
    if includes:
        out = out.filter(pl.col("report").is_in(includes))
    excludes = tuple(exclude_reports)
    if excludes:
        out = out.filter(~pl.col("report").is_in(excludes))
    return sort_scoreboard(out)


def sort_scoreboard(frame: pl.DataFrame) -> pl.DataFrame:
    """Sort by primary score, then worst fold for stability."""

    if frame.height == 0:
        return frame
    return frame.sort(["global_r2", "min_fold_r2", "mean_fold_r2"], descending=[True, True, True])


def top_candidates(frame: pl.DataFrame, *, top_k: int) -> pl.DataFrame:
    """Return top candidates globally."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    return sort_scoreboard(frame).head(top_k)


def top_by_category(frame: pl.DataFrame, *, top_k: int) -> pl.DataFrame:
    """Return top candidates within each report category."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if frame.height == 0:
        return frame
    return (
        sort_scoreboard(frame)
        .with_columns((pl.col("global_r2").rank("ordinal", descending=True).over("category")).alias("category_rank"))
        .filter(pl.col("category_rank") <= top_k)
        .sort(["category", "category_rank"])
    )


def write_scoreboard_report(path: Path, *, config: ReportScoreboardConfig, top: pl.DataFrame, category_top: pl.DataFrame, audit: dict[str, Any]) -> None:
    """Write a concise Markdown report."""

    lines = [
        "# Report Scoreboard",
        "",
        "## Scope",
        "",
        "This report aggregates existing `candidate_summary.csv` files. It is a diagnostic index, not a new validation run and not a deployable selection policy.",
        "",
        "## Filters",
        "",
        f"- `min_rows`: `{config.min_rows}`",
        f"- `categories`: `{', '.join(config.categories) if config.categories else 'all'}`",
        f"- `include_reports`: `{', '.join(config.include_reports) if config.include_reports else 'all'}`",
        f"- `exclude_reports`: `{', '.join(config.exclude_reports) if config.exclude_reports else 'none'}`",
        "",
        "## Top Candidates",
        "",
        _markdown_table(top, columns=("category", "report", "candidate", "family", "rows", "global_r2", "min_fold_r2")),
        "",
        "## Top By Category",
        "",
        _markdown_table(category_top, columns=("category", "category_rank", "report", "candidate", "global_r2", "min_fold_r2")),
        "",
        "## Audit",
        "",
        f"- Reports scanned: `{audit['reports_scanned']}`",
        f"- Candidate rows: `{audit['candidate_rows']}`",
        f"- Filtered rows: `{audit['filtered_rows']}`",
        f"- Skipped reports: `{len(audit['skipped'])}`",
        f"- Status: `{audit['status']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pl.DataFrame, *, columns: Sequence[str]) -> str:
    if frame.height == 0:
        return "_No rows._"
    cols = [column for column in columns if column in frame.columns]
    rows = frame.select(cols).to_dicts()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(_format_cell(row[column]) for column in cols) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    return value
