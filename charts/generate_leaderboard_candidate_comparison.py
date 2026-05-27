from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "charts" / "leaderboard_candidate_score_comparison.csv"
FIGURE_PATH = ROOT / "charts" / "figures" / "leaderboard_candidate_score_comparison.png"


KAGGLE_PUBLIC_TOP = [
    {"label": "Kaggle #1: ms capital", "score": 0.013890, "source": "Kaggle public leaderboard"},
    {"label": "Kaggle #2: Patrick Yam", "score": 0.013273, "source": "Kaggle public leaderboard"},
    {"label": "Kaggle #3: shorturl.at/LKhAD", "score": 0.013163, "source": "Kaggle public leaderboard"},
    {"label": "Kaggle #4: Haoze Hou", "score": 0.011683, "source": "Kaggle public leaderboard"},
    {"label": "Kaggle #5: hyd", "score": 0.011449, "source": "Kaggle public leaderboard"},
]


LOCAL_CANDIDATES = [
    {
        "label": "Local historical: residual-tail",
        "score": 0.015630171202,
        "source": "Local historical validation",
    },
    {
        "label": "Local historical: dynamic RLS",
        "score": 0.015425344,
        "source": "Local historical validation",
    },
    {
        "label": "Local Stage 3: batch mean/std blend",
        "score": 0.014424968604,
        "source": "Local Stage 3 validation",
    },
    {
        "label": "Local Stage 3: residual-tail modes",
        "score": 0.013851999952,
        "source": "Local Stage 3 validation",
    },
    {
        "label": "Local Stage 3: dynamic RLS",
        "score": 0.013836465051,
        "source": "Local Stage 3 validation",
    },
]


def write_csv(rows: list[dict[str, str | float]]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["label", "score", "source"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def render_chart(rows: list[dict[str, str | float]]) -> None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: float(row["score"]), reverse=True)

    labels = [str(row["label"]) for row in ordered]
    scores = [float(row["score"]) for row in ordered]
    colors = []
    for row in ordered:
        source = str(row["source"])
        if source.startswith("Local Stage 3"):
            colors.append("#c65f2e")
        elif source.startswith("Local historical"):
            colors.append("#845d93")
        else:
            colors.append("#2f6f9f")

    fig, ax = plt.subplots(figsize=(12.2, 7.4))
    bars = ax.barh(labels, scores, color=colors, edgecolor="#1d1d1f", linewidth=0.6)
    ax.invert_yaxis()
    ax.set_xlabel("Score / local global_r2")
    ax.set_title("Kaggle public leaderboard vs preserved local candidates")
    ax.set_xlim(0.0, max(scores) * 1.14)
    ax.grid(axis="x", linestyle="--", alpha=0.28)

    for bar, score in zip(bars, scores, strict=True):
        ax.text(
            score + max(scores) * 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.6f}",
            va="center",
            fontsize=9,
        )

    ax.text(
        0.0,
        -0.16,
        "Local bars are offline OOF, historical, or Stage 3 validation scores, not official Kaggle submissions.",
        transform=ax.transAxes,
        fontsize=9,
        color="#4a4a4a",
    )

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = LOCAL_CANDIDATES + KAGGLE_PUBLIC_TOP
    write_csv(rows)
    render_chart(rows)


if __name__ == "__main__":
    main()
