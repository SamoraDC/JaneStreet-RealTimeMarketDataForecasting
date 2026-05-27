"""Competition metrics."""

from collections.abc import Iterable
from math import isfinite

import polars as pl


def weighted_zero_mean_r2(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    weights: Iterable[float],
) -> float:
    """Return sample-weighted zero-mean R2.

    This is the local equivalent of the Jane Street objective:
    1 - sum(w * (y - y_hat)^2) / sum(w * y^2).
    """

    numerator = 0.0
    denominator = 0.0

    for idx, (target, pred, weight) in enumerate(zip(y_true, y_pred, weights, strict=True)):
        target = float(target)
        pred = float(pred)
        weight = float(weight)

        if weight < 0.0:
            raise ValueError(f"weights must be non-negative; got {weight} at index {idx}")
        if not (isfinite(target) and isfinite(pred) and isfinite(weight)):
            raise ValueError(f"non-finite metric input at index {idx}")

        error = target - pred
        numerator += weight * error * error
        denominator += weight * target * target

    if denominator <= 0.0:
        raise ValueError("weighted target energy must be positive")

    return 1.0 - numerator / denominator


def weighted_zero_mean_r2_polars(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    target: str = "responder_6",
    prediction: str = "prediction",
    weight: str = "weight",
) -> float:
    """Return sample-weighted zero-mean R2 from a Polars frame."""

    lazy = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    result = lazy.select(
        (pl.col(weight) * (pl.col(target) - pl.col(prediction)).pow(2)).sum().alias("numerator"),
        (pl.col(weight) * pl.col(target).pow(2)).sum().alias("denominator"),
    ).collect()

    numerator = float(result["numerator"][0])
    denominator = float(result["denominator"][0])
    if denominator <= 0.0:
        raise ValueError("weighted target energy must be positive")

    return 1.0 - numerator / denominator

