from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl


def load_graficos_module():
    path = Path(__file__).resolve().parents[1] / "graficos" / "gerar_graficos.py"
    spec = importlib.util.spec_from_file_location("gerar_graficos", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_weighted_r2_from_components_uses_additive_components() -> None:
    module = load_graficos_module()
    frame = pl.DataFrame({"numerator": [1.0, 2.0], "denominator": [4.0, 6.0]})

    assert module.weighted_r2_from_components(frame) == 0.7


def test_matrix_from_frame_preserves_coordinate_grid() -> None:
    module = load_graficos_module()
    frame = pl.DataFrame(
        {
            "date_bucket": [50, 0, 50, 0],
            "time_bucket": [1, 0, 0, 1],
            "value": [4.0, 1.0, 3.0, 2.0],
        }
    )

    x_values, y_values, matrix = module.matrix_from_frame(frame, "date_bucket", "time_bucket", "value")

    np.testing.assert_array_equal(x_values, np.array([0.0, 50.0]))
    np.testing.assert_array_equal(y_values, np.array([0.0, 1.0]))
    np.testing.assert_array_equal(matrix, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_deterministic_cap_is_stable_and_bounded() -> None:
    module = load_graficos_module()
    frame = pl.DataFrame({"x": list(range(10))})

    capped = module.deterministic_cap(frame, 4)

    assert capped.height <= 4
    assert capped["x"].to_list() == [0, 3, 6, 9]
