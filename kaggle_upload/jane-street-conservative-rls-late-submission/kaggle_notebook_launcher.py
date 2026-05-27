"""Kaggle notebook launcher for the conservative RLS late-submission package.

Paste this file's contents into one Kaggle notebook cell, or run it from an
attached Dataset. It auto-detects the package directory under /kaggle/input.
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path


def find_package_dir() -> Path:
    input_root = Path("/kaggle/input")
    expected = Path("submission/submission.py")
    for candidate in sorted(input_root.glob("*")):
        if (candidate / expected).exists() and (
            candidate / "artifacts/jane_street_submission/meta_rls_experts_alpha10000_f0p995/rls_state.npz"
        ).exists():
            return candidate
    raise FileNotFoundError("Could not find the attached late-submission package under /kaggle/input.")


PACKAGE_DIR = find_package_dir()
os.environ.setdefault("JANE_STREET_BASE_ARTIFACT_DIR", str(PACKAGE_DIR / "artifacts/jane_street_submission/base_models"))
os.environ.setdefault(
    "JANE_STREET_META_ARTIFACT_DIR",
    str(PACKAGE_DIR / "artifacts/jane_street_submission/meta_rls_experts_alpha10000_f0p995"),
)
os.environ.setdefault("JANE_STREET_DEVICE", "cuda")

runpy.run_path(str(PACKAGE_DIR / "submission/submission.py"), run_name="__main__")
