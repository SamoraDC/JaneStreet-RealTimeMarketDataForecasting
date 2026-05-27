"""Build a manual Kaggle Dataset package for late submission.

The package contains the submission entrypoint, the local `janestreet` package,
the conservative RLS meta artifact, final base model artifacts, and tiny vendored
Python modules that are not guaranteed to exist in Kaggle's offline runtime.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_NAME = "dynamic_gateway_rls_experts_alpha10000_f0p995"
PACKAGE_SLUG = "jane-street-conservative-rls-late-submission"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "kaggle_upload" / PACKAGE_SLUG)
    parser.add_argument("--base-artifact-dir", type=Path, default=REPO_ROOT / "artifacts/jane_street_submission/base_models")
    parser.add_argument(
        "--meta-artifact-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/jane_street_submission/meta_rls_experts_alpha10000_f0p995",
    )
    parser.add_argument("--no-zip", action="store_true", help="Do not create a .zip archive next to the package directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    _require_dir(args.base_artifact_dir)
    _require_dir(args.meta_artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _copy_tree(REPO_ROOT / "submission", output_dir / "submission")
    _copy_tree(REPO_ROOT / "src" / "janestreet", output_dir / "src" / "janestreet")
    _copy_tree(args.base_artifact_dir, output_dir / "artifacts/jane_street_submission/base_models")
    _copy_tree(args.meta_artifact_dir, output_dir / "artifacts/jane_street_submission/meta_rls_experts_alpha10000_f0p995")
    _copy_vendor_module("tabm", output_dir / "vendor")
    _copy_vendor_module("rtdl_num_embeddings", output_dir / "vendor")

    _write_text(output_dir / "kaggle_notebook_launcher.py", _launcher_text())
    _write_text(output_dir / "README_KAGGLE_SUBMISSION.md", _readme_text())
    shutil.copy2(
        REPO_ROOT / "notebooks" / "jane_street_conservative_late_submission.ipynb",
        output_dir / "jane_street_conservative_late_submission.ipynb",
    )
    _write_manifest(output_dir, args.base_artifact_dir, args.meta_artifact_dir)
    _remove_pycache(output_dir)

    if not args.no_zip:
        archive = shutil.make_archive(str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name)
        print(f"wrote {archive}")
    print(f"wrote {output_dir}")


def _require_dir(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(path)


def _copy_tree(source: Path, destination: Path) -> None:
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    shutil.copytree(source, destination, dirs_exist_ok=True, ignore=ignore)


def _copy_vendor_module(module_name: str, vendor_dir: Path) -> None:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(module_name)
    vendor_dir.mkdir(parents=True, exist_ok=True)
    source = Path(spec.origin)
    if source.name == "__init__.py":
        _copy_tree(source.parent, vendor_dir / source.parent.name)
        return
    shutil.copy2(source, vendor_dir / source.name)


def _remove_pycache(root: Path) -> None:
    for path in root.rglob("__pycache__"):
        if path.is_dir():
            shutil.rmtree(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _launcher_text() -> str:
    return '''"""Kaggle notebook launcher for the conservative RLS late-submission package.

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
'''


def _readme_text() -> str:
    return """# Jane Street Conservative RLS Late Submission Package

This package is prepared for a manual Kaggle late submission to the Jane Street Real-Time Market Data Forecasting competition.

## Candidate

- Strategy: `dynamic_gateway_rls_experts_alpha10000_f0p995`
- Local Stage 3 score: `global_r2=0.013836465`, `min_fold_r2=0.007030887`
- Historical confirmation: `global_r2=0.015425344`
- Intended first submission: conservative RLS meta-layer over TabM, XGBoost, LightGBM, Ridge-calibrated predictions, and tree ensemble prediction.

## Files

- `submission/submission.py`: Kaggle inference server entrypoint.
- `src/janestreet/`: local package required by the entrypoint.
- `artifacts/jane_street_submission/base_models/`: final TabM and tree artifacts.
- `artifacts/jane_street_submission/meta_rls_experts_alpha10000_f0p995/`: conservative RLS meta-state.
- `vendor/`: small offline modules required by the TabM artifact.
- `kaggle_notebook_launcher.py`: one-cell Kaggle launcher.
- `jane_street_conservative_late_submission.ipynb`: importable Kaggle Notebook.

## Manual Kaggle Upload

1. Create a new private Kaggle Dataset.
2. Upload the contents of this package directory. If you use the `.zip` archive for transfer, make sure the Kaggle Dataset exposes the extracted files rather than only one zip file.
3. Create a new notebook for the competition, or import `jane_street_conservative_late_submission.ipynb`.
4. Attach the competition data and this package Dataset.
5. Set Accelerator to GPU if available.
6. Disable internet.
7. Paste the contents of `kaggle_notebook_launcher.py` into the first notebook cell.
8. Save a notebook version.
9. Submit that notebook version as the late submission.

## Audit Notes

The online update is causal by construction: at `date_id=D`, it updates from cached `D-1` features joined to gateway-provided `responder_*_lag_1`, then predicts the current batch. Local gateway smoke passed, but the official local mock has only one `date_id`; the Kaggle rerun remains the real packaging test.
"""


def _write_manifest(output_dir: Path, base_artifact_dir: Path, meta_artifact_dir: Path) -> None:
    manifest = {
        "package_slug": PACKAGE_SLUG,
        "candidate_name": CANDIDATE_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_stage3_global_r2": 0.013836465,
        "local_stage3_min_fold_r2": 0.007030887,
        "historical_max1398_global_r2": 0.015425344,
        "base_artifact_source": str(base_artifact_dir),
        "meta_artifact_source": str(meta_artifact_dir),
        "submission_entrypoint": "submission/submission.py",
        "kaggle_launcher": "kaggle_notebook_launcher.py",
        "notebook": "jane_street_conservative_late_submission.ipynb",
    }
    _write_text(output_dir / "package_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
