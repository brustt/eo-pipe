"""Sentinel-1 GRD preprocessing pipelines: calibration, border cut, orthorectification.

Three pipelines available:

  1. ``run_calibration_only``   — SARCalibration only (DN → σ⁰/γ⁰/β⁰).
  2. ``run_ortho_only``         — OrthoRectification only (assumes already calibrated).
  3. ``run_full_pipeline``      — Calibration → border cut → orthorectification.

Inputs
------
Fill in the constants below before running.

  S1_TILES     : list of pre-extracted S1 GRD measurement .tif files (one per polarisation)
  REFERENCE    : reference image that defines the target grid (CRS, bounds, pixel size)
  DEM_DIR      : directory of DEM tiles for elevation-aware orthorectification — None to skip
  GEOID_FILE   : geoid .bsb file for altitude correction — None to skip
  STUDY_AREA   : optional GeoPackage/Shapefile for final clip

Run
---
    uv run python scripts/sentinel1_preprocessing.py [calibrate|ortho|full]
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import rootutils

import eo_pipe  # noqa: F401 — triggers step self-registration
from eo_pipe import ParallelBatch, PipelineComposition

# ---------------------------------------------------------------------------
# Inputs — fill in before running
# ---------------------------------------------------------------------------

PROJECT_ROOT = rootutils.find_root(search_from=__file__, indicator="pyproject.toml")
DATA_DIR = PROJECT_ROOT.parent.parent / "data"

S1_TILES: list[Path] = [
    DATA_DIR / "S1" / "S1C_IW_GRDH_1SDV_20260211T060734_20260211T060759_006304_00CABF_54BF/measurement" / "s1c-iw-grd-vh-20260211t060734-20260211t060759-006304-00cabf-002.tiff",
    #DATA_DIR / "S1" / "S1C_IW_GRDH_1SDV_20260211T060734_20260211T060759_006304_00CABF_54BF/measurement" / "s1c-iw-grd-vv-20260211t060734-20260211t060759-006304-00cabf-001.tiff",
]

S1_TILES_CALIBRATED: list[Path] = [
    DATA_DIR / "interim" / "s1_preprocess" / "2026-05-19_10-57-46" / "00_sar_calibrate" / "sar_calibrate_s1c-iw-grd-vh-20260211t060734-20260211t060759-006304-00cabf-002.tiff",
    #DATA_DIR / "interim" / "s1_preprocess" / "2026-05-19_10-57-46" / "00_sar_calibrate" / "sar_calibrate_s1c-iw-grd-vv-20260211t060734-20260211t060759-006304-00cabf-001.tiff",
]

REFERENCE: Path = DATA_DIR / "S2" / "S2B_MSIL2A_20250618T104619_N0511_R051_T31TCK_20250618T134459_RGB_4326.tif"

STUDY_AREA: Optional[Path] = DATA_DIR / "lot_dpt.gpkg"

DEM_DIR: Optional[Path] = DATA_DIR / "SRTM_30_hgt"

GEOID_FILE: Optional[Path] = None  # e.g. DATA_DIR / "geoid" / "egm96.bsb"

WORKSPACE: Path = DATA_DIR / "interim" / "s1_preprocess"

# Calibration
LUT: str = "sigma"          # "sigma" (σ⁰), "gamma" (γ⁰), "beta" (β⁰), "dn"
REMOVENOISE: bool = False   # True only for products with intact noise LUTs

# Orthorectification
INTERPOLATOR: str = "bco"   # "bco" (bicubic), "nn" (nearest), "linear"
GRID_SPACING: float = 4.0   # OTB resampling-grid spacing in pixels
RAM_MB: int = 1024          # OTB RAM budget in MB


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _check_inputs(require_reference: bool = True) -> None:
    errors: list[str] = []

    if not S1_TILES:
        errors.append("S1_TILES is empty — add at least one tile path.")
    else:
        for p in S1_TILES:
            if not p.exists():
                errors.append(f"S1 tile not found: {p}")

    if require_reference and not REFERENCE.exists():
        errors.append(f"Reference image not found: {REFERENCE}")

    if STUDY_AREA is not None and not STUDY_AREA.exists():
        errors.append(f"Study area file not found: {STUDY_AREA}")

    if DEM_DIR is not None and not DEM_DIR.exists():
        errors.append(f"DEM directory not found: {DEM_DIR}")

    if GEOID_FILE is not None and not GEOID_FILE.exists():
        errors.append(f"Geoid file not found: {GEOID_FILE}")

    if errors:
        raise FileNotFoundError(
            "Fix inputs in scripts/sentinel1_preprocessing.py:\n"
            + "\n".join(f"  • {e}" for e in errors)
        )


# ---------------------------------------------------------------------------
# Pipeline 1 — calibration only
# ---------------------------------------------------------------------------


def run_calibration_only(workspace: Path) -> None:
    """SARCalibration only: DN → physical backscatter (σ⁰ / γ⁰ / β⁰).

    Output: one calibrated .tif per input polarisation.
    """
    print("[calibration] running SAR calibration pipeline …")

    ctx = (
        PipelineComposition(workspace=workspace)
        .add_step(
            "sar_calibrate",
            ParallelBatch(),
            lut=LUT,
            removenoise=REMOVENOISE,
            ram_mb=RAM_MB,
        )
        .run(inputs=S1_TILES, save_intermediate=True)
    )

    print("[calibration] done.")
    print(f"[calibration] outputs ({len(ctx.inputs)}):")
    for p in ctx.inputs:
        print(f"  {p}")


# ---------------------------------------------------------------------------
# Pipeline 2 — orthorectification only
# ---------------------------------------------------------------------------


def run_ortho_only(workspace: Path) -> None:
    """OrthoRectification only: assumes inputs are already calibrated.

    Chains: orthorectify → optional clip.
    """
    print("[ortho] running orthorectification pipeline …")

    steps = (
        PipelineComposition(workspace=workspace)
        .add_step(
            "orthorectify",
            ParallelBatch(),
            ref=REFERENCE,
            interpolator=INTERPOLATOR,
            elev_dem=DEM_DIR,
            elev_geoid=GEOID_FILE,
            grid_spacing=GRID_SPACING,
            ram_mb=RAM_MB,
        )
    )

    if STUDY_AREA is not None:
        steps = steps.add_step("clip", ParallelBatch(), shp=STUDY_AREA)

    ctx = steps.run(inputs=S1_TILES, save_intermediate=True)

    print("[ortho] done.")
    print(f"[ortho] outputs ({len(ctx.inputs)}):")
    for p in ctx.inputs:
        print(f"  {p}")


# ---------------------------------------------------------------------------
# Pipeline 3 — full: calibrate → cut borders → orthorectify
# ---------------------------------------------------------------------------


def run_full_pipeline(workspace: Path) -> None:
    """Full S1 GRD preprocessing pipeline.

    Steps:
      1. SARCalibration    — DN → σ⁰ (or configured LUT)
      2. SARBorderCut      — zero sawtooth margins (auto-detected, skipped if clean)
      3. OrthoRectification — reproject to reference grid with terrain correction
      4. Clip (optional)   — mask to study area
    """
    print("[full] running full S1 preprocessing pipeline …")

    steps = (
        PipelineComposition(workspace=workspace)
        # .add_step(
        #     "sar_calibrate",
        #     ParallelBatch(),
        #     lut=LUT,
        #     removenoise=REMOVENOISE,
        #     ram_mb=RAM_MB,
        # )
        .add_step(
            "sar_cut_borders",
            ParallelBatch(),
            ram_mb=RAM_MB,
        )
        # .add_step(
        #     "orthorectify",
        #     ParallelBatch(),
        #     ref=REFERENCE,
        #     interpolator=INTERPOLATOR,
        #     elev_dem=DEM_DIR,
        #     elev_geoid=GEOID_FILE,
        #     grid_spacing=GRID_SPACING,
        #     ram_mb=RAM_MB,
        # )
    )

    if STUDY_AREA is not None:
        steps = steps.add_step("clip", ParallelBatch(), shp=STUDY_AREA)

    ctx = steps.run(inputs=S1_TILES, save_intermediate=True)

    print("[full] done.")
    print(f"[full] outputs ({len(ctx.inputs)}):")
    for p in ctx.inputs:
        print(f"  {p}")

    for step_name in ("sar_calibrate", "sar_cut_borders", "orthorectify"):
        result = ctx.outputs.get(step_name)
        if result:
            print(f"[full] {step_name} → {len(result.outputs)} file(s)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_workspace = WORKSPACE / timestamp
    run_workspace.mkdir(parents=True, exist_ok=True)

    if mode == "calibrate":
        _check_inputs(require_reference=False)
        run_calibration_only(run_workspace)
    elif mode == "ortho":
        _check_inputs(require_reference=True)
        run_ortho_only(run_workspace)
    else:
        _check_inputs(require_reference=True)
        run_full_pipeline(run_workspace)
