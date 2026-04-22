"""Example usage of eo-pipe for Sentinel raster processing.

Two usage patterns are demonstrated:

  1. Standalone utility  — ``downsample_raster`` writes eagerly to disk;
                           useful for a single preprocessing step outside
                           a pipeline.

  2. Direct step calls   — ``StepBase.run()`` returns a lazy ``StepOutput``;
                           nothing is written until ``.flush_all()`` is called.
                           Useful for fine-grained control or custom orchestration.

  3. Pipeline composition — fluent builder that chains steps automatically,
                            flushing each step's outputs before passing them to
                            the next.

Inputs
------
Fill in the three placeholders below before running:

  SENTINEL_TILES   : list of Sentinel-2 (*.tif) or Sentinel-1 (*.tif) tiles
  STUDY_AREA_SHP   : path to a GeoPackage / Shapefile delimiting the study area
  WORKSPACE        : working directory; intermediate outputs land here

Run
---
    uv run python scripts/example.py
"""

from pydantic_core.core_schema import FieldPlainNoInfoSerializerFunction
from pathlib import Path
from typing import Optional
from datetime import datetime
import rootutils

import eo_pipe  # noqa: F401 — triggers step self-registration
from eo_pipe import MergeBatch, ParallelBatch, PipelineComposition
from eo_pipe.steps.raster.clip import ClipStep
from eo_pipe.steps.raster.resample import ResampleStep, downsample_raster

# ---------------------------------------------------------------------------
# Inputs — fill in before running
# ---------------------------------------------------------------------------

PROJECT_ROOT = rootutils.find_root(search_from=__file__, indicator="pyproject.toml")

DATA_FOLDER = PROJECT_ROOT.parent.parent / "data"

SENTINEL_TILES: list[Path] = [
    DATA_FOLDER / "S2/S2C_MSIL2A_20260218T105111_N0512_R051_T31TCK_20260218T145711_RGB.tif",
]

STUDY_AREA_SHP: Path = DATA_FOLDER / "lot_dpt.gpkg"

WORKSPACE: Path = DATA_FOLDER / "interim" / "run_example"

TARGET_RESOLUTION: float = 30.


def _check_inputs() -> None:
    errors: list[str] = []

    if not SENTINEL_TILES:
        errors.append("SENTINEL_TILES is empty — add at least one tile path.")
    else:
        for p in SENTINEL_TILES:
            if not p.exists():
                errors.append(f"Tile not found: {p}")

    if STUDY_AREA_SHP is not None and not STUDY_AREA_SHP.exists():
        errors.append(f"Study area file not found: {STUDY_AREA_SHP}")

    if errors:
        raise FileNotFoundError(
            "Please fill in the input placeholders in scripts/example.py:\n"
            + "\n".join(f"  • {e}" for e in errors)
        )


def run_steps(study_area_file: Path, inputs: list[Path]) -> None:
    """Run clip step."""

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    workspace = WORKSPACE / timestamp
    clip_dir = workspace / "clip"
    resample_dir = workspace / "resample"

    print("[direct] clipping to study area …")
    clip_result = ClipStep().run(
        inputs=inputs,
        output_dir=clip_dir,
        shp=study_area_file,
        save_overlay=False,
    ).flush_all()

    print(f"[direct] clipped {len(clip_result.outputs)} tile(s):")
    for p in clip_result.outputs:
        print(f"  {p}")

    print("[direct] resampling tiles …")
    resample_result = ResampleStep().run(
        inputs=clip_result.outputs,
        output_dir=resample_dir,
        target_resolution=TARGET_RESOLUTION,
        method="average",
    ).flush_all()  # writes to disk here

    resampled_tiles = resample_result.outputs
    print(f"[direct] resampled {len(resampled_tiles)} tile(s):")
    for p in resampled_tiles:
        print(f"  {p}")


# ---------------------------------------------------------------------------
# Example 3 — pipeline composition (fluent builder)
# ---------------------------------------------------------------------------


def run_pipeline_composition() -> None:
    """Chain steps with the fluent ``PipelineComposition`` builder.

    The pipeline threads each step's flushed outputs as inputs to the next.
    With ``save_intermediate=True``, per-step directories are kept on disk
    for inspection; set it to ``False`` (default) to use a temp dir and only
    keep the final outputs.
    """
    if STUDY_AREA_SHP is None:
        print("[pipeline] skipping (STUDY_AREA_SHP not set — clip step requires it)")
        return

    workspace = WORKSPACE / "pipeline"

    print("[pipeline] running composition …")
    ctx = (
        PipelineComposition(workspace=workspace)

        # 1. Clip each resampled tile to the study area.
        .add_step(
            "clip",
            ParallelBatch(),
            shp=STUDY_AREA_SHP,
            save_overlay=True,
        )

        # 2. Resample all tiles to a common resolution — outputs deferred until
        #    the batch strategy calls flush() for each RasterOutput.
        .add_step(
            "resample",
            ParallelBatch(),
            target_resolution=TARGET_RESOLUTION,
            method="average",
        )
        # 3. Merge all clipped tiles into a single mosaic (N → 1).
        .add_step(
            "merge_raster",
            MergeBatch(),
            output_name="mosaic",
            method="first",
        )
        .run(inputs=SENTINEL_TILES, save_intermediate=True)
    )

    print("[pipeline] done.")
    print(f"[pipeline] final outputs ({len(ctx.inputs)}):")
    for p in ctx.inputs:
        print(f"  {p}")

    # Per-step results are accessible from the context after the run.
    clip_result = ctx.outputs.get("clip")
    resample_result = ctx.outputs.get("resample")
    merge_result = ctx.outputs.get("merge_raster")

    if clip_result:
        print(f"[pipeline] clip → {len(clip_result.outputs)} file(s)")
        if clip_result.artifacts:
            print(f"  artifacts: {list(clip_result.artifacts.keys())}")

    if resample_result: 
        print(f"[pipeline] resample → {len(resample_result.outputs)} file(s)")

    if merge_result:
        print(f"[pipeline] merge → {merge_result.outputs}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _check_inputs()
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    run_pipeline_composition()
    # run_steps(STUDY_AREA_SHP, SENTINEL_TILES)