"""Integration tests for PipelineComposition."""

from pathlib import Path

import rasterio as rio
import pytest

from eo_pipe.io.raster_io import RasterWriter
from eo_pipe.pipeline.batch import MergeBatch, ParallelBatch
from eo_pipe.pipeline.composition import PipelineComposition


class TestPipelineComposition:
    # ------------------------------------------------------------------ #
    # Single-step runs                                                     #
    # ------------------------------------------------------------------ #

    def test_single_resample_step(self, single_raster, tmp_path):
        out_dir = tmp_path / "out"
        ctx = (
            PipelineComposition()
            .add_step("resample", ParallelBatch(), target_resolution=0.02)
            .run(inputs=[single_raster], output_dir=out_dir, save_intermediate=False)
        )

        assert len(ctx.inputs) == 1
        assert ctx.inputs[0].exists()

        with rio.open(ctx.inputs[0]) as dst:
            assert dst.width == 32

    def test_two_step_resample_then_filter(self, single_raster, tmp_path):
        out_dir = tmp_path / "out"
        ctx = (
            PipelineComposition()
            .add_step("resample", ParallelBatch(), target_resolution=0.02)
            .add_step("filter", ParallelBatch(), method="median", kernel_size=3)
            .run(inputs=[single_raster], output_dir=out_dir, save_intermediate=False)
        )

        assert ctx.inputs[0].exists()
        with rio.open(ctx.inputs[0]) as dst:
            assert dst.width == 32  # still 32 after filter

    # ------------------------------------------------------------------ #
    # save_intermediate behaviour                                          #
    # ------------------------------------------------------------------ #

    def test_save_intermediate_creates_step_dirs(self, single_raster, tmp_path):
        workspace = tmp_path / "workspace"
        ctx = (
            PipelineComposition(workspace=workspace)
            .add_step("resample", ParallelBatch(), target_resolution=0.02)
            .run(inputs=[single_raster], save_intermediate=True)
        )

        step_dirs = list(workspace.iterdir())
        assert any(d.is_dir() and "resample" in d.name for d in step_dirs)

    def test_no_intermediate_final_copied_to_output_dir(self, single_raster, tmp_path):
        out_dir = tmp_path / "final"
        ctx = (
            PipelineComposition()
            .add_step("resample", ParallelBatch(), target_resolution=0.02)
            .run(inputs=[single_raster], output_dir=out_dir, save_intermediate=False)
        )

        # Final output must live inside out_dir
        for p in ctx.inputs:
            assert str(p).startswith(str(out_dir))

    # ------------------------------------------------------------------ #
    # Multiple inputs / merge batch                                        #
    # ------------------------------------------------------------------ #

    def test_parallel_then_merge(self, two_rasters, tmp_path):
        out_dir = tmp_path / "out"
        ctx = (
            PipelineComposition()
            .add_step("resample", ParallelBatch(), target_resolution=0.02)
            .add_step("merge_raster", MergeBatch(), to_cog=False)
            .run(inputs=two_rasters, output_dir=out_dir, save_intermediate=False)
        )

        # After merging, exactly one output
        assert len(ctx.inputs) == 1
        assert ctx.inputs[0].exists()

    # ------------------------------------------------------------------ #
    # Context & metadata                                                   #
    # ------------------------------------------------------------------ #

    def test_step_results_recorded_in_context(self, single_raster, tmp_path):
        out_dir = tmp_path / "out"
        ctx = (
            PipelineComposition()
            .add_step("resample", ParallelBatch(), target_resolution=0.02)
            .run(inputs=[single_raster], output_dir=out_dir, save_intermediate=False)
        )

        assert "resample" in ctx.outputs

    def test_user_metadata_passed_through(self, single_raster, tmp_path):
        out_dir = tmp_path / "out"
        ctx = (
            PipelineComposition(metadata={"zone": "test_zone", "res": 0.02})
            .add_step("resample", ParallelBatch(), target_resolution=0.02)
            .run(inputs=[single_raster], output_dir=out_dir, save_intermediate=False)
        )

        assert ctx.metadata["zone"] == "test_zone"

    # ------------------------------------------------------------------ #
    # Step lookup by name                                                  #
    # ------------------------------------------------------------------ #

    def test_step_by_string_name(self, single_raster, tmp_path):
        out_dir = tmp_path / "out"
        ctx = (
            PipelineComposition()
            .add_step("resample", target_resolution=0.02)  # no batch → ParallelBatch default
            .run(inputs=[single_raster], output_dir=out_dir, save_intermediate=False)
        )
        assert ctx.inputs[0].exists()

    def test_unknown_step_name_raises(self):
        with pytest.raises(KeyError):
            PipelineComposition().add_step("__not_a_step__")

    # ------------------------------------------------------------------ #
    # Writer injection                                                     #
    # ------------------------------------------------------------------ #

    def test_custom_writer_used_by_step(self, single_raster, tmp_path):
        """Writer passed to PipelineComposition must reach the step."""
        out_dir = tmp_path / "out"
        custom_writer = RasterWriter(compress="lzw", tiled=False)
        ctx = (
            PipelineComposition(writer=custom_writer)
            .add_step("filter", ParallelBatch(), method="median", kernel_size=3)
            .run(inputs=[single_raster], output_dir=out_dir, save_intermediate=False)
        )
        assert ctx.inputs[0].exists()

    def test_preinstantiated_step_keeps_its_writer(self, single_raster, tmp_path):
        """A pre-instantiated step with its own writer is not overwritten by the composition."""
        out_dir = tmp_path / "out"
        step_writer = RasterWriter(compress="lzw")
        from eo_pipe.steps.raster.filter import FilterStep
        step = FilterStep(writer=step_writer)
        assert step._writer is step_writer

        ctx = (
            PipelineComposition()  # default writer
            .add_step(step, ParallelBatch(), method="median", kernel_size=3)
            .run(inputs=[single_raster], output_dir=out_dir, save_intermediate=False)
        )
        # Step's own writer should be unchanged
        assert step._writer is step_writer
        assert ctx.inputs[0].exists()

    # ------------------------------------------------------------------ #
    # Preflight checks                                                     #
    # ------------------------------------------------------------------ #

    def test_preflight_passes_when_all_available(self, single_raster, tmp_path):
        out_dir = tmp_path / "out"
        # resample is always available (pure Python) — should not raise
        ctx = (
            PipelineComposition()
            .add_step("resample", ParallelBatch(), target_resolution=0.02)
            .run(inputs=[single_raster], output_dir=out_dir, preflight=True)
        )
        assert ctx.inputs[0].exists()

    def test_preflight_raises_when_step_unavailable(self, single_raster, tmp_path):
        from unittest.mock import patch
        from eo_pipe.pipeline.base import StepBase

        with patch.object(StepBase, "is_available", return_value=False):
            with pytest.raises(RuntimeError, match="preflight failed"):
                (
                    PipelineComposition()
                    .add_step("resample", ParallelBatch(), target_resolution=0.02)
                    .run(inputs=[single_raster], preflight=True)
                )

    def test_preflight_false_skips_availability_check(self, single_raster, tmp_path):
        from unittest.mock import patch
        from eo_pipe.pipeline.base import StepBase

        out_dir = tmp_path / "out"
        with patch.object(StepBase, "is_available", return_value=False):
            # preflight=False → no RuntimeError despite unavailable step
            ctx = (
                PipelineComposition()
                .add_step("resample", ParallelBatch(), target_resolution=0.02)
                .run(inputs=[single_raster], output_dir=out_dir, preflight=False)
            )
        assert ctx.inputs[0].exists()

    def test_preflight_error_lists_all_unavailable_steps(self, single_raster, tmp_path):
        from unittest.mock import patch
        from eo_pipe.pipeline.base import StepBase

        with patch.object(StepBase, "is_available", return_value=False):
            with pytest.raises(RuntimeError, match="resample") as exc_info:
                (
                    PipelineComposition()
                    .add_step("resample", ParallelBatch(), target_resolution=0.02)
                    .add_step("filter", ParallelBatch(), method="median", kernel_size=3)
                    .run(inputs=[single_raster], preflight=True)
                )
            assert "filter" in str(exc_info.value)

    # ------------------------------------------------------------------ #
    # Empty composition                                                    #
    # ------------------------------------------------------------------ #

    def test_empty_composition_returns_inputs(self, single_raster, tmp_path):
        out_dir = tmp_path / "out"
        ctx = (
            PipelineComposition()
            .run(inputs=[single_raster], output_dir=out_dir, save_intermediate=False)
        )
        assert len(ctx.inputs) == 1
        assert ctx.inputs[0].exists()
        assert str(ctx.inputs[0]).startswith(str(out_dir))
