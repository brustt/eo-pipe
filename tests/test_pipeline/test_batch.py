"""Tests for BatchStrategy subclasses."""

from pathlib import Path

import pytest

from eo_pipe.io.output_types import FlushedOutput
from eo_pipe.pipeline.base import StepBase, StepOutput, StepResult
from eo_pipe.pipeline.batch import MergeBatch, ParallelBatch, SingleBatch


# ---------------------------------------------------------------------------
# Spy step — records calls without touching the filesystem
# ---------------------------------------------------------------------------


class _SpyStep(StepBase):
    name = "_spy"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    def execute(self, inputs, output_dir, **params):
        self.calls.append(list(inputs))
        return StepOutput(
            outputs=[FlushedOutput(output_dir / f"out_{len(self.calls)}.tif")],
            metadata={"call": len(self.calls)},
        )


# ---------------------------------------------------------------------------
# ParallelBatch
# ---------------------------------------------------------------------------


class TestParallelBatch:
    def test_called_once_per_input(self, tmp_path):
        inputs = [tmp_path / "a.tif", tmp_path / "b.tif"]
        step = _SpyStep()
        batch = ParallelBatch()
        batch.apply(step, inputs, tmp_path)
        assert len(step.calls) == 2

    def test_each_call_has_single_input(self, tmp_path):
        inputs = [tmp_path / "a.tif", tmp_path / "b.tif"]
        step = _SpyStep()
        batch = ParallelBatch()
        batch.apply(step, inputs, tmp_path)
        for call_inputs in step.calls:
            assert len(call_inputs) == 1

    def test_outputs_aggregated(self, tmp_path):
        inputs = [tmp_path / "a.tif", tmp_path / "b.tif", tmp_path / "c.tif"]
        step = _SpyStep()
        batch = ParallelBatch()
        result = batch.apply(step, inputs, tmp_path)
        assert len(result.outputs) == 3

    def test_empty_inputs(self, tmp_path):
        step = _SpyStep()
        batch = ParallelBatch()
        result = batch.apply(step, [], tmp_path)
        assert result.outputs == []
        assert step.calls == []


# ---------------------------------------------------------------------------
# MergeBatch
# ---------------------------------------------------------------------------


class TestMergeBatch:
    def test_called_once_with_all_inputs(self, tmp_path):
        inputs = [tmp_path / "a.tif", tmp_path / "b.tif", tmp_path / "c.tif"]
        step = _SpyStep()
        batch = MergeBatch()
        batch.apply(step, inputs, tmp_path)
        assert len(step.calls) == 1
        assert len(step.calls[0]) == 3

    def test_result_propagated(self, tmp_path):
        inputs = [tmp_path / "a.tif"]
        step = _SpyStep()
        batch = MergeBatch()
        result = batch.apply(step, inputs, tmp_path)
        assert len(result.outputs) == 1


# ---------------------------------------------------------------------------
# SingleBatch
# ---------------------------------------------------------------------------


class TestSingleBatch:
    def test_one_input_ok(self, tmp_path):
        inputs = [tmp_path / "a.tif"]
        step = _SpyStep()
        batch = SingleBatch()
        result = batch.apply(step, inputs, tmp_path)
        assert len(result.outputs) == 1

    def test_two_inputs_raises(self, tmp_path):
        inputs = [tmp_path / "a.tif", tmp_path / "b.tif"]
        step = _SpyStep()
        batch = SingleBatch()
        with pytest.raises(ValueError, match="exactly one input"):
            batch.apply(step, inputs, tmp_path)

    def test_zero_inputs_raises(self, tmp_path):
        step = _SpyStep()
        batch = SingleBatch()
        with pytest.raises(ValueError, match="exactly one input"):
            batch.apply(step, [], tmp_path)
