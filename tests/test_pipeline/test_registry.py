"""Tests for StepRegistry."""

import pytest

from eo_pipe.io.output_types import FlushedOutput
from eo_pipe.io.raster_io import RasterWriter
from eo_pipe.pipeline.base import StepBase, StepOutput, StepResult
from eo_pipe.pipeline.registry import StepRegistry


# ---------------------------------------------------------------------------
# Helper: isolated registry to avoid polluting the global one
# ---------------------------------------------------------------------------

class _IsolatedRegistry(StepRegistry):
    _registry = {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStepRegistry:
    def test_register_and_create(self):
        @_IsolatedRegistry.register
        class _FakeStep(StepBase):
            name = "_fake_isolated"
            def execute(self, inputs, output_dir, **params):
                return StepOutput()

        step = _IsolatedRegistry.create("_fake_isolated")
        assert isinstance(step, _FakeStep)

    def test_create_unknown_raises(self):
        with pytest.raises(KeyError, match="not registered"):
            StepRegistry.create("__does_not_exist__")

    def test_register_without_name_raises(self):
        with pytest.raises(TypeError, match="name"):
            @_IsolatedRegistry.register
            class _NoName(StepBase):
                def execute(self, inputs, output_dir, **params):
                    return StepResult()

    def test_available_returns_sorted_list(self):
        names = StepRegistry.available()
        assert names == sorted(names)
        assert isinstance(names, list)

    def test_register_overwrites(self):
        """Registering twice with the same name replaces the old class."""
        @_IsolatedRegistry.register
        class _V1(StepBase):
            name = "_overwrite_test"
            def execute(self, inputs, output_dir, **params):
                return StepResult(metadata={"v": 1})

        @_IsolatedRegistry.register
        class _V2(StepBase):
            name = "_overwrite_test"
            def execute(self, inputs, output_dir, **params):
                return StepResult(metadata={"v": 2})

        step = _IsolatedRegistry.create("_overwrite_test")
        assert isinstance(step, _V2)

    def test_builtin_steps_registered(self):
        """Importing eo_pipe should auto-register all built-in steps."""
        import eo_pipe  # triggers step registration
        names = StepRegistry.available()
        for expected in ("resample", "clip", "filter", "merge_raster"):
            assert expected in names, f"'{expected}' not registered"

    def test_create_with_writer_kwarg(self):
        """writer= passed to create() must be stored on the step instance."""
        import eo_pipe  # ensure steps are registered
        custom_writer = RasterWriter(compress="lzw")
        step = StepRegistry.create("clip", writer=custom_writer)
        assert step._writer is custom_writer
