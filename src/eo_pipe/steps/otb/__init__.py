import os

# OTB env profiles set PROJ_LIB/PROJ_DATA to OTB's bundled (older) database.
# Clear them before rasterio/pyproj initialise so the parent process uses the
# system PROJ installation. OTB subprocesses inherit the original env and find
# their own PROJ correctly. See docs/adr/0002-proj-env-cleared-at-otb-import.md.
os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)

from eo_pipe.steps.otb.base import OTBStepBase
from eo_pipe.steps.otb.orthorectify import OrthoRectifyStep
from eo_pipe.steps.otb.sar_calibrate import SARCalibrationStep
from eo_pipe.steps.otb.sar_border_cut import SARBorderCutStep

__all__ = ["OTBStepBase", "OrthoRectifyStep", "SARCalibrationStep", "SARBorderCutStep"]
