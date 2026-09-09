"""Texture analysis: autocorrelation lengths of a volume, size sweeps and parameter suggestions.

The numbers here describe the image (how far grey values stay correlated along each axis),
which is what the DVC subset has to span; they are not a material property. See
``docs/texture_analysis_plan.md`` for the design and the defects of the scripts it replaces.
"""

from .acf import DEFAULT_MIN_OVERLAP, ESTIMATORS, Autocorrelation, autocorrelation
from .analysis import TextureResult, analyse_texture, analysis_window, result_from_acf
from .boolean_model import analytic_length, boolean_correlation, boolean_spheres
from .boxes import (
    MAX_ANALYSIS_VOXELS,
    Box,
    box_centre,
    box_of_mask,
    box_size,
    centred_window,
    normalise_box,
    whole_box,
)
from .concentric import (
    DEFAULT_COUNT,
    DEFAULT_START,
    DEFAULT_STEP,
    LAG_FRACTION,
    MIN_OVERLAP,
    analyse_cube,
    concentric_boxes,
    concentric_sizes,
    cube_box,
    cube_limits,
    max_lag_for,
    sweep_concentric,
)
from .crossing import THRESHOLD_LABELS, THRESHOLDS, Crossing, correlation_length, lengths
from .profiles import Profile, directional_profiles, radial_profile
from .recommend import Recommendation, recommend_parameters
from .rve import PlateauDecision, SizeLevel, SizeSweep, SubVolume, decide_plateau, sample_positions, size_schedule, sweep_sizes
from .sliding import analyse_range, lag_reach, sliding_autocorrelation

__all__ = [
    "DEFAULT_COUNT",
    "DEFAULT_MIN_OVERLAP",
    "DEFAULT_START",
    "DEFAULT_STEP",
    "ESTIMATORS",
    "LAG_FRACTION",
    "MAX_ANALYSIS_VOXELS",
    "MIN_OVERLAP",
    "THRESHOLDS",
    "THRESHOLD_LABELS",
    "Autocorrelation",
    "Box",
    "Crossing",
    "PlateauDecision",
    "Profile",
    "Recommendation",
    "SizeLevel",
    "SizeSweep",
    "SubVolume",
    "TextureResult",
    "analyse_cube",
    "analyse_range",
    "analyse_texture",
    "analysis_window",
    "analytic_length",
    "autocorrelation",
    "boolean_correlation",
    "boolean_spheres",
    "box_centre",
    "box_of_mask",
    "box_size",
    "centred_window",
    "concentric_boxes",
    "concentric_sizes",
    "correlation_length",
    "cube_box",
    "cube_limits",
    "decide_plateau",
    "directional_profiles",
    "lag_reach",
    "lengths",
    "max_lag_for",
    "normalise_box",
    "radial_profile",
    "recommend_parameters",
    "result_from_acf",
    "sample_positions",
    "size_schedule",
    "sliding_autocorrelation",
    "sweep_concentric",
    "sweep_sizes",
    "whole_box",
]
