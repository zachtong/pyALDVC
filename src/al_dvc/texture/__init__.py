"""Texture analysis: autocorrelation lengths of a volume, size sweeps and parameter suggestions.

The numbers here describe the image (how far grey values stay correlated along each axis),
which is what the DVC subset has to span; they are not a material property. See
``docs/texture_analysis_plan.md`` for the design and the defects of the scripts it replaces.
"""

from .acf import ESTIMATORS, Autocorrelation, autocorrelation
from .analysis import TextureResult, analyse_texture, analysis_window
from .boolean_model import analytic_length, boolean_correlation, boolean_spheres
from .crossing import THRESHOLD_LABELS, THRESHOLDS, Crossing, correlation_length, lengths
from .profiles import Profile, directional_profiles, radial_profile
from .recommend import Recommendation, recommend_parameters
from .rve import PlateauDecision, SizeLevel, SizeSweep, SubVolume, decide_plateau, sample_positions, size_schedule, sweep_sizes

__all__ = [
    "ESTIMATORS",
    "THRESHOLDS",
    "THRESHOLD_LABELS",
    "Autocorrelation",
    "Crossing",
    "PlateauDecision",
    "Profile",
    "Recommendation",
    "SizeLevel",
    "SizeSweep",
    "SubVolume",
    "TextureResult",
    "analyse_texture",
    "analysis_window",
    "analytic_length",
    "autocorrelation",
    "boolean_correlation",
    "boolean_spheres",
    "correlation_length",
    "decide_plateau",
    "directional_profiles",
    "lengths",
    "radial_profile",
    "recommend_parameters",
    "sample_positions",
    "size_schedule",
    "sweep_sizes",
]

from .sliding import (  # noqa: E402
    DEFAULT_MIN_LAG,
    MAX_RANGE_VOXELS,
    analyse_range,
    box_of_mask,
    box_size,
    centred_window,
    lag_reach,
    normalise_box,
    sliding_autocorrelation,
    sweep_concentric,
    sweep_sizes_concentric,
    whole_box,
)

__all__ += [
    "DEFAULT_MIN_LAG",
    "MAX_RANGE_VOXELS",
    "analyse_range",
    "box_of_mask",
    "box_size",
    "centred_window",
    "lag_reach",
    "normalise_box",
    "sliding_autocorrelation",
    "sweep_concentric",
    "sweep_sizes_concentric",
    "whole_box",
]
