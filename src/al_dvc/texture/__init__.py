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

__all__ = [
    "ESTIMATORS",
    "THRESHOLDS",
    "THRESHOLD_LABELS",
    "Autocorrelation",
    "Crossing",
    "Profile",
    "TextureResult",
    "analyse_texture",
    "analysis_window",
    "analytic_length",
    "autocorrelation",
    "boolean_correlation",
    "boolean_spheres",
    "correlation_length",
    "directional_profiles",
    "lengths",
    "radial_profile",
]
