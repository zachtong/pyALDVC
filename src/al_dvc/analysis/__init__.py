"""Statistics of DVC results: summary statistics, node selection, rigid-body motion, noise floor, regions,
profiles, a virtual extensometer and the corrected result the views and exports can show.

Qt-free, so the window, the command line (``al-dvc stats``) and scripts share one implementation. The
stored result is never changed: corrections are views computed on demand. See
``docs/plans/2026-09-23-result-statistics.md`` for the design and the definitions.
"""

from .corrected import CorrectedResult, Correction, corrected_result
from .fields import FIELD_GROUPS, available_groups, field_kind, field_unit
from .motion import MOTION_KINDS, MotionFit, corrected_gradient, fit_motion, remove_motion
from .precision import Homogeneous, NoiseFloor, homogeneous, noise_floor, vsg_size
from .profiles import AxisProfile, Extensometer, axis_profile, extensometer, sample_line
from .regions import REGION_SHAPES, Region, next_region_id, regions_from_dicts, regions_to_dicts
from .selection import FILTER_REASONS, NodeFilter, Selection, select_nodes
from .series import FrameStats, frame_series, frame_stats
from .stats import (
    CI_NAMES,
    STAT_NAMES,
    FieldStats,
    VectorStats,
    effective_sample_size,
    mean_confidence,
    spatial_temporal_std,
    strain_precision,
    summarize,
    vector_stats,
)
from .view import FrameView, frame_view

__all__ = [
    "CI_NAMES",
    "FIELD_GROUPS",
    "FILTER_REASONS",
    "MOTION_KINDS",
    "REGION_SHAPES",
    "STAT_NAMES",
    "AxisProfile",
    "CorrectedResult",
    "Correction",
    "Extensometer",
    "FieldStats",
    "FrameStats",
    "FrameView",
    "Homogeneous",
    "MotionFit",
    "NodeFilter",
    "NoiseFloor",
    "Region",
    "Selection",
    "VectorStats",
    "available_groups",
    "axis_profile",
    "corrected_gradient",
    "corrected_result",
    "effective_sample_size",
    "extensometer",
    "field_kind",
    "field_unit",
    "fit_motion",
    "frame_series",
    "frame_stats",
    "frame_view",
    "homogeneous",
    "mean_confidence",
    "next_region_id",
    "noise_floor",
    "regions_from_dicts",
    "regions_to_dicts",
    "remove_motion",
    "sample_line",
    "select_nodes",
    "spatial_temporal_std",
    "strain_precision",
    "summarize",
    "vector_stats",
    "vsg_size",
]
