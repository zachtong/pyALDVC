"""How precise a measurement is, and what the average deformation of a region is.

* :func:`noise_floor` -- for a static (or known-translation) pair: the bias and noise floor per component as
  defined by the DVC Challenge 2.0 paper, the same after a rigid fit, strain mean and standard deviation,
  MAER/SDER, the virtual strain gauge size, and -- with several static frames -- the spatial and temporal
  standard deviations of the iDICs Good Practices Guide.
* :func:`homogeneous` -- the affine fit of a region's displacement: the homogeneous deformation gradient, its
  polar decomposition and strains, next to the mean of the nodal strains. For a linear strain component the
  two agree and scatter alike under noise (reports/statistics.pdf, page 4b); for a nonlinear quantity -- von
  Mises, a principal magnitude, a Green-Lagrange component -- the mean of the nodal values carries a noise bias
  that grows with the noise (page 4a), the strain of the fit does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.config import length_unit
from ..core.data_structures import PipelineResult
from .motion import MotionFit
from .selection import NodeFilter
from .stats import FieldStats, VectorStats, spatial_temporal_std, strain_precision, summarize, vector_stats
from .view import frame_view

STRAIN_COMPONENTS = ("exx", "eyy", "ezz", "exy", "exz", "eyz")
_EYE = np.eye(3)


def vsg_size(para) -> NDArray[np.float64] | None:
    """Virtual strain gauge size per axis in voxels, ``(L_window - 1) L_step + L_subset`` (iDICs GPG Eq. 7.3):
    the window of nodes a strain value comes from, the node step and the subset span (``winsize + 1``).
    ``None`` for strain methods without a node window (finite elements, the ADMM gradient)."""
    subset = np.asarray(para.winsize, dtype=np.float64) + 1.0
    step = np.asarray(para.winstepsize, dtype=np.float64)
    if para.strain_method == "plane_fit":
        window = 2.0 * np.asarray(para.strain_plane_fit_halfwidth, dtype=np.float64) + 1.0
    elif para.strain_method == "fd":
        window = np.full(3, 3.0)
    else:
        return None
    return (window - 1.0) * step + subset


@dataclass(frozen=True, eq=False)
class NoiseFloor:
    frame: int
    unit: str
    nominal: NDArray[np.float64]
    displacement: VectorStats  # Eq. 1 and 2: the translation goes into the bias
    rigid: VectorStats | None  # the same after a rigid fit: rotation between scans no longer counts as noise
    rigid_fit: MotionFit | None
    strain: dict[str, FieldStats] | None
    maer: float | None
    sder: float | None
    vsg: NDArray[np.float64] | None  # voxels
    spatial_std: NDArray[np.float64] | None
    temporal_std: NDArray[np.float64] | None
    n_frames: int

    def as_dict(self) -> dict:
        return {
            "frame": self.frame + 1,
            "unit": self.unit,
            "nominal": self.nominal.tolist(),
            "displacement": self.displacement.as_dict(),
            "rigid": None if self.rigid is None else self.rigid.as_dict(),
            "rigid_fit": None if self.rigid_fit is None else self.rigid_fit.as_dict(),
            "strain": None if self.strain is None else {k: v.as_dict() for k, v in self.strain.items()},
            "maer": self.maer,
            "sder": self.sder,
            "vsg_voxel": None if self.vsg is None else self.vsg.tolist(),
            "spatial_std": None if self.spatial_std is None else self.spatial_std.tolist(),
            "temporal_std": None if self.temporal_std is None else self.temporal_std.tolist(),
            "frames_pooled": self.n_frames,
        }


def noise_floor(
    result: PipelineResult,
    frame: int = 0,
    node_filter: NodeFilter | None = None,
    nominal=None,
    pool_frames: bool = True,
    region=None,
) -> NoiseFloor:
    """The precision of frame ``frame`` of a static or known-translation sequence (``nominal``: the applied
    displacement, physical units, zero by default). With ``pool_frames`` and several frames, every frame is
    taken as a repeat of the static state for the spatial and temporal standard deviations."""
    nominal = np.zeros(3) if nominal is None else np.asarray(nominal, dtype=np.float64).reshape(3)
    view = frame_view(result, frame, "none", node_filter, region)
    m = view.selection.mask
    displacement = vector_stats(view.displacement()[m], nominal)
    rigid = rigid_fit = None
    if int(m.sum()) >= 3:
        rview = frame_view(result, frame, "rigid", node_filter, region, fit_region=region)
        rigid, rigid_fit = vector_stats(rview.displacement()[m]), rview.fit
    strain = maer = sder = vsg = None
    if view.strain() is not None:
        strain = {c: summarize(view.values(c)[m]) for c in STRAIN_COMPONENTS}
        maer, sder = strain_precision(np.column_stack([view.values(c)[m] for c in STRAIN_COMPONENTS]))
        vsg = vsg_size(result.dvc_para)
    spatial = temporal = None
    n_frames = 1
    if pool_frames and len(result.result_disp) >= 2:
        views = [frame_view(result, k, "none", node_filter, region) for k in range(len(result.result_disp))]
        common = np.logical_and.reduce([v.selection.mask for v in views])
        spatial, temporal = spatial_temporal_std(np.stack([v.displacement()[common] - nominal for v in views]))
        n_frames = len(views)
    return NoiseFloor(
        frame=frame,
        unit=length_unit(result.dvc_para),
        nominal=nominal,
        displacement=displacement,
        rigid=rigid,
        rigid_fit=rigid_fit,
        strain=strain,
        maer=maer,
        sder=sder,
        vsg=vsg,
        spatial_std=spatial,
        temporal_std=temporal,
        n_frames=n_frames,
    )


@dataclass(frozen=True, eq=False)
class Homogeneous:
    """The homogeneous deformation of the nodes used: ``F = R U`` from the affine fit, and its strains."""

    frame: int
    fit: MotionFit
    F: NDArray[np.float64]
    rotation: NDArray[np.float64]
    stretch: NDArray[np.float64]
    green_lagrange: NDArray[np.float64]
    infinitesimal: NDArray[np.float64]
    rotation_deg: float
    nodal_mean: dict[str, float] | None  # mean of the nodal strain components, for comparison

    def as_dict(self) -> dict:
        return {
            "frame": self.frame + 1,
            "fit": self.fit.as_dict(),
            "F": self.F.tolist(),
            "rotation": self.rotation.tolist(),
            "stretch": self.stretch.tolist(),
            "green_lagrange": self.green_lagrange.tolist(),
            "infinitesimal": self.infinitesimal.tolist(),
            "rotation_deg": self.rotation_deg,
            "nodal_mean": self.nodal_mean,
        }


def homogeneous(result: PipelineResult, frame: int = 0, node_filter: NodeFilter | None = None, region=None) -> Homogeneous:
    """:class:`Homogeneous` of frame ``frame`` over the nodes ``node_filter`` keeps (inside ``region``)."""
    from scipy.linalg import polar

    view = frame_view(result, frame, "affine", node_filter, region, fit_region=region)
    fit = view.fit
    F = fit.matrix
    R, U = polar(F, side="right")
    H = F - _EYE
    nodal = None
    if view.strain() is not None:
        m = view.selection.mask
        nodal = {c: summarize(view.values(c)[m]).mean for c in STRAIN_COMPONENTS}
    return Homogeneous(
        frame=frame,
        fit=fit,
        F=F,
        rotation=R,
        stretch=U,
        green_lagrange=0.5 * (F.T @ F - _EYE),
        infinitesimal=0.5 * (H + H.T),
        rotation_deg=fit.rotation_deg,
        nodal_mean=nodal,
    )
