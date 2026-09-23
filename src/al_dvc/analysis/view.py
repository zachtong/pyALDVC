"""One frame of a result as a statistic sees it: the nodes used and, optionally, the rigid motion removed.

The stored result is never changed. A corrected frame is computed on demand: the displacement without the
fitted motion, and -- for a rigid correction -- the strain from the corrected gradient ``R^T (I + H) - I``.
That is exact, not an approximation: the gradient of the plane fit (and of the optional Gaussian smoothing)
is linear in the displacement, so fitting the corrected displacement again gives the same gradient.
A translation changes no strain; an affine correction changes the displacement only (its homogeneous strain
is reported by :func:`al_dvc.analysis.precision.homogeneous`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from ..core.data_structures import PipelineResult, StrainResult
from ..export.export_utils import STD_FIELDS, displacement_physical, field_array
from .motion import MotionFit, corrected_gradient, fit_motion, remove_motion
from .selection import NodeFilter, Selection, select_nodes

_DISP = {"disp_u": 0, "disp_v": 1, "disp_w": 2}


@dataclass(frozen=True, eq=False)
class FrameView:
    result: PipelineResult
    frame: int
    motion: str
    fit: MotionFit | None
    selection: Selection

    @cached_property
    def positions(self) -> NDArray[np.float64]:
        """Reference node positions ``(N, 3)`` in physical units."""
        return self.result.dvc_mesh.coordinates * np.asarray(self.result.dvc_para.voxel_size, dtype=np.float64)

    @cached_property
    def _valid(self) -> NDArray[np.bool_]:
        valid = np.asarray(self.result.dvc_mesh.node_valid, dtype=bool)
        return valid if valid.size == self.result.dvc_mesh.n_nodes else np.ones(self.result.dvc_mesh.n_nodes, dtype=bool)

    def raw_displacement(self) -> NDArray[np.float64]:
        """Cumulative displacement ``(N, 3)`` in physical units, NaN at invalid nodes."""
        U = displacement_physical(self.result.result_disp[self.frame], self.result.dvc_para.voxel_size)
        U = np.array(U, dtype=np.float64, copy=True)
        U[~self._valid] = np.nan
        return U

    @cached_property
    def _displacement(self) -> NDArray[np.float64]:
        U = self.raw_displacement()
        return U if self.fit is None else remove_motion(self.positions, U, self.fit)

    def displacement(self) -> NDArray[np.float64]:
        """Displacement ``(N, 3)`` in physical units without the fitted motion, NaN at invalid nodes."""
        return self._displacement

    @cached_property
    def _strain(self) -> StrainResult | None:
        res = self.result
        if self.frame >= len(res.result_strain):
            return None
        sr = res.result_strain[self.frame]
        if self.fit is None or self.fit.kind != "rigid":
            return sr
        from ..strain.compute_strain import strain_result_from_gradient

        disp = np.column_stack([sr.disp_u, sr.disp_v, sr.disp_w])
        return strain_result_from_gradient(
            remove_motion(self.positions, disp, self.fit),
            corrected_gradient(sr.F, self.fit),
            sr.strain_valid,
            sr.strain_type,
            sr.method,
        )

    def strain(self) -> StrainResult | None:
        """The frame's strain (corrected for a rigid motion), ``None`` when strain was not computed."""
        return self._strain

    def values(self, name: str) -> NDArray[np.float64]:
        """Per-node values ``(N,)`` of a field, NaN where it has none (invalid node, trimmed strain)."""
        if name in _DISP or name == "disp_magnitude":
            U = self.displacement()
            return U[:, _DISP[name]].copy() if name in _DISP else np.linalg.norm(U, axis=1)
        if name == "zncc":
            z = self.result.result_disp[self.frame].zncc
            if z is None:
                raise ValueError("this result has no ZNCC")
            out = np.array(z, dtype=np.float64, copy=True)
            out[~self._valid] = np.nan
            return out
        if name in STD_FIELDS:
            return field_array(self.result, self.frame, name)
        sr = self.strain()
        if sr is None:
            raise ValueError(f"strain field {name!r} requested but strain was not computed")
        return sr.field(name, trimmed=True)


def frame_view(
    result: PipelineResult,
    frame: int,
    motion: str = "none",
    node_filter: NodeFilter | None = None,
    region: NDArray[np.bool_] | None = None,
    fit_region: NDArray[np.bool_] | None = None,
) -> FrameView:
    """Frame ``frame`` with the nodes ``node_filter`` keeps inside ``region``; for ``motion`` other than
    ``"none"`` the motion is fitted and removed from the whole field.

    The fit stands on the nodes ``node_filter`` keeps inside ``fit_region`` -- all of them by default, a fixture
    or any rigid part when one is given -- not on ``region``: removing a rigid motion is one correction of the
    field, and a region's statistics must not lose that region's own rotation.
    """
    if not 0 <= frame < len(result.result_disp):
        raise IndexError(f"frame {frame} out of range (0-{len(result.result_disp) - 1})")
    selection = select_nodes(result, frame, node_filter, region)
    view = FrameView(result=result, frame=frame, motion=motion, fit=None, selection=selection)
    if motion == "none":
        return view
    fit_nodes = selection if (region is None and fit_region is None) else select_nodes(result, frame, node_filter, fit_region)
    m = fit_nodes.mask
    fit = fit_motion(view.positions[m], view.raw_displacement()[m], motion)
    return FrameView(result=result, frame=frame, motion=motion, fit=fit, selection=selection)
