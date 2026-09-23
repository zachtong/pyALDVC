"""A result with the rigid-body motion removed, for everything that draws or exports fields.

:func:`corrected_result` returns an object that behaves like the :class:`~al_dvc.core.data_structures.PipelineResult`
it was made from -- the same mesh, parameters and frames -- except that every frame's cumulative displacement and
strain are those of :func:`al_dvc.analysis.view.frame_view` with the motion removed. Frames are computed when first
asked for, so switching the main viewer to the corrected field costs one frame at a time, and the last
:data:`CACHE_FRAMES` are kept (a rigid correction's strain is a full strain result per frame: every frame of a long
series would not fit in memory). The stored result is never changed. A frame whose motion cannot be fitted (too few
nodes) is left as measured and listed by :meth:`CorrectedResult.uncorrected_frames`.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields, replace

import numpy as np

from ..core.data_structures import PipelineResult
from .motion import MOTION_KINDS
from .regions import Region
from .selection import NodeFilter
from .view import frame_view

CACHE_FRAMES = 6  # corrected frames kept in memory, the most recently used


@dataclass(frozen=True)
class Correction:
    """Which motion to remove, fitted over which nodes."""

    motion: str = "none"
    node_filter: NodeFilter = field(default_factory=NodeFilter)
    fit_region: Region | None = None  # the nodes the fit stands on (a fixture); all nodes by default

    __hash__ = None  # a region's parameters are a dict

    def __post_init__(self) -> None:
        if self.motion not in MOTION_KINDS:
            raise ValueError(f"unknown motion {self.motion!r}; use one of {MOTION_KINDS}")

    def describe(self) -> str:
        where = f"region {self.fit_region.name}" if self.fit_region is not None else "all nodes"
        if self.node_filter.converged_only:
            where = "the measured nodes of " + where
        return f"{self.motion}, fitted over {where}"

    def as_dict(self) -> dict:
        return {
            "motion": self.motion,
            "node_filter": asdict(self.node_filter),
            "fit_region": None if self.fit_region is None else self.fit_region.as_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Correction:
        if not isinstance(d, dict):
            raise ValueError("a correction is a mapping")
        known = {f.name for f in fields(NodeFilter)}
        nf = d.get("node_filter") or {}
        if not isinstance(nf, dict):
            raise ValueError("node_filter must be a mapping")
        region = d.get("fit_region")
        return cls(
            motion=str(d.get("motion", "none")),
            node_filter=NodeFilter(**{k: v for k, v in nf.items() if k in known}),
            fit_region=None if region is None else Region.from_dict(region),
        )


class _Frames:
    """What the corrected frames share: the source, the correction, the last corrected frames and a lock (the viewer
    and an export may ask for frames from two threads). Frames whose motion could not be fitted are remembered for
    good; the corrected ones only while they are among the :data:`CACHE_FRAMES` most recently used."""

    def __init__(self, source: PipelineResult, correction: Correction) -> None:
        self.source = source
        self.correction = correction
        self.fit_mask = None if correction.fit_region is None else correction.fit_region.node_mask(source)
        self.lock = threading.Lock()
        self.views: OrderedDict[int, object] = OrderedDict()
        self.disp: OrderedDict[int, object] = OrderedDict()
        self.strain: OrderedDict[int, object] = OrderedDict()
        self.evaluated: set[int] = set()  # frames whose motion was fitted (or could not be)
        self.failed: set[int] = set()

    def _cached(self, cache: OrderedDict, k: int):
        with self.lock:
            if k in cache:
                cache.move_to_end(k)
                return True, cache[k]
        return False, None

    def _keep(self, cache: OrderedDict, k: int, value):
        with self.lock:
            if k in cache:  # another thread got there first: one copy
                cache.move_to_end(k)
                return cache[k]
            cache[k] = value
            while len(cache) > CACHE_FRAMES:
                cache.popitem(last=False)
            return value

    def view(self, k: int):
        with self.lock:
            if k in self.failed:
                return None
        found, view = self._cached(self.views, k)
        if found:
            return view
        c = self.correction
        try:
            view = frame_view(self.source, k, c.motion, c.node_filter, fit_region=self.fit_mask)
        except ValueError:  # too few nodes to fit: this frame stays as measured
            view = None
        with self.lock:
            self.evaluated.add(k)
            if view is None:
                self.failed.add(k)
                return None
        return self._keep(self.views, k, view)

    def frame(self, k: int):
        found, fr = self._cached(self.disp, k)
        if found:
            return fr
        fr = self.source.result_disp[k]
        view = self.view(k)
        if view is not None:
            vs = np.asarray(self.source.dvc_para.voxel_size, dtype=np.float64)
            fr = replace(fr, U_accum=view.displacement() / vs)  # the corrected cumulative displacement, in voxels
        return self._keep(self.disp, k, fr)

    def strain_of(self, k: int):
        found, sr = self._cached(self.strain, k)
        if found:
            return sr
        view = self.view(k)
        sr = self.source.result_strain[k] if view is None else view.strain()
        return self._keep(self.strain, k, sr)


class _LazyList(Sequence):
    """A read-only list whose items are computed on first access."""

    def __init__(self, n: int, get) -> None:
        self._n = n
        self._get = get

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self._n))]
        i = int(index)
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError(f"frame {index} out of range")
        return self._get(i)


@dataclass(frozen=True)
class CorrectedResult(PipelineResult):
    """A :class:`PipelineResult` whose frames have the motion of ``correction`` removed."""

    correction: Correction | None = None
    source: PipelineResult | None = None

    def uncorrected_frames(self) -> list[int]:
        """The frames left as measured because the motion could not be fitted (every frame not yet fitted is)."""
        state: _Frames = self.result_disp._state  # type: ignore[attr-defined]
        for k in range(len(self.result_disp)):
            if k not in state.evaluated:
                state.view(k)
        return sorted(state.failed)


def corrected_result(result: PipelineResult, correction: Correction) -> PipelineResult:
    """``result`` with ``correction`` applied (``result`` itself when there is nothing to remove)."""
    if correction.motion == "none" or result is None:
        return result
    state = _Frames(result, correction)
    frames = _LazyList(len(result.result_disp), state.frame)
    frames._state = state  # type: ignore[attr-defined]
    strains = _LazyList(len(result.result_strain), state.strain_of)
    return CorrectedResult(
        dvc_para=result.dvc_para,
        dvc_mesh=result.dvc_mesh,
        result_disp=frames,  # type: ignore[arg-type]
        result_strain=strains,  # type: ignore[arg-type]
        frame_schedule=result.frame_schedule,
        volume_shape=result.volume_shape,
        timings=dict(result.timings),
        stopped_early=result.stopped_early,
        stopped_at_frame=result.stopped_at_frame,
        stop_reason=result.stop_reason,
        correction=correction,
        source=result,
    )
