"""Masks drawn on the slice viewer: 2-D shapes on one plane, extruded along its normal.

A :class:`MaskEditor` keeps a boolean volume ``(nz, ny, nx)`` (True = material,
the pipeline's convention) and the list of :class:`MaskOp` that produced it
from a base mask. Every operation is a shape drawn on the XY, XZ or YZ plane
of the viewer, applied to a range of slices along the plane's normal
(one slice, a range, or the whole extent) in ``add`` (union) or ``cut``
(subtract) mode. Because the mask is a pure function of the base and the
operations, undo is "drop the last operation and replay", and a session can
store the operations instead of the volume.

No Qt here: the editor is used by the viewer, by tests and by scripts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

PLANES = ("xy", "xz", "yz")
SHAPES = ("rectangle", "ellipse", "polygon", "brush", "threshold", "invert", "fill", "empty")
MODES = ("add", "cut", "replace")
# in-plane (horizontal, vertical) axes and the normal of each plane, as (nz, ny, nx) array axes
PLANE_AXES = {"xy": ("x", "y", "z"), "xz": ("x", "z", "y"), "yz": ("y", "z", "x")}
AXIS_INDEX = {"z": 0, "y": 1, "x": 2}
MIN_POLYGON_POINTS = 3
BOUNDARY_EPS = 1e-6
MAX_UNDO_REPLAY_OPS = 200


@dataclass(frozen=True)
class MaskOp:
    """One drawing operation.

    Args:
        shape: one of :data:`SHAPES`. ``invert`` / ``fill`` / ``empty`` ignore the geometry.
        plane: viewer plane the shape was drawn on.
        points: in-plane ``(h, v)`` vertices in voxel coordinates (2 corners for
            rectangle / ellipse, >= 3 for polygon, the stroke for brush).
        depth: inclusive ``(first, last)`` slice range along the plane's normal;
            ``None`` extrudes through the whole volume.
        mode: ``add``, ``cut`` or ``replace`` (the shape becomes the mask).
        radius: brush radius in voxels.
        level: intensity threshold of a ``threshold`` op (``None`` = Otsu on the volume).
        keep_largest / fill_holes: clean-up of a ``threshold`` op.
    """

    shape: str
    plane: str = "xy"
    points: tuple[tuple[float, float], ...] = ()
    depth: tuple[int, int] | None = None
    mode: str = "add"
    radius: float = 1.0
    level: float | None = None
    keep_largest: bool = True
    fill_holes: bool = True

    def __post_init__(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError(f"shape must be one of {SHAPES}, got {self.shape!r}")
        if self.plane not in PLANES:
            raise ValueError(f"plane must be one of {PLANES}, got {self.plane!r}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.shape in ("rectangle", "ellipse") and len(self.points) != 2:
            raise ValueError(f"{self.shape} needs exactly two corner points, got {len(self.points)}")
        if self.shape == "polygon" and len(self.points) < MIN_POLYGON_POINTS:
            raise ValueError(f"polygon needs at least {MIN_POLYGON_POINTS} points, got {len(self.points)}")
        if self.shape == "brush" and (len(self.points) < 1 or self.radius <= 0):
            raise ValueError("brush needs at least one point and a positive radius")
        if self.depth is not None and (len(self.depth) != 2 or self.depth[0] > self.depth[1]):
            raise ValueError(f"depth must be (first, last) with first <= last, got {self.depth}")
        object.__setattr__(self, "points", tuple((float(h), float(v)) for h, v in self.points))
        if self.depth is not None:
            object.__setattr__(self, "depth", (int(self.depth[0]), int(self.depth[1])))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["points"] = [list(p) for p in self.points]
        if self.shape != "threshold":
            for key in ("level", "keep_largest", "fill_holes"):
                d.pop(key, None)
        d["depth"] = None if self.depth is None else list(self.depth)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MaskOp":
        return cls(
            shape=str(d["shape"]),
            plane=str(d.get("plane", "xy")),
            points=tuple(tuple(p) for p in d.get("points", ())),
            depth=None if d.get("depth") is None else tuple(d["depth"]),
            mode=str(d.get("mode", "add")),
            radius=float(d.get("radius", 1.0)),
            level=None if d.get("level") is None else float(d["level"]),
            keep_largest=bool(d.get("keep_largest", True)),
            fill_holes=bool(d.get("fill_holes", True)),
        )


# ----------------------------------------------------------------------------- intensity threshold
def otsu_threshold(values, bins: int = 256) -> float:
    """Otsu's threshold of the finite values (the level that best separates two intensity classes)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres) / np.maximum(w0, 1e-300)
    m1 = (np.cumsum((hist * centres)[::-1])[::-1] - hist * centres) / np.maximum(w1, 1e-300)
    between = w0[:-1] * w1[:-1] * (m0[:-1] - m1[:-1]) ** 2  # threshold after bin k: classes <= k and > k
    top = np.flatnonzero(between >= between.max() * (1.0 - 1e-9))  # a plateau when the classes are separated by a gap
    k = int((top[0] + top[-1]) // 2)  # its middle keeps the level away from both classes
    return float(edges[k + 1])


def threshold_region(volume, level: float | None = None, keep_largest: bool = True, fill_holes: bool = True):
    """Boolean volume of the voxels above ``level`` (Otsu when ``None``), optionally cleaned up."""
    from scipy import ndimage

    vol = np.asarray(volume)
    if vol.ndim != 3:
        raise ValueError(f"threshold needs a 3-D volume, got shape {vol.shape}")
    if level is None:
        step = max(1, int(round(vol.size / 2_000_000)))  # sample large volumes for the histogram
        level = otsu_threshold(vol.ravel()[::step])
    region = vol > level
    if fill_holes and region.any():
        region = ndimage.binary_fill_holes(region)
    if keep_largest and region.any():
        labels, n = ndimage.label(region)
        if n > 1:
            sizes = ndimage.sum(region, labels, index=np.arange(1, n + 1))
            region = labels == (int(np.argmax(sizes)) + 1)
    return np.asarray(region, dtype=bool)


# ----------------------------------------------------------------------------- 2-D rasterisation
def _plane_grid(h_len: int, v_len: int) -> tuple[NDArray, NDArray]:
    v, h = np.mgrid[0:v_len, 0:h_len]
    return h.astype(np.float64), v.astype(np.float64)


def rasterise_2d(op: MaskOp, h_len: int, v_len: int) -> NDArray[np.bool_]:
    """Boolean ``(v_len, h_len)`` image of the shape; voxel centres are integer coordinates."""
    if op.shape == "rectangle":
        (h1, v1), (h2, v2) = op.points
        out = np.zeros((v_len, h_len), dtype=bool)
        hs, he = sorted((int(round(h1)), int(round(h2))))
        vs, ve = sorted((int(round(v1)), int(round(v2))))
        hs, he = max(hs, 0), min(he, h_len - 1)
        vs, ve = max(vs, 0), min(ve, v_len - 1)
        if hs <= he and vs <= ve:
            out[vs : ve + 1, hs : he + 1] = True
        return out
    if op.shape == "ellipse":
        (h1, v1), (h2, v2) = op.points
        ch, cv = (h1 + h2) / 2.0, (v1 + v2) / 2.0
        a, b = max(abs(h2 - h1) / 2.0, 0.5), max(abs(v2 - v1) / 2.0, 0.5)
        h, v = _plane_grid(h_len, v_len)
        return ((h - ch) / a) ** 2 + ((v - cv) / b) ** 2 <= 1.0
    if op.shape == "polygon":
        from matplotlib.path import Path as MplPath

        h, v = _plane_grid(h_len, v_len)
        path = MplPath(np.asarray(op.points, dtype=np.float64))
        pts = np.column_stack([h.ravel(), v.ravel()])
        # boundary-inclusive for either vertex orientation (matplotlib's radius sign depends on it)
        inside = path.contains_points(pts, radius=BOUNDARY_EPS) | path.contains_points(pts, radius=-BOUNDARY_EPS)
        return inside.reshape(v_len, h_len)
    if op.shape == "brush":
        return _rasterise_stroke(op.points, op.radius, h_len, v_len)
    raise ValueError(f"{op.shape} has no 2-D geometry")


def _rasterise_stroke(points, radius: float, h_len: int, v_len: int) -> NDArray[np.bool_]:
    """Union of discs of ``radius`` swept along the polyline ``points``."""
    out = np.zeros((v_len, h_len), dtype=bool)
    pts = np.asarray(points, dtype=np.float64)
    segments = [(pts[0], pts[0])] if len(pts) == 1 else list(zip(pts[:-1], pts[1:]))
    r2 = radius * radius
    for p, q in segments:
        lo_h = max(int(np.floor(min(p[0], q[0]) - radius)), 0)
        hi_h = min(int(np.ceil(max(p[0], q[0]) + radius)), h_len - 1)
        lo_v = max(int(np.floor(min(p[1], q[1]) - radius)), 0)
        hi_v = min(int(np.ceil(max(p[1], q[1]) + radius)), v_len - 1)
        if lo_h > hi_h or lo_v > hi_v:
            continue
        v, h = np.mgrid[lo_v : hi_v + 1, lo_h : hi_h + 1]
        d = q - p
        seg2 = float(d @ d)
        if seg2 == 0.0:
            t = np.zeros_like(h, dtype=np.float64)
        else:
            t = np.clip(((h - p[0]) * d[0] + (v - p[1]) * d[1]) / seg2, 0.0, 1.0)
        dist2 = (h - (p[0] + t * d[0])) ** 2 + (v - (p[1] + t * d[1])) ** 2
        out[lo_v : hi_v + 1, lo_h : hi_h + 1] |= dist2 <= r2
    return out


def rasterise(op: MaskOp, shape: tuple[int, int, int]) -> NDArray[np.bool_]:
    """Boolean volume ``(nz, ny, nx)`` covered by ``op`` (geometry shapes only)."""
    nz, ny, nx = shape
    h_axis, v_axis, n_axis = PLANE_AXES[op.plane]
    sizes = {"z": nz, "y": ny, "x": nx}
    img = rasterise_2d(op, sizes[h_axis], sizes[v_axis])  # (v, h)
    n_len = sizes[n_axis]
    first, last = (0, n_len - 1) if op.depth is None else op.depth
    first, last = max(first, 0), min(last, n_len - 1)
    region = np.zeros(shape, dtype=bool)
    if first > last:
        return region
    if op.plane == "xy":  # img (ny, nx), normal z
        region[first : last + 1] = img[None]
    elif op.plane == "xz":  # img (nz, nx), normal y
        region[:, first : last + 1, :] = img[:, None, :]
    else:  # yz: img (nz, ny), normal x
        region[:, :, first : last + 1] = img[:, :, None]
    return region


# ----------------------------------------------------------------------------- editor
@dataclass
class MaskEditor:
    """Boolean mask volume as a base plus a replayable list of operations."""

    shape: tuple[int, int, int]
    base: NDArray[np.bool_] | None = None
    ops: list[MaskOp] = field(default_factory=list)
    volume: NDArray | None = field(default=None, repr=False)  # intensities for ``threshold`` ops
    mask: NDArray[np.bool_] = field(init=False)
    _redo: list[MaskOp] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.shape = tuple(int(s) for s in self.shape)  # type: ignore[assignment]
        if len(self.shape) != 3 or min(self.shape) < 1:
            raise ValueError(f"shape must be (nz, ny, nx) with positive sizes, got {self.shape}")
        if self.base is not None:
            self.base = np.asarray(self.base, dtype=bool)
            if self.base.shape != self.shape:
                raise ValueError(f"base mask shape {self.base.shape} does not match {self.shape}")
        ops, self.ops = list(self.ops), []
        self.mask = self._base_copy()
        for op in ops:
            self.apply(op)

    def _base_copy(self) -> NDArray[np.bool_]:
        return np.zeros(self.shape, dtype=bool) if self.base is None else self.base.copy()

    # ------------------------------------------------------------------ editing
    def apply(self, op: MaskOp) -> NDArray[np.bool_]:
        """Apply one operation in place; returns the mask."""
        if op.shape == "invert":
            np.logical_not(self.mask, out=self.mask)
        elif op.shape == "fill":
            self.mask[...] = True
        elif op.shape == "empty":
            self.mask[...] = False
        else:
            if op.shape == "threshold":
                if self.volume is None:
                    raise ValueError("a threshold operation needs the volume intensities (MaskEditor.volume)")
                region = threshold_region(self.volume, op.level, op.keep_largest, op.fill_holes)
            else:
                region = rasterise(op, self.shape)
            if op.mode == "replace":
                self.mask[...] = region
            elif op.mode == "add":
                self.mask |= region
            else:
                self.mask &= ~region
        self.ops.append(op)
        self._redo.clear()
        return self.mask

    def undo(self) -> bool:
        if not self.ops:
            return False
        op = self.ops.pop()
        self._redo.append(op)
        self._replay()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        op = self._redo.pop()
        redo_stack = list(self._redo)
        self.apply(op)
        self._redo = redo_stack
        return True

    def _replay(self) -> None:
        """Rebuild the mask from the base and the remaining operations (keeps the redo stack)."""
        ops, redo = self.ops, list(self._redo)
        if len(ops) > MAX_UNDO_REPLAY_OPS:  # fold the oldest operations into the base to bound the replay cost
            fold, ops = ops[:-MAX_UNDO_REPLAY_OPS], ops[-MAX_UNDO_REPLAY_OPS:]
            self.ops = []
            self.mask = self._base_copy()
            for op in fold:
                self.apply(op)
            self.base = self.mask.copy()
        self.ops = []
        self.mask = self._base_copy()
        for op in ops:
            self.apply(op)
        self._redo = redo

    def reset(self, base: NDArray[np.bool_] | None = None) -> None:
        """Drop every operation and start again from ``base`` (None: all False)."""
        self.base = None if base is None else np.asarray(base, dtype=bool).copy()
        if self.base is not None and self.base.shape != self.shape:
            raise ValueError(f"base mask shape {self.base.shape} does not match {self.shape}")
        self.ops = []
        self._redo = []
        self.mask = self._base_copy()

    @property
    def can_undo(self) -> bool:
        return bool(self.ops)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def coverage(self) -> float:
        """Fraction of voxels that are material (True)."""
        return float(np.count_nonzero(self.mask)) / float(self.mask.size)

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        """Operations only (the base mask, if any, is not serialised)."""
        return {"shape": list(self.shape), "ops": [op.to_dict() for op in self.ops]}

    @classmethod
    def from_dict(cls, d: dict[str, Any], base: NDArray[np.bool_] | None = None, volume: NDArray | None = None) -> "MaskEditor":
        shape = tuple(int(s) for s in d["shape"])
        return cls(volume=volume, shape=shape, base=base, ops=[MaskOp.from_dict(o) for o in d.get("ops", [])])  # type: ignore[arg-type]
