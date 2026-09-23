"""Regions of a result: which nodes a statistic, a profile or a motion fit stands on.

Every region is defined in the reference (frame-0) configuration, in voxel coordinates ``[x, y, z]`` like the
node coordinates, so a saved region stays valid when the voxel size changes. Bounds are inclusive.

* ``box``      -- ``lo``, ``hi``: the corners ``[x, y, z]``
* ``sphere``   -- ``centre``, ``radius``
* ``cylinder`` -- ``centre`` (a point of the axis), ``radius``, ``axis`` (``x``, ``y`` or ``z``), ``lo``, ``hi`` along it
* ``slab``     -- ``axis``, ``lo``, ``hi``: every node whose coordinate along the axis is in range
* ``prism``    -- an outline drawn on a plane (``xy``, ``xz`` or ``yz``) and extruded along its normal from ``lo``
  to ``hi``: ``outline`` ``rect`` or ``ellipse`` (``points`` = two opposite corners of the box) or ``polygon``
  (``points`` = at least three vertices), in the plane's coordinates (``xz``: ``[x, z]``)

With anisotropic voxels a sphere in voxels is an ellipsoid in physical units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

REGION_SHAPES = ("box", "sphere", "cylinder", "slab", "prism")
OUTLINES = ("rect", "ellipse", "polygon")
AXES = {"x": 0, "y": 1, "z": 2}
PLANES = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}  # (in-plane a, in-plane b, normal) indices of [x, y, z]
DEFAULT_COLORS = ("#f59e0b", "#22d3ee", "#a3e635", "#f472b6", "#c084fc", "#fb7185", "#34d399", "#fbbf24")


def _vec(params: dict, key: str, n: int = 3) -> list[float]:
    v = params.get(key)
    try:
        out = [float(x) for x in v]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be {n} numbers, got {v!r}") from exc
    if len(out) != n or not all(np.isfinite(out)):
        raise ValueError(f"{key} must be {n} finite numbers, got {v!r}")
    return out


def _num(params: dict, key: str, positive: bool = False) -> float:
    try:
        v = float(params.get(key))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number, got {params.get(key)!r}") from exc
    if not np.isfinite(v) or (positive and v <= 0):
        raise ValueError(f"{key} must be {'positive' if positive else 'finite'}, got {v}")
    return v


def _axis(params: dict) -> int:
    a = params.get("axis")
    if a not in AXES:
        raise ValueError(f"axis must be one of {tuple(AXES)}, got {a!r}")
    return AXES[a]


@dataclass(frozen=True, eq=True)
class Region:
    """A named region; ``contains`` says which of the given voxel positions lie inside it."""

    id: int
    name: str
    shape: str
    params: dict[str, Any] = field(default_factory=dict)
    color: str = DEFAULT_COLORS[0]
    visible: bool = True

    __hash__ = None  # the parameters are a dict

    def __post_init__(self) -> None:
        if self.shape not in REGION_SHAPES:
            raise ValueError(f"unknown region shape {self.shape!r}; use one of {REGION_SHAPES}")
        p = self.params
        if self.shape == "box":
            lo, hi = _vec(p, "lo"), _vec(p, "hi")
            if any(h < lo_ for lo_, h in zip(lo, hi)):
                raise ValueError(f"box: hi {hi} below lo {lo}")
        elif self.shape == "sphere":
            _vec(p, "centre")
            _num(p, "radius", positive=True)
        elif self.shape == "cylinder":
            _vec(p, "centre")
            _num(p, "radius", positive=True)
            _axis(p)
            if _num(p, "hi") < _num(p, "lo"):
                raise ValueError("cylinder: hi below lo")
        elif self.shape == "slab":
            _axis(p)
            if _num(p, "hi") < _num(p, "lo"):
                raise ValueError("slab: hi below lo")
        else:
            if p.get("plane") not in PLANES:
                raise ValueError(f"plane must be one of {tuple(PLANES)}, got {p.get('plane')!r}")
            if p.get("outline") not in OUTLINES:
                raise ValueError(f"outline must be one of {OUTLINES}, got {p.get('outline')!r}")
            pts = np.asarray(p.get("points", []), dtype=np.float64)
            need = 3 if p["outline"] == "polygon" else 2
            if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < need or not np.all(np.isfinite(pts)):
                raise ValueError(f"a {p['outline']} needs at least {need} points [a, b], got {p.get('points')!r}")
            if _num(p, "hi") < _num(p, "lo"):
                raise ValueError("prism: hi below lo")

    def contains(self, X) -> NDArray[np.bool_]:
        """``(N,)`` bool: which positions ``X`` ``(N, 3)`` (voxels, ``[x, y, z]``) lie in the region."""
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        p = self.params
        if self.shape == "box":
            return np.all((X >= np.asarray(p["lo"], float)) & (X <= np.asarray(p["hi"], float)), axis=1)
        if self.shape == "sphere":
            return np.linalg.norm(X - np.asarray(p["centre"], float), axis=1) <= float(p["radius"])
        if self.shape == "slab":
            c = X[:, AXES[p["axis"]]]
            return (c >= float(p["lo"])) & (c <= float(p["hi"]))
        if self.shape == "cylinder":
            k = AXES[p["axis"]]
            others = [i for i in range(3) if i != k]
            d = X[:, others] - np.asarray(p["centre"], float)[others]
            along = X[:, k]
            return (np.hypot(d[:, 0], d[:, 1]) <= float(p["radius"])) & (along >= float(p["lo"])) & (along <= float(p["hi"]))
        a, b, n = PLANES[p["plane"]]
        depth = X[:, n]
        inside = (depth >= float(p["lo"])) & (depth <= float(p["hi"]))
        A, B = X[:, a], X[:, b]
        pts = np.asarray(p["points"], dtype=np.float64)
        if p["outline"] in ("rect", "ellipse"):
            (a0, b0), (a1, b1) = pts[0], pts[1]
            lo_a, hi_a, lo_b, hi_b = min(a0, a1), max(a0, a1), min(b0, b1), max(b0, b1)
            if p["outline"] == "rect":
                return inside & (A >= lo_a) & (A <= hi_a) & (B >= lo_b) & (B <= hi_b)
            ca, cb = 0.5 * (lo_a + hi_a), 0.5 * (lo_b + hi_b)
            ra, rb = max(0.5 * (hi_a - lo_a), 1e-12), max(0.5 * (hi_b - lo_b), 1e-12)
            return inside & (((A - ca) / ra) ** 2 + ((B - cb) / rb) ** 2 <= 1.0)
        from matplotlib.path import Path

        path = Path(pts, closed=False)
        out = np.zeros(X.shape[0], dtype=bool)
        if inside.any():
            out[inside] = path.contains_points(np.column_stack([A[inside], B[inside]]))
        return out

    def node_mask(self, result) -> NDArray[np.bool_]:
        """The nodes of ``result`` inside the region."""
        return self.contains(result.dvc_mesh.coordinates)

    def as_dict(self) -> dict:
        return {
            "id": int(self.id),
            "name": str(self.name),
            "shape": self.shape,
            "params": _plain(self.params),
            "color": self.color,
            "visible": bool(self.visible),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Region:
        if not isinstance(d, dict):
            raise ValueError(f"a region is a mapping, got {type(d).__name__}")
        params = d.get("params")
        if not isinstance(params, dict):
            raise ValueError("region params must be a mapping")
        try:
            rid = int(d.get("id"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid region id {d.get('id')!r}") from exc
        return cls(
            id=rid,
            name=str(d.get("name") or f"R{rid}"),
            shape=str(d.get("shape")),
            params=dict(params),
            color=str(d.get("color") or DEFAULT_COLORS[rid % len(DEFAULT_COLORS)]),
            visible=bool(d.get("visible", True)),
        )


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def regions_to_dicts(regions) -> list[dict]:
    return [r.as_dict() for r in regions]


def regions_from_dicts(items) -> list[Region]:
    """Regions from their dictionaries (a session); ``ValueError`` names the first invalid one."""
    if not isinstance(items, list):
        raise ValueError("regions must be a list")
    return [Region.from_dict(d) for d in items]


def next_region_id(regions) -> int:
    return 1 + max((r.id for r in regions), default=0)
