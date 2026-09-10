"""Solve the local steps over sub-boxes of the volume instead of the whole thing.

Every local kernel addresses ``f``, ``gx``, ``gy``, ``gz``, ``mask`` and ``g`` relative to a node
centre -- ``f[z0 + dz, y0 + dy, x0 + dx]`` for the reference, ``sample_volume(g, zw, yw, xw)`` with
``xw = a00 * X + ... + (x0 + P[9])`` for the deformed frame -- and no kernel reduces across nodes.
So a block of nodes can be solved against a *crop* of the volumes, with the crop's origin subtracted
from the node coordinates, and nothing inside the kernels needs to know. That is what this module
does: it plans the blocks, cuts the boxes, and scatters the per-node results back.

What it buys. Today the reference gradients, the reference mask and the deformed frame's
interpolation preparation are all full-volume arrays that live for the whole run, and every one of
them is uploaded whole to the GPU (21 bytes per voxel of VRAM: 22.5 GB at 1024^3, 180 GB at 2048^3).
Per tile they are bounded by the box, so device residency stops scaling with the scan.

What it does not buy. The provider's normalised frames stay whole -- ``get_normalized`` returns a
full-shape array -- so the host floor is still about 13 bytes per voxel. Getting past that needs a
provider that serves boxes, which is a different change.

Contracts kept: coordinates are ``(N, 3)`` ``[x, y, z]``, volumes are ``(nz, ny, nx)``, and boxes are
expressed as an origin ``(z0, y0, x0)`` plus a shape, in that array order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# The reference window of a node spans winsize/2 and the gradient stencil reaches 3 voxels beyond it
# (``GRADIENT_BORDER``); on-the-fly gradients read the same stencil inside the kernel.
REF_STENCIL_PAD = 3
# The deformed sample of a node can leave its reference window by the displacement, by the subset's
# own stretch and by the interpolation margin. TILE_STRAIN_MARGIN is the stretch allowance per axis.
SAMPLE_PAD = 2  # ``SAMPLE_HI_MARGIN`` of interp_kernels: the highest sample a cubic kernel touches
# A B-spline prefilter is not local, so its coefficients are computed once for the whole frame and the
# box is cut out of the coefficients, not out of the grey values. That is exact -- a per-tile prefilter
# would only be approximately the same -- and it is why no B-spline halo appears here.
MIN_TILE_NODES = 1  # a plan must give every node a box, even if that means one node per box

__all__ = ["TileBox", "TilePlan", "plan_tiles", "tile_pad", "whole_box_tile"]


@dataclass(frozen=True)
class TileBox:
    """A sub-box of the volume: ``origin`` ``(z0, y0, x0)`` and ``shape`` ``(nz, ny, nx)``.

    ``open_faces`` records which faces were cut by the *tile* rather than by the volume: a subset
    that leaves the box through an open face is a halo that was too small, while one that leaves
    through a volume face is genuinely out of bounds. The two must never be confused, so the driver
    can tell them apart and retry the first.
    """

    origin: tuple[int, int, int]
    shape: tuple[int, int, int]
    open_faces: tuple[bool, bool, bool, bool, bool, bool] = (False,) * 6  # -z +z -y +y -x +x

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return tuple(slice(o, o + n) for o, n in zip(self.origin, self.shape))  # type: ignore[return-value]

    @property
    def is_whole(self) -> bool:
        return self.origin == (0, 0, 0) and not any(self.open_faces)

    @property
    def has_open_face(self) -> bool:
        return any(self.open_faces)

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.shape))

    def crop(self, vol: NDArray | None) -> NDArray | None:
        """The box's voxels of ``vol``, contiguous. Returns ``vol`` itself for a whole-volume box.

        A 1x1x1 placeholder (an absent mask, on-the-fly gradients) is passed through untouched: it
        is not a volume and cropping it would destroy the shape the kernels recognise it by.
        """
        if vol is None:
            return None
        arr = np.asarray(vol)
        if arr.shape == (1, 1, 1) or self.shape == arr.shape:
            return arr
        return np.ascontiguousarray(arr[self.slices])

    def shift(self, coords_xyz: NDArray) -> NDArray:
        """``coords_xyz`` moved into the box's frame: ``[x, y, z] - [x0, y0, z0]``."""
        z0, y0, x0 = self.origin
        out = np.array(coords_xyz, dtype=np.int64, copy=True)
        out[:, 0] -= x0
        out[:, 1] -= y0
        out[:, 2] -= z0
        return out

    def contains_windows(self, coords_xyz: NDArray, pad: tuple[int, int, int]) -> bool:
        """True when every node's window plus ``pad`` ``(px, py, pz)`` lies inside the box."""
        if coords_xyz.size == 0:
            return True
        z0, y0, x0 = self.origin
        nz, ny, nx = self.shape
        c = np.asarray(coords_xyz, dtype=np.int64)
        for axis, lo, size, p in ((0, x0, nx, pad[0]), (1, y0, ny, pad[1]), (2, z0, nz, pad[2])):
            if int(c[:, axis].min()) - p < lo or int(c[:, axis].max()) + p > lo + size - 1:
                return False
        return True


@dataclass(frozen=True)
class TilePlan:
    """The blocks a local step runs in: ``(node indices, box)`` pairs plus what the plan cost."""

    tiles: list[tuple[NDArray[np.int64], TileBox]]
    ref_pad: tuple[int, int, int]
    def_pad: tuple[int, int, int]
    fallback: str = ""  # why a whole-volume plan was used instead of the requested tiles

    def __len__(self) -> int:
        return len(self.tiles)

    def __iter__(self):
        return iter(self.tiles)

    @property
    def is_whole(self) -> bool:
        return len(self.tiles) == 1 and self.tiles[0][1].is_whole

    @property
    def max_voxels(self) -> int:
        return max((box.n_voxels for _nodes, box in self.tiles), default=0)


def whole_box_tile(shape: tuple[int, int, int]) -> TileBox:
    """The degenerate box: the whole volume, no open faces, so the untiled path is a plan of one."""
    return TileBox(origin=(0, 0, 0), shape=tuple(int(s) for s in shape))  # type: ignore[arg-type]


def tile_pad(
    para,
    half: tuple[int, int, int],
    disp: NDArray | None = None,
    deformed: bool = False,
) -> tuple[int, int, int]:
    """Voxels a box must hold beyond a node centre, per axis ``(px, py, pz)``.

    The reference side needs the subset half-width and the gradient stencil. The deformed side needs
    the same window, plus how far the warp can move it: the displacement actually present in the
    initial guess (not a guess from the configuration -- ``disp`` is measured), the subset's own
    stretch, the interpolation margin and, with a B-spline, the prefilter's reach.
    """
    pad = [int(h) + REF_STENCIL_PAD for h in half]
    sigma = float(getattr(para, "prefilter_sigma", 0.0) or 0.0)
    if sigma > 0:
        smooth = int(4.0 * sigma + 0.5)
        pad = [p + smooth for p in pad]
    if not deformed:
        return tuple(pad)  # type: ignore[return-value]
    strain = float(getattr(para, "tile_strain_margin", 0.25) or 0.0)
    radius = np.broadcast_to(np.asarray(getattr(para, "search_radius", 0), dtype=np.int64), (3,))
    margin = int(getattr(para, "tile_disp_margin", 0) or 0)
    if margin <= 0:
        margin = int(max(np.broadcast_to(np.asarray(para.winstepsize, dtype=np.int64), (3,))))
    moved = [0, 0, 0]
    if disp is not None and np.size(disp):
        d = np.abs(np.asarray(disp, dtype=np.float64))
        finite = d[np.isfinite(d).all(axis=1)] if d.ndim == 2 else d
        if finite.size:
            moved = [int(np.ceil(finite[:, k].max())) for k in range(3)]
    out = []
    for k, (h, p) in enumerate(zip(half, pad)):
        out.append(int(p + moved[k] + margin + int(np.ceil(strain * h)) + int(radius[k]) + SAMPLE_PAD))
    return tuple(out)  # type: ignore[return-value]


def _blocks(n: int, count: int) -> list[NDArray[np.int64]]:
    """``n`` indices split into ``count`` contiguous, near-equal blocks."""
    count = max(1, min(int(count), int(n)))
    return [b for b in np.array_split(np.arange(n, dtype=np.int64), count) if b.size]


def _box_for(coords: NDArray, pad: tuple[int, int, int], shape: tuple[int, int, int]) -> TileBox:
    """The smallest box holding every node's window plus ``pad``, clamped to the volume."""
    nz, ny, nx = (int(s) for s in shape)
    c = np.asarray(coords, dtype=np.int64)
    lo, hi, faces = [], [], []
    for axis, size, p in ((2, nz, pad[2]), (1, ny, pad[1]), (0, nx, pad[0])):  # array order z, y, x
        want_lo = int(c[:, axis].min()) - p
        want_hi = int(c[:, axis].max()) + p + 1
        a, b = max(0, want_lo), min(size, want_hi)
        lo.append(a)
        hi.append(b)
        faces.append(a > 0)  # the low face was cut by the tile, not by the volume
        faces.append(b < size)
    return TileBox(
        origin=(lo[0], lo[1], lo[2]),
        shape=(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]),
        open_faces=tuple(faces),  # type: ignore[arg-type]
    )


def plan_tiles(
    grid_shape: tuple[int, int, int],
    coords_xyz: NDArray,
    volume_shape: tuple[int, int, int],
    para,
    half: tuple[int, int, int],
    disp: NDArray | None = None,
    nodes: NDArray | None = None,
    edge: int | None = None,
) -> TilePlan:
    """Split the node grid into blocks whose boxes fit ``para.tile_local`` voxels per edge.

    ``grid_shape`` is the node lattice ``(nz, ny, nx)`` and ``coords_xyz`` its ``(N, 3)`` centres;
    ``nodes`` restricts the plan to a subset of them (the escape retry). ``edge`` overrides
    ``para.tile_local``. With the target at 0 -- or when even one node per block would not fit -- the
    plan is a single whole-volume box, which is the untiled path by construction.
    """
    volume_shape = tuple(int(s) for s in volume_shape)  # type: ignore[assignment]
    coords_xyz = np.asarray(coords_xyz, dtype=np.int64)
    n_total = coords_xyz.shape[0]
    index = np.arange(n_total, dtype=np.int64) if nodes is None else np.asarray(nodes, dtype=np.int64)
    ref_pad = tile_pad(para, half, deformed=False)
    def_pad = tile_pad(para, half, disp=disp, deformed=True)
    pad = tuple(max(r, d) for r, d in zip(ref_pad, def_pad))
    target = int(edge if edge is not None else getattr(para, "tile_local", 0) or 0)
    whole = TilePlan([(index, whole_box_tile(volume_shape))], ref_pad, def_pad)
    if target <= 0 or index.size == 0:
        return whole
    smallest = [2 * p + 1 for p in pad]  # one node still needs its window and the halo
    if max(smallest) > target:
        return TilePlan(
            whole.tiles,
            ref_pad,
            def_pad,
            fallback=(
                f"tile_local={target} is smaller than one subset plus its halo ({max(smallest)} voxels); "
                "the local steps run on the whole volume"
            ),
        )
    # split each axis of the node lattice until the resulting box edge fits the target
    gz, gy, gx = (int(s) for s in grid_shape)
    if gz * gy * gx != n_total:
        return TilePlan(whole.tiles, ref_pad, def_pad, fallback="the node grid is not a full lattice")
    selected = np.zeros(n_total, dtype=bool)
    selected[index] = True
    # the halo is paid by every box, so the node span a box may cover is the target minus both halos
    counts = []
    for axis, _g in enumerate((gz, gy, gx)):
        p = pad[2 - axis]  # pad is (x, y, z); axis 0 is z
        room = target - 2 * p
        span = int(coords_xyz[:, 2 - axis].max() - coords_xyz[:, 2 - axis].min()) + 1
        counts.append(max(1, int(np.ceil(span / max(room, 1)))))
    tiles: list[tuple[NDArray[np.int64], TileBox]] = []
    lattice = np.arange(n_total, dtype=np.int64).reshape(gz, gy, gx)
    for bz in _blocks(gz, counts[0]):
        for by in _blocks(gy, counts[1]):
            for bx in _blocks(gx, counts[2]):
                block = lattice[np.ix_(bz, by, bx)].ravel()
                block = block[selected[block]]
                if block.size < MIN_TILE_NODES:
                    continue
                tiles.append((block, _box_for(coords_xyz[block], pad, volume_shape)))
    if not tiles:
        return whole
    covered = np.concatenate([t[0] for t in tiles])
    if covered.size != index.size or not np.array_equal(np.sort(covered), np.sort(index)):
        return TilePlan(whole.tiles, ref_pad, def_pad, fallback="the tile plan did not cover every node exactly once")
    plan = TilePlan(tiles, ref_pad, def_pad)
    if plan.max_voxels > target**3:
        logger.info(
            "Tile boxes are larger than tile_local=%d asked for (largest edge %d): the halo of %s voxels is what "
            "sets the floor, so a larger tile_local costs proportionally less of it",
            target,
            max(max(box.shape) for _n, box in tiles),
            pad,
        )
    logger.info(
        "Local steps tiled: %d boxes, largest %s voxels (%.2f GB of reference data), halo %s / %s",
        len(tiles),
        plan.max_voxels,
        17 * plan.max_voxels / 1e9,
        ref_pad,
        def_pad,
    )
    return plan


@dataclass
class ReferenceSource:
    """A reference frame the local steps can ask for one box at a time.

    Holds ``f`` and the boolean mask by reference and builds a :class:`ReferenceBundle` on demand:
    for the whole volume once (cached, which is the untiled behaviour), for a box every time it is
    asked. The gradients of a box are computed from the box, so they never exist at full size --
    that is where the 12 bytes per voxel go. Pair it with ``gradient_mode="on_the_fly"`` and there
    is nothing to recompute per tile at all, only the crop.
    """

    f: NDArray[np.float32]
    mask: NDArray[np.bool_] | None = None
    gradient_mode: str = "stored"
    _whole: object | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in np.asarray(self.f).shape)  # type: ignore[return-value]

    @property
    def has_mask(self) -> bool:
        return self.mask is not None

    def bundle_for(self, box: TileBox):
        """The :class:`ReferenceBundle` of ``box`` (the whole-volume one is built once and kept)."""
        from ..io.volume_ops import build_reference_bundle

        if box.is_whole:
            if self._whole is None:
                self._whole = build_reference_bundle(self.f, self.mask, self.gradient_mode)
            return self._whole
        return build_reference_bundle(box.crop(self.f), box.crop(self.mask), self.gradient_mode)

    def release_whole(self) -> None:
        """Drop the cached whole-volume bundle (the tiled path never wants it)."""
        self._whole = None


def as_source(ref, gradient_mode: str = "stored") -> ReferenceSource:
    """A :class:`ReferenceSource` from either a source or an already-built bundle.

    Callers that hold a bundle keep working: the bundle's own arrays become the source's, so the
    whole-volume path is the same object it always was.
    """
    if isinstance(ref, ReferenceSource):
        return ref
    mask = None if not getattr(ref, "has_mask", True) else np.asarray(ref.mask) > 0
    mode = "stored" if getattr(ref, "stored_gradients", True) else "on_the_fly"
    src = ReferenceSource(f=ref.f, mask=mask, gradient_mode=mode if gradient_mode == "auto" else gradient_mode)
    src._whole = ref  # the caller's bundle is the whole-volume bundle; do not rebuild it
    return src


def local_split(split_index, split_keep, nodes: NDArray[np.int64]):
    """The subset-splitting rows of ``nodes`` as a standalone pair, renumbered from zero.

    ``split_index`` is global (one entry per node, -1 when nothing is split) and ``split_keep`` holds
    one packed row per split node. A tile's kernel needs its own contiguous rows, and the plan a
    tile came from is not necessarily the plan the rows were built with, so they are gathered rather
    than sliced.
    """
    if split_index is None or split_keep is None:
        return None, None
    idx = np.asarray(split_index, dtype=np.int64)[nodes]
    rows = np.flatnonzero(idx >= 0)
    if rows.size == 0:
        return np.full(nodes.size, -1, dtype=np.int64), np.zeros((1, np.asarray(split_keep).shape[1]), dtype=np.uint8)
    local = np.full(nodes.size, -1, dtype=np.int64)
    local[rows] = np.arange(rows.size, dtype=np.int64)
    return local, np.ascontiguousarray(np.asarray(split_keep, dtype=np.uint8)[idx[rows]])


def merge_split(parts: list[tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.uint8]]], n_nodes: int):
    """Concatenate per-tile ``(nodes, local_index, keep_rows)`` into one global pair."""
    split_index = np.full(n_nodes, -1, dtype=np.int64)
    rows: list[NDArray[np.uint8]] = []
    offset = 0
    width = 1
    for nodes, local, keep in parts:
        keep = np.asarray(keep, dtype=np.uint8)
        width = max(width, keep.shape[1])
        taken = np.flatnonzero(np.asarray(local) >= 0)
        if taken.size == 0:
            continue
        split_index[np.asarray(nodes)[taken]] = offset + np.asarray(local)[taken]
        rows.append(keep[: int(np.asarray(local)[taken].max()) + 1])
        offset += rows[-1].shape[0]
    if not rows:
        return split_index, np.zeros((1, width), dtype=np.uint8)
    return split_index, np.ascontiguousarray(np.concatenate(rows, axis=0))
