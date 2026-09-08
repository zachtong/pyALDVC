#!/usr/bin/env python
"""Generate a synthetic test case with a crack, two pores and a region of interest.

A speckled block is opened by a crack that stops at a front, so subsets and the node grid meet
three kinds of boundary at once: the crack, two spherical pores and the edge of the region of
interest. Every frame comes with its own mask, and the exact displacement is written next to the
volumes, so a run can be checked against the truth.

The deformation of frame ``k`` is

    u(X) = t_k + F_k (X - c) + opening_k(X)

with ``t_k`` a rigid translation, ``F_k`` a small stretch and ``opening_k`` the crack: the two
faces move apart by ``+/- a_k`` along x, tapering smoothly to zero at the crack front.

Usage::

    python scripts/make_synthetic_case.py [--out DIR] [--frames 3] [--noise 0.01] [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from al_dvc import __version__  # noqa: E402
from al_dvc.io.volume_io import save_volume  # noqa: E402
from al_dvc.synthetic import generate_speckle_volume  # noqa: E402

SHAPE = (128, 176, 192)  # (nz, ny, nx)
QUICK_SHAPE = (64, 88, 96)
CRACK_X = 96.0  # the crack plane
CRACK_HALF = 1.5  # half thickness of the masked band in the reference (3 voxels)
CRACK_FRONT_Y = 120.0  # the crack ends here; beyond it the material is continuous
TAPER = 40.0  # the opening fades to zero over this distance before the front
PORES = (((48.0, 60.0, 64.0), 12.0), ((150.0, 128.0, 40.0), 9.0))  # ((x, y, z), radius)
OPENING_PER_FRAME = 0.8  # voxels, each face
SPECKLE_SIGMA = 2.0
GREY_MAX = 4000  # uint16 range of the written volumes
STRETCH = (0.5, 99.5)  # percentiles of the reference material mapped to 0 and GREY_MAX
VOID_LEVEL = 0.06  # grey level of the crack and the pores, as a fraction of the material range


def scaled(shape, quick: bool):
    """The geometry scaled to the volume size (the quick case is half of everything)."""
    s = 0.5 if quick else 1.0
    return {
        "crack_x": CRACK_X * s,
        "crack_half": CRACK_HALF if not quick else 1.0,
        "front_y": CRACK_FRONT_Y * s,
        "taper": TAPER * s,
        "pores": tuple(((x * s, y * s, z * s), r * s) for (x, y, z), r in PORES),
    }


def smoothstep(t):
    """0 below 0, 1 above 1, with a continuous first derivative in between."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def opening_profile(y, geo):
    """How much of the full opening the crack has at height ``y`` (1 far behind the front, 0 at it)."""
    return smoothstep((geo["front_y"] - np.asarray(y, dtype=np.float64)) / geo["taper"])


def frame_parameters(k: int):
    """``(opening per face, translation, diagonal stretch)`` of frame ``k``."""
    return OPENING_PER_FRAME * k, np.array([0.30, -0.20, 0.15]) * k, np.array([0.0030, -0.0009, -0.0009]) * k


def displacement(k: int, geo: dict, centre) -> callable:
    """The displacement of frame ``k`` as ``u(x, y, z) -> (u, v, w)`` in voxels."""
    amp, t, diag = frame_parameters(k)
    F = np.diag(diag)
    cx, cy, cz = centre

    def fn(x, y, z):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        dx, dy, dz = x - cx, y - cy, z - cz
        u = t[0] + F[0, 0] * dx
        v = t[1] + F[1, 1] * dy
        w = t[2] + F[2, 2] * dz
        side = np.where(x >= geo["crack_x"], 1.0, -1.0)
        u = u + side * amp * opening_profile(y, geo)
        return u, v, np.broadcast_to(w, u.shape).copy()

    return fn


def material_mask(shape, geo, opened: float = 0.0) -> np.ndarray:
    """``(nz, ny, nx)`` uint8: 1 where there is material, 0 in the crack and the pores.

    ``opened`` widens the crack band by that many voxels on each side (the deformed frames).
    """
    nz, ny, nx = shape
    Z, Y, X = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float64)
    half = geo["crack_half"] + opened * opening_profile(Y, geo)
    crack = (np.abs(X - geo["crack_x"]) <= half) & (Y <= geo["front_y"])
    mask = ~crack
    for (px, py, pz), r in geo["pores"]:
        mask &= ((X - px) ** 2 + (Y - py) ** 2 + (Z - pz) ** 2) > r * r
    return mask.astype(np.uint8)


def warp(ref: np.ndarray, mask: np.ndarray, disp, n_iter: int = 25):
    """``(deformed volume, deformed mask)``: one inverse map, quintic for the grey values, nearest for the mask."""
    nz, ny, nx = ref.shape
    Z, Y, X = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float64)
    Xr, Yr, Zr = X.copy(), Y.copy(), Z.copy()
    for _ in range(n_iter):
        u, v, w = disp(Xr, Yr, Zr)
        Xr, Yr, Zr = X - u, Y - v, Z - w
    coords = np.vstack([Zr.ravel(), Yr.ravel(), Xr.ravel()])
    g = map_coordinates(ref.astype(np.float64), coords, order=5, mode="nearest").reshape(ref.shape)
    m = map_coordinates(mask, coords, order=0, mode="nearest").reshape(ref.shape)
    return g, m


def grey_window(vol: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """``(lo, hi)`` of the contrast stretch: the percentiles of the material, so the speckle fills the range."""
    lo, hi = np.percentile(np.asarray(vol, dtype=np.float64)[mask > 0], STRETCH)
    return float(lo), float(hi)


def to_uint16(vol: np.ndarray, window: tuple[float, float], noise: float, seed: int) -> np.ndarray:
    """Stretch ``window`` to 0 .. GREY_MAX, add read-out noise and clip (all frames share one window)."""
    lo, hi = window
    v = (np.asarray(vol, dtype=np.float64) - lo) / max(hi - lo, 1e-12)
    if noise > 0:
        v = v + np.random.default_rng(seed).normal(0.0, noise, v.shape)
    return np.clip(v * GREY_MAX, 0, GREY_MAX).astype(np.uint16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "pyALDVC_synthetic_crack"))
    ap.add_argument("--frames", type=int, default=3, help="deformed frames after the reference")
    ap.add_argument("--noise", type=float, default=0.01, help="grey-value noise (fraction of the range)")
    ap.add_argument("--quick", action="store_true", help="half-size volume")
    args = ap.parse_args()

    shape = QUICK_SHAPE if args.quick else SHAPE
    geo = scaled(shape, args.quick)
    out = Path(args.out)
    (out / "volumes").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    nz, ny, nx = shape
    centre = ((nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2)

    print(f"pyALDVC {__version__}: synthetic crack case {nx} x {ny} x {nz} voxels, {args.frames} deformed frames")
    speckle = generate_speckle_volume(shape, sigma=SPECKLE_SIGMA, seed=7)
    mask_ref = material_mask(shape, geo)
    window = grey_window(speckle, mask_ref)
    # the crack and the pores are dark in the reference too, as they would be in a scan
    void = window[0] + VOID_LEVEL * (window[1] - window[0])
    ref = np.where(mask_ref > 0, speckle, void)

    frame0 = to_uint16(ref, window, args.noise, 100)
    save_volume(out / "volumes" / "frame_00.tif", frame0)
    save_volume(out / "masks" / "mask_00.tif", mask_ref)
    inside = frame0[mask_ref > 0]
    print(
        f"  frame 00: reference, mask keeps {100 * mask_ref.mean():.1f} % of the voxels, "
        f"material grey {np.percentile(inside, 1):.0f} .. {np.percentile(inside, 99):.0f} of {GREY_MAX}"
    )

    step = 8 if not args.quick else 4
    gz, gy, gx = np.mgrid[0:nz:step, 0:ny:step, 0:nx:step]
    coords = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]).astype(np.float64)
    # the sampled field is convenient but it cannot be interpolated across the crack; the parameters
    # below evaluate the same field exactly at any point (see the README)
    truth = {
        "coordinates_xyz": coords,
        "shape_zyx": np.array(shape),
        "step": step,
        "crack_x": geo["crack_x"],
        "crack_front_y": geo["front_y"],
        "crack_taper": geo["taper"],
        "centre_xyz": np.array(centre),
        "opening_per_face": np.array([frame_parameters(k)[0] for k in range(1, args.frames + 1)]),
        "translation": np.array([frame_parameters(k)[1] for k in range(1, args.frames + 1)]),
        "stretch_diagonal": np.array([frame_parameters(k)[2] for k in range(1, args.frames + 1)]),
    }

    for k in range(1, args.frames + 1):
        disp = displacement(k, geo, centre)
        g, m = warp(ref, mask_ref, disp)
        m &= material_mask(shape, geo, opened=OPENING_PER_FRAME * k)  # the crack is wider once it is open
        g = np.where(m > 0, g, void)
        save_volume(out / "volumes" / f"frame_{k:02d}.tif", to_uint16(g, window, args.noise, 100 + k))
        save_volume(out / "masks" / f"mask_{k:02d}.tif", m.astype(np.uint8))
        u, v, w = disp(coords[:, 0], coords[:, 1], coords[:, 2])
        truth[f"U_{k:02d}"] = np.column_stack([u, v, w])
        print(
            f"  frame {k:02d}: opening {2 * OPENING_PER_FRAME * k:.1f} voxel, "
            f"|u| up to {np.max(np.abs(np.column_stack([u, v, w]))):.2f} voxel, mask keeps {100 * m.mean():.1f} %"
        )

    np.savez_compressed(out / "ground_truth.npz", **truth)
    meta = {
        "generator": "scripts/make_synthetic_case.py",
        "pyaldvc_version": __version__,
        "shape_zyx": list(shape),
        "crack": {"x": geo["crack_x"], "half_width": geo["crack_half"], "front_y": geo["front_y"], "taper": geo["taper"]},
        "pores": [{"centre_xyz": list(c), "radius": r} for c, r in geo["pores"]],
        "opening_per_frame_each_face": OPENING_PER_FRAME,
        "frames": args.frames,
        "noise": args.noise,
    }
    (out / "case.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "README.md").write_text(README.format(nx=nx, ny=ny, nz=nz, frames=args.frames, step=step), encoding="utf-8")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"wrote {out}  ({total / 1e6:.0f} MB)")


README = """# pyALDVC synthetic case: crack, pores, region of interest

A speckled block of {nx} x {ny} x {nz} voxels with three kinds of boundary:

* a **crack** in the plane x = 96 (3 voxels wide in the reference) that stops at a front near y = 120,
* two **spherical pores**,
* the **edge of the region of interest**, which is the mask itself.

`volumes/frame_00.tif` is the reference and `frame_01..{frames:02d}.tif` the deformed states. The crack
opens by 1.6 voxels more in every frame (0.8 per face), the opening fades to zero at the front, and a
small stretch and rigid translation are added on top. `masks/mask_kk.tif` is the region of interest of
the matching frame: 1 = material, 0 = crack or pore. Grey values are uint16.

## How to run it

1. Start the application (`al-dvc-gui`), *Add volumes...* and pick the four files in `volumes/`.
2. For every frame, *Load mask...* in the region-of-interest panel and pick the matching file in
   `masks/`. (A single mask also works: load `mask_00.tif` and press *Copy to all frames*.)
3. Parameters: subset 32, step 8 is a good start; the texture analysis suggests its own.
4. Run. "Split at boundaries" is on by default.

To see what the splitting does, run once with the check box on and once with it off, and look at the
displacement next to the crack: with it off the two faces are averaged together and the jump is
smeared over about one subset.

## Checking the result

`ground_truth.npz` holds the displacement of every frame on a grid of {step} voxels
(`coordinates_xyz`, `U_01` ...) and, more usefully, the parameters of the analytic field. The field
jumps at the crack, so interpolating the sampled grid is wrong within a few voxels of it; evaluate it
instead:

    import numpy as np
    d = np.load("ground_truth.npz")

    def truth(xyz, k):            # xyz: (M, 3) voxel coordinates [x, y, z]; k: frame number
        x, y, z = xyz.T
        c = d["centre_xyz"]
        u = d["translation"][k - 1] + d["stretch_diagonal"][k - 1] * (xyz - c)
        t = np.clip((d["crack_front_y"] - y) / d["crack_taper"], 0, 1)
        taper = t * t * (3 - 2 * t)
        u[:, 0] += np.where(x >= d["crack_x"], 1.0, -1.0) * d["opening_per_face"][k - 1] * taper
        return u

`case.json` records the geometry and the parameters the volumes were made with.
"""


if __name__ == "__main__":
    main()
