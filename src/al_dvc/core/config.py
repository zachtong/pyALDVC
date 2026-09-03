"""DVC parameter container, defaults and validation.

``DVCPara`` is the single source of truth for every tunable of the pipeline.
It replaces the interactive ``input()`` prompts of the MATLAB code
(``funParaInput.m``, ``ReadImage3.m``) with documented, validated fields.

Use :func:`dvcpara_default` to build one (scalars are broadcast to 3-tuples)
and ``dataclasses.replace`` to derive variants.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, fields
from typing import Any, Literal

from .data_structures import FrameSchedule, VOIRange

Triple = tuple[int, int, int]
FTriple = tuple[float, float, float]

# MATLAB main_ALDVC.m: ALVarBetaList = [sqrt(1e-5),1e-2,sqrt(1e-3),1e-1,sqrt(1e-1)] * mean(h)^2 * mu
DEFAULT_BETA_RANGE: tuple[float, ...] = (
    math.sqrt(1e-5),
    1e-2,
    math.sqrt(1e-3),
    1e-1,
    math.sqrt(1e-1),
)

_TRIPLE_INT_FIELDS = ("winsize", "winstepsize", "search_radius", "init_subset", "strain_plane_fit_halfwidth")
_TRIPLE_FLOAT_FIELDS = ("voxel_size",)


@dataclass(frozen=True)
class DVCPara:
    """Immutable AL-DVC parameter set.

    Tuples that describe per-axis quantities are ordered ``(x, y, z)`` --
    the same order as node coordinates -- NOT the array order ``(z, y, x)``.
    """

    # --- 1. Volume of interest & physical units ---
    voi: VOIRange = field(default_factory=VOIRange)
    voxel_size: FTriple = (1.0, 1.0, 1.0)  # physical size of one voxel (x, y, z)
    units: str = "voxel"  # label for exports/plots
    volume_shape: Triple = (0, 0, 0)  # (nz, ny, nx); filled by the pipeline
    prefilter_sigma: float = 0.0  # Gaussian pre-smoothing of every volume (voxels; 0 = off)

    # --- 2. Subset and node grid ---
    winsize: Triple = (32, 32, 32)  # subset size (x, y, z); even integers
    winstepsize: Triple = (16, 16, 16)  # node spacing (x, y, z)

    # --- 3. Initial guess ---
    # "pyramid": coarse-to-fine NCC (robust to large motion; default)
    # "ncc":     single-level NCC with `search_radius`
    # "zero":    U0 = 0 (small motion, fastest)
    # "previous": reuse the previous frame's result (falls back to pyramid on
    #             the first frame / reference switch)
    init_guess_method: Literal["pyramid", "ncc", "zero", "previous"] = "pyramid"
    search_radius: Triple = (8, 8, 8)  # NCC search half-width per axis (voxels)
    init_subset: Triple | None = None  # NCC template size; None -> winsize
    global_shift: bool = True  # rigid pre-shift by phase correlation
    pyramid_levels: int = 0  # 0 = automatic
    pyramid_fine_radius: int = 2  # NCC refinement radius at the finer pyramid levels (voxels of that level)
    ncc_auto_expand: bool = True  # grow search radius when peaks are clipped
    ncc_max_expand: int = 3  # max number of doublings
    init_outlier_threshold: float = 2.0  # universal median test (0 disables)
    init_min_pce: float = 0.0  # drop NCC results with PCE below this (0 disables)

    # --- 4. Local IC-GN (Subproblem 1) ---
    icgn_tol: float = 1e-2  # relative gradient-norm tolerance (MATLAB criterion)
    icgn_dp_tol: float = 1e-3  # parameter-increment tolerance [voxel]; gradient terms scaled by winsize/2
    icgn_patience: int = 5  # stop a node after this many iterations without improvement of the
    # ZNCC (12-DOF) or of the step norm (3-DOF); 0 disables
    icgn_max_iter: int = 100
    interp_method: Literal["cubic", "bspline", "linear"] = "cubic"
    subset_stride: int = 1  # sample every k-th subset voxel along each axis (k^3 fewer voxels per iteration)
    min_valid_ratio: float = 0.5  # min fraction of mask-valid voxels per subset
    local_outlier_threshold: float = 2.0  # median test after the local pass (0 disables); MATLAB default.
    # Flags ~10-15 % of nodes on noisy anisotropic scans, which
    # measurably lowers the local error there (see docs/design.md)
    hessian_cond_max: float = 1e12  # reject subsets whose Hessian is worse conditioned

    # --- 5. ADMM (Subproblem 2 + iterations) ---
    use_global_step: bool = True
    mu: float = 1e-3
    beta: float | None = None  # None -> auto-tune (L-curve) per reference frame
    beta_range: tuple[float, ...] = DEFAULT_BETA_RANGE
    beta_criterion: str = "matlab"  # L-curve score: 'matlab' = |u-u_hat| + h^2 |F-grad u_hat| (discrete
    # minimum, MATLAB parity) or 'normalized' (z-scored sum, refined)
    admm_max_iter: int = 4  # total ADMM steps including the first local+global pass
    admm_tol: float = 1e-2  # stop when RMS update of U (voxels) drops below
    subpb2_method: Literal["fem", "fd"] = "fem"
    gauss_pt_order: int = 2
    dual_update: Literal["accumulate", "reset"] = "accumulate"
    global_solver: Literal["auto", "pcg", "direct"] = "auto"
    pcg_tol: float = 1e-8
    pcg_max_iter: int = 500
    alpha: float = 0.0  # extra Laplacian smoothing weight (0 = off)

    # --- 6. Post-smoothing (Gaussian, sigma in node units; 0 = off) ---
    disp_smoothing: float = 0.0
    strain_smoothing: float = 0.0

    # --- 7. Strain ---
    strain_method: Literal["plane_fit", "fem", "fd", "direct"] = "plane_fit"
    strain_plane_fit_halfwidth: Triple = (1, 1, 1)  # in nodes; (1,1,1) = 3x3x3 window
    strain_type: Literal["infinitesimal", "green_lagrange", "euler_almansi", "hencky"] = "infinitesimal"
    strain_edge_trim: bool = True  # flag nodes whose fitting window is incomplete

    # --- 8. Multi-frame tracking ---
    reference_mode: Literal["accumulative", "incremental"] = "accumulative"
    frame_schedule: FrameSchedule | None = None
    cumulative_interp: Literal["linear", "cubic"] = "cubic"

    # --- 9. Compute ---
    backend: Literal["numba", "numpy"] = "numba"
    gradient_mode: Literal["stored", "on_the_fly"] = "stored"  # "on_the_fly": no gradient volumes (-12 bytes/voxel),
    # about 15-20 % slower local step
    n_threads: int = 0  # 0 = all cores
    store_local_result: bool = True  # keep U_local / F_local in FrameResult
    verbose: bool = True

    def __post_init__(self) -> None:
        """Broadcast scalars to (x, y, z) triples and validate.

        Runs for every construction path -- ``DVCPara(...)``,
        ``dvcpara_default(...)`` and ``dataclasses.replace(...)`` -- so a
        scalar ``winsize=24`` or a dict ``voi`` is always normalised.
        """
        for name in _TRIPLE_INT_FIELDS:
            object.__setattr__(self, name, _as_triple(getattr(self, name), name, int))
        for name in _TRIPLE_FLOAT_FIELDS:
            object.__setattr__(self, name, _as_triple(getattr(self, name), name, float))
        if isinstance(self.voi, dict):
            object.__setattr__(self, "voi", VOIRange(**{k: tuple(v) for k, v in self.voi.items()}))
        object.__setattr__(self, "beta_range", tuple(float(b) for b in self.beta_range))
        object.__setattr__(self, "volume_shape", tuple(int(v) for v in self.volume_shape))
        validate_dvcpara(self)


def _as_triple(value: Any, name: str, kind: type) -> tuple:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (kind(value),) * 3
    try:
        seq = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a scalar or a length-3 sequence (got {value!r})") from exc
    if len(seq) == 1:
        return (kind(seq[0]),) * 3
    if len(seq) != 3:
        raise ValueError(f"{name} must have 3 entries (x, y, z) (got {len(seq)})")
    return tuple(kind(v) for v in seq)


def dvcpara_default(**overrides: Any) -> DVCPara:
    """Return a validated ``DVCPara`` with defaults, optionally overridden.

    Scalars given for per-axis fields (``winsize``, ``winstepsize``,
    ``search_radius``, ``voxel_size``, ``strain_plane_fit_halfwidth``,
    ``init_subset``) are broadcast to all three axes.
    ``voi`` may be given as a ``VOIRange`` or a dict ``{"x": (..), ...}``.
    """
    known = {f.name for f in fields(DVCPara)}
    unknown = set(overrides) - known
    if unknown:
        raise TypeError(f"Unknown DVCPara field(s): {sorted(unknown)}")
    return DVCPara(**overrides)  # __post_init__ broadcasts and validates


def validate_dvcpara(p: DVCPara) -> None:
    """Raise ``ValueError`` / ``TypeError`` on inconsistent parameters."""
    for k, w in zip("xyz", p.winsize):
        if not isinstance(w, (int,)) or w < 4 or w % 2 != 0:
            raise ValueError(f"winsize[{k}]={w} must be an even integer >= 4.")
    for k, h in zip("xyz", p.winstepsize):
        if not isinstance(h, int) or h < 1:
            raise ValueError(f"winstepsize[{k}]={h} must be a positive integer.")
    if any(h > w for h, w in zip(p.winstepsize, p.winsize)):
        warnings.warn(
            f"winstepsize {p.winstepsize} exceeds winsize {p.winsize}: subsets do not "
            "overlap, the global step will be weakly constrained.",
            UserWarning,
            stacklevel=3,
        )
    for k, s in zip("xyz", p.voxel_size):
        if not (s > 0) or not math.isfinite(s):
            raise ValueError(f"voxel_size[{k}]={s} must be positive and finite.")
    if p.prefilter_sigma < 0:
        raise ValueError("prefilter_sigma must be >= 0 (0 disables pre-smoothing).")
    for k, r in zip("xyz", p.search_radius):
        if r < 0:
            raise ValueError(f"search_radius[{k}]={r} must be >= 0.")
    if p.init_subset is not None:
        for k, s in zip("xyz", p.init_subset):
            if s < 4 or s % 2 != 0:
                raise ValueError(f"init_subset[{k}]={s} must be an even integer >= 4.")
    if p.init_guess_method not in ("pyramid", "ncc", "zero", "previous"):
        raise ValueError(f"init_guess_method must be pyramid|ncc|zero|previous (got {p.init_guess_method!r}).")
    if p.pyramid_levels < 0:
        raise ValueError("pyramid_levels must be >= 0 (0 = automatic).")
    if p.ncc_max_expand < 0:
        raise ValueError("ncc_max_expand must be >= 0.")
    if p.init_outlier_threshold < 0 or p.local_outlier_threshold < 0:
        raise ValueError("outlier thresholds must be >= 0 (0 disables).")

    if not (0 < p.icgn_tol < 1):
        raise ValueError(f"icgn_tol must be in (0, 1) (got {p.icgn_tol}).")
    if not (0 < p.icgn_dp_tol < 1):
        raise ValueError(f"icgn_dp_tol must be in (0, 1) (got {p.icgn_dp_tol}).")
    if p.icgn_patience < 0:
        raise ValueError("icgn_patience must be >= 0 (0 disables stall detection).")
    if p.icgn_max_iter < 1:
        raise ValueError("icgn_max_iter must be >= 1.")
    if int(p.pyramid_fine_radius) < 1:
        raise ValueError("pyramid_fine_radius must be >= 1.")
    if int(p.subset_stride) < 1:
        raise ValueError(f"subset_stride must be >= 1 (got {p.subset_stride}).")
    if int(p.subset_stride) > max(1, min(int(w) for w in p.winsize) // 4):
        raise ValueError(
            f"subset_stride {p.subset_stride} leaves fewer than 5 samples per axis for winsize {p.winsize}; "
            "use a larger subset or a smaller stride."
        )
    if p.interp_method not in ("cubic", "bspline", "linear"):
        raise ValueError(f"interp_method must be cubic|bspline|linear (got {p.interp_method!r}).")
    if not (0 < p.min_valid_ratio <= 1):
        raise ValueError("min_valid_ratio must be in (0, 1].")
    if p.hessian_cond_max <= 1:
        raise ValueError("hessian_cond_max must be > 1.")

    if p.mu <= 0:
        raise ValueError(f"mu must be positive (got {p.mu}).")
    if p.beta is not None and p.beta <= 0:
        raise ValueError("beta must be positive or None (auto).")
    if len(p.beta_range) < 1 or any(b <= 0 for b in p.beta_range):
        raise ValueError("beta_range must contain positive values.")
    if p.beta_criterion not in ("matlab", "normalized"):
        raise ValueError(f"beta_criterion must be 'matlab' or 'normalized' (got {p.beta_criterion!r}).")
    if p.admm_max_iter < 1:
        raise ValueError("admm_max_iter must be >= 1.")
    if p.admm_tol <= 0:
        raise ValueError("admm_tol must be positive.")
    if p.subpb2_method not in ("fem", "fd"):
        raise ValueError("subpb2_method must be 'fem' or 'fd'.")
    if p.gauss_pt_order not in (2, 3):
        raise ValueError("gauss_pt_order must be 2 or 3.")
    if p.dual_update not in ("accumulate", "reset"):
        raise ValueError("dual_update must be 'accumulate' or 'reset'.")
    if p.global_solver not in ("auto", "pcg", "direct"):
        raise ValueError("global_solver must be auto|pcg|direct.")
    if p.pcg_tol <= 0 or p.pcg_max_iter < 1:
        raise ValueError("pcg_tol must be > 0 and pcg_max_iter >= 1.")
    if p.alpha < 0:
        raise ValueError("alpha must be >= 0.")
    if p.disp_smoothing < 0 or p.strain_smoothing < 0:
        raise ValueError("smoothing sigmas must be >= 0.")

    if p.strain_method not in ("plane_fit", "fem", "fd", "direct"):
        raise ValueError("strain_method must be plane_fit|fem|fd|direct.")
    if any(h < 1 for h in p.strain_plane_fit_halfwidth):
        raise ValueError("strain_plane_fit_halfwidth entries must be >= 1.")
    if p.strain_type not in ("infinitesimal", "green_lagrange", "euler_almansi", "hencky"):
        raise ValueError("strain_type must be infinitesimal|green_lagrange|euler_almansi|hencky.")

    if p.reference_mode not in ("accumulative", "incremental"):
        raise ValueError("reference_mode must be 'accumulative' or 'incremental'.")
    if p.frame_schedule is not None:
        if not isinstance(p.frame_schedule, FrameSchedule):
            raise TypeError("frame_schedule must be a FrameSchedule instance.")
    if p.cumulative_interp not in ("linear", "cubic"):
        raise ValueError("cumulative_interp must be 'linear' or 'cubic'.")
    if p.backend not in ("numba", "numpy"):
        raise ValueError("backend must be 'numba' or 'numpy'.")
    if p.gradient_mode not in ("stored", "on_the_fly"):
        raise ValueError("gradient_mode must be 'stored' or 'on_the_fly'.")
    if p.gradient_mode == "on_the_fly" and p.backend == "numpy":
        raise ValueError("gradient_mode='on_the_fly' needs the numba backend.")
    if p.n_threads < 0:
        raise ValueError("n_threads must be >= 0 (0 = all cores).")


def para_to_dict(p: DVCPara) -> dict[str, Any]:
    """JSON/YAML-friendly dict of a parameter set."""
    out: dict[str, Any] = {}
    for f in fields(DVCPara):
        v = getattr(p, f.name)
        if isinstance(v, VOIRange):
            v = {"x": list(v.x), "y": list(v.y), "z": list(v.z)}
        elif isinstance(v, FrameSchedule):
            v = {"ref_indices": list(v.ref_indices)}
        elif isinstance(v, tuple):
            v = list(v)
        out[f.name] = v
    return out


def para_from_dict(d: dict[str, Any]) -> DVCPara:
    """Inverse of :func:`para_to_dict` (tolerates missing keys)."""
    d = dict(d)
    if "frame_schedule" in d and isinstance(d["frame_schedule"], dict):
        d["frame_schedule"] = FrameSchedule(ref_indices=tuple(int(i) for i in d["frame_schedule"]["ref_indices"]))
    if "volume_shape" in d and d["volume_shape"] is not None:
        d["volume_shape"] = tuple(int(v) for v in d["volume_shape"])
    return dvcpara_default(**d)
