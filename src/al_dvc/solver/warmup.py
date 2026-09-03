"""JIT warm-up: compile every Numba kernel on a tiny synthetic problem.

Call :func:`warmup` once at application start (GUI, batch script) so the
first real frame does not pay the ~10-20 s compilation cost. With
``JIT_CACHE`` enabled the compiled kernels are also cached on disk.
"""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


def warmup(verbose: bool = False) -> float:
    """Compile the kernels by running the full pipeline on a 40^3 volume.

    Returns the elapsed seconds.
    """
    from ..core.config import dvcpara_default
    from ..core.pipeline import run_aldvc
    from ..synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian

    t0 = time.perf_counter()
    vol = generate_speckle_volume((40, 40, 40), sigma=1.5, seed=0)
    g = warp_volume_lagrangian(vol, affine_displacement(None, (0.6, -0.4, 0.3)), n_iter=3, order=3)
    for interp in ("cubic", "bspline", "linear"):
        para = dvcpara_default(winsize=8, winstepsize=8, search_radius=2, interp_method=interp,
                               admm_max_iter=2, verbose=False, global_shift=(interp == "cubic"),
                               init_guess_method="pyramid" if interp == "cubic" else "ncc")
        run_aldvc(para, [vol, g], compute_strain=(interp == "cubic"))
    dt = time.perf_counter() - t0
    if verbose:
        logger.info("Numba kernels compiled in %.1fs", dt)
    return dt
