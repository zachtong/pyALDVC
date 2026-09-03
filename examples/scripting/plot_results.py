"""Re-plot exported results without recomputing anything.

    python examples/scripting/plot_results.py results/aldvc.npz exx 1 out.png
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from al_dvc.core.data_structures import DVCMesh
from al_dvc.export import load_npz_result
from al_dvc.viz import plot_field_slices


def main(npz: str, field: str = "exx", frame: int = 1, out: str | None = None) -> None:
    d = load_npz_result(npz)
    grid = tuple(int(s) for s in d["grid_shape"])
    mesh = DVCMesh(coordinates=d["coordinates"], elements=d["elements"], grid_shape=grid,
                   x0=d["x0"], y0=d["y0"], z0=d["z0"], spacing=tuple(float(s) for s in d["spacing"]),
                   node_valid=d["node_valid"])
    key = f"{field}_{frame}"
    if key not in d:
        raise SystemExit(f"{key} not in archive; available: {sorted(k for k in d if k.endswith(f'_{frame}'))}")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    plot_field_slices(d[key].reshape(grid), mesh, title=f"{field} (frame {frame})", fig=fig, axes=axes)
    out = out or f"{field}_{frame}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1] if len(a) > 1 else "exx", int(a[2]) if len(a) > 2 else 1, a[3] if len(a) > 3 else None)
