"""Write ``reports/large_volume.pdf``: what a large scan costs, before and after the optimisation work.

Inputs are the two JSON files ``scripts/bench_large_volume.py`` writes::

    python scripts/bench_large_volume.py --label before --out reports/large_volume_before.json
    ... make the changes ...
    python scripts/bench_large_volume.py --label after  --out reports/large_volume_after.json
    python scripts/make_large_volume_report.py

Every timing is wall-clock on the thread Qt would be blocking, and every memory figure is a
``tracemalloc`` peak reported per voxel so it extrapolates to the scan you care about. The last page
states what is still not solved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from al_dvc import __version__  # noqa: E402
from al_dvc.io.volume_ops import GRADIENT_AUTO_BYTES, memory_model  # noqa: E402

BEFORE, AFTER = "#ef4444", "#2563eb"
SCANS = ((512, "512^3"), (1024, "1024^3"), (2048, "2048^3"))  # what the per-voxel figures extrapolate to


def series(data: dict, group: str, key: str):
    """``(voxels, values)`` of one measurement across the sizes a run covers."""
    xs, ys = [], []
    for entry in data["sizes"].values():
        block = entry.get(group) or {}
        if key in block and isinstance(block[key], (int, float)):
            xs.append(entry["voxels"])
            ys.append(block[key])
    order = np.argsort(xs)
    return np.asarray(xs)[order], np.asarray(ys)[order]


def page_interaction(pdf, before: dict, after: dict) -> list[str]:
    """What the user feels: drawing a region, moving a slider, undoing."""
    rows = [
        ("texture_window", "draw_rectangle_s", "draw a region rectangle\n(texture window, whole edit)"),
        ("region_viewer", "slider_move_s", "move a slice slider"),
        ("region_viewer", "draw_brush_one_slice_s", "brush stroke on one slice"),
        ("mask_editor", "undo_s", "undo one operation"),
    ]
    fig, axes = plt.subplots(1, len(rows), figsize=(13, 4.2))
    lines = []
    for ax, (group, key, title) in zip(axes, rows):
        for data, colour, label in ((before, BEFORE, "before"), (after, AFTER, "after")):
            xs, ys = series(data, group, key)
            if xs.size:
                ax.plot(xs / 1e6, 1000 * ys, "o-", color=colour, label=label, lw=1.8, ms=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("volume [M voxels]", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3, which="both")
        xb, yb = series(before, group, key)
        xa, ya = series(after, group, key)
        if xb.size and xa.size:
            name = title.splitlines()[0]
            lines.append(f"{name:<34s} {1000 * yb[-1]:8.1f} -> {1000 * ya[-1]:7.1f} ms  ({yb[-1] / ya[-1]:5.1f}x)")
    axes[0].set_ylabel("time [ms]")
    axes[0].legend(fontsize=8)
    fig.suptitle("What a drawing gesture costs, against the size of the scan (log-log)", fontsize=11)
    fig.text(0.5, 0.005, "measured on the Qt thread; a 3 GB scan is about 15x the largest point here", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    pdf.savefig(fig)
    plt.close(fig)
    return lines


def page_full_volume_work(pdf, before: dict, after: dict) -> None:
    """The passes over the whole volume that used to run per edit and per redraw."""
    rows = [
        ("node_counting", "subset_valid_fraction_s", "subset_valid_fraction\n(every masked run, twice per reference)"),
        ("voi_and_lattice", "plan_lattice_cut_s", "lattice preview with the mesh cut\n(every redraw of the main viewer)"),
        ("mask_editor", "apply_rectangle_one_slice_s", "one-slice mask operation"),
        ("region_viewer", "redraw_s", "one region-viewer redraw"),
    ]
    fig, axes = plt.subplots(1, len(rows), figsize=(13, 4.2))
    for ax, (group, key, title) in zip(axes, rows):
        for data, colour, label in ((before, BEFORE, "before"), (after, AFTER, "after")):
            xs, ys = series(data, group, key)
            if xs.size:
                ax.plot(xs / 1e6, 1000 * ys, "o-", color=colour, label=label, lw=1.8, ms=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("volume [M voxels]", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("time [ms]")
    axes[0].legend(fontsize=8)
    fig.suptitle("Work that scaled with the whole volume and had no reason to", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    pdf.savefig(fig)
    plt.close(fig)


def page_memory(pdf, before: dict, after: dict) -> list[str]:
    """Bytes per voxel: the figures that decide whether a scan can be analysed at all."""
    fig, (ax_peak, ax_model) = plt.subplots(1, 2, figsize=(13, 4.6))
    lines = []
    for data, colour, label in ((before, BEFORE, "before"), (after, AFTER, "after")):
        xs, ys = series(data, "node_counting", "subset_valid_fraction_peak_bytes_per_voxel")
        if xs.size:
            ax_peak.plot(xs / 1e6, ys, "o-", color=colour, label=f"{label}: subset_valid_fraction", lw=1.8, ms=4)
        xs, ys = series(data, "reference_bundle", "resident_stored_bytes_per_voxel")
        if xs.size:
            ax_peak.plot(xs / 1e6, ys, "s--", color=colour, label=f"{label}: reference bundle (stored)", lw=1.4, ms=4)
        xs, ys = series(data, "reference_bundle", "resident_on_the_fly_bytes_per_voxel")
        if xs.size:
            ax_peak.plot(xs / 1e6, ys, "^:", color=colour, label=f"{label}: reference bundle (on the fly)", lw=1.4, ms=4)
    ax_peak.set_xscale("log")
    ax_peak.set_xlabel("volume [M voxels]", fontsize=9)
    ax_peak.set_ylabel("bytes per voxel")
    ax_peak.set_title("Measured allocation, per voxel", fontsize=10)
    ax_peak.grid(alpha=0.3, which="both")
    ax_peak.legend(fontsize=7)

    labels, olds, news = [], [], []
    for edge, name in SCANS:
        shape = (edge,) * 3
        n = float(edge) ** 3
        old = 25.0 * n + 24.1 * n  # steady masked run plus the summed-area table it had to build
        new = memory_model(shape, "auto", "cubic", True)["total_bytes"]
        labels.append(f"{name}\n{2 * n / 1e9:.1f} GB uint16")
        olds.append(old / 1e9)
        news.append(new / 1e9)
        lines.append(f"{name:<8s} masked run: {old / 1e9:7.1f} GB -> {new / 1e9:6.1f} GB")
    x = np.arange(len(labels))
    ax_model.bar(x - 0.2, olds, 0.4, color=BEFORE, label="before")
    ax_model.bar(x + 0.2, news, 0.4, color=AFTER, label="after")
    for i, (o, v) in enumerate(zip(olds, news)):
        ax_model.text(i - 0.2, o, f"{o:.0f}", ha="center", va="bottom", fontsize=7, color=BEFORE)
        ax_model.text(i + 0.2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7, color=AFTER)
    ax_model.axhline(64, color="k", ls="--", lw=1)
    ax_model.text(len(labels) - 0.5, 66, "64 GB", ha="right", fontsize=8)
    ax_model.set_yscale("log")
    ax_model.set_xticks(x)
    ax_model.set_xticklabels(labels, fontsize=8)
    ax_model.set_ylabel("resident volume memory [GB]")
    ax_model.set_title("A masked run, peak volume memory", fontsize=10)
    ax_model.legend(fontsize=8)
    ax_model.grid(alpha=0.3, axis="y", which="both")
    fig.suptitle("Memory: what decides whether the scan can be analysed at all", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    pdf.savefig(fig)
    plt.close(fig)
    return lines


def page_notes(pdf, before: dict, after: dict, interaction: list[str], memory: list[str]) -> None:
    """The numbers in words, and what is still not solved."""
    fig = plt.figure(figsize=(13, 8.5))
    fig.text(0.5, 0.96, "Large volumes: what changed, and what did not", ha="center", fontsize=13)
    text = [
        f"pyALDVC {__version__}   |   {after.get('platform', '')}",
        "",
        "INTERACTION (largest measured size; a 3 GB scan is about 15x that)",
        *[f"  {line}" for line in interaction],
        "",
        "MEMORY",
        *[f"  {line}" for line in memory],
        "",
        "WHAT WAS WRONG",
        "  subset_valid_fraction built a full (nz+1, ny+1, nx+1) int64 summed-area table through three",
        "    chained cumsums: 24.1 bytes per voxel, 26 GB on a 1024^3 scan, and it runs twice per reference",
        "    since subset splitting became the default. A masked run past ~512^3 could not allocate at all.",
        "  Every mask operation allocated a full boolean volume and combined it into the mask, even a brush",
        "    dab on one slice. The bounding box and the voxel count were recomputed by each of the twelve",
        "    callers per edit, four passes each.",
        "  Both slice viewers cleared and rebuilt every matplotlib artist per redraw, and sent the whole",
        "    slice to a pane a few hundred pixels wide: 2.69 s per canvas draw at 2048^2.",
        "  effective_voi and the lattice preview repeated whole-volume passes on every slider tick, the",
        "    preview starting with a full uint8 copy of a boolean mask.",
        "  VolumeEntry.load cached every frame the user clicked on and nothing ever released it, and a run",
        "    then materialised the whole sequence again to hand the solver a list.",
        "  A run without a mask allocated an all-ones uint8 volume, in RAM and again in VRAM.",
        "  The three gradient volumes (12 bytes per voxel) were kept whatever the size of the scan.",
        "",
        "WHAT IS STILL NOT SOLVED",
        "  The floor is now the volumes themselves: two normalised float32 frames in the provider, the",
        "    deformed frame's interpolation preparation and, on a masked run, the copy the NCC search reads.",
        "    That is about 13 bytes per voxel and it scales with the scan, so 2048^3 is still ~112 GB.",
        "  Bringing it down needs a provider that serves sub-boxes rather than whole frames, and a solver",
        "    that walks the node grid in tiles. The kernels are ready for it -- every one of them addresses",
        "    the volumes relative to a node centre -- but the provider is not.",
        "  On the GPU the same six volumes are uploaded whole: 9 bytes per voxel with on-the-fly gradients,",
        "    9.7 GB at 1024^3 and 77 GB at 2048^3. Tiling is what removes that ceiling.",
        "  gradient_mode='auto' changes the answer near a volume face: the stencil needs three voxels of",
        "    context, so nodes closer than that are refused rather than given a zero gradient. It fires above",
        f"    {GRADIENT_AUTO_BYTES / 1024**3:.0f} GB of gradients (about 880^3); below that nothing changed.",
        "  The display decimation is a display change: a region sample counts as inside only when every",
        "    voxel it covers is, so thin exclusions survive, but a thin inclusion is drawn thinner than it is.",
    ]
    fig.text(0.06, 0.90, "\n".join(text), va="top", fontsize=8.5, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", default=str(ROOT / "reports" / "large_volume_before.json"))
    ap.add_argument("--after", default=str(ROOT / "reports" / "large_volume_after.json"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "large_volume.pdf"))
    args = ap.parse_args(argv)
    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        interaction = page_interaction(pdf, before, after)
        page_full_volume_work(pdf, before, after)
        memory = page_memory(pdf, before, after)
        page_notes(pdf, before, after, interaction, memory)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
