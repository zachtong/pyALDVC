"""Visual report of session files (format 3): what is saved, the round trip, moved projects, sizes and times.

    python scripts/make_session_report.py [--out reports/session.pdf] [--quick]

Everything is measured when the script runs, in a temporary folder:

* what a session saves and restores (a table), and what it leaves out on purpose;
* a completed run (strain computed in the post-processing window, a texture analysis) saved and read
  back: the largest absolute difference of every array (all zero: bit for bit, dtypes included);
* the moved-project scenarios, each with what the user gets;
* file size and save / open times against nodes x frames (synthetic results of 10k, 50k and 100k
  nodes), the time the UI thread is busy, and the longest pause of the UI thread while a large session
  is saved and opened on the worker;
* two ways of storing the masks (bit-packed vs one byte per voxel, both compressed);
* the main window and the post-processing window before saving and after reopening in a fresh window
  (offscreen captures, also written as PNG next to the report).
"""

from __future__ import annotations

import argparse
import io
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

from al_dvc import __version__  # noqa: E402
from al_dvc.core.config import dvcpara_default, para_to_dict  # noqa: E402
from al_dvc.core.data_structures import FrameResult, FrameSchedule, PipelineResult, StrainResult  # noqa: E402
from al_dvc.gui.app_state import AppState  # noqa: E402
from al_dvc.gui.session import apply_session, load_session, prepare_save, save_session, write_session  # noqa: E402
from al_dvc.io.session_bundle import pack_mask  # noqa: E402
from al_dvc.io.session_serialize import load_object_npz, load_result_npz, save_object_npz, save_result_npz  # noqa: E402
from al_dvc.io.volume_io import save_volume  # noqa: E402
from al_dvc.mesh.grid_mesh import mesh_setup  # noqa: E402
from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian  # noqa: E402

SHAPE = (40, 44, 48)
RUN_SHAPE = (56, 64, 72)  # the demonstration run: enough nodes for a fit window of 5
SIZES_FULL = {"small": ((22, 22, 21), 3), "medium": ((37, 37, 37), 5), "large": ((47, 46, 46), 10)}
SIZES_QUICK = {"small": ((12, 12, 12), 2), "medium": ((22, 22, 21), 3), "large": ((30, 30, 30), 4)}
MASK_EDGES_FULL = (256, 512)
MASK_EDGES_QUICK = (128, 192)

SAVED = [
    ("Volumes", "session.json", "order, labels, identities; path + relative path + fingerprint (never the voxels)"),
    ("Masks / region of interest", "masks.npz", "every distinct composed mask, bit-packed (frames sharing one: stored once)"),
    ("Parameters", "session.json", "every DVCPara field (advanced, backend), checkpoint option"),
    ("Output folder, exported archive", "session.json", "absolute + relative path"),
    ("Run results", "results.npz", "all frames: U, F, U_accum, local pass, U0, ZNCC, U_std, status, outliers,"),
    ("", "", "split fractions, ADMM and beta sweep, mesh, schedule, timings, parameters used"),
    ("Result <-> volume mapping", "session.json", "result_uids + volume identities: fields stay on their frames"),
    ("Strain (post-processing)", "results.npz", "every frame, with method, measure, fit window, smoothing, edge trim"),
    ("Statistics", "session.json", "regions, fit region, displayed correction, all tab settings"),
    ("Texture analysis", "texture.npz", "autocorrelation + RVE sweep; region (masks.npz), centre, cube, factor,"),
    ("", "", "sweep settings, whether each result still describes the input, plot settings"),
    ("Display", "session.json", "field, frame, colormap, range, overlay, slices, layout, same scale, background,"),
    ("", "", "grid, subset, mask tint"),
    ("3-D view", "session.json", "mode, surfaces, iso level, cut-away, warp, arrows, slices, outline, background,"),
    ("", "", "camera preset / turn / tilt / zoom, animation kind / axis / direction / speed / smooth"),
    ("Windows", "session.json", "centre tab; post-processing tab, frame, strain and display controls, open or not;"),
    ("", "", "texture step, open or not"),
]
NOT_SAVED = [
    "the volumes' voxels (gigabytes; referenced and found again instead);",
    "application preferences: theme, language, window geometry, notifications (they belong to the application);",
    "the drawing tool of the moment, undo histories, export dialog choices;",
    "statistics computed on demand (series over frames, noise floor, profiles, line): recomputed from the saved",
    "  result and settings when their tab is used; a running job (a run, a strain computation) is not saved.",
]


# ---------------------------------------------------------------------------- helpers
def _pump(n: int = 20) -> None:
    from PySide6.QtWidgets import QApplication

    for _ in range(n):
        QApplication.processEvents()


def _frames(shape=SHAPE, n=3):
    centre = tuple((s - 1) / 2 for s in shape[::-1])
    ref = generate_speckle_volume(shape, sigma=2.0, seed=51)
    out = [ref]
    for k in range(1, n):
        disp = affine_displacement(np.diag([0.005 * k, -0.003 * k, 0.004 * k]), (0.4 * k, -0.3, 0.2), centre)
        out.append(warp_volume_lagrangian(ref, disp))
    return out


def _write(folder: Path, frames) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, vol in enumerate(frames):
        save_volume(folder / f"v{k}.npy", vol)
        paths.append(folder / f"v{k}.npy")
    return paths


def _leaves(a, b, where="result"):
    """``(path, a array, b array)`` for every array of two result trees, walked together."""
    if isinstance(a, np.ndarray):
        yield where, a, b
    elif is_dataclass(a):
        for f in fields(a):
            yield from _leaves(getattr(a, f.name), getattr(b, f.name), f"{where}.{f.name}")
    elif isinstance(a, dict):
        for k in a:
            yield from _leaves(a[k], b[k], f"{where}[{k!r}]")
    elif isinstance(a, (list, tuple)):
        for i, (x, y) in enumerate(zip(a, b)):
            yield from _leaves(x, y, f"{where}[{i}]")


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return 0.0
    if a.dtype.kind == "f":
        both_nan = np.isnan(a) & np.isnan(b)
        if np.any(np.isnan(a) != np.isnan(b)):
            return float("inf")
        d = np.abs(np.where(both_nan, 0.0, a.astype(np.float64) - b.astype(np.float64)))
        return float(np.nanmax(d)) if d.size else 0.0
    return float(np.max(np.abs(a.astype(np.int64) - b.astype(np.int64))))


def _verification_rows(a, b, prefix: str) -> list[tuple]:
    """One row per array kind (frames pooled): path, dtype, count, largest |difference|, bytes identical."""
    groups: dict = {}
    for path, x, y in _leaves(a, b, prefix):
        key = re.sub(r"\[\d+\]", "[*]", path)
        g = groups.setdefault(key, {"dtype": str(x.dtype), "n": 0, "diff": 0.0, "same": True})
        g["n"] += 1
        g["diff"] = max(g["diff"], _max_abs(x, y))
        same = x.dtype == y.dtype and x.shape == y.shape
        same = same and np.ascontiguousarray(x).tobytes() == np.ascontiguousarray(y).tobytes()
        g["same"] = g["same"] and same
    return [(k, g["dtype"], g["n"], g["diff"], "yes" if g["same"] else "NO") for k, g in groups.items()]


def _synthetic(grid, n_frames, seed=0) -> PipelineResult:
    """A result of ``grid`` nodes and ``n_frames`` frames with every array a run fills (smooth fields plus noise)."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = grid
    mesh = mesh_setup(10.0 + 8 * np.arange(nx), 10.0 + 8 * np.arange(ny), 10.0 + 8 * np.arange(nz))
    n = mesh.n_nodes
    mesh = replace(mesh, node_valid=np.ones(n, bool))
    X = mesh.coordinates
    para = dvcpara_default(winsize=16, winstepsize=8, verbose=False)
    frames, strains = [], []
    for k in range(n_frames):
        U = np.column_stack([1e-3 * (k + 1) * X[:, 0], -5e-4 * X[:, 1], 2e-4 * X[:, 2]]) + 1e-3 * rng.standard_normal((n, 3))
        F = 1e-3 * rng.standard_normal((n, 3, 3))
        frames.append(
            FrameResult(
                U=U,
                F=F,
                U_accum=U.copy(),
                U_local=U + 1e-4 * rng.standard_normal((n, 3)),
                F_local=F + 1e-5,
                U0=np.round(U),
                zncc=0.9 + 0.05 * rng.random(n),
                U_std=1e-3 * rng.random((n, 3)),
                status=np.zeros(n, np.int8),
                outlier=rng.random(n) < 0.01,
            )
        )
        e = [1e-3 * rng.standard_normal(n) for _ in range(12)]
        strains.append(
            StrainResult(
                U[:, 0],
                U[:, 1],
                U[:, 2],
                F,
                *e[:6],
                np.column_stack(e[6:9]),
                e[9],
                e[10],
                e[11],
                1 + e[0],
                e[1],
                np.ones(n, bool),
                "infinitesimal",
                "plane_fit",
            )  # fmt: skip
        )
    return PipelineResult(para, mesh, frames, strains, FrameSchedule(tuple(0 for _ in range(n_frames))), (400, 400, 400))


def _grab(widget, path: Path):
    widget.grab().save(str(path))
    return mpimg.imread(str(path))


# ---------------------------------------------------------------------------- sections
def completed_run(root: Path):
    """A real run in the window: strain, a region, display and 3-D settings, a texture analysis; saved, closed,
    reopened in a fresh window. Returns the verification rows, the screenshots and the session path."""
    from al_dvc.gui.app import MainWindow
    from al_dvc.gui.mask_editor import MaskOp
    from al_dvc.gui.names import select_key

    frames = _frames(RUN_SHAPE)
    paths = _write(root / "data", frames)
    window = MainWindow()
    window.resize(1440, 900)
    window.show()
    st = window.state
    st.add_volume_paths([str(p) for p in paths])
    st.wait_for_shapes(10_000)
    nz, ny, nx = RUN_SHAPE
    st.set_mask_display(target="all")  # a drawn region of interest on every frame (stored once in masks.npz)
    st.apply_mask_op(MaskOp("ellipse", "xy", ((0.06 * nx, 0.06 * ny), (0.94 * nx, 0.94 * ny))))
    st.set_params(winsize=16, winstepsize=6, search_radius=4, admm_max_iter=2, verbose=False)
    st.set_output_dir(root / "out")
    st.write_checkpoints = False
    window.run_panel.start()
    window.run_panel.wait(600_000)
    _pump()
    sw = window.open_strain_window()
    sw.resize(1200, 760)
    select_key(sw.method, "plane_fit")
    select_key(sw.measure, "green_lagrange")
    sw.fit_window.setValue(5)
    sw.disp_smoothing.setValue(0.5)
    sw.edge_trim.setChecked(False)
    sw.compute()
    sw.wait(300_000)
    _pump()
    sw.show_tab("analysis")
    grip = sw.analysis.regions_panel.add("sphere")
    select_key(sw.analysis.motion, "rigid")
    select_key(sw.analysis.fit_region_combo, grip.id)
    sw.analysis.apply_main.setChecked(True)
    sw.analysis.wait()
    sw.show_tab("strain")
    select_key(sw.field, "exx")
    sw.frame_slider.setValue(1)
    st.set_current_frame(2)
    window.results_panel.select_field("exx")
    st.set_display(colormap="magma", overlay_alpha=0.6)
    window.viewer.set_layout("row")
    select_key(window.view3d.mode, "surface")
    window.view3d.iso_levels.setValue(3)
    tw = window.open_texture_window()
    _pump()
    tw.go_to_step(tw.TAB_ACF)
    tw.cube_size.setValue(24)
    tw.analyse()
    tw.wait(300_000)
    _pump(40)
    shots = {"main_before": _grab(window, root / "main_before.png"), "post_before": _grab(sw, root / "post_before.png")}
    before_result, before_texture = st.results, {"result": tw.result}
    session = window.save_session_path(root / "proj" / "completed.aldvc")
    st.dirty = False
    for w in (sw, tw, window):
        w.close()
    _pump()

    fresh = MainWindow()
    fresh.resize(1440, 900)
    fresh.show()
    t0 = time.perf_counter()
    missing = fresh.open_session_path(str(session))
    reopen_s = time.perf_counter() - t0
    _pump(40)
    sw2 = fresh.strain_window
    sw2.resize(1200, 760)
    _pump(20)
    shots["main_after"] = _grab(fresh, root / "main_after.png")
    shots["post_after"] = _grab(sw2, root / "post_after.png")
    rows = _verification_rows(before_result, fresh.state.results, "result")
    rows += _verification_rows(before_texture, {"result": fresh.texture_window.result}, "texture")
    checks = {
        "missing volumes": len(missing),
        "run state": fresh.state.run_state.value,
        "unsaved changes after opening": fresh.state.dirty,
        "strain in post-processing": bool(fresh.state.results.result_strain) and not sw2.is_stale,
        "valid strain nodes, frame 1": int(fresh.state.results.result_strain[0].strain_valid.sum()),
        "regions": len(fresh.state.regions),
        "display correction": fresh.state.correction_text(),
        "texture analysis current": not fresh.texture_window.is_stale,
        "reopen (read + apply) [s]": round(reopen_s, 2),
    }
    fresh.state.dirty = False
    for w in (fresh.strain_window, fresh.texture_window, fresh):
        w.close()
    _pump()
    return rows, shots, checks, session


def relocation_scenarios(root: Path) -> list[tuple[str, str, str]]:
    """Each moved-project scenario: what happened to the files, and what the user gets on opening."""
    frames = _frames((24, 26, 28))
    result = _synthetic((4, 4, 4), 2)
    rows = []

    def project(base: Path, sub="data", names=("v0.npy", "v1.npy", "v2.npy")) -> Path:
        (base / sub).mkdir(parents=True, exist_ok=True)
        paths = []
        for name, vol in zip(names, frames):
            (base / sub / name).parent.mkdir(parents=True, exist_ok=True)
            save_volume(base / sub / name, vol)
            paths.append(base / sub / name)
        state = AppState()
        state.add_volume_paths([str(p) for p in paths])
        state.wait_for_shapes(10_000)
        state.set_results(result)
        return save_session(state, base / "proj" / "s.aldvc")

    def outcome(session, locate=None) -> str:
        data = load_session(session)
        state = AppState()
        missing = apply_session(data, state, session, locate_folder_cb=locate)
        hows = sorted({v["how"] for v in data.volumes})
        restored = "results restored" if state.results is not None else "NO results"
        return f"{3 - len(missing)}/3 found ({', '.join(hows)}); {restored}; {len(missing)} missing"

    s = project(root / "a")
    shutil.move(str(root / "a"), str(root / "a_moved" / "renamed"))
    rows.append(("(a) session + volumes moved together", "data/ outside proj/", outcome(root / "a_moved/renamed/proj/s.aldvc")))
    s = project(root / "b")
    (root / "b_desk").mkdir()
    shutil.move(str(s), str(root / "b_desk" / "s.aldvc"))
    rows.append(("(b) session moved alone", "volumes untouched", outcome(root / "b_desk" / "s.aldvc")))
    s = project(root / "c")
    shutil.move(str(root / "c" / "data"), str(s.parent / "data"))
    rows.append(("(c) volumes moved next to the session", "into proj/data", outcome(s)))
    s = project(root / "c2", names=("v0.npy", "series/day1/v1.npy", "series/day2/v2.npy"))
    for name in ("v0.npy", "series/day1/v1.npy", "series/day2/v2.npy"):
        (s.parent / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(root / "c2" / "data" / name), str(s.parent / name))
    rows.append(("(c') data content moved into proj/", "one found, the move applied", outcome(s)))
    s = project(root / "d")
    shutil.rmtree(root / "d" / "data")
    rows.append(("(d) volumes deleted", "nothing to find", outcome(s)))
    s = project(root / "e")
    shutil.move(str(root / "e" / "data"), str(root / "e_archive"))
    rows.append(
        ("(e) volumes elsewhere, user points at", "e_archive/ (headless: given)", outcome(s, lambda _m: str(root / "e_archive")))
    )
    s = project(root / "f")
    shutil.rmtree(root / "f" / "data")
    save_volume(s.parent / "v0.npy", np.zeros((8, 8, 8), np.float32))
    rows.append(("(f) another file with the same name", "proj/v0.npy, other size", outcome(s)))
    # an older session (format 2, drawn masks in <name>_masks) moved as a whole
    legacy = root / "g"
    paths = _write(legacy / "data", frames)
    (legacy / "s_masks").mkdir()
    save_volume(legacy / "s_masks" / "mask_00.npy", np.ones(frames[0].shape, np.uint8))
    doc = {
        "format": 2,
        "para_revision": 2,
        "volumes": [{"path": f"data/{p.name}", "mask": "s_masks/mask_00.npy" if k == 0 else None} for k, p in enumerate(paths)],
        "para": para_to_dict(dvcpara_default()),
        "output_dir": "out",
    }
    import json

    (legacy / "s.aldvc").write_text(json.dumps(doc), encoding="utf-8")
    shutil.move(str(legacy), str(root / "g_moved"))
    data = load_session(root / "g_moved" / "s.aldvc")
    state = AppState()
    missing = apply_session(data, state, root / "g_moved" / "s.aldvc")
    mask_ok = state.volumes[0].load_mask() is not None and bool(state.volumes[0].load_mask().all())
    again = save_session(state, root / "g_moved" / "s3.aldvc")
    back = load_session(again)
    embedded = back.volumes[0]["mask"] is not None and bool(back.volumes[0]["mask"].all())
    rows.append(
        (
            "(g) format 2 session moved with its mask folder",
            "saved again",
            f"{3 - len(missing)}/3 found, mask {'read' if mask_ok else 'NOT read'}; saved as format {back.format}, "
            f"mask {'inside' if embedded else 'NOT inside'}",
        )
    )
    return rows


def scaling(root: Path, sizes: dict) -> list[dict]:
    frames = [np.zeros((8, 8, 8), np.float32)]
    rows = []
    for name, (grid, n_frames) in sizes.items():
        result = _synthetic(grid, n_frames, seed=len(rows))
        paths = _write(root / name / "data", frames * (n_frames + 1))
        state = AppState()
        state.add_volume_paths([str(p) for p in paths])
        state.wait_for_shapes(10_000)
        state.set_results(result)
        target = root / name / "s.aldvc"
        t0 = time.perf_counter()
        plan = prepare_save(state, target)
        t1 = time.perf_counter()
        write_session(plan)
        t2 = time.perf_counter()
        data = load_session(target)
        t3 = time.perf_counter()
        fresh = AppState()
        apply_session(data, fresh, target)
        t4 = time.perf_counter()
        raw = sum(int(a.nbytes) for a in _all_arrays(result))
        rows.append(
            {
                "name": name,
                "nodes": result.dvc_mesh.n_nodes,
                "frames": n_frames,
                "raw_mb": raw / 1e6,
                "file_mb": target.stat().st_size / 1e6,
                "prepare_s": t1 - t0,
                "write_s": t2 - t1,
                "load_s": t3 - t2,
                "apply_s": t4 - t3,
            }
        )
    return rows


def _all_arrays(obj):
    seen = set()
    for _p, a, _b in _leaves(obj, obj):
        if id(a) not in seen:
            seen.add(id(a))
            yield a


def responsiveness(root: Path, grid, n_frames) -> dict:
    """The longest gap between the ticks of a 10 ms UI-thread timer while a large session is saved and opened on the
    worker (the window's own path: File > Save session / Open session)."""
    from PySide6.QtCore import QElapsedTimer, QTimer

    from al_dvc.gui.app import MainWindow

    result = _synthetic(grid, n_frames, seed=7)
    paths = _write(root / "resp" / "data", [np.zeros((8, 8, 8), np.float32)] * (n_frames + 1))
    window = MainWindow()
    window.state.add_volume_paths([str(p) for p in paths])
    window.state.wait_for_shapes(10_000)
    window.state.set_results(result)
    _pump()
    clock, gaps = QElapsedTimer(), []
    timer = QTimer()
    timer.setInterval(10)

    def tick():
        gaps.append(clock.restart())

    timer.timeout.connect(tick)
    out = {}
    for kind in ("save", "open"):
        gaps.clear()
        clock.start()
        timer.start()
        t0 = time.perf_counter()
        if kind == "save":
            window.save_session_path(root / "resp" / "s.aldvc", wait=False)
        else:
            window.open_session_path(str(root / "resp" / "s.aldvc"), wait=False)
        while window.sessions.busy:
            _pump(1)
            time.sleep(0.002)
        out[f"{kind}_s"] = time.perf_counter() - t0
        _pump(5)
        timer.stop()
        out[f"{kind}_max_gap_ms"] = max(gaps[1:] or [0])
    window.state.dirty = False
    window.close()
    _pump()
    return out


def mask_storage(edges) -> list[dict]:
    rows = []
    for edge in edges:
        z, y, x = np.ogrid[:edge, :edge, :edge]
        c = edge / 2
        mask = ((x - c) ** 2 + (y - c) ** 2 + ((z - c) * 1.3) ** 2) < (0.42 * edge) ** 2
        for method in ("bit-packed", "one byte per voxel"):
            buf = io.BytesIO()
            t0 = time.perf_counter()
            if method == "bit-packed":
                bits, _meta = pack_mask(mask)
                np.savez_compressed(buf, m=bits)
            else:
                np.savez_compressed(buf, m=mask)
            t1 = time.perf_counter()
            buf.seek(0)
            with np.load(buf) as npz:
                back = npz["m"]
            if method == "bit-packed":
                back = np.unpackbits(back, count=mask.size).reshape(mask.shape).view(bool)
            t2 = time.perf_counter()
            assert np.array_equal(back, mask)
            rows.append(
                {
                    "edge": edge,
                    "method": method,
                    "raw_mb": mask.nbytes / 1e6,
                    "mb": len(buf.getvalue()) / 1e6,
                    "write_s": t1 - t0,
                    "read_s": t2 - t1,
                }
            )
    return rows


# ---------------------------------------------------------------------------- pages
def _text_page(pdf, title: str, lines: list[str], size: float = 8.4) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.05, 0.96, title, fontsize=15, weight="bold", va="top")
    fig.text(0.05, 0.915, "\n".join(lines), fontsize=size, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def _table_page(pdf, title: str, header: list[str], rows: list[list], widths=None, note: str = "", size: float = 7.4) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.05, 0.96, title, fontsize=13, weight="bold", va="top")
    if note:
        fig.text(0.05, 0.925, note, fontsize=8.2, va="top", wrap=True)
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.84 if note else 0.87])
    ax.axis("off")
    table = ax.table(
        cellText=[[str(c) for c in r] for r in rows], colLabels=header, loc="upper left", colWidths=widths, cellLoc="left"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(size)
    table.scale(1.0, 1.25)
    for (r, _c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#dbe4f0")
    pdf.savefig(fig)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "reports" / "session.pdf"))
    ap.add_argument("--quick", action="store_true", help="smaller synthetic results and masks")
    args = ap.parse_args(argv)
    sizes = SIZES_QUICK if args.quick else SIZES_FULL
    edges = MASK_EDGES_QUICK if args.quick else MASK_EDGES_FULL
    out = Path(args.out)
    shots_dir = out.parent / "session_screens"
    shots_dir.mkdir(parents=True, exist_ok=True)

    from al_dvc.gui.app import create_application

    _app = create_application([sys.argv[0]])
    with tempfile.TemporaryDirectory(prefix="pyaldvc_session_report_") as tmp:
        root = Path(tmp)
        print("completed run ...")
        rows, shots, checks, session = completed_run(root / "run")
        members = sorted(zipfile.ZipFile(session).namelist())
        member_sizes = {i.filename: i.file_size for i in zipfile.ZipFile(session).infolist()}
        for name in ("main_before", "main_after", "post_before", "post_after"):
            shutil.copy(root / "run" / f"{name}.png", shots_dir / f"{name}.png")
        print("relocation scenarios ...")
        scenarios = relocation_scenarios(root / "moves")
        print("scaling ...")
        scale = scaling(root / "scale", sizes)
        big_grid, big_frames = sizes["large"]
        print("responsiveness ...")
        resp = responsiveness(root, big_grid, big_frames)
        print("masks ...")
        masks = mask_storage(edges)
        # the texture archive alone, as a check of the generic encoder on another record tree
        from al_dvc.texture import analyse_texture

        tex = analyse_texture(_frames((32, 32, 32), 1)[0])
        buf = io.BytesIO()
        save_object_npz(buf, {"result": tex}, "texture")
        buf.seek(0)
        tex_back = load_object_npz(buf, "texture", dict)["result"]
        assert all(d == 0.0 for _p, _t, _n, d, _s in _verification_rows(tex, tex_back, "t"))
        buf = io.BytesIO()
        save_result_npz(buf, _synthetic((6, 6, 6), 2))
        buf.seek(0)
        load_result_npz(buf)

    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(out)) as pdf:
        big = scale[-1]
        _text_page(
            pdf,
            "Session files (format 3)",
            [
                f"pyALDVC {__version__} -- al_dvc.gui.session, al_dvc.io.session_bundle / session_serialize / session_paths",
                "",
                "A .aldvc session is one zip file:",
                "  session.json  what the user set up and saw (human-readable JSON)",
                "  results.npz   the computed result, faithfully (generic dataclass encoder, arrays deduplicated,",
                "                allow_pickle=False)",
                "  masks.npz     every distinct composed mask, bit-packed and compressed",
                "  texture.npz   the texture analysis",
                "Written to a temporary file next to the target and moved over it when complete: a failed save",
                "leaves the previous file untouched. Opening sniffs zip vs JSON: formats 1 and 2 still open (with",
                "their <name>_masks folders); saving always writes format 3. A newer format is refused.",
                "",
                "Saving and opening run on a worker thread with a progress bar in the status bar; the UI thread",
                "only takes the snapshot and applies the session.",
                "",
                f"Completed run reopened in a fresh window (bundle members: {', '.join(members)}):",
                *(f"  {k}: {v}" for k, v in checks.items()),
                "",
                f"Largest case measured: {big['nodes']:,} nodes x {big['frames']} frames -> {big['file_mb']:.0f} MB,",
                f"  written in {big['write_s']:.2f} s, read in {big['load_s']:.2f} s on the worker;",
                f"  UI thread busy {1000 * big['prepare_s']:.0f} ms (snapshot) + {1000 * big['apply_s']:.0f} ms (apply);",
                f"  longest UI pause during save {resp['save_max_gap_ms']:.0f} ms, during open {resp['open_max_gap_ms']:.0f} ms.",
                "",
                "Not saved, on purpose:",
                *(f"  - {line}" for line in NOT_SAVED),
                "",
                "Sizes of the members of the completed-run bundle (bytes):",
                *(f"  {k}: {v:,}" for k, v in member_sizes.items()),
            ],
        )
        _table_page(
            pdf, "What a session saves and restores", ["item", "where", "content"], [list(r) for r in SAVED], [0.2, 0.13, 0.67]
        )
        _table_page(
            pdf,
            "Round trip of a completed run: every array read back",
            ["array (frames pooled)", "dtype", "count", "max |diff|", "bytes equal"],
            [[p, t, n, f"{d:g}", s] for p, t, n, d, s in rows],
            [0.52, 0.1, 0.08, 0.14, 0.14],
            note="A real run (3 frames, a drawn region of interest) with strain computed in the post-processing window "
            "(plane fit, Green-Lagrange, fit window 5, displacement smoothing 0.5) and a texture analysis, saved from the "
            "window and opened in a fresh one.",
            size=6.4,
        )
        _table_page(
            pdf,
            "Moved projects: what the user gets",
            ["scenario", "what moved", "outcome on opening"],
            [list(r) for r in scenarios],
            [0.31, 0.21, 0.48],
            note="Order: saved path; relative to the session; a folder beside the session with the old folder's name; the "
            "session's folder; the move of one volume applied to the others; the folder the user points at (asked once). A "
            "candidate other than the saved path must match name and size. Missing volumes stay in the list, marked; "
            "results, masks and settings are restored anyway.",
        )
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 8))
        labels = [f"{r['nodes'] / 1000:.0f}k x {r['frames']}" for r in scale]
        axes[0, 0].bar(labels, [r["raw_mb"] for r in scale], color="#9aa9c2", label="arrays in memory")
        axes[0, 0].bar(labels, [r["file_mb"] for r in scale], color="#4c72b0", width=0.5, label="session file")
        axes[0, 0].set_ylabel("MB")
        axes[0, 0].set_title("size (nodes x frames)", fontsize=10)
        axes[0, 0].legend(fontsize=7)
        for key, color, lab in (("write_s", "#dd8452", "write (worker)"), ("load_s", "#55a868", "read (worker)")):
            axes[0, 1].plot(labels, [r[key] for r in scale], "o-", color=color, label=lab)
        axes[0, 1].set_ylabel("s")
        axes[0, 1].set_title("save / open time", fontsize=10)
        axes[0, 1].legend(fontsize=7)
        axes[1, 0].bar(labels, [1000 * r["prepare_s"] for r in scale], color="#8172b3", label="snapshot (save)")
        axes[1, 0].bar(
            labels,
            [1000 * r["apply_s"] for r in scale],
            bottom=[1000 * r["prepare_s"] for r in scale],
            color="#c44e52",
            label="apply (open)",
        )
        axes[1, 0].set_ylabel("ms")
        axes[1, 0].set_title("time on the UI thread (no window)", fontsize=10)
        axes[1, 0].legend(fontsize=7)
        mlabels = sorted({f"{m['edge']}^3" for m in masks}, key=lambda s: int(s.split("^")[0]))
        for k, method in enumerate(("bit-packed", "one byte per voxel")):
            vals = [m["mb"] for m in masks if m["method"] == method]
            axes[1, 1].bar(np.arange(len(vals)) + 0.38 * k, vals, 0.38, label=method)
        axes[1, 1].set_xticks(np.arange(len(mlabels)) + 0.19, mlabels)
        axes[1, 1].set_ylabel("MB (compressed)")
        axes[1, 1].set_title("a mask of a sphere, two encodings", fontsize=10)
        axes[1, 1].legend(fontsize=7)
        fig.suptitle("Sizes and times", fontsize=13, weight="bold")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        _table_page(
            pdf,
            "Sizes and times: numbers",
            ["case", "nodes", "frames", "arrays MB", "file MB", "snapshot s", "write s", "read s", "apply s"],
            [
                [
                    r["name"],
                    f"{r['nodes']:,}",
                    r["frames"],
                    f"{r['raw_mb']:.1f}",
                    f"{r['file_mb']:.1f}",
                    f"{r['prepare_s']:.3f}",
                    f"{r['write_s']:.2f}",
                    f"{r['load_s']:.2f}",
                    f"{r['apply_s']:.3f}",
                ]
                for r in scale
            ]  # fmt: skip
            + [["", "", "", "", "", "", "", "", ""], ["mask", "edge", "method", "raw MB", "file MB", "", "write s", "read s", ""]]
            + [
                [
                    "",
                    m["edge"],
                    m["method"],
                    f"{m['raw_mb']:.1f}",
                    f"{m['mb']:.2f}",
                    "",
                    f"{m['write_s']:.2f}",
                    f"{m['read_s']:.2f}",
                    "",
                ]
                for m in masks
            ]
            + [
                ["", "", "", "", "", "", "", "", ""],
                [
                    "window",
                    f"{big['nodes']:,}",
                    big["frames"],
                    "",
                    "",
                    "",
                    f"save {resp['save_s']:.1f}",
                    f"open {resp['open_s']:.1f}",
                    "",
                ],
                ["UI pause", "", "", "", "", "", f"{resp['save_max_gap_ms']:.0f} ms", f"{resp['open_max_gap_ms']:.0f} ms", ""],
            ],
            note="Synthetic results with every array a run fills (displacement, gradients, local pass, initial guess, ZNCC, "
            "uncertainty, status, outliers, strain). Floating-point arrays are stored uncompressed unless a sample shrinks by "
            "15 % (measured noise barely compresses: 4 % at zlib level 1 for 3.4 x the time); integer and boolean arrays are "
            "deflated. The window rows: the File menu path, on the worker, with a 10 ms UI timer measuring the longest pause.",
            size=6.8,
        )
        for key, title in (("main", "Main window"), ("post", "Post-processing window")):
            fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))
            for ax, when in zip(axes, ("before", "after")):
                ax.imshow(shots[f"{key}_{when}"])
                ax.axis("off")
                ax.set_title(
                    f"{title}: {'before saving' if when == 'before' else 'after reopening in a fresh window'}", fontsize=10
                )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        _text_page(
            pdf,
            "Limitations",
            [
                "- The volumes are referenced, not embedded: a project moved without its volumes opens with the",
                "  frames marked missing (results, masks, settings restored); the grey values need the files.",
                "- A candidate found elsewhere must match the saved size in bytes; a volume re-exported with the",
                "  same name but another size is not adopted (by design). Folders of slices are matched by their",
                "  file count. Sessions of formats 1 and 2 carry no fingerprint: their files are matched by name.",
                "- A file changed in place (same path, other size) is used and reported, not refused.",
                "- The masks are held in memory once a session is open (one array per distinct mask); a session",
                "  with many different full-size masks needs that memory.",
                "- A session is as large as its result: about the size of the arrays in memory (floats are",
                "  stored uncompressed when they do not compress). Saving writes a temporary copy next to the",
                "  file, so the folder needs that much free space.",
                "- Statistics computed on demand (series, noise floor, profiles, line) are recomputed, not stored;",
                "  a job running at save time (a run, a strain computation) is not part of the session.",
                "- Opening a session of a newer pyALDVC is refused rather than guessed at.",
            ],
        )
    print(f"wrote {out}")
    print(f"screenshots in {shots_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
