"""Visual report of the volume-size check: sizes from file headers, refused before the run, flagged at import.

    python scripts/make_volume_size_report.py [--out reports/volume_size.pdf] [--quick] [--dataset DIR]

A sequence whose volumes do not all have the reference's size used to fail when the run reached the
odd frame, after the frames before it had been solved. Now every file is sized from its header when it
is added (``read_volume_shape``, off the UI thread in the window), the odd ones are marked and reported,
and the run -- GUI, CLI or batch -- is refused before the first frame. The report measures

* the header read against a full load, per file format;
* the time until a mixed sequence is refused, before (the old lazy check) and now;
* the volume table after adding such a sequence (offscreen capture);

and lists what the header cannot tell. ``--dataset DIR`` adds the header read of every volume in a real
folder (a cloud drive that has not synced a file downloads it whole on that read).
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import textwrap
import time
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
from al_dvc.io.volume_io import (  # noqa: E402
    FileVolumeProvider,
    VolumeShapeError,
    load_volume,
    read_volume_shape,
    resolve_volume_paths,
    save_volume,
    shape_text,
)

SIZE_FULL = (384, 384, 384)  # uint16: 113 MB per file
SIZE_QUICK = (128, 128, 128)
RUN_FULL = (96, 104, 112)
RUN_QUICK = (56, 60, 64)


def _write_formats(root: Path, shape) -> dict[str, Path]:
    """The same uint16 volume in every format the header reader knows."""
    import h5py
    import tifffile
    from scipy.io import savemat

    vol = np.random.default_rng(0).integers(0, 60000, shape, dtype=np.uint16)
    files: dict[str, Path] = {}
    files["TIFF (tifffile)"] = root / "shaped.tif"
    tifffile.imwrite(str(files["TIFF (tifffile)"]), vol)
    files["TIFF (ImageJ)"] = root / "imagej.tif"
    tifffile.imwrite(str(files["TIFF (ImageJ)"]), vol, imagej=True)
    files["TIFF (page per slice)"] = root / "pages.tif"
    with tifffile.TiffWriter(str(files["TIFF (page per slice)"])) as tw:
        for k in range(shape[0]):
            tw.write(vol[k], metadata=None, contiguous=False)
    files[".npy"] = root / "vol.npy"
    np.save(files[".npy"], vol)
    files[".npz"] = root / "vol.npz"
    np.savez(files[".npz"], vol=vol)
    files["HDF5"] = root / "vol.h5"
    save_volume(files["HDF5"], vol)
    files[".mat v5"] = root / "v5.mat"
    savemat(str(files[".mat v5"]), {"vol": np.transpose(vol, (2, 1, 0))})
    files[".mat v7.3"] = root / "v73.mat"
    with h5py.File(str(files[".mat v7.3"]), "w", userblock_size=512) as f:
        f.create_dataset("vol", data=vol)
    text = b"MATLAB 7.3 MAT-file, HDF5 schema 1.00 ."
    with open(files[".mat v7.3"], "r+b") as fh:
        fh.write(text.ljust(116, b" ") + b"\x00" * 8 + b"\x00\x02IM")
    folder = root / "slices"
    folder.mkdir()
    for k in range(shape[0]):
        tifffile.imwrite(str(folder / f"s{k:04d}.tif"), vol[k])
    files["TIFF slice folder"] = folder
    return files


def _time_formats(files: dict[str, Path], repeats: int) -> list[dict]:
    rows = []
    for name, path in files.items():
        header_s = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            shape = read_volume_shape(path)
            header_s.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        loaded = load_volume(path).shape
        load_s = time.perf_counter() - t0
        rows.append(
            {"name": name, "header": statistics.median(header_s), "load": load_s, "shape": shape, "agrees": shape == loaded}
        )
    return rows


def _write_sequence(root: Path, shape) -> list[Path]:
    """Four frames of a small rigid shift; the third has one slice too many (as seen on real micro-CT data)."""
    from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian

    ref = generate_speckle_volume(shape, sigma=2.0, seed=5)
    centre = tuple((s - 1) / 2 for s in shape[::-1])
    frames = [ref]
    for k in (1, 2, 3):
        frames.append(warp_volume_lagrangian(ref, affine_displacement(np.zeros((3, 3)), (0.4 * k, -0.2 * k, 0.1 * k), centre)))
    frames[2] = np.concatenate([frames[2], frames[2][-1:]], axis=0)  # one slice more
    paths = [root / f"scan{k}.npy" for k in range(4)]
    for p, v in zip(paths, frames):
        save_volume(p, v.astype(np.float32))
    return paths


def _time_to_refusal(paths: list[Path]) -> tuple[float, float, str]:
    """Seconds until the mixed sequence is refused: the old lazy check (a run) and the check made now,
    and the message the check gives now."""
    from al_dvc.core.config import dvcpara_default
    from al_dvc.core.pipeline import run_aldvc

    para = dvcpara_default(winsize=16, winstepsize=8, search_radius=4, backend="numba", verbose=False)
    real_check = FileVolumeProvider._check_shapes
    FileVolumeProvider._check_shapes = lambda self, ref: None  # the provider as it was: frames checked when read
    t0 = time.perf_counter()
    try:
        run_aldvc(para, FileVolumeProvider(paths), compute_strain=False)
    except ValueError:
        pass
    finally:
        FileVolumeProvider._check_shapes = real_check
    before = time.perf_counter() - t0
    new_msg = ""
    t0 = time.perf_counter()
    try:
        FileVolumeProvider(paths)
    except VolumeShapeError as exc:
        new_msg = str(exc)
    after = time.perf_counter() - t0
    return before, after, new_msg


def _table_capture(root: Path) -> tuple[np.ndarray, list[str]]:
    """The window after adding a reference and three deformed volumes, two of another size."""
    from PySide6.QtWidgets import QApplication

    from al_dvc.gui.app import MainWindow, create_application

    create_application([sys.argv[0]])
    shapes = {"scan0.npy": (40, 44, 48), "scan1.npy": (40, 44, 48), "scan2.npy": (41, 44, 48), "scan3.npy": (39, 44, 49)}
    folder = root / "table"
    folder.mkdir()
    for name, shape in shapes.items():
        save_volume(folder / name, np.random.default_rng(len(name)).random(shape, dtype=np.float32))
    window = MainWindow()
    window.resize(1440, 900)
    window.show()
    logged: list[str] = []
    window.state.log_message.connect(lambda message, level: logged.append(f"[{level}] {message}"))
    window.volume_panel.import_files([str(folder / "scan0.npy")])
    window.volume_panel.import_files([str(folder / n) for n in ("scan1.npy", "scan2.npy", "scan3.npy")])
    window.state.wait_for_shapes(30_000)
    for _ in range(20):
        QApplication.processEvents()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
        shot_name = fh.name
    window.volume_panel._list.setMinimumHeight(250)  # every row in the picture
    for _ in range(10):
        QApplication.processEvents()
    window.volume_panel.grab().save(shot_name)
    shot = mpimg.imread(shot_name)
    os.unlink(shot_name)
    window.run_panel.start()  # refused
    for _ in range(5):
        QApplication.processEvents()
    window.state.dirty = False
    window.close()
    return shot, [line for line in logged if "[error]" in line]


def _dataset_rows(folder: Path) -> list[str]:
    rows = []
    for p in resolve_volume_paths(str(folder)):
        t0 = time.perf_counter()
        shape = read_volume_shape(p)
        dt = time.perf_counter() - t0
        rows.append(f"  {p.name:<28s} {shape_text(shape) if shape else '?':>20s}   header read {1e3 * dt:9.1f} ms")
    return rows


def _wrapped(lines: list[str], width: int = 96) -> list[str]:
    return [out for line in lines for out in textwrap.wrap(line, width, initial_indent="  ", subsequent_indent="    ")]


def _text_page(pdf, title: str, lines: list[str], size: float = 8.5) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.05, 0.95, title, fontsize=15, weight="bold", va="top")
    fig.text(0.05, 0.90, "\n".join(lines), fontsize=size, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "reports" / "volume_size.pdf"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dataset", help="a folder of real volumes to size from their headers")
    args = ap.parse_args(argv)
    size = SIZE_QUICK if args.quick else SIZE_FULL
    run_shape = RUN_QUICK if args.quick else RUN_FULL

    with tempfile.TemporaryDirectory(prefix="pyaldvc_size_report_") as tmp:
        root = Path(tmp)
        formats_dir = root / "formats"
        formats_dir.mkdir()
        rows = _time_formats(_write_formats(formats_dir, size), 5)
        seq_dir = root / "sequence"
        seq_dir.mkdir()
        before, after, new_msg = _time_to_refusal(_write_sequence(seq_dir, run_shape))
        shot, errors = _table_capture(root)
    dataset = _dataset_rows(Path(args.dataset)) if args.dataset else []

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(out)) as pdf:
        nbytes = int(np.prod(size)) * 2
        lines = [
            f"pyALDVC {__version__} -- volume sizes (io.volume_io.read_volume_shape, gui volume table, run panel)",
            "",
            "Every volume and mask of a sequence must have the reference's size. The window used to",
            "record only the file paths when volumes were added, and the streaming provider checked a",
            "frame when the run reached it: a scan with one slice too many failed the run after the",
            "frames before it had been solved. On a real micro-CT sequence (scans of 1856 slices,",
            "one of 1857) the run failed 2 min 25 s after it started, the first frame having taken",
            "a minute; the window now marks that scan when it is added.",
            "",
            "Now:",
            "  * read_volume_shape() sizes a file from its header; no voxel is read;",
            "  * the window sizes every volume as it is added, on a background thread (a cloud drive",
            "    downloads an unsynced file on the first read of any byte), shows every size in the",
            "    Shape column (x \u00d7 y \u00d7 z), marks the ones that differ from the reference and tells",
            "    the user once per import (a dialog; the console line below when headless);",
            "  * Run refuses such a sequence and names the volumes; a mask file of another size is",
            "    not attached;",
            "  * FileVolumeProvider (GUI runs, al-dvc run, batch sessions) checks every header before",
            "    reading a voxel and raises VolumeShapeError naming every volume that differs.",
            "",
            f"Mixed sequence of 4 frames {shape_text(run_shape)} (the third has one slice more):",
            f"  refused after {before:8.2f} s before (the run solved frame 1, then read frame 2)",
            f"  refused after {after:8.4f} s now   (four headers; nothing read, nothing solved)",
            "",
            "Message (al-dvc run, batch sessions, the run worker):",
            *_wrapped((new_msg or "(none)").splitlines()),
            "",
            "Window, after adding scan0 then scan1-3 (console errors; the dialog when not headless):",
            *_wrapped(errors),
        ]
        _text_page(pdf, "Volume sizes: checked when the volumes are added", lines)

        fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.6), gridspec_kw={"width_ratios": [3, 2]})
        names = [r["name"] for r in rows]
        y = np.arange(len(rows))
        axes[0].barh(y - 0.2, [1e3 * r["load"] for r in rows], height=0.4, color="#c44e52", label="full load")
        axes[0].barh(y + 0.2, [1e3 * r["header"] for r in rows], height=0.4, color="#4c72b0", label="header only")
        axes[0].set_xscale("log")
        axes[0].set_yticks(y, names, fontsize=7)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("time (ms, log)")
        axes[0].legend(fontsize=7)
        axes[0].set_title(f"sizing a {shape_text(size)} uint16 volume ({nbytes / 1e6:.0f} MB)", fontsize=9)
        axes[1].axis("off")
        table = [
            [r["name"], f"{1e3 * r['header']:.2f}", f"{r['load'] / max(r['header'], 1e-9):,.0f}x", "yes" if r["agrees"] else "NO"]
            for r in rows
        ]
        tab = axes[1].table(cellText=table, colLabels=["format", "header ms", "faster", "= load"], loc="center", fontsize=7)
        tab.auto_set_font_size(False)
        tab.set_fontsize(6.5)
        tab.scale(1.0, 1.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 6))
        ax.imshow(shot)
        ax.axis("off")
        ax.set_title("Volume table after the import: sizes read from the headers, two marked (offscreen capture)", fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        limits = [
            "What the header tells, per format:",
            "  TIFF (any writer, BigTIFF, compressed, one page per slice), .npy, .npz (the member",
            "  load_volume reads), HDF5 (the dataset it reads), .mat v5 and v7.3 (the variable it",
            "  picks), NIfTI and NRRD (with their optional readers): the exact size, tested against",
            "  load_volume for every case in tests/test_volume_shape.py.",
            "  Slice folders: the slice count and the first slice's size (load_volume refuses",
            "  slices of different sizes itself).",
            "",
            "What it cannot tell (the size is checked when the volume is read, as before):",
            "  DICOM folders, a MATLAB cell or struct, an .npz member that is not an array, a missing",
            "  optional reader, a header the reader does not understand. The table shows '?'.",
            "",
            "Boundary conditions:",
            "  * a file changed on disk after it was added is caught by the provider's header check",
            "    at the start of the run, or by the check when the frame is read;",
            "  * sizes still being read when Run is pressed are checked by the provider, before the",
            "    first frame (the window does not wait);",
            "  * a cloud drive (Box Drive, OneDrive) downloads a file it has not synced on the first",
            "    read, header included: 26-45 s per 3.7-GB scan on Box Drive. The read runs on a",
            "    daemon thread (closing the window never waits for it); the table shows '\u2026'.",
            "",
            "Limitation: volumes of different sizes are refused, not analysed. Cropping or padding the",
            "deformed volumes to the reference's grid (the padding masked out) would let such",
            "sequences run; that is a separate feature.",
        ]
        if dataset:
            limits += ["", f"Real folder {args.dataset}:"] + dataset
        _text_page(pdf, "Formats, boundary conditions, limitations", limits)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
