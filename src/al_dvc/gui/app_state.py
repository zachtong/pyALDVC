"""Application state shared by every panel of the pyALDVC window.

One ``AppState`` instance holds the loaded volumes and masks, the current
parameter set, the run status, the results and the display settings, and
announces changes through Qt signals. Panels never talk to each other: they
read and write the state and react to its signals.
"""

from __future__ import annotations

import contextlib
import enum
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QCoreApplication, QObject, Qt, QThread, Signal

from al_dvc.core.config import DVCPara, dvcpara_default
from al_dvc.core.data_structures import PipelineResult, VOIRange, voi_from_mask

from .mask_editor import FULL_BASE, MaskEditor, MaskOp, ThresholdCancelled, threshold_region

logger = logging.getLogger(__name__)
_NUMBER = re.compile(r"(\d+)")


def natural_key(path: str | Path) -> tuple:
    """Sort key that orders the numbers inside a file name numerically: frame2 before frame10."""
    parts = _NUMBER.split(Path(path).name)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def lexical_key(path: str | Path) -> str:
    """Sort key that compares file names character by character (000, 001, ..., as the file system lists them)."""
    return Path(path).name.lower()


SIZE_SEP = " \u00d7 "  # multiplication sign between the x, y and z sizes the window shows


def size_text(shape) -> str:
    """A ``(nz, ny, nx)`` shape as the window shows a volume's size: ``nx \u00d7 ny \u00d7 nz`` (x, y, z)."""
    from al_dvc.io.volume_io import shape_text

    return shape_text(shape, SIZE_SEP)


class RunState(enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    DONE = "done"
    FAILED = "failed"


# How much of the sequence the window keeps in memory while the user browses it. A frame is only
# dropped when it came from a file; PYALDVC_CACHE_FRAMES overrides the count.
GUI_CACHE_BYTES = 8 * 1024**3
CACHE_MIN_FRAMES = 2  # the reference and the frame on screen
CACHE_MAX_FRAMES = 4


@dataclass(eq=False)  # identity equality: a field-wise one would compare the volumes themselves
class VolumeEntry:
    """One frame of the sequence: a file (loaded on demand) or an in-memory array.

    A file-backed entry caches what it read, and :meth:`AppState.release_frames` drops that cache
    when too many frames are resident. An array-backed entry is the only copy there is and is never
    released.
    """

    path: str | None = None
    array: NDArray | None = None
    mask_path: str | None = None
    mask: NDArray[np.bool_] | None = None
    label: str = ""
    mask_ops: dict | None = None  # drawing operations that produced ``mask`` (undo history; legacy sessions)
    uid: str = field(default_factory=lambda: uuid.uuid4().hex)  # identity, so results stay attached to their volumes
    header_shape: tuple[int, int, int] | None = None  # the file's size, read from its header when it was added
    shape_pending: bool = False  # the header is being read, off the UI thread

    @property
    def shape(self) -> tuple[int, int, int] | None:
        """The frame's size ``(nz, ny, nx)``: the array's when it is in memory, else the file header's."""
        if self.array is not None:
            return tuple(int(s) for s in self.array.shape)  # type: ignore[return-value]
        return self.header_shape

    def load(self) -> NDArray:
        if self.array is None:
            if not self.path:
                raise ValueError("volume entry has neither a file nor an array")
            from al_dvc.io.volume_io import load_volume

            arr = load_volume(self.path)  # bind locally first: a release must not blank a live read
            self.array = arr
            self.header_shape = tuple(int(s) for s in arr.shape)  # type: ignore[assignment]  # known after a release
            return arr
        return self.array

    @property
    def resident(self) -> bool:
        """True when the frame's voxels are in memory."""
        return self.array is not None

    @property
    def releasable(self) -> bool:
        """A file-backed frame can be read again; an array-backed one cannot."""
        return bool(self.path)

    def release(self) -> bool:
        """Drop the cached volume. Whoever is still using it keeps it alive; returns True if dropped."""
        if not self.releasable or self.array is None:
            return False
        self.array = None
        return True

    def load_mask(self) -> NDArray[np.bool_] | None:
        if self.mask is None and self.mask_path:
            from al_dvc.io.volume_io import load_volume

            self.mask = np.asarray(load_volume(self.mask_path)) > 0
        return self.mask

    @property
    def name(self) -> str:
        return self.label or (Path(self.path).name if self.path else "array")


def write_mask_file(path: str | Path, mask) -> Path:
    """Write ``mask`` as a uint8 volume (1 = material). A contiguous boolean array is written through a
    view, so nothing volume-sized is allocated for the conversion."""
    from al_dvc.io.volume_io import save_volume

    out = Path(path)
    m = np.asarray(mask)
    data = m.view(np.uint8) if m.dtype == np.bool_ and m.flags.c_contiguous else m.astype(np.uint8)
    save_volume(out, data)
    return out


class SaveMaskWorker(QThread):
    """``write_mask_file`` on a worker thread. The mask it holds is the one written, whatever is drawn meanwhile."""

    finished_path = Signal(object)  # the Path written
    failed = Signal(str)

    def __init__(self, path: Path, mask, revision: int, parent=None) -> None:
        super().__init__(parent)
        self.path = Path(path)
        self.mask = mask
        self.revision = int(revision)

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            out = write_mask_file(self.path, self.mask)
        except Exception as exc:  # surface to the UI
            logger.exception("Saving the mask failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished_path.emit(out)


class AutoMaskWorker(QThread):
    """``threshold_region`` on a worker thread, with stage progress and a cancel that lands at the next stage."""

    progress = Signal(float, str)
    finished_region = Signal(object)  # the boolean volume
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, volume, op: MaskOp, frame: int, parent=None) -> None:
        super().__init__(parent)
        self._volume = volume
        self.op = op
        self.frame = int(frame)
        self._stop = False

    def cancel(self) -> None:
        self._stop = True

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            region = threshold_region(
                self._volume,
                self.op.level,
                self.op.keep_largest,
                self.op.fill_holes,
                progress=self.progress.emit,
                stop=lambda: self._stop,
            )
        except ThresholdCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:  # surface to the UI
            logger.exception("Automatic mask failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        if self._stop:  # a cancel during the last stage is honoured, not published
            self.cancelled.emit()
            return
        self.finished_region.emit(region)


class AppState(QObject):
    """Observable state of the application."""

    volumes_changed = Signal()
    current_frame_changed = Signal(int)
    params_changed = Signal()
    run_state_changed = Signal(object)  # RunState
    progress_updated = Signal(float, str)
    results_changed = Signal()
    display_changed = Signal()
    log_message = Signal(str, str)  # (message, level)
    output_dir_changed = Signal(str)
    mask_changed = Signal()
    auto_mask_state = Signal(str)  # "started" | "finished" | "cancelled" | "failed"
    save_mask_state = Signal(str, str)  # ("started" | "finished" | "failed", path)
    shapes_changed = Signal()  # a volume's size became known
    shape_check_finished = Signal(object)  # the uids of a group of volumes whose sizes have all been read
    regions_changed = Signal()  # the post-processing regions were added, edited or removed
    correction_changed = Signal()  # the motion removed from what the views draw and the field exports write
    analysis_restored = Signal()  # a session replaced the regions, the correction and the statistics settings
    _shape_read = Signal(int, str, object, float)  # (batch, uid, shape or None, seconds), from the reading thread
    _shape_batch_done = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.volumes: list[VolumeEntry] = []
        self.current_frame: int = 0
        self.para: DVCPara = dvcpara_default()
        self.run_state: RunState = RunState.IDLE
        self.progress: float = 0.0
        self.progress_message: str = ""
        self.results: PipelineResult | None = None
        self.result_uids: list[str] = []  # ``uid`` of the volumes the results describe (frame k <-> result k-1)
        self.results_path: str | None = None  # the last exported results archive (session pointer)
        self.session_generation: int = 0  # bumped whenever the session context is replaced or the sequence changes
        # bumped whenever the *effective* mask may have changed: a drawing, a different current frame,
        # a different target, a different sequence. Caches of anything derived from the mask key on it.
        self.mask_revision: int = 0
        self._voi_cache: tuple | None = None
        self.dirty: bool = False  # unsaved edits (volumes, masks, parameters, output folder)
        self.output_dir: Path = Path("aldvc_results")
        self.session_path: Path | None = None
        self.write_checkpoints: bool = False  # advanced option; results live in memory and are exported afterwards
        # display
        self.display_field: str = "disp_magnitude"
        self.colormap: str = "turbo"
        self.color_auto: bool = True
        self.color_min: float = 0.0
        self.color_max: float = 1.0
        self.overlay_alpha: float = 0.75
        self.show_overlay: bool = True
        # Which volume is drawn under the field. None follows the selected frame -- the frame the
        # field belongs to -- and an index pins it; 0 (the reference) is the pairing that matches
        # where the field is actually drawn, because the node grid never leaves the reference
        # configuration. See docs/field_configuration.md.
        self.background_frame: int | None = None
        self.slice_index: dict[str, int | None] = {"z": None, "y": None, "x": None}
        self.slice_layout: str = "grid"  # arrangement of the three slices: "grid" (XY / XZ left, YZ top-right), "row", "column"
        self.slice_equal_scale: bool = False  # same voxels-per-pixel scale on the three planes
        self.show_mesh: bool = True  # the node grid on the slices (planned before a run, the result mesh after)
        self.show_subset_window: bool = False  # the subset of the crosshair node and of the node under the pointer
        # mask drawing
        self.mask_editor: MaskEditor | None = None
        self.mask_target: str = "current"  # "current" | "all"
        self.show_mask: bool = True
        self.mask_alpha: float = 0.35
        # post-processing: regions (reference configuration, voxels), the correction the views and the field
        # exports apply (None: the result as measured) and the statistics controls, all kept with the session.
        # The stored result never changes: a correction is a view of it (al_dvc.analysis.corrected).
        self.regions: list = []  # al_dvc.analysis.Region
        self.display_correction = None  # al_dvc.analysis.Correction | None
        self.analysis_settings: dict = {}
        self._display_cache: tuple | None = None
        # volume sizes, read from the file headers by background threads
        self._shape_batches: dict[int, list[str]] = {}
        self._shape_batch_seq = 0
        self._shape_read.connect(self._on_shape_read, Qt.ConnectionType.QueuedConnection)
        self._shape_batch_done.connect(self._on_shape_batch_done, Qt.ConnectionType.QueuedConnection)

    # ------------------------------------------------------------------ volumes
    @property
    def busy(self) -> bool:
        """A run is active: the sequence, the session and the parameters it captured must not change."""
        return self.run_state in (RunState.RUNNING, RunState.STOPPING)

    def _locked(self, what: str) -> bool:
        if self.busy:
            self.log(f"{what}: the volume list is locked while a run is active (stop it first)", "warning")
            return True
        return False

    def _sequence_changed(self) -> None:
        """The sequence is different: a run started before this must not publish into it."""
        self.session_generation += 1
        self.mask_revision += 1  # which frame is the reference may have changed with the list
        self.dirty = True

    def mark_clean(self) -> None:
        self.dirty = False

    def add_volume_paths(self, paths: list[str]) -> None:
        if not paths or self._locked("add volumes"):
            return
        added = [VolumeEntry(path=str(p)) for p in paths]
        self.volumes.extend(added)
        self._sequence_changed()
        self.volumes_changed.emit()
        if self.results is not None:
            self.results_changed.emit()  # the new rows have no result; the views re-map
        self.start_shape_check(added)

    def set_volume_arrays(self, arrays: list[NDArray], labels: list[str] | None = None) -> None:
        self.volumes = [
            VolumeEntry(array=np.asarray(a), label=(labels[i] if labels else f"frame {i}")) for i, a in enumerate(arrays)
        ]
        self.current_frame = 0
        self.results = None
        self.result_uids = []
        self.mask_editor = None
        self._sequence_changed()
        self.volumes_changed.emit()
        self.results_changed.emit()
        self.mask_changed.emit()
        self.start_shape_check(self.volumes)  # arrays are sized already: this only reports a mismatch

    def remove_volume(self, index: int) -> None:
        if not (0 <= index < len(self.volumes)) or self._locked("remove volume"):
            return
        del self.volumes[index]
        self.current_frame = min(self.current_frame, max(0, len(self.volumes) - 1))
        self.mask_editor = None
        self._sequence_changed()
        self.volumes_changed.emit()
        self.current_frame_changed.emit(self.current_frame)
        self.mask_changed.emit()
        if self.results is not None:
            self.results_changed.emit()  # the results keep their identity; the removed row's field is gone

    def move_volume(self, index: int, new_index: int) -> None:
        """Reorder the sequence (frame 0 stays the reference of the run); the moved frame stays current."""
        n = len(self.volumes)
        if not (0 <= index < n) or not (0 <= new_index < n) or index == new_index or self._locked("reorder volumes"):
            return
        entry = self.volumes.pop(index)
        self.volumes.insert(new_index, entry)
        self.current_frame = new_index
        self.mask_editor = None
        self._sequence_changed()
        self.volumes_changed.emit()
        self.current_frame_changed.emit(new_index)
        self.mask_changed.emit()
        if self.results is not None:
            self.results_changed.emit()  # every row keeps the result it was computed for

    def sort_volumes(self, natural: bool = True) -> None:
        """Reorder the sequence by file name, numbers compared numerically (``natural``) or character by
        character; the selected volume stays selected."""
        if len(self.volumes) < 2 or self._locked("sort volumes"):
            return
        key = natural_key if natural else lexical_key
        order = sorted(range(len(self.volumes)), key=lambda i: key(self.volumes[i].name))
        if order == list(range(len(self.volumes))):
            return
        current = self.volumes[self.current_frame] if self.current_frame < len(self.volumes) else None
        self.volumes = [self.volumes[i] for i in order]
        self.current_frame = self.volumes.index(current) if current is not None else 0
        self.mask_editor = None
        self._sequence_changed()
        self.volumes_changed.emit()
        self.current_frame_changed.emit(self.current_frame)
        self.mask_changed.emit()
        if self.results is not None:
            self.results_changed.emit()

    def clear_volumes(self) -> None:
        if self._locked("clear volumes"):
            return
        self.volumes = []
        self.current_frame = 0
        self.mask_editor = None
        self._sequence_changed()
        self.volumes_changed.emit()
        self.mask_changed.emit()
        if self.results is not None:
            self.results_changed.emit()

    def set_mask(self, index: int, path: str | None = None, mask: NDArray[np.bool_] | None = None) -> bool:
        """Attach a mask file or array to frame ``index``. Returns False, attaching nothing and logging why,
        when a mask file's size is not the volume's (an array is checked when a run starts)."""
        entry = self.volumes[index]
        problem = self.mask_size_problem(index, path) if path is not None else ""
        if problem:
            self.log(problem, "error")
            return False
        entry.mask_path = path
        entry.mask = None if mask is None else np.asarray(mask, dtype=bool)
        entry.mask_ops = None
        if index == self.current_frame:
            self.mask_editor = None
        self._mask_changed()
        self.volumes_changed.emit()
        self.mask_changed.emit()
        return True

    # ------------------------------------------------------------------ volume sizes
    def reference_shape(self) -> tuple[int, int, int] | None:
        """The size of the reference volume (frame 0), ``None`` while it is not known."""
        return self.volumes[0].shape if self.volumes else None

    def shape_mismatches(self) -> list[tuple[int, tuple[int, int, int]]]:
        """``(index, shape)`` of every frame whose size is known and is not the reference's.

        A run needs every frame at the reference's size. Empty while the reference's size is not known.
        """
        ref = self.reference_shape()
        if ref is None:
            return []
        out = []
        for i, entry in enumerate(self.volumes[1:], start=1):
            shape = entry.shape
            if shape is not None and shape != ref:
                out.append((i, shape))
        return out

    def mismatch_fields(self, mismatches: list[tuple[int, tuple[int, int, int]]], sep: str = "; ") -> dict:
        """The ``n``, ``name``, ``size`` and ``files`` of a message that reports ``mismatches`` (``files`` joined by ``sep``)."""
        ref = self.volumes[0]
        return {
            "n": len(mismatches),
            "name": ref.name,
            "size": size_text(ref.shape),
            "files": sep.join(f"{self.volumes[i].name}: {size_text(shape)}" for i, shape in mismatches),
        }

    def mask_size_problem(self, index: int, path: str | os.PathLike) -> str:
        """Why the mask file ``path`` cannot be the mask of frame ``index``; "" when it can (or a size is not known)."""
        from al_dvc.io import volume_io

        entry = self.volumes[index]
        want = entry.shape
        if want is None:
            return ""
        try:
            got = volume_io.read_volume_shape(path)
        except OSError:  # a missing file is reported when the mask is read
            return ""
        if got is None or got == want:
            return ""
        return self.tr("The mask {name} is {size} voxels but the volume {volume} is {ref}; the mask was not attached.").format(
            name=Path(path).name, size=size_text(got), volume=entry.name, ref=size_text(want)
        )

    def start_shape_check(self, entries: list[VolumeEntry] | None = None) -> int:
        """Read the size of every file among ``entries`` (default: all) from its header, on a background thread.

        The header is a few kilobytes, but a cloud drive (Box Drive, OneDrive) downloads a file it has not synced
        yet on the first read of any byte -- 26-45 s per 3.7-GB scan on Box Drive -- so the read never runs on the
        UI thread. The thread is a daemon: it only reads, and a download still in progress must not keep the
        application from closing. Each size arrives through :attr:`shapes_changed`; :attr:`shape_check_finished`
        carries the uids of ``entries`` once all of theirs are in (at once when every size is known already).
        Returns the id of the batch.
        """
        entries = list(self.volumes if entries is None else entries)
        self._shape_batch_seq += 1
        batch = self._shape_batch_seq
        uids = [e.uid for e in entries]
        todo = [(e.uid, str(e.path)) for e in entries if e.path and e.shape is None]
        if not todo:
            self.shape_check_finished.emit(uids)
            return batch
        for e in entries:
            if e.path and e.shape is None:
                e.shape_pending = True
        self.shapes_changed.emit()  # the rows say the sizes are being read
        self._shape_batches[batch] = uids
        threading.Thread(target=self._read_shapes, args=(batch, todo), name=f"pyaldvc-sizes-{batch}", daemon=True).start()
        return batch

    def _read_shapes(self, batch: int, todo: list[tuple[str, str]]) -> None:
        """Thread body: one header after another; the results are queued to the UI thread. The batch always
        ends, so no row is left waiting whatever happens here."""
        from al_dvc.io import volume_io

        try:
            for uid, path in todo:
                t0 = time.perf_counter()
                try:
                    shape = volume_io.read_volume_shape(path)
                except Exception as exc:  # a missing or unreadable file: the size stays unknown, the load says why
                    logger.debug("size of %s not read: %s", path, exc)
                    shape = None
                self._shape_read.emit(batch, uid, shape, time.perf_counter() - t0)
        except RuntimeError:  # the window, and this state with it, went away during a download
            return
        except Exception:
            logger.exception("reading the volume sizes failed")
        with contextlib.suppress(RuntimeError):
            self._shape_batch_done.emit(batch)

    def _on_shape_read(self, _batch: int, uid: str, shape, seconds: float) -> None:
        from al_dvc.io import volume_io

        entry = next((e for e in self.volumes if e.uid == uid), None)
        if entry is None:
            return  # removed while its size was being read
        entry.shape_pending = False
        if shape is not None:
            entry.header_shape = tuple(int(s) for s in shape)  # type: ignore[assignment]
        if seconds >= volume_io.SLOW_HEADER_READ_S:  # the "..." stayed that long: say why
            self.log(
                self.tr(
                    "Reading the size of {name} took {s:.0f} s: a cloud or network drive probably downloaded the whole file."
                ).format(name=entry.name, s=seconds)
            )
        self.shapes_changed.emit()

    def _on_shape_batch_done(self, batch: int) -> None:
        uids = self._shape_batches.pop(batch, None)
        if uids is None:
            return
        for entry in self.volumes:
            if entry.uid in uids:
                entry.shape_pending = False
        self.shape_check_finished.emit(list(uids))

    def shape_check_pending(self) -> bool:
        return bool(self._shape_batches)

    def wait_for_shapes(self, timeout_ms: int = 10_000) -> bool:
        """Process events until every size being read has been reported (tests, scripts); False on a timeout."""
        deadline = time.monotonic() + timeout_ms / 1000.0
        while self._shape_batches:
            if time.monotonic() > deadline:
                return False
            QCoreApplication.processEvents()
            QThread.msleep(5)
        QCoreApplication.processEvents()
        return True

    def _mask_changed(self) -> None:
        self.mask_revision += 1
        self.dirty = True

    # ------------------------------------------------------------------ mask drawing
    def current_mask(self) -> NDArray[np.bool_] | None:
        """The mask shown for the current frame: the live editor's, else the frame's own."""
        if self.mask_editor is not None:
            return self.mask_editor.mask
        if not self.volumes or self.current_frame >= len(self.volumes):
            return None
        try:
            return self.volumes[self.current_frame].load_mask()
        except Exception as exc:
            self.log(f"cannot load the mask of frame {self.current_frame}: {exc}", "error")
            return None

    def reference_mask(self) -> NDArray[np.bool_] | None:
        """The mask of the reference frame (frame 0): the live editor's when it applies to frame 0, else its own."""
        if not self.volumes:
            return None
        if self.mask_editor is not None and (self.current_frame == 0 or self.mask_target == "all"):
            return self.mask_editor.mask
        try:
            return self.volumes[0].load_mask()
        except Exception as exc:
            self.log(f"cannot load the mask of the reference frame: {exc}", "error")
            return None

    def effective_voi(self) -> VOIRange | None:
        """The analysed box: ``para.voi`` when set, else the region of interest's bounding box grown by the
        subset half-width and the search range (``None`` = whole volume).

        Cached on ``mask_revision``: the slice viewer asks for it on every slider tick and the answer
        costs four passes over the mask volume.
        """
        if self.para.voi is not None and not self.para.voi.is_whole:
            return self.para.voi
        key = (self.mask_revision, tuple(self.para.winsize), tuple(np.atleast_1d(self.para.search_radius).tolist()))
        if self._voi_cache is not None and self._voi_cache[0] == key:
            return self._voi_cache[1]
        mask = self.reference_mask()
        voi = None if mask is None else voi_from_mask(mask, self.para.winsize, self.para.search_radius)
        self._voi_cache = (key, voi)
        return voi

    def ensure_mask_editor(self, base: str = "current") -> MaskEditor:
        """The editor for the current frame, created on first use.

        ``base``: ``current`` starts from the frame's mask when it has one (else empty),
        ``empty`` from all False, ``full`` from all True.
        """
        shape = self.volume_shape()
        if shape is None:
            raise ValueError("load a volume before drawing a mask")
        if self.mask_editor is not None and tuple(self.mask_editor.shape) == tuple(shape):
            return self.mask_editor
        base_mask = None
        if base == "current":
            base_mask = self.current_mask()
            if base_mask is not None and base_mask.shape != tuple(shape):
                self.log(f"mask shape {base_mask.shape} differs from the volume {tuple(shape)}; starting empty", "warning")
                base_mask = None
        elif base == "full":
            base_mask = FULL_BASE  # symbolic: the editor never materialises a second volume
        volume = None
        try:
            volume = self.volume_array(self.current_frame)
        except Exception as exc:  # the editor still works for geometric shapes
            self.log(f"mask editor: cannot load the frame for threshold operations: {exc}", "warning")
        self.mask_editor = MaskEditor(shape, base=base_mask, volume=volume)
        return self.mask_editor

    def _target_frames(self) -> list[int]:
        return list(range(len(self.volumes))) if self.mask_target == "all" else [self.current_frame]

    def _push_mask(self) -> None:
        """The editor's mask onto the target frames: the current frame shares the editor's array, the other
        frames (target ``all``) get their own copy so a later edit cannot change them behind the user's back."""
        ed = self.mask_editor
        if ed is None:
            return
        ops = ed.to_dict()
        for i in self._target_frames():
            if i < len(self.volumes):
                self.volumes[i].mask = ed.mask if i == self.current_frame else ed.mask.copy()
                self.volumes[i].mask_ops = ops
        self._mask_copy_backup = None  # a new drawing supersedes an undoable copy
        self._mask_changed()
        self.mask_changed.emit()

    def copy_mask_to_all_frames(self) -> bool:
        """Give every frame a copy of the current frame's mask (explicit, reversible with :meth:`undo_mask`)."""
        mask = self.current_mask()
        if mask is None or len(self.volumes) < 2:
            return False
        ops = self.mask_editor.to_dict() if self.mask_editor is not None else self.volumes[self.current_frame].mask_ops
        self._mask_copy_backup = [(v.mask_path, v.mask, v.mask_ops) for v in self.volumes]
        for i, entry in enumerate(self.volumes):
            if i == self.current_frame:
                continue
            entry.mask_path = None
            entry.mask = mask.copy()
            entry.mask_ops = ops
        self._mask_changed()
        self.log(f"mask of frame {self.current_frame} copied to the other {len(self.volumes) - 1} frame(s)")
        self.volumes_changed.emit()
        self.mask_changed.emit()
        return True

    def _undo_mask_copy(self) -> bool:
        backup = getattr(self, "_mask_copy_backup", None)
        if not backup or len(backup) != len(self.volumes):
            self._mask_copy_backup = None
            return False
        for entry, (path, mask, ops) in zip(self.volumes, backup):
            entry.mask_path, entry.mask, entry.mask_ops = path, mask, ops
        self._mask_copy_backup = None
        self._mask_changed()
        self.log("mask copy undone: every frame has its previous mask again")
        self.volumes_changed.emit()
        self.mask_changed.emit()
        return True

    def apply_mask_op(self, op: MaskOp) -> None:
        self.ensure_mask_editor().apply(op)
        self._push_mask()

    # ------------------------------------------------------------------ the automatic mask, off the UI thread
    def auto_mask_running(self) -> bool:
        """A job is owned, from its start to its terminal signal (which is what the buttons need to know)."""
        return getattr(self, "_auto_mask", None) is not None

    def auto_mask_thread_running(self) -> bool:
        """The thread itself is alive -- what a shutdown has to wait for, not what the buttons show."""
        w = getattr(self, "_auto_mask", None)
        return w is not None and w.isRunning()

    def start_auto_mask(self, op: MaskOp | None = None):
        """Compute the threshold mask of the current frame on a worker thread; returns the worker, or None.

        The threshold used to run on the UI thread: past a few hundred voxels an edge the window froze
        for tens of seconds and nothing could be cancelled. The worker computes the region and hands it
        to the editor as an ordinary operation when it is done; if the frame was changed meanwhile the
        result is discarded rather than applied to the wrong frame.
        """
        if not self.volumes or self.auto_mask_running():
            return None
        op = op if op is not None else MaskOp("threshold", mode="replace")
        if op.shape != "threshold":
            raise ValueError(f"the automatic mask is a threshold operation, got {op.shape!r}")
        frame = int(self.current_frame)
        worker = AutoMaskWorker(self.volume_array(frame), op, frame, parent=self)
        worker.finished_region.connect(lambda region, w=worker: self._auto_mask_done(w, region))
        worker.failed.connect(lambda message, w=worker: self._auto_mask_ended(w, "failed", message))
        worker.cancelled.connect(lambda w=worker: self._auto_mask_ended(w, "cancelled"))
        worker.finished.connect(worker.deleteLater)  # the Qt object goes once the thread has really ended
        self._auto_mask = worker
        worker.start()
        self.auto_mask_state.emit("started")  # after start(), so a listener that asks the thread sees it alive
        return worker

    def cancel_auto_mask(self) -> None:
        if self.auto_mask_running():
            self._auto_mask.cancel()

    def _auto_mask_done(self, worker, region) -> None:
        if worker is not getattr(self, "_auto_mask", None):
            return  # a job that was replaced; its result is nobody's
        shape = self.volume_shape()
        if worker.frame != self.current_frame or shape is None or tuple(region.shape) != tuple(shape):
            self.log("Automatic mask discarded: the frame changed while it was being computed", "warning")
            self._auto_mask_ended(worker, "cancelled")
            return
        try:
            self.ensure_mask_editor().apply_computed(worker.op, region)
            self._push_mask()
        except Exception as exc:
            self._auto_mask_ended(worker, "failed", f"{type(exc).__name__}: {exc}")
            return
        self._auto_mask_ended(worker, "finished")

    def _auto_mask_ended(self, worker, state: str, message: str = "") -> None:
        if worker is getattr(self, "_auto_mask", None):
            self._auto_mask = None
        if message:
            self.log(f"Automatic mask failed: {message}", "error")
        self.auto_mask_state.emit(state)

    def undo_mask(self) -> bool:
        if getattr(self, "_mask_copy_backup", None):
            return self._undo_mask_copy()
        ok = self.mask_editor is not None and self.mask_editor.undo()
        if ok:
            self._push_mask()
        return ok

    def redo_mask(self) -> bool:
        ok = self.mask_editor is not None and self.mask_editor.redo()
        if ok:
            self._push_mask()
        return ok

    def reset_mask(self, base: str = "empty") -> None:
        """Start the drawing again from ``empty`` or ``full``."""
        self.mask_editor = None
        self.ensure_mask_editor(base)
        self._push_mask()

    def remove_mask(self, index: int | None = None) -> None:
        """Drop the mask (file, array, drawing) of one frame or of the current frame."""
        i = self.current_frame if index is None else index
        if 0 <= i < len(self.volumes):
            entry = self.volumes[i]
            entry.mask_path = None
            entry.mask = None
            entry.mask_ops = None
        if i == self.current_frame:
            self.mask_editor = None
        self._mask_changed()
        self.volumes_changed.emit()
        self.mask_changed.emit()

    def save_mask(self, path: str | Path) -> Path:
        """Write the current mask as a volume file (uint8, 1 = material) and attach the file to the target frames.

        Synchronous: the CLI, sessions and tests use it. The window uses :meth:`start_save_mask`.
        """
        mask = self.current_mask()
        if mask is None:
            raise ValueError("there is no mask to save")
        out = write_mask_file(path, mask)
        self._attach_saved_mask(out, mask, self.mask_revision)
        return out

    def _attach_saved_mask(self, out: Path, mask, revision: int) -> None:
        """Bookkeeping after a mask file was written: the file becomes the frames' mask.

        ``revision`` is the mask revision the written array came from. If the user drew while the file
        was being written, the drawing is kept and the file simply holds the mask as it was -- the
        editor is not reset over the newer drawing, and the log says so.
        """
        for i in self._target_frames():
            if i < len(self.volumes):
                self.volumes[i].mask_path = str(out)
                self.volumes[i].mask = mask if i == self.current_frame else mask.copy()
                self.volumes[i].mask_ops = None  # the file is the composed mask: nothing to replay on top of it
        if self.mask_editor is not None:
            if revision == self.mask_revision:
                self.mask_editor.reset(mask)  # drawing continues from the saved file; the undo history restarts
            else:
                self.log("The mask file holds the mask as it was when saving started; the edits since are not in it", "warning")
                self.volumes[self.current_frame].mask = self.mask_editor.mask  # the frame keeps the newer drawing
        self.dirty = True
        self.volumes_changed.emit()

    # ------------------------------------------------------------------ saving the mask, off the UI thread
    def save_mask_running(self) -> bool:
        return getattr(self, "_save_mask", None) is not None

    def save_mask_thread_running(self) -> bool:
        w = getattr(self, "_save_mask", None)
        return w is not None and w.isRunning()

    def start_save_mask(self, path: str | Path):
        """Write the current mask on a worker thread; returns the worker, or None when nothing can be saved.

        The write used to run on the UI thread with no feedback: a mask of a large scan froze the window
        for the seconds the file took. The worker writes a snapshot of the mask as it is now; drawing
        stays possible meanwhile, and a drawing made during the write is kept rather than lost to the
        history reset that follows a save. There is no cancel: a half-written file would be worse than
        a wait.
        """
        if self.save_mask_running():
            return None
        mask = self.current_mask()
        if mask is None:
            raise ValueError("there is no mask to save")
        if self.mask_editor is not None and self.mask_editor.mask is mask:
            mask = self.mask_editor.snapshot()  # the editor copies on its next edit; the worker's array stays
        worker = SaveMaskWorker(Path(path), mask, int(self.mask_revision), parent=self)
        worker.finished_path.connect(lambda out, w=worker: self._save_mask_done(w, out))
        worker.failed.connect(lambda message, w=worker: self._save_mask_ended(w, "failed", message))
        worker.finished.connect(worker.deleteLater)
        self._save_mask = worker
        worker.start()
        self.save_mask_state.emit("started", str(path))
        return worker

    def _save_mask_done(self, worker, out) -> None:
        if worker is not getattr(self, "_save_mask", None):
            return
        try:
            self._attach_saved_mask(Path(out), np.asarray(worker.mask, dtype=bool), worker.revision)
        except Exception as exc:
            self._save_mask_ended(worker, "failed", f"{type(exc).__name__}: {exc}")
            return
        self._save_mask_ended(worker, "finished")

    def _save_mask_ended(self, worker, state: str, message: str = "") -> None:
        if worker is getattr(self, "_save_mask", None):
            self._save_mask = None
        if message:
            self.log(f"saving the mask failed: {message}", "error")
        self.save_mask_state.emit(state, str(worker.path))

    def set_mask_display(self, show: bool | None = None, alpha: float | None = None, target: str | None = None) -> None:
        if show is not None:
            self.show_mask = bool(show)
        if alpha is not None:
            self.mask_alpha = float(alpha)
        if target is not None:
            if target not in ("current", "all"):
                raise ValueError(f"mask target must be 'current' or 'all', got {target!r}")
            self.mask_target = target  # where the next drawing operations go; copying now is an explicit action
            self.mask_revision += 1  # the target decides whether the live editor is the reference's mask
        self.mask_changed.emit()

    def cache_limit(self) -> int:
        """How many frames may stay resident: a byte budget, floored at the reference plus the current one."""
        override = os.environ.get("PYALDVC_CACHE_FRAMES")
        if override:
            try:
                return max(CACHE_MIN_FRAMES, int(override))
            except ValueError:
                self.log(f"PYALDVC_CACHE_FRAMES={override!r} is not a number; using the default", "warning")
        resident = [v.array for v in self.volumes if v.array is not None]
        if not resident:
            return CACHE_MAX_FRAMES
        nbytes = max(int(a.nbytes) for a in resident)
        return int(min(CACHE_MAX_FRAMES, max(CACHE_MIN_FRAMES, GUI_CACHE_BYTES // max(nbytes, 1))))

    def release_frames(self, keep: str | None = None) -> int:
        """Drop cached volumes until at most :meth:`cache_limit` frames are resident; returns how many.

        The reference (frame 0), the frame on screen and ``keep`` -- the one just served -- are never
        dropped, and nothing is dropped while a run is going: the worker reads through the entries.
        Releasing only drops this cache's reference; anything still using the array keeps it alive.
        """
        if self.busy:
            return 0
        protected = {self.volumes[0].uid if self.volumes else None, keep}
        if 0 <= self.current_frame < len(self.volumes):
            protected.add(self.volumes[self.current_frame].uid)
        limit = self.cache_limit()
        resident = [(i, v) for i, v in enumerate(self.volumes) if v.resident]
        droppable = [(i, v) for i, v in resident if v.releasable and v.uid not in protected]
        n_drop = len(resident) - limit
        if n_drop <= 0 or not droppable:
            return 0
        droppable.sort(key=lambda iv: -abs(iv[0] - self.current_frame))  # furthest from what is on screen first
        dropped = 0
        for _i, entry in droppable[:n_drop]:
            dropped += int(entry.release())
        if dropped:
            self.log(f"released {dropped} cached frame(s); {limit} kept in memory", "debug")
        return dropped

    def volume_array(self, index: int) -> NDArray:
        entry = self.volumes[index]
        arr = entry.load()
        self.release_frames(keep=entry.uid)
        return arr

    def volume_shape(self) -> tuple[int, int, int] | None:
        if not self.volumes:
            return None
        try:
            return tuple(int(s) for s in self.volume_array(0).shape)  # type: ignore[return-value]
        except Exception as exc:  # unreadable file: report, do not crash
            self.log(f"cannot read {self.volumes[0].name}: {exc}", "error")
            return None

    @property
    def display_frame(self) -> int:
        """Result frame of the selected volume (0 when it has none): frame k is the deformed volume k against the
        reference. Selecting a volume in the list selects its result (like pyALDIC); :meth:`result_frame` says
        whether the selected volume has one at all."""
        frame = self.result_frame()
        return 0 if frame is None else frame

    def result_frame(self) -> int | None:
        """Index into ``results.result_disp`` for the selected volume, ``None`` when it has no computed result
        (the reference, a volume added or reordered after the run, a frame the run never reached)."""
        res = self.results
        if res is None or not res.result_disp or not self.volumes or self.current_frame >= len(self.volumes):
            return None
        uid = self.volumes[self.current_frame].uid
        if self.result_uids:
            if uid not in self.result_uids:
                return None
            k = self.result_uids.index(uid)
        else:
            k = self.current_frame
        if k < 1 or k - 1 >= len(res.result_disp):
            return None
        return k - 1

    def volume_for_result(self, frame: int) -> int | None:
        """Index in ``volumes`` of the deformed volume that result frame ``frame`` describes (``None`` if gone);
        frame -1 is the reference state, i.e. the reference volume."""
        if frame < 0:
            return 0 if self.volumes else None
        if self.result_uids:
            if frame + 1 >= len(self.result_uids):
                return None
            uid = self.result_uids[frame + 1]
            for i, v in enumerate(self.volumes):
                if v.uid == uid:
                    return i
            return None
        return frame + 1 if frame + 1 < len(self.volumes) else None

    def set_current_frame(self, index: int) -> None:
        if self.volumes and 0 <= index < len(self.volumes) and index != self.current_frame:
            self.current_frame = index
            self.mask_editor = None
            self._mask_changed()  # a different frame can mean a different effective mask
            self.current_frame_changed.emit(index)
            self.mask_changed.emit()

    # ------------------------------------------------------------------ parameters
    def set_param(self, name: str, value: Any) -> None:
        """Replace one parameter; raises ``ValueError`` (validation) without changing the state."""
        new = replace(self.para, **{name: value})
        self.para = new
        self.dirty = True
        self.params_changed.emit()

    def set_params(self, **values: Any) -> None:
        self.para = replace(self.para, **values)
        self.dirty = True
        self.params_changed.emit()

    def set_para(self, para: DVCPara) -> None:
        self.para = para
        self.dirty = True
        self.params_changed.emit()

    # ------------------------------------------------------------------ run
    def set_run_state(self, state: RunState) -> None:
        self.run_state = state
        self.run_state_changed.emit(state)

    def set_progress(self, fraction: float, message: str = "") -> None:
        self.progress = float(fraction)
        self.progress_message = message
        self.progress_updated.emit(self.progress, message)

    def set_results(self, results: PipelineResult | None, uids: list[str] | None = None) -> None:
        """Publish results. ``uids`` names the volumes they were computed from (a run passes the ones it
        captured); without it a replacement (strain added) keeps the existing identity and a fresh result
        takes the current sequence."""
        self.results = results
        if uids is not None:
            self.result_uids = list(uids)
        elif results is None:
            self.result_uids = []
        elif not self.result_uids:
            self.result_uids = [v.uid for v in self.volumes]
        if results is not None and results.result_disp:
            self.show_mask = False  # a field is on the slices now; the region tint would only muddy it
            if self.result_frame() is None:
                first = self.volume_for_result(0)
                if first is not None:
                    self.set_current_frame(first)  # the first deformed volume carries the first result
        self.results_changed.emit()
        self.mask_changed.emit()

    def set_output_dir(self, path: str | Path) -> None:
        self.output_dir = Path(path)
        self.dirty = True
        self.output_dir_changed.emit(str(self.output_dir))

    # ------------------------------------------------------------------ post-processing
    def set_regions(self, regions) -> None:
        """Replace the post-processing regions (a list of :class:`al_dvc.analysis.Region`)."""
        self.regions = list(regions)
        self.dirty = True
        self.regions_changed.emit()

    def set_display_correction(self, correction) -> None:
        """Draw and export the fields with ``correction`` (an :class:`al_dvc.analysis.Correction`) applied;
        ``None`` or a correction that removes nothing shows them as measured."""
        if correction is not None and correction.motion == "none":
            correction = None
        if correction == self.display_correction:
            return
        self.display_correction = correction
        self._display_cache = None
        self.dirty = True
        self.correction_changed.emit()
        self.display_changed.emit()

    def display_result(self) -> PipelineResult | None:
        """What the views draw and the field exports (CSV, ParaView, images, report) write: ``results`` with
        :attr:`display_correction` applied, computed frame by frame on first use and kept while neither changes.
        The archives (npz, mat) always hold ``results`` as measured."""
        res, corr = self.results, self.display_correction
        if res is None or corr is None:
            return res
        cache = self._display_cache
        if cache is not None and cache[0] is res and cache[1] == corr:
            return cache[2]
        from al_dvc.analysis.corrected import corrected_result

        shown = corrected_result(res, corr)
        self._display_cache = (res, corr, shown)
        return shown

    def correction_text(self) -> str:
        """A short note for what the views show, "" when the fields are as measured."""
        corr = self.display_correction
        if corr is None or self.results is None:
            return ""
        what = {
            "translation": self.tr("translation removed"),
            "rigid": self.tr("rigid motion removed"),
            "affine": self.tr("affine motion removed"),
        }.get(corr.motion, corr.motion)
        if corr.fit_region is not None:
            what += " " + self.tr("(fitted over {region})").format(region=corr.fit_region.name)
        return what

    # ------------------------------------------------------------------ display
    def set_display(self, **values: Any) -> None:
        for key, val in values.items():
            if not hasattr(self, key):
                raise AttributeError(key)
            setattr(self, key, val)
        self.display_changed.emit()

    def background_index(self) -> int | None:
        """Index of the volume to draw under the field, ``None`` without a sequence.

        ``background_frame`` is honoured while it still points at a frame; a sequence that shrank
        falls back to the selected frame rather than to a frame that is no longer there.
        """
        if not self.volumes:
            return None
        pinned = self.background_frame
        if pinned is not None and 0 <= int(pinned) < len(self.volumes):
            return int(pinned)
        return min(self.current_frame, len(self.volumes) - 1)

    def set_slice(self, axis: str, index: int | None) -> None:
        self.slice_index[axis] = index
        self.display_changed.emit()

    # ------------------------------------------------------------------ misc
    def log(self, message: str, level: str = "info") -> None:
        getattr(logger, level if level in ("debug", "info", "warning", "error") else "info")(message)
        self.log_message.emit(message, level)

    def reset(self) -> None:
        self.volumes = []
        self.current_frame = 0
        self.para = dvcpara_default()
        self.results = None
        self.result_uids = []
        self.results_path = None
        self.session_path = None
        self.run_state = RunState.IDLE
        self.mask_editor = None
        self._mask_copy_backup = None
        self.regions = []
        self.display_correction = None
        self.analysis_settings = {}
        self._display_cache = None
        self.session_generation += 1
        self.mask_revision += 1
        self.dirty = False
        self.analysis_restored.emit()
        self.regions_changed.emit()
        self.correction_changed.emit()
        self.volumes_changed.emit()
        self.mask_changed.emit()
        self.params_changed.emit()
        self.results_changed.emit()
        self.run_state_changed.emit(self.run_state)
