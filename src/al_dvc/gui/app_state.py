"""Application state shared by every panel of the pyALDVC window.

One ``AppState`` instance holds the loaded volumes and masks, the current
parameter set, the run status, the results and the display settings, and
announces changes through Qt signals. Panels never talk to each other: they
read and write the state and react to its signals.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, Signal

from al_dvc.core.config import DVCPara, dvcpara_default
from al_dvc.core.data_structures import PipelineResult, VOIRange, voi_from_mask

from .mask_editor import MaskEditor, MaskOp

logger = logging.getLogger(__name__)


class RunState(enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    DONE = "done"
    FAILED = "failed"


@dataclass
class VolumeEntry:
    """One frame of the sequence: a file (loaded on demand) or an in-memory array."""

    path: str | None = None
    array: NDArray | None = None
    mask_path: str | None = None
    mask: NDArray[np.bool_] | None = None
    label: str = ""
    mask_ops: dict | None = None  # drawing operations that produced ``mask`` (session persistence)

    def load(self) -> NDArray:
        if self.array is None:
            if not self.path:
                raise ValueError("volume entry has neither a file nor an array")
            from al_dvc.io.volume_io import load_volume

            self.array = load_volume(self.path)
        return self.array

    def load_mask(self) -> NDArray[np.bool_] | None:
        if self.mask is None and self.mask_path:
            from al_dvc.io.volume_io import load_volume

            self.mask = np.asarray(load_volume(self.mask_path)) > 0
        return self.mask

    @property
    def name(self) -> str:
        return self.label or (Path(self.path).name if self.path else "array")


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

    def __init__(self) -> None:
        super().__init__()
        self.volumes: list[VolumeEntry] = []
        self.current_frame: int = 0
        self.para: DVCPara = dvcpara_default()
        self.run_state: RunState = RunState.IDLE
        self.progress: float = 0.0
        self.progress_message: str = ""
        self.results: PipelineResult | None = None
        self.output_dir: Path = Path("aldvc_results")
        self.session_path: Path | None = None
        self.write_checkpoints: bool = False  # advanced option; results live in memory and are exported afterwards
        # display
        self.display_field: str = "disp_magnitude"
        self.display_frame: int = 0
        self.colormap: str = "turbo"
        self.color_auto: bool = True
        self.color_min: float = 0.0
        self.color_max: float = 1.0
        self.overlay_alpha: float = 0.75
        self.show_overlay: bool = True
        self.slice_index: dict[str, int | None] = {"z": None, "y": None, "x": None}
        self.slice_layout: str = "row"  # arrangement of the three slices ("row", "column", "grid")
        self.slice_equal_scale: bool = False  # same voxels-per-pixel scale on the three planes
        # mask drawing
        self.mask_editor: MaskEditor | None = None
        self.mask_target: str = "current"  # "current" | "all"
        self.show_mask: bool = True
        self.mask_alpha: float = 0.35

    # ------------------------------------------------------------------ volumes
    def add_volume_paths(self, paths: list[str]) -> None:
        for p in paths:
            self.volumes.append(VolumeEntry(path=str(p)))
        self.volumes_changed.emit()

    def set_volume_arrays(self, arrays: list[NDArray], labels: list[str] | None = None) -> None:
        self.volumes = [
            VolumeEntry(array=np.asarray(a), label=(labels[i] if labels else f"frame {i}")) for i, a in enumerate(arrays)
        ]
        self.current_frame = 0
        self.results = None
        self.mask_editor = None
        self.volumes_changed.emit()
        self.results_changed.emit()
        self.mask_changed.emit()

    def remove_volume(self, index: int) -> None:
        if 0 <= index < len(self.volumes):
            del self.volumes[index]
            self.current_frame = min(self.current_frame, max(0, len(self.volumes) - 1))
            self.volumes_changed.emit()

    def move_volume(self, index: int, new_index: int) -> None:
        """Reorder the sequence (frame 0 stays the reference of the run); the moved frame stays current."""
        n = len(self.volumes)
        if not (0 <= index < n) or not (0 <= new_index < n) or index == new_index:
            return
        entry = self.volumes.pop(index)
        self.volumes.insert(new_index, entry)
        self.current_frame = new_index
        self.mask_editor = None
        self.volumes_changed.emit()
        self.current_frame_changed.emit(new_index)
        self.mask_changed.emit()

    def clear_volumes(self) -> None:
        self.volumes = []
        self.current_frame = 0
        self.mask_editor = None
        self.volumes_changed.emit()
        self.mask_changed.emit()

    def set_mask(self, index: int, path: str | None = None, mask: NDArray[np.bool_] | None = None) -> None:
        entry = self.volumes[index]
        entry.mask_path = path
        entry.mask = None if mask is None else np.asarray(mask, dtype=bool)
        entry.mask_ops = None
        if index == self.current_frame:
            self.mask_editor = None
        self.volumes_changed.emit()
        self.mask_changed.emit()

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
        subset half-width and the search range (``None`` = whole volume)."""
        if self.para.voi is not None and not self.para.voi.is_whole:
            return self.para.voi
        mask = self.reference_mask()
        if mask is None:
            return None
        return voi_from_mask(mask, self.para.winsize, self.para.search_radius)

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
            base_mask = np.ones(shape, dtype=bool)
        self.mask_editor = MaskEditor(shape, base=base_mask)
        return self.mask_editor

    def _target_frames(self) -> list[int]:
        return list(range(len(self.volumes))) if self.mask_target == "all" else [self.current_frame]

    def _push_mask(self) -> None:
        ed = self.mask_editor
        if ed is None:
            return
        ops = ed.to_dict()
        for i in self._target_frames():
            if i < len(self.volumes):
                self.volumes[i].mask = ed.mask
                self.volumes[i].mask_ops = ops
        self.mask_changed.emit()

    def apply_mask_op(self, op: MaskOp) -> None:
        self.ensure_mask_editor().apply(op)
        self._push_mask()

    def undo_mask(self) -> bool:
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
        self.volumes_changed.emit()
        self.mask_changed.emit()

    def save_mask(self, path: str | Path) -> Path:
        """Write the current mask as a volume file (uint8, 1 = material) and attach the file to the target frames."""
        mask = self.current_mask()
        if mask is None:
            raise ValueError("there is no mask to save")
        from al_dvc.io.volume_io import save_volume

        out = Path(path)
        save_volume(out, mask.astype(np.uint8))
        for i in self._target_frames():
            if i < len(self.volumes):
                self.volumes[i].mask_path = str(out)
                self.volumes[i].mask = mask
        self.volumes_changed.emit()
        return out

    def set_mask_display(self, show: bool | None = None, alpha: float | None = None, target: str | None = None) -> None:
        if show is not None:
            self.show_mask = bool(show)
        if alpha is not None:
            self.mask_alpha = float(alpha)
        if target is not None:
            if target not in ("current", "all"):
                raise ValueError(f"mask target must be 'current' or 'all', got {target!r}")
            self.mask_target = target
            self._push_mask()
        self.mask_changed.emit()

    def volume_array(self, index: int) -> NDArray:
        return self.volumes[index].load()

    def volume_shape(self) -> tuple[int, int, int] | None:
        if not self.volumes:
            return None
        try:
            return tuple(int(s) for s in self.volumes[0].load().shape)  # type: ignore[return-value]
        except Exception as exc:  # unreadable file: report, do not crash
            self.log(f"cannot read {self.volumes[0].name}: {exc}", "error")
            return None

    def set_current_frame(self, index: int) -> None:
        if self.volumes and 0 <= index < len(self.volumes) and index != self.current_frame:
            self.current_frame = index
            self.mask_editor = None
            self.current_frame_changed.emit(index)
            self.mask_changed.emit()

    # ------------------------------------------------------------------ parameters
    def set_param(self, name: str, value: Any) -> None:
        """Replace one parameter; raises ``ValueError`` (validation) without changing the state."""
        new = replace(self.para, **{name: value})
        self.para = new
        self.params_changed.emit()

    def set_params(self, **values: Any) -> None:
        self.para = replace(self.para, **values)
        self.params_changed.emit()

    def set_para(self, para: DVCPara) -> None:
        self.para = para
        self.params_changed.emit()

    # ------------------------------------------------------------------ run
    def set_run_state(self, state: RunState) -> None:
        self.run_state = state
        self.run_state_changed.emit(state)

    def set_progress(self, fraction: float, message: str = "") -> None:
        self.progress = float(fraction)
        self.progress_message = message
        self.progress_updated.emit(self.progress, message)

    def set_results(self, results: PipelineResult | None) -> None:
        self.results = results
        self.display_frame = 0
        self.results_changed.emit()

    def set_output_dir(self, path: str | Path) -> None:
        self.output_dir = Path(path)
        self.output_dir_changed.emit(str(self.output_dir))

    # ------------------------------------------------------------------ display
    def set_display(self, **values: Any) -> None:
        for key, val in values.items():
            if not hasattr(self, key):
                raise AttributeError(key)
            setattr(self, key, val)
        self.display_changed.emit()

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
        self.session_path = None
        self.run_state = RunState.IDLE
        self.mask_editor = None
        self.volumes_changed.emit()
        self.mask_changed.emit()
        self.params_changed.emit()
        self.results_changed.emit()
        self.run_state_changed.emit(self.run_state)
