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
from al_dvc.core.data_structures import PipelineResult

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
        self.write_checkpoints: bool = True
        # display
        self.display_field: str = "disp_magnitude"
        self.display_frame: int = 0
        self.colormap: str = "viridis"
        self.color_auto: bool = True
        self.color_min: float = 0.0
        self.color_max: float = 1.0
        self.overlay_alpha: float = 0.75
        self.show_overlay: bool = True
        self.slice_index: dict[str, int | None] = {"z": None, "y": None, "x": None}

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
        self.volumes_changed.emit()
        self.results_changed.emit()

    def remove_volume(self, index: int) -> None:
        if 0 <= index < len(self.volumes):
            del self.volumes[index]
            self.current_frame = min(self.current_frame, max(0, len(self.volumes) - 1))
            self.volumes_changed.emit()

    def clear_volumes(self) -> None:
        self.volumes = []
        self.current_frame = 0
        self.volumes_changed.emit()

    def set_mask(self, index: int, path: str | None = None, mask: NDArray[np.bool_] | None = None) -> None:
        entry = self.volumes[index]
        entry.mask_path = path
        entry.mask = None if mask is None else np.asarray(mask, dtype=bool)
        self.volumes_changed.emit()

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
            self.current_frame_changed.emit(index)

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
        self.volumes_changed.emit()
        self.params_changed.emit()
        self.results_changed.emit()
        self.run_state_changed.emit(self.run_state)
