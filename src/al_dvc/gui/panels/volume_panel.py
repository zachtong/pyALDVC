"""Volume list: the frames of the sequence with their region-of-interest status.

A table (frame, name, shape, region) like pyALDIC's image list: the reference frame's mask is
the region of interest of the analysis, a mask on a deformed frame only excludes its own voxels.
Files and folders can be dropped on the panel; rows can be reordered; a context menu offers the
per-frame actions.

The Shape column shows every volume's size as soon as it is added, read from the file header off the UI
thread. A volume whose size is not the reference's is marked, the user is told at once, and the run refuses
to start: it used to fail only when it reached that frame.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..app_state import AppState, lexical_key, natural_key, size_text
from ..mask_editor import mask_coverage as _coverage
from ..settings_store import gui_settings
from ..theme import current_colors
from ..theme_manager import connect_theme
from ..widgets import headless

VOLUME_FILTER = "Volumes (*.tif *.tiff *.mat *.npy *.npz *.h5 *.hdf5 *.nii *.nii.gz *.nrrd);;All files (*)"
COLUMNS = ("thumb", "index", "name", "shape", "region")
THUMB_SIZE = 44  # px, middle XY slice of a loaded volume
NATURAL_SORT_KEY = "ui/natural_sort"
FLAG = "\u26a0"  # warning sign in front of a size that is not the reference's
PENDING = "\u2026"  # the size is being read

__all__ = ["VolumePanel", "natural_key", "lexical_key", "select_single_type", "thumbnail_pixmap"]


def _kind(path: Path) -> str:
    """The file type of a volume path: the (compound) extension, or ``dir`` for a folder of slices."""
    if path.is_dir():
        return "dir"
    name = path.name.lower()
    return ".nii.gz" if name.endswith(".nii.gz") else path.suffix.lower()


def select_single_type(paths: list[str]) -> tuple[list[str], dict[str, int]]:
    """Keep the files of one supported volume type (the most numerous one) and count what is left out.

    A sequence is one kind of file; stray files of other types in the same folder (notes, previews,
    exports) must not become frames. Returns ``(kept, skipped)`` with ``skipped`` counting per type.
    """
    from al_dvc.io.volume_io import VOLUME_EXT

    supported = set(VOLUME_EXT) | {".nii.gz", "dir"}
    kinds = {p: _kind(Path(p)) for p in paths}
    counts = Counter(k for k in kinds.values() if k in supported)
    if not counts:
        return [], dict(Counter(kinds.values()))
    keep = max(counts, key=lambda k: (counts[k], k))
    kept = [p for p in paths if kinds[p] == keep]
    skipped = Counter(k if k else "(no extension)" for p, k in kinds.items() if k != keep)
    return kept, dict(skipped)


def thumbnail_pixmap(volume, size: int = THUMB_SIZE):
    """``QPixmap`` of the middle XY slice of ``volume`` (percentile-stretched grey), ``None`` when it fails."""
    try:
        import numpy as np
        from PySide6.QtGui import QImage, QPixmap

        vol = np.asarray(volume)
        if vol.ndim != 3 or vol.size == 0:
            return None
        sl = np.asarray(vol[vol.shape[0] // 2], dtype=np.float64)
        finite = sl[np.isfinite(sl)]
        if finite.size == 0:
            return None
        lo, hi = np.percentile(finite, (1, 99))
        if hi <= lo:
            hi = lo + 1.0
        img = np.clip((sl - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        img = np.ascontiguousarray(img[::-1])  # origin at the bottom, like the slice viewer
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        return QPixmap.fromImage(qimg).scaled(
            size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
    except Exception:  # a thumbnail must never break the panel
        return None


class _VolumeTable(QTableWidget):
    """Read-only single-selection table; ``count()`` keeps the old list API."""

    def count(self) -> int:
        return self.rowCount()


class VolumePanel(QWidget):
    """Frames of the sequence (files or arrays) with their optional masks."""

    session_dropped = Signal(str)  # a session file (.aldvc) was dropped here: the window opens it

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._list = _VolumeTable(0, len(COLUMNS))
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._list.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._list.verticalHeader().setVisible(False)
        self._list.setShowGrid(False)
        self._list.setMinimumHeight(110)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header = self._list.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(0, THUMB_SIZE + 8)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self._list.verticalHeader().setDefaultSectionSize(THUMB_SIZE + 6)
        self._thumbs: dict[int, QPixmap] = {}
        # sizes arrive one file at a time from the reading thread; one refresh serves all that are queued
        self._size_refresh = QTimer(self)
        self._size_refresh.setSingleShot(True)
        self._size_refresh.setInterval(0)
        self._size_refresh.timeout.connect(self.refresh)
        header.setHighlightSections(False)
        self._btn_add = QPushButton()
        self._btn_folder = QPushButton()
        self._natural_sort = QCheckBox()
        self._natural_sort.setChecked(bool(gui_settings().value(NATURAL_SORT_KEY, True, type=bool)))
        self._btn_mask = QPushButton()
        self._btn_remove = QPushButton()
        self._btn_up = QPushButton()
        self._btn_down = QPushButton()
        for b in (self._btn_up, self._btn_down):
            b.setFixedWidth(36)
            b.setStyleSheet("padding: 4px 2px;")
        self._info = QLabel()
        self._info.setWordWrap(True)
        self._info.setObjectName("hint")
        self._roi_hint = QLabel()
        self._roi_hint.setWordWrap(True)
        self._roi_hint.setObjectName("hint")
        self._placeholder = QLabel(self._list)
        self._placeholder.setObjectName("placeholder")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        self.setAcceptDrops(True)
        self._list.setAcceptDrops(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row1 = QHBoxLayout()
        row1.addWidget(self._btn_add)
        row1.addWidget(self._btn_folder)
        layout.addLayout(row1)
        layout.addWidget(self._list)
        row_sort = QHBoxLayout()
        row_sort.addWidget(self._natural_sort)
        row_sort.addStretch(1)
        layout.addLayout(row_sort)
        row2 = QHBoxLayout()
        row2.addWidget(self._btn_mask)
        row2.addWidget(self._btn_remove)
        row2.addWidget(self._btn_up)
        row2.addWidget(self._btn_down)
        layout.addLayout(row2)
        layout.addWidget(self._info)
        layout.addWidget(self._roi_hint)

        self._btn_add.clicked.connect(self._on_add_files)
        self._natural_sort.toggled.connect(self._on_natural_sort)
        self._btn_folder.clicked.connect(self._on_add_folder)
        self._btn_mask.clicked.connect(self._on_set_mask)
        self._btn_remove.clicked.connect(self._on_remove)
        self._btn_up.clicked.connect(lambda: self._move(-1))
        self._btn_down.clicked.connect(lambda: self._move(+1))
        self._list.currentCellChanged.connect(lambda row, _c, _pr, _pc: self._on_row_changed(row))
        self._list.customContextMenuRequested.connect(self._on_context_menu)
        self._state.volumes_changed.connect(self.refresh)
        self._state.mask_changed.connect(self.refresh)
        self._state.shapes_changed.connect(self._size_refresh.start)
        self._state.shape_check_finished.connect(self._report_sizes)
        self._state.current_frame_changed.connect(self._select_row)
        connect_theme(self._on_theme_changed)
        self.retranslate_ui()
        self.refresh()

    def _on_theme_changed(self, _name: str) -> None:
        self.refresh()  # a size that differs from the reference is flagged in the theme's danger colour

    # ------------------------------------------------------------------ drag and drop
    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt API
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.add_dropped(paths)
            event.acceptProposedAction()

    def add_dropped(self, paths: list[str]) -> int:
        """Add dropped files and folders; a dropped folder replaces the sequence like Add folder. Returns the count."""
        from al_dvc.io.volume_io import resolve_volume_paths

        sessions = [p for p in paths if str(p).lower().endswith(".aldvc")]
        if sessions:  # a session is not a volume: it replaces the whole document
            self.session_dropped.emit(str(sessions[0]))
            return 0
        files: list[str] = []
        replace = False
        for p in paths:
            path = Path(p)
            if path.is_dir():
                replace = True
                try:
                    files += [str(q) for q in resolve_volume_paths(str(path))]
                except Exception as exc:
                    self._state.log(f"{path}: {exc}", "error")
            elif path.is_file():
                files.append(str(path))
        if not files:
            self._state.log(self.tr("Nothing usable was dropped (volumes or a folder of volumes)."), "warning")
            return 0
        return self.import_files(files, replace=replace)

    def sort_key(self):
        return natural_key if self._natural_sort.isChecked() else lexical_key

    def import_files(self, files: list[str], replace: bool = False) -> int:
        """Add the files of one volume type, in the chosen order; ``replace`` drops the current sequence first.
        Returns how many frames were added."""
        kept, skipped = select_single_type([str(f) for f in files])
        if not kept:
            self._state.log(
                self.tr("No volume files among the selection ({types}).").format(
                    types=", ".join(f"{n} {k}" for k, n in sorted(skipped.items()))
                ),
                "warning",
            )
            return 0
        if replace and self._state.volumes:
            n_old = len(self._state.volumes)
            self._state.clear_volumes()
            if self._state.volumes:  # locked by a run
                return 0
            self._state.log(self.tr("The previous {n} volume(s) were replaced.").format(n=n_old))
        self._state.add_volume_paths(sorted(kept, key=self.sort_key()))
        kind = _kind(Path(kept[0]))
        self._state.log(
            self.tr("{n} {kind} volume(s) added").format(n=len(kept), kind=self.tr("slice-folder") if kind == "dir" else kind)
        )
        if skipped:
            self._state.log(
                self.tr("Skipped {n} file(s) of other types: {types}").format(
                    n=sum(skipped.values()), types=", ".join(f"{k} ({v})" for k, v in sorted(skipped.items()))
                ),
                "warning",
            )
        return len(kept)

    # ------------------------------------------------------------------ actions
    def _on_natural_sort(self, natural: bool) -> None:
        gui_settings().setValue(NATURAL_SORT_KEY, bool(natural))
        self._state.sort_volumes(bool(natural))

    def _on_add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, self.tr("Add volumes"), "", VOLUME_FILTER)
        if files:
            self.import_files(files, replace=False)

    def _on_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, self.tr("Folder of volumes (replaces the sequence)"))
        if not folder:
            return
        self.import_folder(folder)

    def import_folder(self, folder: str) -> int:
        """The volumes of ``folder`` (one type) become the sequence; whatever was listed before is dropped."""
        from al_dvc.io.volume_io import resolve_volume_paths

        try:
            paths = resolve_volume_paths(folder)
        except Exception as exc:
            self._state.log(f"{folder}: {exc}", "error")
            return 0
        if not paths:
            self._state.log(self.tr("No volume files found in {folder}").format(folder=folder), "warning")
            return 0
        return self.import_files([str(p) for p in paths], replace=True)

    def _on_set_mask(self) -> None:
        row = self._list.currentRow()
        if row < 0:
            return
        path, _ = QFileDialog.getOpenFileName(self, self.tr("Mask volume (True = material)"), "", VOLUME_FILTER)
        if not path:
            return
        problem = self._state.mask_size_problem(row, path)
        if problem:
            self._state.log(problem, "error")
            if not headless():
                QMessageBox.warning(self, self.tr("Mask not attached"), problem)
            return
        self._state.set_mask(row, path=path)

    def _on_remove(self) -> None:
        row = self._list.currentRow()
        if row >= 0:
            self._state.remove_volume(row)

    def _move(self, delta: int) -> None:
        row = self._list.currentRow()
        if row >= 0:
            self._state.move_volume(row, row + delta)

    def _on_context_menu(self, pos) -> None:
        row = self._list.rowAt(pos.y())
        if row < 0 or row >= len(self._state.volumes):
            return
        self._list.setCurrentCell(row, 0)
        menu = QMenu(self)
        entry = self._state.volumes[row]
        has_mask = entry.mask_path is not None or entry.mask is not None
        act_mask = menu.addAction(self.tr("Set mask from file..."))
        act_clear = menu.addAction(self.tr("Remove mask"))
        act_clear.setEnabled(has_mask)
        menu.addSeparator()
        act_up = menu.addAction(self.tr("Move up"))
        act_up.setEnabled(row > 0)
        act_down = menu.addAction(self.tr("Move down"))
        act_down.setEnabled(row < len(self._state.volumes) - 1)
        menu.addSeparator()
        act_remove = menu.addAction(self.tr("Remove frame"))
        chosen = menu.exec(self._list.viewport().mapToGlobal(pos))
        if chosen is act_mask:
            self._on_set_mask()
        elif chosen is act_clear:
            self._state.remove_mask(row)
        elif chosen is act_up:
            self._move(-1)
        elif chosen is act_down:
            self._move(+1)
        elif chosen is act_remove:
            self._state.remove_volume(row)

    def _on_row_changed(self, row: int) -> None:
        if row >= 0:
            self._state.set_current_frame(row)
            self._update_info(row)

    def _select_row(self, index: int) -> None:
        if 0 <= index < self._list.rowCount() and self._list.currentRow() != index:
            self._list.setCurrentCell(index, 0)

    # ------------------------------------------------------------------ view
    def _region_text(self, index: int, entry) -> str:
        has_mask = entry.mask_path is not None or entry.mask is not None
        if index == 0:
            if not has_mask:
                return self.tr("whole volume")
            mask = self._state.reference_mask()
            return self.tr("ROI {pct:.0f}%").format(pct=100.0 * _coverage(mask)) if mask is not None else self.tr("ROI")
        return self.tr("own mask") if has_mask else "-"

    def _size_cell(self, entry, differs: bool, ref) -> tuple[str, str]:
        """Text and tooltip of the Shape column."""
        shape = entry.shape
        if shape is None:
            if entry.shape_pending:
                return PENDING, self.tr("Reading the size from the file header...")
            return "?", self.tr("The size is known once the volume is read")
        if differs:
            return f"{FLAG} {size_text(shape)}", self.tr(
                "Differs from the reference volume ({size} voxels, x \u00d7 y \u00d7 z)"
            ).format(size=size_text(ref))
        return size_text(shape), self.tr("Size in voxels (x \u00d7 y \u00d7 z)")

    def refresh(self) -> None:
        differs = {i for i, _shape in self._state.shape_mismatches()}
        ref = self._state.reference_shape()
        self._list.blockSignals(True)
        self._list.setRowCount(len(self._state.volumes))
        for i, entry in enumerate(self._state.volumes):
            name = Path(entry.path).name if entry.path else entry.name
            if entry.missing:
                name = FLAG + " " + name
            size, size_tip = self._size_cell(entry, i in differs, ref)
            cells = ["", str(i), name, size, self._region_text(i, entry)]
            path_tip = entry.path or self.tr("in-memory array")
            if entry.missing:
                path_tip = self.tr("File not found: {path}").format(path=entry.path)
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(size_tip if c == 3 else path_tip)
                if c in (1, 3, 4):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if (i in differs and c in (2, 3)) or (entry.missing and c == 2):
                    item.setForeground(QColor(current_colors().DANGER))
                self._list.setItem(i, c, item)
            self._list.setCellWidget(i, 0, self._thumbnail_label(entry))
        self._list.blockSignals(False)
        if self._state.volumes:
            row = min(self._state.current_frame, len(self._state.volumes) - 1)
            if self._list.currentRow() != row:
                self._list.setCurrentCell(row, 0)
        self._update_info(self._list.currentRow())
        has = bool(self._state.volumes)
        self._placeholder.setVisible(not has)
        self._update_roi_hint()
        for b in (self._btn_mask, self._btn_remove, self._btn_up, self._btn_down):
            b.setEnabled(has)

    def _thumbnail_label(self, entry) -> QLabel:
        """A small grey-scale picture of the middle XY slice (only for volumes already in memory)."""
        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pix = thumbnail_pixmap(entry.array, THUMB_SIZE) if entry.array is not None else None
        if pix is not None:
            label.setPixmap(pix)
        else:
            label.setText("\u25a1")  # empty square: not loaded yet
        return label

    def _update_info(self, row: int) -> None:
        n = len(self._state.volumes)
        if n == 0:
            self._info.setText(self.tr("Add at least two volumes (the first one is the reference)."))
            return
        parts = [self.tr("{n} frames").format(n=n)]
        if 0 <= row < n:
            entry = self._state.volumes[row]
            text = Path(entry.path).name if entry.path else entry.name
            if entry.shape is not None:
                text += f": {size_text(entry.shape)}"
            if entry.array is not None:
                text += f" {entry.array.dtype}"
            parts.append(text)
        differ = len(self._state.shape_mismatches())
        if differ:
            parts.append(FLAG + " " + self.tr("{n} volume(s) differ in size from the reference").format(n=differ))
        self._info.setText("   ".join(parts))

    def _report_sizes(self, uids: list[str]) -> None:
        """Tell the user, once for each group of volumes added, which of them do not have the reference's size."""
        volumes = self._state.volumes
        mismatches = self._state.shape_mismatches()
        if not mismatches:
            return
        batch = set(uids)
        if volumes[0].uid not in batch:  # a new reference concerns every volume; otherwise only the ones just added
            mismatches = [(i, shape) for i, shape in mismatches if volumes[i].uid in batch]
        if not mismatches:
            return
        self._state.log(
            self.tr("{n} volume(s) differ in size from the reference {name} ({size}): {files}").format(
                **self._state.mismatch_fields(mismatches)
            ),
            "error",
        )
        if headless():
            return
        QMessageBox.warning(
            self,
            self.tr("Volumes of different sizes"),
            self.tr(
                "Every volume must have the size of the reference volume (frame 0, {name}): {size} voxels "
                "(x \u00d7 y \u00d7 z).\n\nThese volumes differ:\n{files}\n\nRemove them from the list, or replace "
                "them with volumes of the reference size, before running."
            ).format(**self._state.mismatch_fields(mismatches, sep="\n")),
        )

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._placeholder.setGeometry(self._list.rect().adjusted(8, 8, -8, -8))

    def _update_roi_hint(self) -> None:
        if not self._state.volumes:
            self._roi_hint.setText("")
            return
        mask = self._state.reference_mask()
        if mask is None:
            self._roi_hint.setText(
                self.tr("No region of interest: the whole volume is analysed. Draw one on the slices to crop.")
            )
        else:
            frac = 100.0 * _coverage(mask)
            self._roi_hint.setText(self.tr("Region of interest covers {pct:.0f}% of the reference volume.").format(pct=frac))

    def retranslate_ui(self) -> None:
        self._btn_add.setText(self.tr("Add volumes..."))
        self._btn_add.setToolTip(self.tr("Append volume files of one type to the sequence"))
        self._btn_folder.setText(self.tr("Add folder..."))
        self._btn_folder.setToolTip(self.tr("The volumes of a folder become the sequence (the current list is replaced)"))
        self._natural_sort.setText(self.tr("Natural order (1, 2, ..., 10)"))
        self._natural_sort.setToolTip(
            self.tr(
                "Numbers inside file names compare numerically (frame2 before frame10); "
                "unticked, names compare character by character (000, 001, ...)"
            )
        )
        self._btn_mask.setText(self.tr("Set mask..."))
        self._btn_remove.setText(self.tr("Remove"))
        self._btn_up.setText("\u25b2")  # up-pointing triangle
        self._btn_down.setText("\u25bc")
        self._btn_up.setToolTip(self.tr("Move the frame up (frame 0 is the reference)"))
        self._btn_down.setToolTip(self.tr("Move the frame down"))
        self._list.setHorizontalHeaderLabels(["", self.tr("#"), self.tr("Volume"), self.tr("Shape"), self.tr("ROI")])
        self._list.horizontalHeaderItem(3).setToolTip(self.tr("Size in voxels (x \u00d7 y \u00d7 z)"))
        self._placeholder.setText(self.tr("Drop volume files or a folder here\n(TIFF, npy, npz, mat)"))
        self._update_roi_hint()
        self.refresh()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Delete:
            self._on_remove()
        else:
            super().keyPressEvent(event)
