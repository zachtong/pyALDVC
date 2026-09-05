"""Volume list: the frames of the sequence with their region-of-interest status.

A table (frame, name, shape, region) like pyALDIC's image list: the reference frame's mask is
the region of interest of the analysis, a mask on a deformed frame only excludes its own voxels.
Files and folders can be dropped on the panel; rows can be reordered; a context menu offers the
per-frame actions.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..app_state import AppState

VOLUME_FILTER = "Volumes (*.tif *.tiff *.mat *.npy *.npz *.h5 *.hdf5 *.nii *.nii.gz *.nrrd);;All files (*)"
COLUMNS = ("thumb", "index", "name", "shape", "region")
THUMB_SIZE = 44  # px, middle XY slice of a loaded volume


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
        header.setHighlightSections(False)
        self._btn_add = QPushButton()
        self._btn_folder = QPushButton()
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
        row2 = QHBoxLayout()
        row2.addWidget(self._btn_mask)
        row2.addWidget(self._btn_remove)
        row2.addWidget(self._btn_up)
        row2.addWidget(self._btn_down)
        layout.addLayout(row2)
        layout.addWidget(self._info)
        layout.addWidget(self._roi_hint)

        self._btn_add.clicked.connect(self._on_add_files)
        self._btn_folder.clicked.connect(self._on_add_folder)
        self._btn_mask.clicked.connect(self._on_set_mask)
        self._btn_remove.clicked.connect(self._on_remove)
        self._btn_up.clicked.connect(lambda: self._move(-1))
        self._btn_down.clicked.connect(lambda: self._move(+1))
        self._list.currentCellChanged.connect(lambda row, _c, _pr, _pc: self._on_row_changed(row))
        self._list.customContextMenuRequested.connect(self._on_context_menu)
        self._state.volumes_changed.connect(self.refresh)
        self._state.mask_changed.connect(self.refresh)
        self._state.current_frame_changed.connect(self._select_row)
        self.retranslate_ui()
        self.refresh()

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
        """Add dropped files and folders (a folder contributes every volume file it holds); returns the count."""
        from al_dvc.io.volume_io import resolve_volume_paths

        files: list[str] = []
        for p in paths:
            path = Path(p)
            if path.is_dir():
                try:
                    files += [str(q) for q in resolve_volume_paths(str(path))]
                except Exception as exc:
                    self._state.log(f"{path}: {exc}", "error")
            elif path.is_file():
                files.append(str(path))
        if files:
            self._state.add_volume_paths(sorted(files))
        else:
            self._state.log(self.tr("Nothing usable was dropped (volumes or a folder of volumes)."), "warning")
        return len(files)

    # ------------------------------------------------------------------ actions
    def _on_add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, self.tr("Add volumes"), "", VOLUME_FILTER)
        if files:
            self._state.add_volume_paths(sorted(files))

    def _on_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, self.tr("Add a folder of volumes"))
        if not folder:
            return
        from al_dvc.io.volume_io import resolve_volume_paths

        try:
            paths = resolve_volume_paths(folder)
        except Exception as exc:
            self._state.log(f"{folder}: {exc}", "error")
            return
        if paths:
            self._state.add_volume_paths([str(p) for p in paths])
        else:
            self._state.log(self.tr("No volume files found in {folder}").format(folder=folder), "warning")

    def _on_set_mask(self) -> None:
        row = self._list.currentRow()
        if row < 0:
            return
        path, _ = QFileDialog.getOpenFileName(self, self.tr("Mask volume (True = material)"), "", VOLUME_FILTER)
        if path:
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
            return self.tr("ROI {pct:.0f}%").format(pct=100.0 * float(mask.mean())) if mask is not None else self.tr("ROI")
        return self.tr("own mask") if has_mask else "-"

    def refresh(self) -> None:
        self._list.blockSignals(True)
        self._list.setRowCount(len(self._state.volumes))
        for i, entry in enumerate(self._state.volumes):
            name = Path(entry.path).name if entry.path else entry.name
            shape = f"{tuple(entry.array.shape)}" if entry.array is not None else ""
            cells = ["", str(i), name, shape, self._region_text(i, entry)]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(entry.path or self.tr("in-memory array"))
                if c in (1, 3, 4):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
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
            if entry.array is not None:
                a = entry.array
                parts.append(f"{Path(entry.path).name if entry.path else entry.name}: {tuple(a.shape)} {a.dtype}")
            elif entry.path:
                parts.append(Path(entry.path).name)
        self._info.setText("   ".join(parts))

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
            frac = 100.0 * float(mask.mean())
            self._roi_hint.setText(self.tr("Region of interest covers {pct:.0f}% of the reference volume.").format(pct=frac))

    def retranslate_ui(self) -> None:
        self._btn_add.setText(self.tr("Add volumes..."))
        self._btn_folder.setText(self.tr("Add folder..."))
        self._btn_mask.setText(self.tr("Set mask..."))
        self._btn_remove.setText(self.tr("Remove"))
        self._btn_up.setText("\u25b2")  # up-pointing triangle
        self._btn_down.setText("\u25bc")
        self._btn_up.setToolTip(self.tr("Move the frame up (frame 0 is the reference)"))
        self._btn_down.setToolTip(self.tr("Move the frame down"))
        self._list.setHorizontalHeaderLabels(["", self.tr("#"), self.tr("Volume"), self.tr("Shape"), self.tr("ROI")])
        self._placeholder.setText(self.tr("Drop volume files or a folder here\n(TIFF, npy, npz, mat)"))
        self._update_roi_hint()
        self.refresh()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Delete:
            self._on_remove()
        else:
            super().keyPressEvent(event)
