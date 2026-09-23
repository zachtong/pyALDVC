"""The result pages of the Statistics tab -- series over frames, regions, profile, line, noise floor -- and what
they show: the summary and region tables, the homogeneous deformation, the noise floor and the charts. A mixin of
:class:`AnalysisTab`."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from al_dvc.analysis.stats import CI_NAMES, STAT_NAMES

from .analysis_charts import PALETTE, draw_curves, draw_line, draw_profile, draw_region_bars, fmt, series_curve
from .analysis_jobs import CurrentFrame
from .names import field_name
from .theme import COLORS
from .widgets import combo, dspin, guard_wheel, spin

REDUCTIONS = ("mean_std", "mean_ci", "median_robust", "rms", "max", "min", "n")
PROFILE_AXES = ("x", "y", "z")
TABLE_COLUMNS = STAT_NAMES + CI_NAMES
REGION_COLUMNS = ("n", "mean", "std", "ci95", "n_eff", "median", "min", "max")
COORD_LIMIT = 1e6


def _field_names(series) -> list[str]:
    """The fields of a series (a frame whose motion could not be fitted has none)."""
    return list(next((fs.stats for fs in series if fs.stats), {}))


class AnalysisPagesMixin:
    """Result pages of :class:`~al_dvc.gui.analysis_tab.AnalysisTab` (which provides the controls and the results)."""

    def _long_row(self, kind: str, row: QHBoxLayout) -> None:
        run = QPushButton()
        run.setProperty("class", "btn-primary")
        cancel = QPushButton()
        cancel.setEnabled(False)
        bar = QProgressBar()
        bar.setRange(0, 1000)
        bar.setTextVisible(False)
        bar.setFixedWidth(120)
        self._run[kind], self._cancel[kind], self._bars[kind] = run, cancel, bar
        row.addWidget(run)
        row.addWidget(cancel)
        row.addWidget(bar)

    @staticmethod
    def _figure() -> tuple[Figure, FigureCanvas]:
        fig = Figure(figsize=(9, 3), facecolor=COLORS.BG_CANVAS)
        return fig, FigureCanvas(fig)

    def _page(self) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        return page, layout

    def _build_series_page(self) -> QWidget:
        page, layout = self._page()
        row = QHBoxLayout()
        self._long_row("series", row)
        self.reduction = combo([])
        for key in REDUCTIONS:
            self.reduction.addItem(key, key)
        self.compare_regions = QCheckBox()
        self._min_share_label = QLabel()
        self.min_share = spin(0, 100, 5, width=56)
        self.min_share.setSuffix(" %")
        self.series_info = QLabel()
        self.series_info.setObjectName("hint")
        row.addWidget(self.reduction)
        row.addWidget(self.compare_regions)
        row.addWidget(self._min_share_label)
        row.addWidget(self.min_share)
        row.addWidget(self.series_info, 1)
        layout.addLayout(row)
        self.series_figure, self.series_canvas = self._figure()
        layout.addWidget(self.series_canvas, 1)
        guard_wheel(page)
        return page

    def _build_regions_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        self.regions_table = QTableWidget(0, len(REGION_COLUMNS))
        self.regions_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.regions_table, 1)
        self.regions_figure, self.regions_canvas = self._figure()
        layout.addWidget(self.regions_canvas, 1)
        return page

    def _build_profile_page(self) -> QWidget:
        page, layout = self._page()
        row = QHBoxLayout()
        self._profile_axis_label = QLabel()
        self.profile_axis = combo(list(PROFILE_AXES), width=60)
        self.profile_axis.setCurrentText("z")
        row.addWidget(self._profile_axis_label)
        row.addWidget(self.profile_axis)
        self._long_row("profiles", row)
        self.profile_info = QLabel()
        self.profile_info.setObjectName("hint")
        row.addWidget(self.profile_info, 1)
        layout.addLayout(row)
        self.profile_figure, self.profile_canvas = self._figure()
        layout.addWidget(self.profile_canvas, 1)
        guard_wheel(page)
        return page

    def _build_line_page(self) -> QWidget:
        page, layout = self._page()
        row = QHBoxLayout()
        self._line_labels = [QLabel(), QLabel()]
        self.line_spins = [dspin(-COORD_LIMIT, COORD_LIMIT, 1, width=64) for _ in range(6)]
        for i in range(2):
            row.addWidget(self._line_labels[i])
            for w in self.line_spins[3 * i : 3 * i + 3]:
                row.addWidget(w)
        self._btn_pick_line = QPushButton()
        row.addWidget(self._btn_pick_line)
        row.addStretch(1)
        layout.addLayout(row)
        row2 = QHBoxLayout()
        self._long_row("line", row2)
        self.line_info = QLabel()
        self.line_info.setObjectName("hint")
        self.line_info.setWordWrap(True)
        row2.addWidget(self.line_info, 1)
        layout.addLayout(row2)
        self.line_figure, self.line_canvas = self._figure()
        layout.addWidget(self.line_canvas, 1)
        guard_wheel(page)
        return page

    def _build_noise_page(self) -> QWidget:
        page, layout = self._page()
        nrow = QHBoxLayout()
        self._nominal_label = QLabel()
        nrow.addWidget(self._nominal_label)
        self.nominal = [dspin(-1e6, 1e6, 4) for _ in range(3)]
        for w in self.nominal:
            nrow.addWidget(w)
        run = QPushButton()
        run.setProperty("class", "btn-primary")
        self._run["noise"] = run
        nrow.addWidget(run)
        self.noise_info = QLabel()
        self.noise_info.setObjectName("hint")
        self.noise_info.setWordWrap(True)
        nrow.addWidget(self.noise_info, 1)
        layout.addLayout(nrow)
        self.noise_table = QTableWidget(0, 2)
        self.noise_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.noise_table, 1)
        guard_wheel(page)
        return page

    @staticmethod
    def _cell(text: str, right: bool = True) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        if right:
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return item

    @staticmethod
    def _stat_text(st, key: str) -> str:
        v = getattr(st, key)
        if key == "n":
            return f"{v:,}"
        if key == "n_eff":
            return fmt(v, 3)
        return fmt(v)

    def _show_table(self, stats: dict) -> None:
        names = list(stats)
        self.table.setRowCount(len(names))
        self.table.setVerticalHeaderLabels([self._label_with_unit(n) for n in names])
        for r, name in enumerate(names):
            for c, key in enumerate(TABLE_COLUMNS):
                self.table.setItem(r, c, self._cell(self._stat_text(stats[name], key)))
        self.table.resizeColumnsToContents()

    def _region_rows(self, cur: CurrentFrame) -> list[tuple[str, str, object]]:
        """``(label, colour, FieldStats)`` of the region comparison."""
        out = []
        for region, st in cur.region_rows:
            if region is None:
                out.append((self.tr("All nodes"), COLORS.TEXT_SECONDARY, st))
            else:
                out.append((region.name, region.color, st))
        return out

    def _show_regions(self, cur: CurrentFrame) -> None:
        rows = self._region_rows(cur)
        table = self.regions_table
        table.setRowCount(len(rows))
        table.setVerticalHeaderLabels([label for label, _c, _s in rows])
        for r, (_label, _color, st) in enumerate(rows):
            for c, key in enumerate(REGION_COLUMNS):
                table.setItem(r, c, self._cell(self._stat_text(st, key)))
        table.resizeColumnsToContents()
        draw_region_bars(self.regions_figure, rows, self._label_with_unit(cur.compare_field))
        self.regions_canvas.draw_idle()

    def _show_homogeneous(self, hom) -> None:
        table = self.hom_table
        if hom is None:
            table.setRowCount(0)
            return
        comps = ("exx", "eyy", "ezz", "exy", "exz", "eyz")
        idx = {"exx": (0, 0), "eyy": (1, 1), "ezz": (2, 2), "exy": (0, 1), "exz": (0, 2), "eyz": (1, 2)}
        rows = []
        for c in comps:
            i, j = idx[c]
            nodal = None if hom.nodal_mean is None else hom.nodal_mean.get(c)
            rows.append((field_name(c), fmt(hom.infinitesimal[i, j]), fmt(hom.green_lagrange[i, j]), fmt(nodal)))
        rows.append((self.tr("Rotation [deg]"), fmt(hom.rotation_deg), "", ""))
        rows.append((self.tr("Volume ratio det F"), fmt(float(np.linalg.det(hom.F))), "", ""))
        rows.append((self.tr("Residual RMS [{unit}]").format(unit=self._length_unit()), fmt(hom.fit.residual_rms), "", ""))
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(
            ["", self.tr("Affine fit, infinitesimal"), self.tr("Affine fit, Green-Lagrange"), self.tr("Mean of nodal strains")]
        )
        table.setRowCount(len(rows))
        for r, cells in enumerate(rows):
            for c, text in enumerate(cells):
                table.setItem(r, c, self._cell(text, right=bool(c)))
        table.resizeColumnsToContents()

    def _draw_profile(self) -> None:
        cur = self.current
        others = self.long["profiles"][1] if self.long_is_current("profiles") else None
        profile = None if cur is None or cur.error else cur.profile
        axis = self.profile_axis.currentText()
        field = self.compare_field()
        draw_profile(
            self.profile_figure,
            profile,
            self.frame(),
            others,
            f"{axis} [{self._length_unit()}]",
            self._label_with_unit(field),
        )
        self.profile_canvas.draw_idle()
        if others is None and "profiles" not in self.long:
            where = self.stats_region()
            self.profile_info.setText(
                self.tr("Mean of each node layer along {axis} ({where}); band: one std.").format(
                    axis=axis, where=self.tr("all nodes") if where is None else where.name
                )
            )

    def _draw_line(self) -> None:
        cur = self.current
        series = self.long["line"][1] if self.long_is_current("line") else None
        line = None if cur is None or cur.error else cur.line
        draw_line(
            self.line_figure,
            line,
            self.frame(),
            series,
            self.tr("Distance from the first point [{unit}]").format(unit=self._length_unit()),
            self._label_with_unit(self.compare_field()),
        )
        self.line_canvas.draw_idle()
        if series is not None:
            ext = series.extensometer
            gaps = int(np.isnan(ext.strain).sum())
            self.line_info.setText(
                self.tr("Extensometer: L0 = {L0} {unit}, last frame strain {e}").format(
                    L0=fmt(ext.L0), unit=self._length_unit(), e=fmt(ext.strain[-1] if ext.strain.size else None)
                )
                + (self.tr("; {g} frames without a value at an end point").format(g=gaps) if gaps else "")
            )
        elif "line" not in self.long:
            self.line_info.setText(
                self.tr(
                    "Over all frames: the line in every frame, the field at its two ends, and the virtual extensometer "
                    "(the length between the ends, L/L0 - 1)."
                )
            )

    def _series_curves(self) -> list[tuple]:
        """``(label, colour, frames, centre, band)`` of the series on screen."""
        if self.series is None:
            return []
        runs = self.series[1]
        red = self.reduction.currentData() or "mean_std"
        curves = []
        if len(runs) == 1:  # one region: every field of the group
            _region, series = runs[0]
            for i, name in enumerate(_field_names(series)):
                x, y, band = series_curve(series, name, red)
                curves.append((field_name(name), PALETTE[i % len(PALETTE)], x, y, band))
        else:  # regions compared: the field on the slices, one curve per region
            name = self.compare_field()
            for region, series in runs:
                if not series:
                    continue
                x, y, band = series_curve(series, name, red)
                label = self.tr("All nodes") if region is None else region.name
                color = COLORS.TEXT_SECONDARY if region is None else region.color
                curves.append((label, color, x, y, band))
        return curves

    def _draw_series(self) -> None:
        curves = self._series_curves()
        runs = self.series[1] if self.series is not None else []
        names = _field_names(runs[0][1]) if runs else []
        unit = self._unit(self.compare_field() if len(runs) > 1 else (names[0] if names else None))
        draw_curves(self.series_figure, curves, self.tr("Frame"), unit)
        self.series_canvas.draw_idle()
        if self.series_is_current() and runs:
            series = runs[0][1]
            gaps = sum(int(fs.flag != "ok") for _r, s in runs for fs in s)
            text = self.tr("{n} frames").format(n=len(series))
            if len(runs) > 1:
                text += ", " + self.tr("{r} regions").format(r=len(runs))
            self.series_info.setText(text + (self.tr(", {g} without enough nodes (gaps)").format(g=gaps) if gaps else ""))

    def _show_noise(self) -> None:
        if self.noise is None:
            self.noise_table.setRowCount(0)
            return
        _source, nf = self.noise
        unit = nf.unit

        def vec(v) -> str:
            return "-" if v is None else ", ".join(fmt(x) for x in v)

        rows = [
            (self.tr("Nodes"), f"{nf.displacement.n:,}"),
            (self.tr("Bias (mean error) x, y, z [{unit}]").format(unit=unit), vec(nf.displacement.bias)),
            (self.tr("Noise floor (std) x, y, z [{unit}]").format(unit=unit), vec(nf.displacement.noise)),
            (self.tr("u_rms [{unit}]").format(unit=unit), fmt(nf.displacement.u_rms)),
        ]
        if nf.rigid is not None:
            rows.append((self.tr("Noise floor after a rigid fit x, y, z [{unit}]").format(unit=unit), vec(nf.rigid.noise)))
            rows.append((self.tr("Rigid rotation between the scans [deg]"), fmt(nf.rigid_fit.rotation_deg)))
        if nf.spatial_std is not None:
            rows.append(
                (self.tr("Spatial std x, y, z over {k} frames [{unit}]").format(k=nf.n_frames, unit=unit), vec(nf.spatial_std))
            )
        if nf.temporal_std is not None:
            rows.append((self.tr("Temporal std x, y, z [{unit}]").format(unit=unit), vec(nf.temporal_std)))
        if nf.strain is not None:
            rows.append((self.tr("Strain mean exx ... eyz"), vec([s.mean for s in nf.strain.values()])))
            rows.append((self.tr("Strain std exx ... eyz"), vec([s.std for s in nf.strain.values()])))
            rows.append(("MAER / SDER", f"{fmt(nf.maer)} / {fmt(nf.sder)}"))
        if nf.vsg is not None:
            rows.append((self.tr("Virtual strain gauge x, y, z [voxel]"), vec(nf.vsg)))
        self.noise_table.setHorizontalHeaderLabels([self.tr("Quantity"), self.tr("Value")])
        self.noise_table.setRowCount(len(rows))
        for r, (name, value) in enumerate(rows):
            self.noise_table.setItem(r, 0, QTableWidgetItem(name))
            self.noise_table.setItem(r, 1, QTableWidgetItem(value))
        self.noise_table.resizeColumnsToContents()
        self.noise_info.setText(
            self.tr("Frame {k}; bias = mean error, noise floor = its std (DVC Challenge 2.0 Eq. 1-2).").format(k=nf.frame + 1)
        )
