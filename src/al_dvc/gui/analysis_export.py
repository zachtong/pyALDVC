"""The exports of the Statistics tab: the table to the clipboard, the data of the tab on screen as CSV, a JSON
summary of everything up to date, the chart on screen as PNG. A mixin of :class:`AnalysisTab`."""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QFileDialog

from al_dvc.analysis.export_stats import (
    stats_metadata,
    write_extensometer_csv,
    write_line_csv,
    write_profile_csv,
    write_region_series_csv,
    write_regions_csv,
    write_series_csv,
    write_summary_json,
)

from .widgets import headless

TAB_SUMMARY, TAB_HIST, TAB_SERIES, TAB_REGIONS, TAB_PROFILE, TAB_LINE, TAB_NOISE, TAB_HOM = range(8)


class AnalysisExportMixin:
    """Exports of :class:`~al_dvc.gui.analysis_tab.AnalysisTab` (which provides the widgets and the results)."""

    def _target(self, name: str, flt: str) -> Path | None:
        out = Path(self._state.output_dir)
        if headless():
            out.mkdir(parents=True, exist_ok=True)
            return out / name
        path, _ = QFileDialog.getSaveFileName(self, self.tr("Export statistics"), str(out / name), flt)
        return Path(path) if path else None

    def _meta(self) -> dict:
        stats_r, fit_r = self.stats_region(), self.fit_region()
        return stats_metadata(
            self._state.results,
            self.node_filter(),
            self.motion_kind(),
            region="whole" if stats_r is None else stats_r.name,
            fit_region="whole" if fit_r is None else fit_r.name,
        )

    def copy_table(self) -> str:
        """The table on screen (the summary, or the comparison of the regions) as tab-separated text, also put on
        the clipboard."""
        table = self.regions_table if self.results_tabs.currentIndex() == TAB_REGIONS else self.table
        headers = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
        lines = ["\t".join([""] + headers)]
        for r in range(table.rowCount()):
            head = table.verticalHeaderItem(r)
            cells = [table.item(r, c).text() if table.item(r, c) else "" for c in range(table.columnCount())]
            lines.append("\t".join([head.text() if head else ""] + cells))
        text = "\n".join(lines)
        QGuiApplication.clipboard().setText(text)
        self.status.setText(self.tr("Table copied."))
        return text

    def _csv_kind(self) -> str:
        return {TAB_REGIONS: "regions", TAB_PROFILE: "profile", TAB_LINE: "line"}.get(self.results_tabs.currentIndex(), "series")

    def _csv_ready(self, kind: str) -> bool:
        if kind == "series":
            return self.series_is_current()
        cur = self.current
        if cur is None or cur.error:
            return False
        return {"regions": bool(cur.region_rows), "profile": cur.profile is not None, "line": cur.line is not None}[kind]

    def export_csv(self, path: str | Path | None = None) -> Path | None:
        """The data of the tab on screen as CSV: the series over frames (one row per frame, per region when they
        are compared), the regions of the current frame, the profile (every frame when computed) or the line (and
        the extensometer and the end points next to it, when computed over the frames)."""
        kind = self._csv_kind()
        if not self._csv_ready(kind):
            self.status.setText(
                self.tr("Compute the series over frames first (tab Over frames).")
                if kind == "series"
                else self.tr("Nothing to save yet.")
            )
            return None
        default = {
            "series": "statistics.csv",
            "regions": "statistics_regions.csv",
            "profile": "statistics_profile.csv",
            "line": "statistics_line.csv",
        }
        target = Path(path) if path else self._target(default[kind], "CSV (*.csv)")
        if target is None:
            return None
        meta = self._meta()
        cur = self.current
        try:
            if kind == "series":
                runs = self.series[1]
                if len(runs) == 1:
                    out = write_series_csv(target, runs[0][1], self.fields(), meta)
                else:
                    named = [(self.tr("All nodes") if r is None else r.name, s) for r, s in runs]
                    out = write_region_series_csv(target, named, self.fields(), meta)
            elif kind == "regions":
                rows = [(label, st) for label, _c, st in self._region_rows(cur)]
                out = write_regions_csv(target, rows, cur.compare_field, self.frame(), meta)
            elif kind == "profile":
                profiles = self.long["profiles"][1] if self.long_is_current("profiles") else [(self.frame(), cur.profile)]
                out = write_profile_csv(target, profiles, cur.compare_field, meta)
            else:
                p0, p1 = self.line_points()
                series = self.long["line"][1] if self.long_is_current("line") else None
                if series is None:
                    lines = [(self.frame(), cur.line[0], cur.line[1])]
                else:
                    lines = [(int(k), series.distance, row) for k, row in zip(series.frames, series.values)]
                out = write_line_csv(target, lines, cur.compare_field, p0, p1, meta)
                if series is not None:
                    ext_path = out.with_name(out.stem + "_extensometer.csv")
                    write_extensometer_csv(ext_path, series.extensometer, meta, field=series.field, ends=series.ends)
                    self._state.log(self.tr("Statistics saved: {path}").format(path=ext_path))
        except OSError as exc:
            self.status.setText(self.tr("Cannot write the file: {error}").format(error=exc))
            return None
        self._state.log(self.tr("Statistics saved: {path}").format(path=out))
        return out

    def export_json(self, path: str | Path | None = None) -> Path | None:
        """The current frame (with the regions, the profile and the line) and every long result that is up to
        date, as JSON."""
        cur = self.current
        if cur is None or cur.selection is None:
            return None
        target = Path(path) if path else self._target("statistics.json", "JSON (*.json)")
        if target is None:
            return None
        payload: dict = {
            "meta": self._meta(),
            "regions": [r.as_dict() for r in self._state.regions],
            "frame": {
                "frame": self.frame() + 1,
                **cur.selection.as_dict(),
                "motion": None if cur.fit is None else cur.fit.as_dict(),
                "stats": cur.stats,
            },
            "homogeneous": cur.homogeneous,
            "compare": {
                "field": cur.compare_field,
                "regions": [{"region": label, "stats": st} for label, _c, st in self._region_rows(cur)],
            },
            "profile": cur.profile,
        }
        if cur.line is not None:
            p0, p1 = self.line_points()
            payload["line"] = {"p0_voxel": list(p0), "p1_voxel": list(p1), "distance": cur.line[0], "values": cur.line[1]}
        if self.series_is_current():
            runs = self.series[1]
            if len(runs) == 1:
                payload["frames"] = runs[0][1]
            else:
                payload["frames_by_region"] = {(self.tr("All nodes") if r is None else r.name): s for r, s in runs}
        if self.noise_is_current():
            payload["noise_floor"] = self.noise[1]
        if self.long_is_current("profiles"):
            payload["profiles"] = [{"frame": k + 1, **p.as_dict()} for k, p in self.long["profiles"][1]]
        if self.long_is_current("line"):
            payload["line_over_frames"] = self.long["line"][1]
        try:
            out = write_summary_json(target, payload)
        except (OSError, ValueError) as exc:
            self.status.setText(self.tr("Cannot write the file: {error}").format(error=exc))
            return None
        self._state.log(self.tr("Statistics saved: {path}").format(path=out))
        return out

    def _chart(self) -> tuple[Figure, str] | None:
        return {
            TAB_HIST: (self.hist_figure, "statistics_histogram.png"),
            TAB_SERIES: (self.series_figure, "statistics_series.png"),
            TAB_REGIONS: (self.regions_figure, "statistics_regions.png"),
            TAB_PROFILE: (self.profile_figure, "statistics_profile.png"),
            TAB_LINE: (self.line_figure, "statistics_line.png"),
        }.get(self.results_tabs.currentIndex())

    def export_png(self, path: str | Path | None = None) -> Path | None:
        """The chart on screen as PNG."""
        chart = self._chart()
        if chart is None:
            return None
        figure, name = chart
        target = Path(path) if path else self._target(name, "PNG (*.png)")
        if target is None:
            return None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(target, dpi=150, facecolor=figure.get_facecolor())
        except OSError as exc:
            self.status.setText(self.tr("Cannot write the file: {error}").format(error=exc))
            return None
        self._state.log(self.tr("Statistics saved: {path}").format(path=target))
        return target
