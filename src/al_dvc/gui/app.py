"""pyALDVC main window and application entry point (``al-dvc-gui``).

Layout: a left column with the volume list, the parameters and the run
controls; the three-plane slice viewer in the centre; result display and
export controls on the right. Every panel talks to one :class:`AppState` and
reacts to its signals, so the panels are independent of each other.
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import QEvent, QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from al_dvc import __version__

from .app_state import AppState, RunState
from .i18n import SUPPORTED_LANGUAGES, LanguageManager
from .kernel_warmup import START_DELAY_MS, KernelWarmup
from .panels.param_panel import ParamPanel
from .panels.results_panel import ResultsPanel
from .panels.run_panel import RunPanel
from .panels.view3d import View3DPanel
from .panels.viewer import SliceViewer
from .panels.volume_panel import VolumePanel
from .session import SESSION_SUFFIX, SessionError, apply_session, load_session, save_session
from .theme import build_stylesheet
from .widgets import ConsoleLog
from .window_chrome import enable_dark_title_bar

logger = logging.getLogger(__name__)

SESSION_FILTER = f"pyALDVC session (*{SESSION_SUFFIX})"
LEFT_MIN_WIDTH = 420  # px: the widest row of the left column fits beside the scrollbar without clipping
RIGHT_MIN_WIDTH = 340
MIN_WINDOW_WIDTH = 1200
MIN_WINDOW_HEIGHT = 680
SETTINGS_ORG = "pyALDVC"
SETTINGS_APP = "gui"


class _Section(QWidget):
    """A titled block of the left column."""

    def __init__(self, title: str, body: QWidget) -> None:
        super().__init__()
        self.label = QLabel(title)
        self.label.setObjectName("sectionTitle")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.addWidget(self.label)
        layout.addWidget(body)


class MainWindow(QMainWindow):
    """The application window."""

    def __init__(self, state: AppState | None = None) -> None:
        super().__init__()
        self.state = state or AppState()
        enable_dark_title_bar(self)
        self.resize(1440, 900)

        self.volume_panel = VolumePanel(self.state)
        self.param_panel = ParamPanel(self.state)
        self.run_panel = RunPanel(self.state)
        self.viewer = SliceViewer(self.state)
        self.view3d = View3DPanel(self.state)
        self.center_tabs = QTabWidget()
        self.center_tabs.setObjectName("viewTabs")
        self.center_tabs.tabBar().setExpanding(True)
        self.center_tabs.addTab(self.viewer, "")
        self.center_tabs.addTab(self.view3d, "")
        self.results_panel = ResultsPanel(self.state)

        self.console = ConsoleLog()
        self._sections = {
            "volumes": _Section("", self.volume_panel),
            "roi": _Section("", self.viewer.mask_tools),
            "parameters": _Section("", self.param_panel),
            "run": _Section("", self.run_panel),
        }
        # left column: data and parameters (scrolls; wide enough for every button row)
        left_body = QWidget()
        left_layout = QVBoxLayout(left_body)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_layout.addWidget(self._sections["volumes"])
        left_layout.addWidget(self._sections["roi"])
        left_layout.addWidget(self._sections["parameters"])
        left_layout.addStretch(1)
        left = QScrollArea()
        left.setWidgetResizable(True)
        left.setWidget(left_body)
        left.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left.setMinimumWidth(LEFT_MIN_WIDTH)
        # right column: run controls on top, results in the middle, the console at the bottom (pyALDIC layout)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        right_layout.addWidget(self._sections["run"])
        results_scroll = QScrollArea()
        results_scroll.setWidgetResizable(True)
        results_scroll.setWidget(self.results_panel)
        results_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_layout.addWidget(results_scroll, 1)
        console_box = QWidget()
        console_layout = QVBoxLayout(console_box)
        console_layout.setContentsMargins(8, 0, 8, 6)
        console_layout.addWidget(self.console)
        right_layout.addWidget(console_box, 0)
        right.setMinimumWidth(RIGHT_MIN_WIDTH)
        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self.center_tabs)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([LEFT_MIN_WIDTH + 20, 760, RIGHT_MIN_WIDTH + 20])
        self.setCentralWidget(splitter)
        self._splitter = splitter
        self._left_column = left
        self._right_column = right
        self.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
        self.state.log_message.connect(self.console.append_log)
        self.results_panel.strain_requested.connect(self._on_strain)
        self.results_panel.export_requested.connect(self._on_export_requested)

        self._actions: dict[str, QAction] = {}
        self._menus = {}
        self._build_menu_bar()
        self.state.progress_updated.connect(lambda _f, msg: self.statusBar().showMessage(msg))
        self.state.run_state_changed.connect(self._on_run_state_changed)
        self.state.log_message.connect(self._on_log_message)
        self.retranslate_ui()
        QTimer.singleShot(START_DELAY_MS + 1500, self.param_panel.refresh_backend_status)

    # ------------------------------------------------------------------ messages
    @property
    def headless(self) -> bool:
        """True under the offscreen platform (tests, self-test): dialogs would block forever."""
        app = QApplication.instance()
        return app is not None and app.platformName() == "offscreen"

    def _message(self, kind: str, title: str, text: str) -> bool:
        """Show a dialog, or log it when headless. ``question`` returns the answer (headless: True)."""
        if self.headless:
            self.state.log(f"{title}: {text}", "warning" if kind in ("warning", "critical") else "info")
            return True
        if kind == "question":
            return QMessageBox.question(self, title, text) == QMessageBox.StandardButton.Yes
        getattr(QMessageBox, kind)(self, title, text)
        return True

    # ------------------------------------------------------------------ menus
    def _build_menu_bar(self) -> None:
        bar = self.menuBar()
        self._menus["file"] = bar.addMenu("")
        for key, slot in [
            ("new", self._on_new_session),
            ("open", self._on_open_session),
            ("save", self._on_save_session),
            ("save_as", self._on_save_session_as),
            ("add_volumes", self.volume_panel._on_add_files),
            ("batch", self._on_batch),
            ("exit", self.close),
        ]:
            act = QAction(self)
            act.triggered.connect(slot)
            std = {
                "new": QKeySequence.StandardKey.New,
                "open": QKeySequence.StandardKey.Open,
                "save": QKeySequence.StandardKey.Save,
            }
            if key in std:
                act.setShortcut(std[key])
            elif key == "save_as":
                act.setShortcut(QKeySequence("Ctrl+Shift+S"))
            self._actions[key] = act
            self._menus["file"].addAction(act)
            if key in ("save_as", "batch"):
                self._menus["file"].addSeparator()
        self._menus["view"] = bar.addMenu("")
        for key, slot, shortcut in [
            ("left_column", self._toggle_left, "Ctrl+1"),
            ("right_column", self._toggle_right, "Ctrl+2"),
            ("reset_layout", self.reset_layout, ""),
        ]:
            act = QAction(self)
            act.triggered.connect(slot)
            if key != "reset_layout":
                act.setCheckable(True)
                act.setChecked(True)
            if shortcut:
                act.setShortcut(QKeySequence(shortcut))
            self._actions[key] = act
            self._menus["view"].addAction(act)
        self._menus["view"].addSeparator()
        self._menus["language"] = self._menus["view"].addMenu("")
        group = QActionGroup(self)
        group.setExclusive(True)
        for code, name in SUPPORTED_LANGUAGES.items():
            act = QAction(name, self)
            act.setCheckable(True)
            act.setData(code)
            act.triggered.connect(lambda _c=False, c=code: self._on_language_selected(c))
            group.addAction(act)
            self._menus["language"].addAction(act)
            self._actions[f"lang_{code}"] = act
        self._menus["analysis"] = bar.addMenu("")
        for key, slot, shortcut in [
            ("run", self.run_panel.start, "F5"),
            ("stop", self.run_panel.stop, "Esc"),
            ("strain", self._on_strain, "Ctrl+T"),
            ("export", self._on_export_requested, "Ctrl+E"),
        ]:
            act = QAction(self)
            act.triggered.connect(slot)
            act.setShortcut(QKeySequence(shortcut))
            self._actions[key] = act
            self._menus["analysis"].addAction(act)
            if key == "stop":
                self._menus["analysis"].addSeparator()
        self._menus["help"] = bar.addMenu("")
        for key, slot in [("self_test", self._on_self_test), ("about", self._on_about)]:
            act = QAction(self)
            act.triggered.connect(slot)
            self._actions[key] = act
            self._menus["help"].addAction(act)
        self._sync_language_check()

    def _sync_language_check(self) -> None:
        mgr = _language_manager()
        code = mgr.code if mgr is not None else "en"
        for key, act in self._actions.items():
            if key.startswith("lang_"):
                act.setChecked(key == f"lang_{code}")

    def _on_language_selected(self, code: str) -> None:
        mgr = _language_manager()
        if mgr is not None:
            mgr.load(code)

    # ------------------------------------------------------------------ sessions
    def _on_new_session(self) -> None:
        if self.state.run_state in (RunState.RUNNING, RunState.STOPPING):
            return
        self.state.reset()
        self.setWindowTitle(f"pyALDVC {__version__}")

    def _on_open_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, self.tr("Open session"), "", SESSION_FILTER)
        if path:
            self.open_session_path(path)

    def open_session_path(self, path: str) -> list[str]:
        try:
            data = load_session(path)
            missing = apply_session(data, self.state, path)
        except SessionError as exc:
            self._message("critical", self.tr("Cannot open session"), str(exc))
            return []
        self.setWindowTitle(f"pyALDVC {__version__} - {Path(path).name}")
        if missing:
            self._message(
                "warning",
                self.tr("Missing volumes"),
                self.tr("{n} volume file(s) of the session were not found:\n{files}").format(
                    n=len(missing), files="\n".join(missing[:8])
                ),
            )
        self.state.log(self.tr("Session loaded: {path}").format(path=path))
        return missing

    def _on_save_session(self) -> None:
        if self.state.session_path is None:
            self._on_save_session_as()
        else:
            self.save_session_path(self.state.session_path)

    def _on_save_session_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, self.tr("Save session"), "", SESSION_FILTER)
        if path:
            self.save_session_path(path)

    def save_session_path(self, path: str | Path) -> Path | None:
        results_path = None
        if self.state.results is not None:
            candidate = Path(self.state.output_dir) / "aldvc.npz"
            if candidate.exists():
                results_path = str(candidate)
        try:
            p = save_session(self.state, path, results_path)
        except SessionError as exc:
            self._message("critical", self.tr("Cannot save session"), str(exc))
            return None
        self.setWindowTitle(f"pyALDVC {__version__} - {p.name}")
        self.state.log(self.tr("Session saved: {path}").format(path=p))
        return p

    # ------------------------------------------------------------------ help
    # ------------------------------------------------------------------ layout
    def _toggle_left(self, visible: bool) -> None:
        self._left_column.setVisible(bool(visible))

    def _toggle_right(self, visible: bool) -> None:
        self._right_column.setVisible(bool(visible))

    def reset_layout(self) -> None:
        """Default column widths and both columns visible (also clears the remembered geometry)."""
        for key in ("left_column", "right_column"):
            self._actions[key].setChecked(True)
        self._left_column.setVisible(True)
        self._right_column.setVisible(True)
        self._splitter.setSizes([LEFT_MIN_WIDTH + 20, 760, RIGHT_MIN_WIDTH + 20])
        self.resize(1440, 900)
        QSettings(SETTINGS_ORG, SETTINGS_APP).remove("main")

    def restore_layout(self) -> None:
        """Window geometry and splitter sizes from the previous session (QSettings), when any."""
        settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        geometry = settings.value("main/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        sizes = settings.value("main/splitter")
        if sizes:
            try:
                self._splitter.setSizes([int(s) for s in sizes])
            except (TypeError, ValueError):
                pass

    def save_layout(self) -> None:
        settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        settings.setValue("main/geometry", self.saveGeometry())
        settings.setValue("main/splitter", [int(s) for s in self._splitter.sizes()])

    def open_strain_window(self):
        """The (single) strain post-processing window, created on first use and raised afterwards."""
        from .strain_window import StrainWindow

        window = getattr(self, "strain_window", None)
        if window is None:
            window = StrainWindow(self.state, self)
            self.strain_window = window
            window.export_requested.connect(lambda: self.open_export_dialog(preselect_strain=True))
        window.show()
        window.raise_()
        window.activateWindow()
        return window

    def _on_strain(self) -> None:
        self.open_strain_window()

    def open_export_dialog(self, preselect_strain: bool = False):
        """The (single) export dialog, created on first use and raised afterwards."""
        from .dialogs.export_dialog import ExportDialog

        dialog = getattr(self, "export_dialog", None)
        if dialog is None:
            dialog = ExportDialog(self.state, self, preselect_strain=preselect_strain)
            self.export_dialog = dialog
        dialog.refresh()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return dialog

    def _on_export_requested(self) -> None:
        """Export entry point shared by the results panel, the menu and the strain window."""
        if self.state.results is None:
            self.state.log(self.tr("Nothing to export yet: run an analysis first."), "warning")
            return
        self.open_export_dialog()

    def open_batch_dialog(self):
        """The (single) batch dialog, created on first use and shown non-modally."""
        from .dialogs.batch_dialog import BatchDialog

        dialog = getattr(self, "batch_dialog", None)
        if dialog is None:
            dialog = BatchDialog(self.state, self, open_session=self.open_session_path)
            self.batch_dialog = dialog
        dialog.show()
        dialog.raise_()
        return dialog

    def _on_batch(self) -> None:
        self.open_batch_dialog()

    def _on_self_test(self) -> None:
        from .self_test import run_self_test

        report = Path(self.state.output_dir) / "self_test.txt"
        ok = run_self_test(report) == 0
        self._message(
            "information",
            self.tr("Self-test"),
            (self.tr("All checks passed.") if ok else self.tr("Some checks failed.")) + f"\n{report}",
        )

    def _on_about(self) -> None:
        self._message(
            "about",
            self.tr("About pyALDVC"),
            self.tr(
                "pyALDVC {version}\nAugmented Lagrangian Digital Volume Correlation.\n"
                "https://github.com/zachtong/pyALDVC\nBSD 3-Clause license."
            ).format(version=__version__),
        )

    # ------------------------------------------------------------------ events
    def _on_run_state_changed(self, state: RunState) -> None:
        suffix = {RunState.RUNNING: self.tr(" [running]"), RunState.STOPPING: self.tr(" [stopping]")}.get(state, "")
        base = f"pyALDVC {__version__}"
        if self.state.session_path is not None:
            base += f" - {self.state.session_path.name}"
        self.setWindowTitle(base + suffix)

    def _on_log_message(self, message: str, level: str) -> None:
        if level == "error":
            self.statusBar().showMessage(message, 10000)

    def changeEvent(self, event) -> None:  # noqa: N802
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslate_ui()
            for panel in (self.volume_panel, self.param_panel, self.run_panel, self.viewer, self.results_panel, self.console):
                panel.retranslate_ui()
            window = getattr(self, "strain_window", None)
            if window is not None:
                window.retranslate_ui()
            dialog = getattr(self, "export_dialog", None)
            if dialog is not None:
                dialog.retranslate_ui()
        super().changeEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        if not self.headless:
            self.save_layout()
        dialog = getattr(self, "batch_dialog", None)
        if dialog is not None and dialog.running:
            dialog.stop()
            dialog.wait(60_000)
        if self.state.run_state in (RunState.RUNNING, RunState.STOPPING):
            if not self._message("question", self.tr("Analysis running"), self.tr("A run is in progress. Stop it and close?")):
                event.ignore()
                return
            self.run_panel.stop()
            self.run_panel.wait(60_000)
        event.accept()

    def retranslate_ui(self) -> None:
        self._sections["volumes"].label.setText(self.tr("Volumes"))
        self._sections["parameters"].label.setText(self.tr("Parameters"))
        self._sections["run"].label.setText(self.tr("Run"))
        self._sections["roi"].label.setText(self.tr("Region of interest"))
        self.center_tabs.setTabText(0, self.tr("Slices"))
        self.center_tabs.setTabText(1, self.tr("3-D view"))
        self._menus["file"].setTitle(self.tr("&File"))
        self._menus["view"].setTitle(self.tr("&View"))
        self._menus["language"].setTitle(self.tr("Language"))
        self._menus["analysis"].setTitle(self.tr("&Analysis"))
        self._menus["help"].setTitle(self.tr("&Help"))
        texts = {
            "new": self.tr("New session"),
            "open": self.tr("Open session..."),
            "save": self.tr("Save session"),
            "save_as": self.tr("Save session as..."),
            "add_volumes": self.tr("Add volumes..."),
            "batch": self.tr("Batch run..."),
            "exit": self.tr("Exit"),
            "left_column": self.tr("Data and parameters column"),
            "right_column": self.tr("Results column"),
            "reset_layout": self.tr("Reset window layout"),
            "run": self.tr("Run AL-DVC"),
            "stop": self.tr("Stop"),
            "strain": self.tr("Strain post-processing..."),
            "export": self.tr("Export results..."),
            "self_test": self.tr("Run self-test"),
            "about": self.tr("About pyALDVC"),
        }
        for key, text in texts.items():
            self._actions[key].setText(text)
        self._on_run_state_changed(self.state.run_state)
        self._sync_language_check()
        dialog = getattr(self, "batch_dialog", None)
        if dialog is not None:
            dialog.retranslate_ui()


# ---------------------------------------------------------------------- application
def _language_manager() -> LanguageManager | None:
    app = QApplication.instance()
    return getattr(app, "_pyaldvc_lang_mgr", None) if app is not None else None


def configure_matplotlib() -> None:
    """Canvas fonts and sizes consistent with the Qt theme (Segoe UI / YaHei, small labels)."""
    import matplotlib

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.dpi": 100,
        }
    )


def user_data_dir() -> Path:
    base = Path.home() / (".pyaldvc" if sys.platform != "win32" else "AppData/Local/pyALDVC")
    base.mkdir(parents=True, exist_ok=True)
    return base


def _configure_logging() -> None:
    log_path = user_data_dir() / "pyaldvc.log"
    # A windowed (console-less) frozen executable has no standard streams.
    handlers: list[logging.Handler] = [logging.StreamHandler()] if sys.stderr is not None else []
    try:
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    except OSError:
        pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=handlers)


def _global_exception_hook(exc_type, exc_value, exc_tb) -> None:
    text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    logger.error("Unhandled exception:\n%s", text)
    app = QApplication.instance()
    if app is not None:
        QMessageBox.critical(None, "pyALDVC", f"{exc_type.__name__}: {exc_value}\n\n{text[-1500:]}")


def create_application(argv: list[str] | None = None) -> QApplication:
    """A configured ``QApplication`` (style, stylesheet, language)."""
    configure_matplotlib()
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setOrganizationName("pyALDVC")
    app.setApplicationName("pyALDVC")
    app.setStyle("Fusion")
    app.setStyleSheet(build_stylesheet())
    if getattr(app, "_pyaldvc_lang_mgr", None) is None:
        mgr = LanguageManager(app)
        mgr.load(LanguageManager.resolve_language())
        app._pyaldvc_lang_mgr = mgr  # type: ignore[attr-defined]
    return app


def _session_path_from_argv(argv: list[str]) -> str | None:
    for arg in argv[1:]:
        if arg.endswith(SESSION_SUFFIX) and Path(arg).is_file():
            return arg
    return None


def main(argv: list[str] | None = None) -> int:
    """Launch the GUI (``al-dvc-gui [session.aldvc] [--self-test]``)."""
    import multiprocessing

    multiprocessing.freeze_support()
    argv = list(sys.argv if argv is None else argv)
    if "--self-test" in argv:
        from .self_test import run_self_test

        i = argv.index("--self-test")
        report = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("-") else "pyaldvc_self_test.txt"
        return run_self_test(Path(report))
    _configure_logging()
    sys.excepthook = _global_exception_hook
    app = create_application(argv)
    window = MainWindow()
    window.show()
    warmup = KernelWarmup(window)
    warmup.compiled.connect(lambda s: window.state.log(window.tr("Kernels compiled in {s:.0f} s").format(s=s)))
    QTimer.singleShot(START_DELAY_MS, warmup.start)
    session = _session_path_from_argv(argv)
    if session:
        QTimer.singleShot(0, lambda: window.open_session_path(session))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
