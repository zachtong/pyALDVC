"""Light and dark themes: palettes and stylesheet, the Settings menu, live switching, the saved choice."""

import hashlib
import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.name == "nt":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

from matplotlib.colors import to_hex  # noqa: E402
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from al_dvc.gui.app import MainWindow, create_application  # noqa: E402
from al_dvc.gui.theme import (  # noqa: E402
    ARROWS_DIR,
    DARK,
    DEFAULT_THEME,
    LIGHT,
    SIDE_COLUMN,
    THEMES,
    build_stylesheet,
    current_colors,
    current_theme,
    set_current_theme,
    style_text,
)
from al_dvc.gui.theme_manager import ENV_THEME, SETTINGS_KEY, startup_theme, theme_manager, themed  # noqa: E402

# the dark stylesheet of release 1.0.1 (arrow folder written "<arrows>"); scripts/make_theme_report.py checks it too
DARK_101_SHA256 = "0eeab19fa14d850782cff78c637b598cd6460d8384de4957c3321405d37a0efc"


@pytest.fixture(scope="module")
def qapp():
    return create_application(["pytest"])


@pytest.fixture(scope="module", autouse=True)
def no_leftover_windows(qapp):
    """Delete the hidden windows earlier test modules left in this process. A theme switch restyles every
    widget of the application; in one process (CI runs the suite so) the closed windows of a few hundred tests
    added up to thousands of widgets, and a switch took minutes."""
    from PySide6.QtCore import QEvent

    for widget in QApplication.topLevelWidgets():
        if not widget.isVisible():
            widget.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    yield


@pytest.fixture
def dark_again(qapp):
    """Every test starts dark with nothing saved, and leaves it that way (other modules expect the default)."""
    QSettings().remove(SETTINGS_KEY)
    theme_manager().set_theme("dark", persist=False)
    yield theme_manager()
    theme_manager().set_theme("dark", persist=False)
    QSettings().remove(SETTINGS_KEY)


def _pump(n: int = 10) -> None:
    for _ in range(n):
        QApplication.processEvents()


def _luminance(hex_colour: str) -> float:
    rgb = [int(hex_colour.lstrip("#")[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
    lin = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(a: str, b: str) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _icon_colours(icon, size: int = 18) -> set[str]:
    image = icon.pixmap(size).toImage()
    return {
        image.pixelColor(x, y).name()
        for x in range(image.width())
        for y in range(image.height())
        if image.pixelColor(x, y).alpha() == 255
    }


# ---------------------------------------------------------------------- palettes and stylesheet (no window)
def test_the_dark_theme_is_the_default_and_its_stylesheet_is_the_original_one(dark_again, monkeypatch):
    monkeypatch.delenv(ENV_THEME, raising=False)
    assert set(THEMES) == {"dark", "light"} and THEMES[DEFAULT_THEME] is DARK
    assert startup_theme() == "dark"  # nothing saved
    assert current_theme() == "dark" and current_colors() is DARK
    css = build_stylesheet(DARK)
    assert css == build_stylesheet()  # the theme in use
    # the original dark theme: no transparent containers, no tinted side columns, the style's own table selection
    assert SIDE_COLUMN not in css and "QComboBoxPrivateContainer" not in css
    assert "\nQAbstractItemView {" not in css and "#0b0f1a" in css and "spin_up.svg" in css
    with pytest.raises(ValueError):
        set_current_theme("sepia")


def test_the_dark_stylesheet_is_the_one_of_release_1_0_1():
    """The dark theme is kept byte for byte; change the hash only when the dark theme is changed on purpose."""
    css = build_stylesheet(DARK).replace(ARROWS_DIR.as_posix(), "<arrows>")  # the arrows live where al_dvc is installed
    assert hashlib.sha256(css.encode("utf-8")).hexdigest() == DARK_101_SHA256


def test_the_light_stylesheet_is_white_with_grey_side_columns_and_dark_arrows():
    css = build_stylesheet(LIGHT)
    assert "QMainWindow,\nQWidget {\n    background: #ffffff;" in css  # the main background
    assert f"QWidget#{SIDE_COLUMN} {{\n    background: #f6f7f9;" in css  # the side columns
    assert "selection-background-color: #e0e7ff;" in css and "selection-color: #111827;" in css
    assert "color: #111827;" in css and "#0b0f1a" not in css and "#e2e8f0" not in css  # nothing of the dark theme
    for c in (DARK, LIGHT):  # every arrow the stylesheets name is shipped
        for name in (c.ARROW_UP, c.ARROW_DOWN, c.ARROW_UP_HOVER, c.ARROW_DOWN_HOVER, c.COMBO_ARROW_HOVER):
            assert (ARROWS_DIR / name).is_file(), name
            assert (ARROWS_DIR / name).as_posix() in build_stylesheet(c)


def test_the_light_palette_is_readable():
    c = LIGHT
    for bg in (c.BG_DARKEST, c.BG_SIDE_COLUMN, c.BG_PANEL, c.BG_INPUT, c.BG_HOVER, c.SELECTION):
        assert _contrast(c.TEXT_PRIMARY, bg) >= 7.0, bg
        assert _contrast(c.TEXT_SECONDARY, bg) >= 4.5, bg
        assert _contrast(c.TEXT_MUTED, bg) >= 3.0, bg  # hints: small, but never faint
    assert _contrast(c.CANVAS_TEXT, c.BG_CANVAS) >= 7.0  # ticks and labels on the white canvas
    for accent in (c.ACCENT, c.ACCENT_PRESSED):
        assert _contrast("#ffffff", accent) >= 4.5  # white text on the accent buttons
    assert _contrast("#ffffff", c.ACCENT_HOVER) >= 4.4  # under the pointer: the dark theme's own accent
    for semantic in (c.DANGER, c.WARNING):
        assert _contrast(semantic, c.BG_DARKEST) >= 4.5
    assert _contrast(c.SUCCESS, c.BG_DARKEST) >= 3.0


def test_style_templates_take_the_colours_of_the_theme():
    template = "QLabel {{ color: {TEXT_PRIMARY}; border: 1px solid {BORDER}; }}"
    assert style_text(template, DARK) == "QLabel { color: #e2e8f0; border: 1px solid #1e293b; }"
    assert style_text(template, LIGHT) == "QLabel { color: #111827; border: 1px solid #d1d5db; }"


# ---------------------------------------------------------------------- the Settings menu
def test_settings_menu_holds_theme_language_and_notifications(qapp, dark_again):
    window = MainWindow()
    try:
        titles = [a.text() for a in window.menuBar().actions()]
        assert titles == ["&File", "&View", "&Settings", "&Analysis", "&Help"]  # Settings right after View
        settings = window._menus["settings"]
        entries = settings.actions()
        assert window._menus["theme"].menuAction() in entries and window._menus["language"].menuAction() in entries
        assert window._actions["notify"] in entries
        assert window._menus["theme"].title() == "Theme" and window._menus["language"].title() == "Language"
        themes = window._menus["theme"].actions()
        assert [a.text() for a in themes] == ["Dark", "Light"]
        assert all(a.isCheckable() for a in themes) and themes[0].actionGroup().isExclusive()
        assert window._actions["theme_dark"].isChecked() and not window._actions["theme_light"].isChecked()
        assert len(window._menus["language"].actions()) == 7
        view = window._menus["view"].actions()  # moved out of View
        assert window._actions["notify"] not in view and window._menus["language"].menuAction() not in view
        assert window._menus["theme"].menuAction() not in view
        assert [a.text() for a in view if a.text()] == ["Data and parameters column", "Results column", "Reset window layout"]
        notify = window._actions["notify"]  # same toggle and key as before, now under Settings
        assert notify.text() == "Notify when a task finishes" and notify.isCheckable()
        assert notify.isChecked() == window.notifications_enabled  # (not toggled here: that key is the user's own)
    finally:
        window.close()


def test_the_settings_menu_is_translated(qapp, dark_again):
    window = MainWindow()
    mgr = qapp._pyaldvc_lang_mgr
    try:
        mgr.load("de")
        _pump()
        assert window._menus["settings"].title() == "&Einstellungen" and window._menus["theme"].title() == "Farbschema"
        assert window._actions["theme_light"].text() == "Hell" and window._actions["theme_dark"].text() == "Dunkel"
    finally:
        mgr.load("en")
        _pump()
        window.close()
    assert window._menus["settings"].title() == "&Settings"


# ---------------------------------------------------------------------- live switching
def test_switching_to_light_and_back_restyles_the_application(qapp, dark_again):
    window = MainWindow()
    window.show()
    rng = np.random.default_rng(1)
    vol = (rng.random((24, 28, 32)) * 255).astype(np.uint8)
    window.state.set_volume_arrays([vol, vol], ["a", "b"])
    window.state.log("check the colours", "warning")
    _pump()
    dark_css = qapp.styleSheet()
    assert dark_css == build_stylesheet(DARK)
    ax = window.viewer.axes[0]
    assert to_hex(window.viewer.figure.get_facecolor()) == DARK.BG_CANVAS
    button = window.viewer.mask_tools.tool_buttons["rectangle"]
    assert _icon_colours(button.icon()) == {DARK.TEXT_SECONDARY}
    try:
        window._actions["theme_light"].trigger()  # the menu entry: live, no restart
        _pump()
        assert current_theme() == "light" and qapp.styleSheet() == build_stylesheet(LIGHT)
        assert "background: #ffffff;" in qapp.styleSheet()
        assert window._actions["theme_light"].isChecked() and not window._actions["theme_dark"].isChecked()
        assert QSettings().value(SETTINGS_KEY) == "light"  # remembered for the next start
        # the slice viewer: white canvas, black ticks, labels, spines and titles
        ax = window.viewer.axes[0]
        tick = ax.xaxis.get_major_ticks()[0]
        assert to_hex(window.viewer.figure.get_facecolor()) == "#ffffff" and to_hex(ax.get_facecolor()) == "#ffffff"
        assert to_hex(tick.label1.get_color()) == LIGHT.CANVAS_TEXT == "#111827"
        assert to_hex(tick.tick1line.get_color()) == "#111827" and to_hex(ax.title.get_color()) == "#111827"
        assert to_hex(ax.spines["left"].get_edgecolor()) == "#111827"
        # icons, the console and the side-column overlay follow
        assert _icon_colours(button.icon()) == {LIGHT.TEXT_SECONDARY}
        html = window.console._view.toHtml()
        assert LIGHT.WARNING in html and DARK.WARNING not in html
        assert LIGHT.BG_SIDE_COLUMN in window.sticky_headers.styleSheet()
        assert window.view3d.background_key() == "white"  # the 3-D view's default background follows
        # and back: the dark theme exactly as before
        window._actions["theme_dark"].trigger()
        _pump()
        assert qapp.styleSheet() == dark_css
        assert to_hex(window.viewer.figure.get_facecolor()) == DARK.BG_CANVAS
        assert to_hex(window.viewer.axes[0].xaxis.get_major_ticks()[0].label1.get_color()) == DARK.CANVAS_TEXT
        assert _icon_colours(button.icon()) == {DARK.TEXT_SECONDARY}
        assert window.view3d.background_key() == "dark" and QSettings().value(SETTINGS_KEY) == "dark"
    finally:
        window.state.dirty = False
        window.close()


def test_secondary_windows_follow_a_live_switch(qapp, dark_again):
    window = MainWindow()
    window.show()
    rng = np.random.default_rng(2)
    vol = (rng.random((32, 32, 32)) * 255).astype(np.uint8)
    window.state.set_volume_arrays([vol, vol], ["a", "b"])
    post = window.open_strain_window()
    texture = window.open_texture_window()
    guide = window.open_guide_window()
    export = window.open_export_dialog()
    batch = window.open_batch_dialog()
    _pump()
    try:
        # the user picks a 3-D background: it stays whatever the theme
        panel = window.view3d
        grey = panel.background.findData("grey")
        panel.background.setCurrentIndex(grey)
        panel.background.activated.emit(grey)
        home = texture.toolbar_profiles._actions["home"]
        assert _icon_colours(home.icon(), 24) == {DARK.TEXT_PRIMARY}  # light toolbar icons on the dark theme
        theme_manager().set_theme("light")
        _pump()
        light = current_colors()
        assert to_hex(post.canvas.figure.get_facecolor()) == "#ffffff"  # post-processing: strain canvas
        assert to_hex(post.analysis.canvas.figure.get_facecolor()) == "#ffffff"  # ... and statistics
        assert to_hex(post.analysis.hist_figure.get_facecolor()) == "#ffffff"
        assert to_hex(post.canvas.axes[0].xaxis.get_major_ticks()[0].label1.get_color()) == light.CANVAS_TEXT
        assert to_hex(texture.region.figure.get_facecolor()) == "#ffffff"  # texture: region slices
        assert texture.plot_background.currentData() == "white"  # plots on white: not chosen by hand
        assert to_hex(texture.fig_profiles.get_facecolor()) == "#ffffff"
        assert light.BG_PANEL in texture.steps.styleSheet() and light.TEXT_PRIMARY in texture._suggestion.styleSheet()
        assert _icon_colours(home.icon(), 24) == {light.TEXT_PRIMARY}  # dark toolbar icons on the light theme
        assert light.TEXT_PRIMARY in guide._text["title"].styleSheet()  # the guide
        assert light.TEXT_SECONDARY in guide._text["acf_formula"].text()
        assert panel.background_key() == "grey"  # the user's choice stays
        for dialog in (export, batch, guide):  # the application stylesheet reaches every window
            assert dialog.palette().color(dialog.backgroundRole()).name() == light.BG_DARKEST
        theme_manager().set_theme("dark")
        _pump()
        assert to_hex(post.canvas.figure.get_facecolor()) == DARK.BG_CANVAS
        assert texture.plot_background.currentData() == "dark" and panel.background_key() == "grey"
        assert DARK.TEXT_PRIMARY in guide._text["title"].styleSheet()
    finally:
        window.state.dirty = False
        window.close()


def test_a_themed_widget_is_restyled_on_a_switch(qapp, dark_again):
    from PySide6.QtWidgets import QLabel

    label = themed(QLabel("x"), "color: {TEXT_PRIMARY};")
    label.show()  # a top-level window of its own
    try:
        assert label.styleSheet() == f"color: {DARK.TEXT_PRIMARY};"
        theme_manager().set_theme("light")
        assert label.styleSheet() == f"color: {LIGHT.TEXT_PRIMARY};"
    finally:
        label.close()


# ---------------------------------------------------------------------- the saved choice
def test_the_saved_theme_is_applied_at_start_up(qapp, dark_again, monkeypatch):
    monkeypatch.delenv(ENV_THEME, raising=False)
    QSettings().setValue(SETTINGS_KEY, "light")
    assert startup_theme() == "light"
    app = create_application(["pytest"])  # what al-dvc does before it builds the window
    assert current_theme() == "light" and app.styleSheet() == build_stylesheet(LIGHT)
    window = MainWindow()
    try:
        assert window._actions["theme_light"].isChecked()
        assert to_hex(window.viewer.figure.get_facecolor()) == "#ffffff"
        assert window.view3d.background_key() == "white"
    finally:
        window.close()
    QSettings().setValue(SETTINGS_KEY, "sepia")  # an unknown value: the default
    assert startup_theme() == "dark"
    create_application(["pytest"])
    assert current_theme() == "dark" and qapp.styleSheet() == build_stylesheet(DARK)


def test_the_environment_pins_the_start_theme(qapp, dark_again, monkeypatch):
    QSettings().setValue(SETTINGS_KEY, "dark")
    monkeypatch.setenv(ENV_THEME, "light")  # a script that captures the light theme whatever the user chose
    assert startup_theme() == "light"
    monkeypatch.setenv(ENV_THEME, "sepia")  # not a theme: the saved choice counts
    assert startup_theme() == "dark"
