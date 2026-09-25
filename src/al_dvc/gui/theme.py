"""Colour themes of the AL-DVC GUI: dark navy (the default) and light, both with the indigo accent.

A frozen :class:`Colors` palette per theme (:data:`DARK`, :data:`LIGHT`, by name in :data:`THEMES`),
the theme in use (:func:`current_theme`, :func:`current_colors`, :func:`set_current_theme`) and
:func:`build_stylesheet`, which returns the complete QSS of a palette. Qt-free: the canvases read
their colours here when they draw. Switching the theme of the running application (stylesheet,
icons, canvases, the saved choice) is :mod:`al_dvc.gui.theme_manager`'s job.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

ARROWS_DIR = Path(__file__).parent / "arrows"
SIDE_COLUMN = "sideColumn"  # object name of the side columns, which the light theme sets apart in grey


@dataclass(frozen=True)
class Colors:
    """The palette of one theme; the defaults are the dark navy theme."""

    # Backgrounds
    BG_DARKEST: str = "#0b0f1a"
    BG_SIDEBAR: str = "#0f1322"
    BG_PANEL: str = "#141929"
    BG_INPUT: str = "#1a1f33"
    BG_HOVER: str = "#1e2440"
    BG_CANVAS: str = "#0c0d12"

    # Accent
    ACCENT: str = "#6366f1"
    ACCENT_HOVER: str = "#818cf8"
    ACCENT_PRESSED: str = "#4f46e5"

    # Text
    TEXT_PRIMARY: str = "#e2e8f0"
    TEXT_SECONDARY: str = "#94a3b8"
    TEXT_MUTED: str = "#64748b"

    # Borders
    BORDER: str = "#1e293b"
    BORDER_FOCUS: str = "#6366f1"

    # Semantic
    DANGER: str = "#ef4444"
    DANGER_HOVER: str = "#f87171"
    SUCCESS: str = "#22c55e"
    WARNING: str = "#eab308"

    # What differs between the themes beyond the colours above (the defaults give the dark theme exactly)
    DANGER_PRESSED: str = "#dc2626"
    SCROLL_HANDLE: str = "#1e2440"
    BG_SIDE_COLUMN: str = "#0b0f1a"  # side columns; the same as BG_DARKEST: not set apart
    SELECTION: str = ""  # selected rows of tables and lists; empty: the style's own highlight
    SELECTION_TEXT: str = ""
    CANVAS_TEXT: str = "#94a3b8"  # ticks, axis labels and titles of the matplotlib canvases
    CANVAS_SPINE: str = "#1e293b"  # their axes frames
    # arrows of the spin and combo boxes (files in ARROWS_DIR)
    ARROW_UP: str = "spin_up.svg"
    ARROW_DOWN: str = "spin_down.svg"
    ARROW_UP_HOVER: str = "spin_up_hover.svg"  # on the accent-coloured button under the pointer
    ARROW_DOWN_HOVER: str = "spin_down_hover.svg"
    COMBO_ARROW_HOVER: str = "spin_down_hover.svg"  # on a combo box under the pointer
    # the views with a background choice of their own start on these entries
    VIEW3D_BACKGROUND: str = "dark"  # view3d_scene.BACKGROUNDS
    PLOT_BACKGROUND: str = "dark"  # the texture window's plot backgrounds
    IS_DARK: bool = True  # dark window chrome (the Windows title bar)


DARK = Colors()
LIGHT = Colors(
    BG_DARKEST="#ffffff",
    BG_SIDEBAR="#f6f7f9",
    BG_PANEL="#eef0f4",
    BG_INPUT="#ffffff",
    BG_HOVER="#e8eaf6",
    BG_CANVAS="#ffffff",
    ACCENT="#4f46e5",
    ACCENT_HOVER="#6366f1",
    ACCENT_PRESSED="#4338ca",
    TEXT_PRIMARY="#111827",
    TEXT_SECONDARY="#4b5563",
    TEXT_MUTED="#6b7280",
    BORDER="#d1d5db",
    BORDER_FOCUS="#4f46e5",
    DANGER="#dc2626",
    DANGER_HOVER="#ef4444",
    SUCCESS="#16a34a",
    WARNING="#b45309",
    DANGER_PRESSED="#b91c1c",
    SCROLL_HANDLE="#c4c9d2",
    BG_SIDE_COLUMN="#f6f7f9",
    SELECTION="#e0e7ff",
    SELECTION_TEXT="#111827",
    CANVAS_TEXT="#111827",
    CANVAS_SPINE="#111827",
    ARROW_UP="spin_up_light.svg",
    ARROW_DOWN="spin_down_light.svg",
    COMBO_ARROW_HOVER="combo_down_hover_light.svg",
    VIEW3D_BACKGROUND="white",
    PLOT_BACKGROUND="white",
    IS_DARK=False,
)
THEMES: dict[str, Colors] = {"dark": DARK, "light": LIGHT}
DEFAULT_THEME = "dark"
_current = {"name": DEFAULT_THEME}  # the theme in use (one per process, like the application stylesheet)


def current_theme() -> str:
    """Name of the theme in use (``"dark"`` or ``"light"``)."""
    return _current["name"]


def current_colors() -> Colors:
    """The palette of the theme in use; canvases call this when they draw, never at import."""
    return THEMES[_current["name"]]


def set_current_theme(name: str) -> Colors:
    """Make ``name`` the theme in use (the colours only: the application is restyled by the theme manager)."""
    if name not in THEMES:
        raise ValueError(f"unknown theme {name!r}; available: {sorted(THEMES)}")
    _current["name"] = name
    return THEMES[name]


def style_text(template: str, colors: Colors | None = None) -> str:
    """``template`` with its ``{FIELD}`` placeholders (names of :class:`Colors`) filled from ``colors``
    (default: the theme in use); literal braces are doubled, as in ``str.format``."""
    return template.format_map(asdict(colors if colors is not None else current_colors()))


def _columns_apart(c: Colors) -> bool:
    """True for a palette whose side columns have a colour of their own (the light theme)."""
    return c.BG_SIDE_COLUMN != c.BG_DARKEST


def _containers(c: Colors) -> str:
    """Rules placed right after the one that gives every widget the window colour, for a palette whose side
    columns are set apart: the containers become transparent, so a side column or a group box shows through
    them, and the windows and pop-up frames stay opaque. Empty for the dark theme."""
    if not _columns_apart(c):
        return ""
    return f"""/* containers are transparent: a side column or a group box shows through them */
QWidget {{
    background: transparent;
}}

QMainWindow,
QDialog,
QComboBoxPrivateContainer {{
    background: {c.BG_DARKEST};
}}

QWidget#{SIDE_COLUMN} {{
    background: {c.BG_SIDE_COLUMN};
}}
"""


def _views_and_tracks(c: Colors) -> str:
    """Rules appended for a palette whose side columns are set apart, where the dark theme relies on opaque
    containers and the style: tables and lists on the input colour with a tinted selection, tab panes and
    scroll-bar tracks transparent. Empty for the dark theme."""
    if not _columns_apart(c):
        return ""
    return f"""
/* ============================================================
   Tables and lists, tab panes, scroll-bar tracks
   ============================================================ */
QAbstractItemView {{
    background: {c.BG_INPUT};
    selection-background-color: {c.SELECTION or c.ACCENT};
    selection-color: {c.SELECTION_TEXT or c.TEXT_PRIMARY};
}}

QTabWidget::pane {{
    background: transparent;
}}

QScrollBar:vertical,
QScrollBar:horizontal {{
    background: transparent;
}}
"""


def build_stylesheet(colors: Colors | None = None) -> str:
    """Return the complete QSS stylesheet of ``colors`` (default: the theme in use)."""
    c = colors if colors is not None else current_colors()
    # Arrow SVG paths — Qt QSS requires forward-slash separators on all platforms
    _up = (ARROWS_DIR / c.ARROW_UP).as_posix()
    _up_hover = (ARROWS_DIR / c.ARROW_UP_HOVER).as_posix()
    _down = (ARROWS_DIR / c.ARROW_DOWN).as_posix()
    _down_hover = (ARROWS_DIR / c.ARROW_DOWN_HOVER).as_posix()
    _combo_hover = (ARROWS_DIR / c.COMBO_ARROW_HOVER).as_posix()
    return f"""
/* ============================================================
   Global
   ============================================================ */
* {{
    color: {c.TEXT_PRIMARY};
    font-family: "Segoe UI", "SF Pro Text", "Helvetica Neue", "Microsoft YaHei UI", "Microsoft YaHei",
        "PingFang SC", "Hiragino Sans GB", "Source Han Sans SC", "Noto Sans CJK SC", "Yu Gothic UI", "Malgun Gothic", sans-serif;
    font-size: 13px;
}}

QMainWindow,
QWidget {{
    background: {c.BG_DARKEST};
}}
{_containers(c)}
/* ============================================================
   Sidebar panels
   ============================================================ */
QWidget#leftSidebar,
QWidget#rightSidebar {{
    background: {c.BG_SIDEBAR};
    border-right: 1px solid {c.BORDER};
}}

QWidget#rightSidebar {{
    border-right: none;
    border-left: 1px solid {c.BORDER};
}}

/* ============================================================
   Panel / Card sections
   ============================================================ */
QFrame[frameShape="1"],
QToolButton#tool {{
    background: {c.BG_INPUT};
    border: 1px solid {c.BORDER};
    border-radius: 4px;
    padding: 2px;
}}

QToolButton#tool:hover {{
    background: {c.BG_HOVER};
    border-color: {c.TEXT_MUTED};
}}

QToolButton#tool:checked {{
    background: {c.ACCENT};
    border-color: {c.ACCENT};
}}

QToolButton#tool:disabled {{
    background: {c.BG_PANEL};
    border-color: {c.BORDER};
}}

QLabel#placeholder {{
    color: {c.TEXT_MUTED};
    font-size: 12px;
    background: transparent;
    border: 1px dashed {c.BORDER};
    border-radius: 6px;
}}

QLabel#hint {{
    color: {c.TEXT_MUTED};
    font-size: 11px;
}}

/* a state the user must not forget, e.g. the fields are shown with the rigid motion removed */
QLabel#badge {{
    color: {c.WARNING};
    font-size: 11px;
    border: 1px solid {c.WARNING};
    border-radius: 4px;
    padding: 2px 6px;
}}

/* Typography, three levels: sectionTitle and QGroupBox titles (level 1, 12 px bold primary), sectionHeader of the
   folding sub-sections (level 2, 11 px bold secondary), field labels 12 px regular, hints 11 px muted. */
QLabel#sectionTitle {{
    color: {c.TEXT_PRIMARY};
    font-size: 12px;
    font-weight: bold;
}}

QPushButton#sectionHeader {{
    text-align: left;
    padding: 4px 8px;
    color: {c.TEXT_SECONDARY};
    font-size: 11px;
    font-weight: bold;
    background: transparent;
    border: none;
    border-bottom: 1px solid {c.BORDER};
    border-radius: 0;
}}

QPushButton#sectionHeader:hover {{
    color: {c.TEXT_PRIMARY};
    background: {c.BG_HOVER};
}}

QPlainTextEdit#console,
QTextEdit#console {{
    background: {c.BG_DARKEST};
    border: 1px solid {c.BORDER};
    border-radius: 4px;
    padding: 4px;
    font-family: 'Consolas', 'Cascadia Mono', monospace;
    font-size: 11px;
}}

QGroupBox {{
    background: {c.BG_PANEL};
    border: 1px solid {c.BORDER};
    border-radius: 6px;
    padding: 16px 10px 10px 10px;  /* the title sits in the top padding, clear of the first row */
    margin: 6px 0px 4px 0px;
}}

QGroupBox#analysisBox {{
    background: {c.BG_INPUT};
    border: 1px solid {c.ACCENT};
}}

QGroupBox::title {{
    color: {c.TEXT_SECONDARY};
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px;
    color: {c.TEXT_PRIMARY};
    font-weight: bold;
    font-size: 12px;
}}

/* ============================================================
   Labels
   ============================================================ */
QLabel {{
    background: transparent;
    border: none;
    color: {c.TEXT_PRIMARY};
}}

QLabel[class="secondary"] {{
    color: {c.TEXT_SECONDARY};
}}

QLabel[class="muted"] {{
    color: {c.TEXT_MUTED};
    font-size: 11px;
}}

QLabel[class="heading"] {{
    font-weight: bold;
    font-size: 14px;
}}

/* ============================================================
   Input fields: QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox
   ============================================================ */
QLineEdit,
QSpinBox,
QDoubleSpinBox,
QComboBox {{
    background: {c.BG_INPUT};
    border: 1px solid {c.BORDER};
    border-radius: 4px;
    padding: 5px 8px;
    color: {c.TEXT_PRIMARY};
    selection-background-color: {c.ACCENT};
    min-height: 20px;
}}

QLineEdit:focus,
QSpinBox:focus,
QDoubleSpinBox:focus,
QComboBox:focus {{
    border-color: {c.BORDER_FOCUS};
}}

QLineEdit:hover,
QSpinBox:hover,
QDoubleSpinBox:hover,
QComboBox:hover {{
    background: {c.BG_HOVER};
}}

QLineEdit:disabled,
QSpinBox:disabled,
QDoubleSpinBox:disabled,
QComboBox:disabled {{
    color: {c.TEXT_MUTED};
    background: {c.BG_PANEL};
}}

/* SpinBox buttons */
QSpinBox::up-button,
QDoubleSpinBox::up-button,
QSpinBox::down-button,
QDoubleSpinBox::down-button {{
    background: {c.BG_HOVER};
    border: none;
    width: 16px;
    border-radius: 2px;
}}

QSpinBox::up-button:hover,
QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover,
QDoubleSpinBox::down-button:hover {{
    background: {c.ACCENT};
}}

QSpinBox::up-arrow,
QDoubleSpinBox::up-arrow {{
    image: url("{_up}");
    width: 8px;
    height: 6px;
}}

QSpinBox::up-arrow:hover,
QDoubleSpinBox::up-arrow:hover {{
    image: url("{_up_hover}");
}}

QSpinBox::down-arrow,
QDoubleSpinBox::down-arrow {{
    image: url("{_down}");
    width: 8px;
    height: 6px;
}}

QSpinBox::down-arrow:hover,
QDoubleSpinBox::down-arrow:hover {{
    image: url("{_down_hover}");
}}

/* ComboBox dropdown button */
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    border: none;
    width: 24px;
    background: transparent;
}}

QComboBox::down-arrow {{
    image: url("{_down}");
    width: 8px;
    height: 6px;
}}

QComboBox::down-arrow:hover {{
    image: url("{_combo_hover}");
}}

/* ComboBox popup list */
QComboBox QAbstractItemView {{
    background: {c.BG_PANEL};
    border: 1px solid {c.BORDER};
    border-radius: 4px;
    padding: 4px;
    selection-background-color: {c.ACCENT};
    selection-color: #ffffff;
    outline: none;
}}

QComboBox QAbstractItemView::item {{
    padding: 4px 8px;
    min-height: 22px;
    border-radius: 3px;
}}

QComboBox QAbstractItemView::item:hover {{
    background: {c.BG_HOVER};
}}

/* ============================================================
   Buttons — default
   ============================================================ */
QPushButton {{
    background: {c.BG_INPUT};
    border: 1px solid {c.BORDER};
    border-radius: 4px;
    padding: 6px 14px;
    color: {c.TEXT_PRIMARY};
    font-weight: 500;
    min-height: 22px;
}}

QPushButton:hover {{
    background: {c.BG_HOVER};
    border-color: {c.TEXT_MUTED};
}}

QPushButton:pressed {{
    background: {c.BG_PANEL};
}}

QPushButton:disabled {{
    color: {c.TEXT_MUTED};
    background: {c.BG_PANEL};
    border-color: {c.BORDER};
}}

/* Primary action button */
QPushButton[class="btn-primary"] {{
    background: {c.ACCENT};
    border: 1px solid {c.ACCENT};
    color: #ffffff;
    font-weight: bold;
}}

QPushButton[class="btn-primary"]:hover {{
    background: {c.ACCENT_HOVER};
    border-color: {c.ACCENT_HOVER};
}}

QPushButton[class="btn-primary"]:pressed {{
    background: {c.ACCENT_PRESSED};
    border-color: {c.ACCENT_PRESSED};
}}

QPushButton[class="btn-primary"]:disabled {{
    background: {c.BG_HOVER};
    border-color: {c.BORDER};
    color: {c.TEXT_MUTED};
}}

/* Danger button */
QPushButton[class="btn-danger"] {{
    background: transparent;
    border: 1px solid {c.DANGER};
    color: {c.DANGER};
}}

QPushButton[class="btn-danger"]:hover {{
    background: {c.DANGER};
    color: #ffffff;
}}

QPushButton[class="btn-danger"]:pressed {{
    background: {c.DANGER_PRESSED};
    border-color: {c.DANGER_PRESSED};
}}

/* ============================================================
   Toggle buttons (checkable QPushButton)
   ============================================================ */
QPushButton:checkable {{
    background: {c.BG_INPUT};
    border: 1px solid {c.BORDER};
}}

QPushButton:checkable:checked {{
    background: {c.ACCENT};
    border-color: {c.ACCENT};
    color: #ffffff;
}}

QPushButton:checkable:checked:hover {{
    background: {c.ACCENT_HOVER};
}}

/* ============================================================
   QCheckBox / QRadioButton
   ============================================================ */
QCheckBox,
QRadioButton {{
    background: transparent;
    spacing: 6px;
}}

QCheckBox::indicator,
QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {c.BORDER};
    background: {c.BG_INPUT};
}}

QCheckBox::indicator {{
    border-radius: 3px;
}}

QRadioButton::indicator {{
    border-radius: 8px;
}}

QCheckBox::indicator:checked,
QRadioButton::indicator:checked {{
    background: {c.ACCENT};
    border-color: {c.ACCENT};
}}

QCheckBox::indicator:hover,
QRadioButton::indicator:hover {{
    border-color: {c.TEXT_MUTED};
}}

/* ============================================================
   Progress bar
   ============================================================ */
QProgressBar {{
    background: {c.BG_INPUT};
    border: 1px solid {c.BORDER};
    border-radius: 4px;
    text-align: center;
    color: {c.TEXT_PRIMARY};
    min-height: 18px;
    font-size: 11px;
}}

QProgressBar::chunk {{
    background: {c.ACCENT};
    border-radius: 3px;
}}

/* ============================================================
   Scroll bars
   ============================================================ */
QScrollBar:vertical {{
    background: {c.BG_DARKEST};
    width: 8px;
    margin: 0;
    border: none;
}}

QScrollBar::handle:vertical {{
    background: {c.SCROLL_HANDLE};
    min-height: 30px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical:hover {{
    background: {c.TEXT_MUTED};
}}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0;
    border: none;
}}

QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{
    background: none;
}}

QScrollBar:horizontal {{
    background: {c.BG_DARKEST};
    height: 8px;
    margin: 0;
    border: none;
}}

QScrollBar::handle:horizontal {{
    background: {c.SCROLL_HANDLE};
    min-width: 30px;
    border-radius: 4px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {c.TEXT_MUTED};
}}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{
    width: 0;
    border: none;
}}

QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {{
    background: none;
}}

/* ============================================================
   Sliders
   ============================================================ */
QSlider::groove:horizontal {{
    background: {c.BG_INPUT};
    border: 1px solid {c.BORDER};
    height: 4px;
    border-radius: 2px;
}}

QSlider::sub-page:horizontal {{
    background: {c.ACCENT};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {c.ACCENT};
    border: 2px solid {c.ACCENT_HOVER};
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 8px;
}}

QSlider::handle:horizontal:hover {{
    background: {c.ACCENT_HOVER};
    border-color: #a5b4fc;
}}

QSlider::groove:vertical {{
    background: {c.BG_INPUT};
    border: 1px solid {c.BORDER};
    width: 4px;
    border-radius: 2px;
}}

QSlider::sub-page:vertical {{
    background: {c.ACCENT};
    border-radius: 2px;
}}

QSlider::handle:vertical {{
    background: {c.ACCENT};
    border: 2px solid {c.ACCENT_HOVER};
    width: 14px;
    height: 14px;
    margin: 0 -6px;
    border-radius: 8px;
}}

/* ============================================================
   Tab widget
   ============================================================ */
QTabWidget::pane {{
    background: {c.BG_PANEL};
    border: 1px solid {c.BORDER};
    border-top: none;
}}

QTabBar::tab {{
    background: {c.BG_SIDEBAR};
    border: 1px solid {c.BORDER};
    border-bottom: none;
    padding: 6px 14px;
    color: {c.TEXT_SECONDARY};
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}

QTabBar::tab:selected {{
    background: {c.BG_PANEL};
    color: {c.TEXT_PRIMARY};
    border-bottom: 2px solid {c.ACCENT};
}}

QTabBar::tab:hover:!selected {{
    background: {c.BG_HOVER};
    color: {c.TEXT_PRIMARY};
}}

/* the Slices / 3-D view switch of the centre column: a prominent segmented control */
QTabWidget#viewTabs::pane {{
    border: 1px solid {c.BORDER};
    border-top: 2px solid {c.ACCENT};
}}

QTabWidget#viewTabs > QTabBar::tab {{
    font-size: 13px;
    font-weight: bold;
    padding: 8px 24px;
    min-height: 18px;
    color: {c.TEXT_SECONDARY};
    background: {c.BG_SIDEBAR};
    border: 1px solid {c.BORDER};
    border-bottom: none;
    margin-right: 3px;
}}

QTabWidget#viewTabs > QTabBar::tab:selected {{
    color: white;
    background: {c.ACCENT};
    border-color: {c.ACCENT};
}}

QTabWidget#viewTabs > QTabBar::tab:hover:!selected {{
    color: {c.TEXT_PRIMARY};
    background: {c.BG_HOVER};
}}

/* ============================================================
   Console / log text
   ============================================================ */
QPlainTextEdit[class="console"],
QTextEdit[class="console"] {{
    background: {c.BG_DARKEST};
    border: 1px solid {c.BORDER};
    border-radius: 4px;
    color: {c.TEXT_SECONDARY};
    font-family: "Consolas", "Cascadia Code", "Fira Code", monospace;
    font-size: 12px;
    padding: 6px;
    selection-background-color: {c.ACCENT};
}}

/* ============================================================
   Tooltips
   ============================================================ */
QToolTip {{
    background: {c.BG_PANEL};
    border: 1px solid {c.BORDER};
    border-radius: 4px;
    color: {c.TEXT_PRIMARY};
    padding: 4px 8px;
    font-size: 12px;
}}

/* ============================================================
   Splitter handle
   ============================================================ */
QSplitter::handle {{
    background: {c.BORDER};
}}

QSplitter::handle:horizontal {{
    width: 1px;
}}

QSplitter::handle:vertical {{
    height: 1px;
}}

/* ============================================================
   Menu bar and menus
   ============================================================ */
QMenuBar {{
    background: {c.BG_SIDEBAR};
    border-bottom: 1px solid {c.BORDER};
    padding: 2px;
}}

QMenuBar::item {{
    background: transparent;
    padding: 4px 10px;
    border-radius: 3px;
}}

QMenuBar::item:selected {{
    background: {c.BG_HOVER};
}}

QMenu {{
    background: {c.BG_PANEL};
    border: 1px solid {c.BORDER};
    border-radius: 4px;
    padding: 4px;
}}

QMenu::item {{
    padding: 6px 28px 6px 12px;
    border-radius: 3px;
}}

QMenu::item:selected {{
    background: {c.ACCENT};
    color: #ffffff;
}}

QMenu::separator {{
    height: 1px;
    background: {c.BORDER};
    margin: 4px 8px;
}}

/* ============================================================
   Status bar
   ============================================================ */
QStatusBar {{
    background: {c.BG_SIDEBAR};
    border-top: 1px solid {c.BORDER};
    color: {c.TEXT_MUTED};
    font-size: 11px;
}}
""" + _views_and_tracks(c)
