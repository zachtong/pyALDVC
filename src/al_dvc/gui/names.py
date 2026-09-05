"""User-facing names of the choice values and result fields.

The solver works with short internal keys (``plane_fit``, ``on_the_fly``, ``disp_magnitude``...).
The GUI never shows those: every combo box stores the key as item data and displays the
translated label from here, and result fields are named the way a report would name them.
``tr_noop`` marks the strings for the translation tables; :func:`label` translates at display time.
"""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox

from .i18n import tr, tr_noop

CHOICES: dict[str, dict[str, str]] = {
    "interp": {
        "cubic": tr_noop("Cubic"),
        "bspline": tr_noop("B-spline"),
        "linear": tr_noop("Linear"),
    },
    "init": {
        "pyramid": tr_noop("Pyramid search"),
        "ncc": tr_noop("Single-level search"),
        "zero": tr_noop("Zero displacement"),
        "previous": tr_noop("Previous frame"),
    },
    "tracking": {
        "accumulative": tr_noop("Accumulative (every frame vs. the first)"),
        "incremental": tr_noop("Incremental (every frame vs. the previous)"),
    },
    "solver": {
        "local": tr_noop("Local DVC"),
        "aldvc": tr_noop("AL-DVC"),
    },
    "discretisation": {
        "fem": tr_noop("Finite elements"),
        "fd": tr_noop("Finite differences"),
    },
    "gradient": {
        "stored": tr_noop("Precomputed (fast, more memory)"),
        "on_the_fly": tr_noop("On the fly (less memory)"),
    },
    "backend": {
        "auto": tr_noop("Automatic"),
        "cuda": tr_noop("GPU (CUDA)"),
        "numba": tr_noop("CPU"),
    },
    "strain_method": {
        "plane_fit": tr_noop("Plane fitting"),
        "fem": tr_noop("Finite elements"),
        "fd": tr_noop("Finite differences"),
        "direct": tr_noop("Solver gradient (direct)"),
    },
    "strain_type": {
        "infinitesimal": tr_noop("Infinitesimal"),
        "green_lagrange": tr_noop("Green-Lagrange"),
        "euler_almansi": tr_noop("Euler-Almansi"),
        "hencky": tr_noop("Hencky (logarithmic)"),
    },
    "view3d_mode": {
        "slices": tr_noop("Slices"),
        "points": tr_noop("Points"),
        "surface": tr_noop("Iso-surface"),
        "warped": tr_noop("Deformed lattice"),
    },
    "estimator": {
        "overlap": tr_noop("Overlap-corrected"),
        "window": tr_noop("Finite window"),
    },
    "animation": {
        "orbit": tr_noop("Orbit"),
        "frames": tr_noop("Frames"),
        "slice": tr_noop("Slice sweep"),
        "warp": tr_noop("Deformed lattice"),
    },
    "direction": {
        "ccw": tr_noop("Counter-clockwise"),
        "cw": tr_noop("Clockwise"),
    },
    "camera": {
        "iso": tr_noop("Isometric"),
        "xy": "XY",
        "xz": "XZ",
        "yz": "YZ",
    },
    "background": {
        "dark": tr_noop("Dark"),
        "black": tr_noop("Black"),
        "grey": tr_noop("Grey"),
        "white": tr_noop("White"),
    },
}

FIELDS: dict[str, str] = {
    "disp_u": tr_noop("Displacement u (x)"),
    "disp_v": tr_noop("Displacement v (y)"),
    "disp_w": tr_noop("Displacement w (z)"),
    "disp_magnitude": tr_noop("Displacement magnitude"),
    "disp_std_u": tr_noop("Uncertainty of u"),
    "disp_std_v": tr_noop("Uncertainty of v"),
    "disp_std_w": tr_noop("Uncertainty of w"),
    "disp_std": tr_noop("Uncertainty magnitude"),
    "exx": tr_noop("Strain exx"),
    "eyy": tr_noop("Strain eyy"),
    "ezz": tr_noop("Strain ezz"),
    "exy": tr_noop("Shear strain exy"),
    "exz": tr_noop("Shear strain exz"),
    "eyz": tr_noop("Shear strain eyz"),
    "e1": tr_noop("Principal strain e1 (max)"),
    "e2": tr_noop("Principal strain e2"),
    "e3": tr_noop("Principal strain e3 (min)"),
    "max_shear": tr_noop("Maximum shear strain"),
    "von_mises": tr_noop("Von Mises strain"),
    "volumetric": tr_noop("Volumetric strain"),
    "det_F": tr_noop("Volume ratio det F"),
    "rotation_deg": tr_noop("Rotation angle [deg]"),
}

STATUS: dict[str, str] = {  # keys: ``core.data_structures.STATUS_NAMES`` values
    "converged": tr_noop("converged"),
    "max_iter": tr_noop("iteration limit"),
    "out_of_bounds": tr_noop("out of bounds"),
    "invalid_subset": tr_noop("invalid subset"),
    "singular": tr_noop("singular Hessian"),
    "nan": tr_noop("not a number"),
    "skipped": tr_noop("skipped"),
    "stalled": tr_noop("stalled"),
}


def label(group: str, key: str) -> str:
    """Translated display name of ``key`` in ``group`` (the key itself when unknown)."""
    text = CHOICES.get(group, {}).get(key)
    return tr(text) if text else str(key)


def field_name(key: str) -> str:
    text = FIELDS.get(key)
    return tr(text) if text else str(key)


def status_name(name: str) -> str:
    text = STATUS.get(str(name))
    return tr(text) if text else str(name).replace("_", " ")


def fill_combo(combo: QComboBox, group: str, keys=None) -> None:
    """Populate ``combo`` with the keys of ``group`` as item data and their labels as text."""
    current = combo.currentData()
    combo.blockSignals(True)
    combo.clear()
    for key in keys if keys is not None else CHOICES[group]:
        combo.addItem(label(group, key), key)
    if current is not None:
        i = combo.findData(current)
        if i >= 0:
            combo.setCurrentIndex(i)
    combo.blockSignals(False)


def retranslate_combo(combo: QComboBox, group: str) -> None:
    """Refresh the item texts of a combo filled by :func:`fill_combo` after a language change."""
    for i in range(combo.count()):
        combo.setItemText(i, label(group, combo.itemData(i)))


def select_key(combo: QComboBox, key) -> bool:
    """Select the item whose data is ``key``; False when absent."""
    i = combo.findData(key)
    if i < 0:
        return False
    combo.setCurrentIndex(i)
    return True


__all__ = ["CHOICES", "FIELDS", "STATUS", "field_name", "fill_combo", "label", "retranslate_combo", "select_key", "status_name"]
