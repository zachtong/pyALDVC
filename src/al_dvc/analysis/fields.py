"""The quantities a statistic can be taken of, grouped as a user thinks of them, with their units."""

from __future__ import annotations

from ..core.config import length_unit
from ..core.data_structures import PipelineResult
from ..export.export_utils import DISP_FIELDS, STD_FIELDS

FIELD_GROUPS: dict[str, tuple[str, ...]] = {
    "displacement": DISP_FIELDS,
    "uncertainty": STD_FIELDS,
    "strain": ("exx", "eyy", "ezz", "exy", "exz", "eyz"),
    "principal": ("e1", "e2", "e3"),
    "equivalent": ("von_mises", "max_shear", "volumetric"),
    "rotation": ("rotation_deg",),
    "det_F": ("det_F",),
    "zncc": ("zncc",),
}
STRAIN_GROUPS = ("strain", "principal", "equivalent", "rotation", "det_F")
_LENGTH = set(DISP_FIELDS) | set(STD_FIELDS)
_STRAIN = set(FIELD_GROUPS["strain"]) | set(FIELD_GROUPS["principal"]) | set(FIELD_GROUPS["equivalent"])


def available_groups(result: PipelineResult | None) -> list[str]:
    """The groups ``result`` has values for, in display order."""
    if result is None or not result.result_disp:
        return []
    groups = ["displacement"]
    if result.result_disp[0].U_std is not None:
        groups.append("uncertainty")
    if result.result_strain:
        groups += list(STRAIN_GROUPS)
    if result.result_disp[0].zncc is not None:
        groups.append("zncc")
    return groups


def field_kind(name: str) -> str:
    """``length``, ``strain``, ``angle``, ``ratio`` (det F) or ``score`` (ZNCC)."""
    if name in _LENGTH:
        return "length"
    if name in _STRAIN:
        return "strain"
    if name == "rotation_deg":
        return "angle"
    if name == "det_F":
        return "ratio"
    if name == "zncc":
        return "score"
    raise KeyError(f"unknown field {name!r}")


def field_unit(name: str, para) -> str:
    """The unit a field's values are in: the length unit, ``deg``, or ``""`` (dimensionless)."""
    kind = field_kind(name)
    if kind == "length":
        return length_unit(para)
    return "deg" if kind == "angle" else ""
