"""Translation bookkeeping: extract ``tr()`` source strings from the GUI code and audit the JSON tables.

No Qt here, so the maintenance script and the tests can run anywhere. Source strings are the
first argument of ``self.tr(...)`` / ``tr(...)`` calls (string literals, including implicit
concatenations); a table is complete when every source string has a non-empty entry.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path

GUI_DIR = Path(__file__).parent
TRANSLATIONS_DIR = GUI_DIR / "translations"

__all__ = ["Audit", "audit_table", "extract_sources", "gui_python_files", "update_table"]


def gui_python_files(root: Path = GUI_DIR) -> list[Path]:
    """Every ``.py`` file of the GUI package (translations live there only)."""
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _string_of(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):  # f-strings are not translatable source strings
        return None
    return None


class _TrVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.sources: dict[str, set[str]] = {}
        self.file = ""

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast API
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else (func.id if isinstance(func, ast.Name) else "")
        if name in ("tr", "tr_noop") and node.args:
            text = _string_of(node.args[0])
            if text:
                self.sources.setdefault(text, set()).add(self.file)
        self.generic_visit(node)


def extract_sources(files: list[Path] | None = None) -> dict[str, set[str]]:
    """``{source string: {files}}`` for every translatable string in ``files`` (default: the GUI package)."""
    visitor = _TrVisitor()
    for path in files if files is not None else gui_python_files():
        visitor.file = path.name
        visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    return visitor.sources


@dataclass
class Audit:
    code: str
    total: int
    missing: list[str] = field(default_factory=list)
    empty: list[str] = field(default_factory=list)
    obsolete: list[str] = field(default_factory=list)

    @property
    def translated(self) -> int:
        return self.total - len(self.missing) - len(self.empty)

    @property
    def coverage(self) -> float:
        return self.translated / self.total if self.total else 1.0

    def summary(self) -> str:
        return (
            f"{self.code}: {self.translated}/{self.total} translated ({100 * self.coverage:.1f}%), "
            f"{len(self.missing)} missing, {len(self.empty)} empty, {len(self.obsolete)} obsolete"
        )


def load_json_table(code: str, directory: Path = TRANSLATIONS_DIR) -> dict[str, str]:
    path = directory / f"{code}.json"
    if not path.is_file():
        return {}
    return {str(k): str(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def audit_table(code: str, sources: dict[str, set[str]] | None = None, directory: Path = TRANSLATIONS_DIR) -> Audit:
    """Compare the table of ``code`` with the source strings of the GUI code."""
    sources = extract_sources() if sources is None else sources
    table = load_json_table(code, directory)
    audit = Audit(code=code, total=len(sources))
    for text in sorted(sources):
        if text not in table:
            audit.missing.append(text)
        elif not table[text].strip():
            audit.empty.append(text)
    audit.obsolete = sorted(k for k in table if k not in sources)
    return audit


def update_table(
    code: str, sources: dict[str, set[str]] | None = None, directory: Path = TRANSLATIONS_DIR, prune: bool = False
) -> Audit:
    """Add the missing source strings to the table with an empty value (and optionally drop obsolete ones)."""
    sources = extract_sources() if sources is None else sources
    table = load_json_table(code, directory)
    for text in sources:
        table.setdefault(text, "")
    if prune:
        table = {k: v for k, v in table.items() if k in sources}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{code}.json").write_text(
        json.dumps(dict(sorted(table.items())), ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return audit_table(code, sources, directory)
