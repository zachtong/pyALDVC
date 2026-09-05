"""Runtime language switching without Qt Linguist tooling.

Translations live in ``translations/<code>.json`` as ``{"source": "target"}``
dictionaries keyed by the English source string. :class:`JsonTranslator` is
a ``QTranslator`` that answers ``tr()`` / ``QCoreApplication.translate()``
from such a dictionary (an empty answer makes Qt fall back to the source),
so no ``.ts``/``.qm`` compilation step is needed. The chosen language is
persisted with ``QSettings`` and applied at the next start.

Widgets rebuild their visible strings in ``retranslate_ui`` and the main
window calls it on ``QEvent.Type.LanguageChange``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Final

from PySide6.QtCore import QCoreApplication, QLocale, QObject, QSettings, QTranslator, Signal

logger = logging.getLogger(__name__)

SETTINGS_KEY: Final[str] = "pyaldvc/language"
ENV_LANGUAGE: Final[str] = "PYALDVC_LANGUAGE"  # environment override (tests, batch scripts)
TRANSLATIONS_DIR: Final[Path] = Path(__file__).parent / "translations"
SUPPORTED_LANGUAGES: Final[dict[str, str]] = {
    "en": "English",
    "zh_CN": "简体中文",
    "zh_TW": "繁體中文",
    "ja": "日本語",
    "de": "Deutsch",
    "fr": "Français",
    "es": "Español",
}
# locale name -> shipped languages to try, in order, before the bare language code and English
FALLBACK_CHAIN: Final[dict[str, tuple[str, ...]]] = {
    "zh": ("zh_CN",),
    "zh_TW": ("zh_TW", "zh_CN"),
    "zh_HK": ("zh_TW", "zh_CN"),
    "zh_MO": ("zh_TW", "zh_CN"),
    "zh_SG": ("zh_CN",),
}


class JsonTranslator(QTranslator):
    """``QTranslator`` backed by a source -> target dictionary."""

    def __init__(self, table: dict[str, str], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._table = dict(table)

    def translate(self, context: str, source: str, disambiguation: str | None = None, n: int = -1) -> str:  # noqa: N802
        return self._table.get(source, "")

    def isEmpty(self) -> bool:  # noqa: N802
        return not self._table


def load_table(code: str) -> dict[str, str]:
    """The translation dictionary of ``code`` (empty for English or a missing file)."""
    if code == "en":
        return {}
    path = TRANSLATIONS_DIR / f"{code}.json"
    if not path.is_file():
        logger.warning("No translation file for %s (%s)", code, path)
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}


class LanguageManager(QObject):
    """Installs and switches the application language."""

    language_changed = Signal(str)

    def __init__(self, app: QCoreApplication) -> None:
        super().__init__(app)
        self._app = app
        self._translator: JsonTranslator | None = None
        self._code = "en"

    @property
    def code(self) -> str:
        return self._code

    @staticmethod
    def base_language(name: str) -> str:
        """The shipped language for a locale name (``de_AT`` -> ``de``, ``zh_HK`` -> ``zh_TW``), else ``en``."""
        if name in SUPPORTED_LANGUAGES:
            return name
        base = name.split("_")[0]
        for candidate in FALLBACK_CHAIN.get(name, ()) + FALLBACK_CHAIN.get(base, ()) + (base,):
            if candidate in SUPPORTED_LANGUAGES:
                return candidate
        return "en"

    @staticmethod
    def resolve_language() -> str:
        """Persisted choice, else the system locale mapped onto a shipped language, else English."""
        forced = os.environ.get(ENV_LANGUAGE, "")
        if forced in SUPPORTED_LANGUAGES:  # tests and scripts pin the language regardless of the saved choice
            return forced
        saved = QSettings().value(SETTINGS_KEY, "", type=str)
        if saved in SUPPORTED_LANGUAGES:
            return saved
        return LanguageManager.base_language(QLocale.system().name())  # e.g. "zh_CN", "de_AT"

    def load(self, code: str) -> None:
        if code not in SUPPORTED_LANGUAGES:
            raise ValueError(f"unsupported language {code!r}; supported: {sorted(SUPPORTED_LANGUAGES)}")
        if self._translator is not None:
            self._app.removeTranslator(self._translator)
            self._translator = None
        table = load_table(code)
        if table:
            self._translator = JsonTranslator(table, self)
            self._app.installTranslator(self._translator)
        self._code = code
        QSettings().setValue(SETTINGS_KEY, code)
        self.language_changed.emit(code)


def tr_noop(source: str) -> str:
    """Mark a string for the translation tables without translating it (translate later with :func:`tr`)."""
    return source


def tr(source: str) -> str:
    """Translate a string outside a ``QObject`` (same table as ``self.tr``)."""
    return QCoreApplication.translate("pyALDVC", source)
