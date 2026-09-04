"""Translation tables: extraction of source strings, completeness of every shipped language, translator behaviour."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.name == "nt":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

from al_dvc.gui.i18n import FALLBACK_CHAIN, SUPPORTED_LANGUAGES, JsonTranslator, LanguageManager, load_table  # noqa: E402
from al_dvc.gui.i18n_tools import audit_table, extract_sources, update_table  # noqa: E402


def test_extraction_finds_gui_strings():
    sources = extract_sources()
    assert "Run AL-DVC" in sources and "app.py" in sources["Run AL-DVC"] or "run_panel.py" in sources["Run AL-DVC"]
    assert "Same scale" in sources and "Strain post-processing..." in sources
    assert all(isinstance(k, str) and k for k in sources)
    assert not any(k.startswith('f"') for k in sources)  # f-strings are never translatable sources


@pytest.mark.parametrize("code", [c for c in SUPPORTED_LANGUAGES if c != "en"])
def test_every_shipped_language_is_complete(code):
    audit = audit_table(code)
    assert audit.total > 300
    assert audit.missing == [], f"{code}: missing {audit.missing[:5]}"
    assert audit.empty == [], f"{code}: empty {audit.empty[:5]}"
    assert audit.obsolete == [], f"{code}: obsolete {audit.obsolete[:5]}"
    table = load_table(code)
    # format placeholders survive translation
    for src, dst in table.items():
        for token in ("{n}", "{path}", "{error}", "{s:.0f}", "{s:.1f}", "{pct:.0f}"):
            if token in src:
                assert token in dst, f"{code}: placeholder {token} lost in {src!r}"
        assert src.count("&") == 0 or dst.count("&") >= 1 or "&" not in src.split(" ")[0]


def test_update_table_adds_missing_keys(tmp_path):
    sources = {"Hello": {"x.py"}, "World": {"x.py"}}
    (tmp_path / "xx.json").write_text('{"Hello": "Hallo", "Old": "alt"}', encoding="utf-8")
    audit = update_table("xx", sources, directory=tmp_path)
    assert audit.missing == [] and audit.empty == ["World"] and audit.obsolete == ["Old"]
    audit = update_table("xx", sources, directory=tmp_path, prune=True)
    assert audit.obsolete == []


def test_translator_answers_and_falls_back():
    tr = JsonTranslator({"Run": "Lancer"})
    assert tr.translate("ctx", "Run") == "Lancer"
    assert tr.translate("ctx", "Unknown") == ""  # Qt falls back to the source text
    assert not tr.isEmpty() and JsonTranslator({}).isEmpty()


def test_supported_languages_have_native_names_and_fallbacks():
    assert set(SUPPORTED_LANGUAGES) >= {"en", "zh_CN", "zh_TW", "ja", "de", "fr", "es"}
    assert SUPPORTED_LANGUAGES["zh_TW"] == "繁體中文" and SUPPORTED_LANGUAGES["ja"] == "日本語"
    assert FALLBACK_CHAIN["zh_HK"][0] == "zh_TW"
    assert LanguageManager.base_language("de_AT") == "de"
    assert LanguageManager.base_language("fr_CA") == "fr"
    assert LanguageManager.base_language("pt_BR") == "en"
    assert LanguageManager.base_language("zh_HK") == "zh_TW"
    assert LanguageManager.base_language("zh_SG") == "zh_CN"
