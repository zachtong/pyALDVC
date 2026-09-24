"""The compute backend in words: the parameter panel, the status bar and the self-test, without a GPU needed.

The probe is replaced by its cached answer (available / kind / reason) and ``sys.frozen`` by the portable
version's flag, so every case is exercised on any machine.
"""

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.name == "nt":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

pytest.importorskip("PySide6")

from al_dvc.gui import backend_status as bs  # noqa: E402
from al_dvc.solver import cuda_kernels as ck  # noqa: E402

COMMAND = 'pip install "al-dvc[gpu]"'
PORTABLE_SENTENCE = f"CPU (portable version). For NVIDIA GPU acceleration, install pyALDVC with pip: {COMMAND}"
MISSING_SENTENCE = f"CPU. GPU acceleration is not installed: {COMMAND}"
NO_MODULE = "ModuleNotFoundError: No module named 'numba.cuda'"
ROW_STACK = "AttributeError: module 'numpy' has no attribute 'row_stack'"


def _probe(monkeypatch, *, available=False, kind="", reason="", frozen=False, installed=True, device="RTX Test") -> None:
    """The cached probe answer, the frozen flag and the numba-cuda distribution, as the case needs them."""
    monkeypatch.setattr(ck, "_available", available)
    monkeypatch.setattr(ck, "_unavailable_kind", kind)
    monkeypatch.setattr(ck, "_unavailable_reason", reason)
    monkeypatch.setattr(ck, "device_name", lambda: device if available else "")
    monkeypatch.setattr(sys, "frozen", frozen, raising=False)
    monkeypatch.setattr(bs, "gpu_backend_installed", lambda: installed)


@pytest.mark.parametrize(
    ("kind", "frozen", "installed", "case"),
    [
        ("missing", True, False, bs.PORTABLE),  # the portable version is CPU-only whatever the probe says
        ("no_device", True, False, bs.PORTABLE),
        ("error", True, True, bs.PORTABLE),
        ("missing", False, False, bs.MISSING),
        ("missing", False, True, bs.ERROR),  # numba-cuda is installed but does not import: broken, not missing
        ("no_device", False, True, bs.NO_DEVICE),
        ("error", False, True, bs.ERROR),
        ("error", False, False, bs.MISSING),  # numba's own CUDA module failed: the extra is the fix
    ],
)
def test_the_case_comes_from_the_probe_kind_and_the_frozen_flag(monkeypatch, kind, frozen, installed, case):
    _probe(monkeypatch, kind=kind, reason="SomeError: detail", frozen=frozen, installed=installed)
    status = bs.backend_status()
    assert status.case == case and not status.on_gpu and status.reason == "SomeError: detail"


def test_a_working_gpu_is_named_even_in_the_frozen_bundle(monkeypatch):
    _probe(monkeypatch, available=True, frozen=True)
    status = bs.backend_status()
    assert status.on_gpu and status.device == "RTX Test"
    text = bs.describe(status, translate=None)
    assert text.label == text.line == "GPU: RTX Test" and text.tooltip == ""


def test_the_english_texts_of_every_cpu_case():
    """No raw exception on the line; the technical reason only in the tooltip, for the cases that have one."""
    en = {case: bs.describe(bs.BackendStatus(case, reason=ROW_STACK), translate=None) for case in bs._TEXTS}
    assert {t.label for t in en.values()} == {"CPU"}
    assert en[bs.PORTABLE].line == f"CPU (portable version). For GPU acceleration, install with pip:\n{COMMAND}"
    assert en[bs.PORTABLE].tooltip == PORTABLE_SENTENCE
    assert en[bs.MISSING].line == f"CPU. GPU acceleration is not installed:\n{COMMAND}"
    assert en[bs.MISSING].tooltip == MISSING_SENTENCE
    assert en[bs.NO_DEVICE].line == "CPU. No NVIDIA GPU or driver was found."
    assert en[bs.NO_DEVICE].tooltip == f"CPU. No NVIDIA GPU or driver was found.\n{ROW_STACK}"
    assert en[bs.ERROR].line == "CPU. The GPU backend did not start."
    assert en[bs.ERROR].tooltip == f"CPU. The GPU backend did not start.\n{ROW_STACK}"
    for text in en.values():
        assert "Error" not in text.line  # the reason never reaches the line
    # the pip command is on a line of its own: the word wrap never cuts it
    assert en[bs.PORTABLE].line.splitlines()[-1] == COMMAND == en[bs.MISSING].line.splitlines()[-1]
    # a long reason is cut in the tooltip
    long = bs.describe(bs.BackendStatus(bs.ERROR, reason="x" * 5000), translate=None)
    assert len(long.tooltip) < bs.REASON_CHARS + 100


def test_every_language_keeps_the_pip_command_and_its_line():
    from al_dvc.gui.i18n import load_table

    for code in ("zh_CN", "zh_TW", "ja", "de", "fr", "es"):
        table = load_table(code)
        for pair in bs._TEXTS.values():
            for source in pair:
                assert "{command}" not in source or "{command}" in table[source], (code, source)
                assert source.count("\n") == table[source].count("\n"), (code, source)


@pytest.fixture(scope="module")
def qapp_lang():
    from al_dvc.gui.app import create_application

    return create_application(["pytest"])


@pytest.fixture
def window(qapp_lang):
    from al_dvc.gui.app import MainWindow

    w = MainWindow()
    yield w
    w.state.dirty = False
    w.close()


def _messages(window) -> list[tuple[str, str]]:
    seen: list[tuple[str, str]] = []
    window.state.log_message.connect(lambda message, level: seen.append((message, level)))
    return seen


def test_the_portable_version_says_cpu_and_how_to_get_the_gpu(window, monkeypatch):
    _probe(monkeypatch, kind="missing", reason=NO_MODULE, frozen=True, installed=False)
    window._refresh_backend()
    line = window.param_panel.backend_status
    assert line.text() == f"CPU (portable version). For GPU acceleration, install with pip:\n{COMMAND}"
    assert line.toolTip() == PORTABLE_SENTENCE
    assert "ModuleNotFoundError" not in line.text() + line.toolTip()  # the old "CPU only (ModuleNotFoundError ...)"
    assert window._backend_label.text() == "CPU" and window._backend_label.toolTip() == PORTABLE_SENTENCE


@pytest.mark.parametrize(
    ("kind", "installed", "line_text"),
    [
        ("missing", False, f"CPU. GPU acceleration is not installed:\n{COMMAND}"),
        ("no_device", True, "CPU. No NVIDIA GPU or driver was found."),
    ],
)
def test_a_pip_install_without_the_gpu_says_why(window, monkeypatch, kind, installed, line_text):
    _probe(monkeypatch, kind=kind, reason="RuntimeError: numba.cuda reports no usable CUDA device", installed=installed)
    seen = _messages(window)
    window._refresh_backend()
    assert window.param_panel.backend_status.text() == line_text
    assert window._backend_label.text() == "CPU"
    assert not [m for m, level in seen if level == "warning"]  # a CPU machine is not a problem


def test_a_gpu_backend_that_did_not_start_keeps_the_reason_in_the_tooltip_and_the_log(window, monkeypatch):
    _probe(monkeypatch, kind="error", reason=ROW_STACK, installed=True)
    seen = _messages(window)
    window._refresh_backend()
    line = window.param_panel.backend_status
    assert line.text() == "CPU. The GPU backend did not start."
    assert ROW_STACK in line.toolTip() and ROW_STACK in window._backend_label.toolTip()
    assert window._backend_label.text() == "CPU"
    assert any(ROW_STACK in m and level == "warning" for m, level in seen)


def test_a_working_gpu_shows_its_name(window, monkeypatch):
    _probe(monkeypatch, available=True, device="NVIDIA Test 9000")
    window._refresh_backend()
    assert window.param_panel.backend_status.text() == "GPU: NVIDIA Test 9000"
    assert window._backend_label.text() == "GPU: NVIDIA Test 9000"


def test_the_backend_line_follows_the_language(window, monkeypatch, qapp_lang):
    from PySide6.QtWidgets import QApplication

    from al_dvc.gui.i18n import load_table

    _probe(monkeypatch, kind="missing", reason=NO_MODULE, frozen=True, installed=False)
    window._refresh_backend()
    mgr = qapp_lang._pyaldvc_lang_mgr
    source = "CPU (portable version). For GPU acceleration, install with pip:\n{command}"
    try:
        mgr.load("zh_CN")
        for _ in range(10):
            QApplication.processEvents()
        assert window.param_panel.backend_status.text() == load_table("zh_CN")[source].format(command=COMMAND)
    finally:
        mgr.load("en")
        for _ in range(10):
            QApplication.processEvents()
    assert window.param_panel.backend_status.text() == source.format(command=COMMAND)


@pytest.mark.parametrize("kind", ["missing", "no_device", "error"])
def test_the_self_test_passes_the_portable_version_with_the_same_words(monkeypatch, kind):
    from al_dvc.gui import self_test as st

    _probe(monkeypatch, kind=kind, reason=ROW_STACK, frozen=True, installed=False)
    assert st.check_cuda() == PORTABLE_SENTENCE


def test_the_self_test_still_fails_a_broken_gpu_install(monkeypatch):
    from al_dvc.gui import self_test as st

    _probe(monkeypatch, kind="missing", reason="ImportError: numba_cuda is broken", installed=True)
    with pytest.raises(st.CheckFailed, match="installed but did not start"):
        st.check_cuda()
    _probe(monkeypatch, kind="missing", reason=NO_MODULE, installed=False)
    assert st.check_cuda() == MISSING_SENTENCE
