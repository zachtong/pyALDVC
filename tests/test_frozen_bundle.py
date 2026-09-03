"""Drive a built PyInstaller bundle from outside and check that it works.

Opt in by pointing ``PYALDVC_FROZEN_EXE`` at the console executable::

    python tools/build_exe.py --no-zip
    set PYALDVC_FROZEN_EXE=dist-exe\\pyALDVC\\pyALDVC-console.exe
    pytest tests/test_frozen_bundle.py -v

Skipped otherwise. The bundle must run as a separate process: importing
``al_dvc`` here would exercise the development install, which is exactly
what these tests are not about.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.frozen

EXE_ENV = "PYALDVC_FROZEN_EXE"
TIMEOUT = 900


@pytest.fixture(scope="session")
def frozen_exe() -> Path:
    raw = os.environ.get(EXE_ENV)
    if not raw:
        pytest.skip(f"{EXE_ENV} is not set")
    exe = Path(raw)
    if not exe.is_file():
        pytest.fail(f"{EXE_ENV} points at {exe}, which does not exist")
    return exe


def _run_self_test(exe: Path, workdir: Path) -> tuple[int, str, str]:
    report = workdir / "报告 report.txt"  # non-ASCII and a space, deliberately
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.run(
        [str(exe), "--self-test", str(report)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(workdir),
        timeout=TIMEOUT,
        env=env,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    text = report.read_text(encoding="utf-8") if report.is_file() else ""
    return proc.returncode, text, output


def test_bundle_self_test_passes(frozen_exe, tmp_path):
    code, text, output = _run_self_test(frozen_exe, tmp_path)
    assert text, f"the bundle wrote no report; it died early.\n{output[-4000:]}"
    assert "[FAIL]" not in text, text
    assert "all checks passed" in text
    assert code == 0, output[-2000:]


def test_bundle_works_from_non_ascii_cwd(frozen_exe, tmp_path):
    work = tmp_path / "数据 folder"
    work.mkdir()
    code, text, _output = _run_self_test(frozen_exe, work)
    assert code == 0 and "all checks passed" in text
