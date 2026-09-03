"""Build the portable Windows bundle, then zip it for release.

    python tools/build_exe.py                 # build + zip
    python tools/build_exe.py --no-zip        # build only
    python tools/build_exe.py --clean         # discard the previous build cache

Output goes to ``dist-exe/pyALDVC/`` and ``dist-exe/pyALDVC-<version>-win64.zip``
(not ``dist/``, which the PyPI workflow publishes from). PyInstaller reports a
missing hidden import as a WARNING and carries on; the interesting log lines
are printed after the build instead of being left in the scrollback.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "packaging" / "pyaldvc.spec"
DIST = ROOT / "dist-exe"
WORK = ROOT / "build-exe"
BUNDLE = DIST / "pyALDVC"

INTERESTING = re.compile(
    r"^\d+ WARNING: (Hidden import .* not found|Library not found|Cannot find |lib not found)|^\d+ ERROR:",
)


RMTREE_ATTEMPTS = 10
RMTREE_DELAY_S = 2.0


def _rmtree(path: Path) -> None:
    """Remove a previous build tree, tolerating transient locks on Windows.

    OneDrive's sync engine and antivirus scanners hold handles on freshly
    written files; an emptied directory can then stay "access denied" for a
    while. After the retries the tree is renamed out of the way instead so the
    build can proceed (the stale copy is gitignored and can be deleted later).
    """
    if not path.exists():
        return
    for attempt in range(1, RMTREE_ATTEMPTS + 1):
        try:
            shutil.rmtree(path)
            return
        except PermissionError as exc:
            if attempt == RMTREE_ATTEMPTS:
                stale = path.with_name(f"{path.name}.stale-{int(time.time())}")
                try:
                    path.rename(stale)
                except OSError as rename_exc:
                    raise SystemExit(
                        f"could not remove {path} ({exc.filename}: {exc.strerror}) nor rename it ({rename_exc.strerror}). "
                        "Something holds the directory open (OneDrive sync, an antivirus scan, an Explorer window, "
                        "a still-running pyALDVC.exe). Close it or build outside the synced folder."
                    ) from rename_exc
                print(f"  {path.name}: still locked ({exc.filename}); moved the stale tree to {stale.name}")
                return
            print(f"  {path.name}: locked ({exc.filename}); retrying in {RMTREE_DELAY_S:.0f} s ...")
            time.sleep(RMTREE_DELAY_S)


def _version() -> str:
    text = (ROOT / "src" / "al_dvc" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.M)
    if match is None:
        raise SystemExit("could not read __version__ from src/al_dvc/__init__.py")
    return match.group(1)


def _check_environment() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        raise SystemExit(
            "PyInstaller is not installed in this interpreter.\n"
            f"  {sys.executable} -m pip install -r packaging/requirements-build.txt"
        ) from None
    if "conda" in sys.prefix.lower() or "anaconda" in sys.prefix.lower():
        print(f"note: building from a conda environment ({sys.prefix}); a clean venv is the reference build environment.\n")


def _dll_search_env() -> dict:
    """``os.environ`` with this interpreter's own DLL directories first on PATH."""
    env = dict(os.environ)
    prefix = Path(sys.prefix)
    candidates = [prefix / "Library" / "bin", prefix / "DLLs", prefix]
    front = [str(p) for p in candidates if p.is_dir()]
    env["PATH"] = os.pathsep.join(front + [env.get("PATH", "")])
    return env


def _build(clean: bool) -> None:
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--distpath", str(DIST), "--workpath", str(WORK), str(SPEC)]
    if clean:
        cmd.insert(3, "--clean")
    print("$", " ".join(cmd), "\n")
    proc = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", env=_dll_search_env()
    )
    log = (proc.stdout or "") + (proc.stderr or "")
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "build.log").write_text(log, encoding="utf-8")
    flagged = [line for line in log.splitlines() if INTERESTING.match(line)]
    if flagged:
        print("--- missing imports / libraries reported by PyInstaller ---")
        for line in flagged:
            print(" ", line)
        print()
    if proc.returncode != 0:
        print(log[-6000:], file=sys.stderr)
        raise SystemExit(f"PyInstaller failed with exit code {proc.returncode}")
    print(f"build log: {WORK / 'build.log'}")


def _report_size() -> None:
    files = [f for f in BUNDLE.rglob("*") if f.is_file()]
    total = sum(f.stat().st_size for f in files)
    print(f"\nbundle: {BUNDLE}\n  {len(files)} files, {total / 1024**2:.0f} MB uncompressed")
    for f in sorted(files, key=lambda f: f.stat().st_size, reverse=True)[:8]:
        print(f"    {f.stat().st_size / 1024**2:7.1f} MB  {f.relative_to(BUNDLE)}")


def _zip() -> Path:
    out = DIST / f"pyALDVC-{_version()}-win64.zip"
    if out.exists():
        out.unlink()
    print(f"\nzipping -> {out.name} ...")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in sorted(BUNDLE.rglob("*")):
            if f.is_file():
                zf.write(f, Path("pyALDVC") / f.relative_to(BUNDLE))
    print(f"  {out.stat().st_size / 1024**2:.0f} MB")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args(argv)
    _check_environment()
    if args.clean and WORK.exists():
        _rmtree(WORK)
    if BUNDLE.exists():
        _rmtree(BUNDLE)
    _build(args.clean)
    if not (BUNDLE / "pyALDVC.exe").is_file():
        raise SystemExit(f"build finished but {BUNDLE / 'pyALDVC.exe'} is missing")
    _report_size()
    if not args.no_zip:
        _zip()
    print("\nself-test:  dist-exe\\pyALDVC\\pyALDVC-console.exe --self-test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
