#!/usr/bin/env python
"""Check that every guard in ``tests/test_large_volume_guards.py`` fails when its optimisation is undone.

A regression guard that passes both before and after the change it protects is worse than no test at
all: it reads as coverage and asserts nothing. So for each optimisation this reverts it in the source,
runs the one test that should notice, and puts the source back.

It edits files under ``src/`` and restores them in a ``finally``, so it **refuses to run unless the
working tree is clean** -- then a hard kill costs one ``git checkout -- src`` and nothing else.

Usage::

    python scripts/check_guards.py [-k <substring>]

Exit status is 0 when every guard caught its regression.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST = "tests/test_large_volume_guards.py"

# (label, file, the optimised code, the code before it, the test that must fail)
BREAKS: list[tuple[str, str, str, str, str]] = [
    (
        "box() does not cache its whole-volume scan",
        "src/al_dvc/gui/mask_editor.py",
        """        if not self._box_valid:
            self._box = box_of_mask(self.mask) if self.mask.any() else None
            self._box_valid = True
        return self._box""",
        """        return box_of_mask(self.mask) if self.mask.any() else None""",
        "test_one_texture_window_edit_scans_the_volume_once",
    ),
    (
        "a depth-limited op rasterises the whole volume",
        "src/al_dvc/gui/mask_editor.py",
        """    first, last = (0, n_len - 1) if op.depth is None else op.depth""",
        """    first, last = (0, n_len - 1)""",
        "test_a_mask_operation_allocates_nothing_volume_sized",
    ),
    (
        'base="full" materialises ones() and then copies it',
        "src/al_dvc/gui/mask_editor.py",
        """            self._full_base, self.base = True, None
            return""",
        """            self._full_base, self.base = False, np.ones(self.shape, dtype=bool)
            return""",
        "test_an_editor_over_the_whole_volume_holds_one_boolean_volume",
    ),
    (
        "the mask placeholder becomes a real ones() volume",
        "src/al_dvc/io/volume_ops.py",
        """        m = np.ones((1, 1, 1), dtype=np.uint8)""",
        """        m = np.ones(f.shape, dtype=np.uint8)""",
        "test_a_run_without_a_mask_carries_no_mask_volume",
    ),
    (
        "an edit repaints once per changed thing instead of once per edit",
        "src/al_dvc/gui/region_viewer.py",
        """        if self._hold > 0:""",
        """        if False:""",
        "test_one_edit_repaints_the_region_viewer_once",
    ),
    (
        "the whole-volume reference bundle is rebuilt on every request",
        "src/al_dvc/solver/tiling.py",
        """        if box.is_whole:
            if self._whole is None:
                self._whole = build_reference_bundle(self.f, self.mask, self.gradient_mode)
            return self._whole""",
        """        if box.is_whole:
            return build_reference_bundle(self.f, self.mask, self.gradient_mode)""",
        "test_the_reference_gradients_are_not_computed_twice_per_reference",
    ),
    (
        "the grey window is resampled on every redraw",
        "src/al_dvc/gui/panels/viewer.py",
        """        self._empty.setVisible(False)
        self._update_config_label()""",
        """        self._empty.setVisible(False)
        if self._volume is not None:
            self._vmin, self._vmax = grey_limits(self._volume)
        self._update_config_label()""",
        "test_the_grey_window_is_sampled_once_per_volume",
    ),
    (
        "the lattice plan is not cached",
        "src/al_dvc/gui/panels/viewer.py",
        """            if self._plan_cache is not None and self._plan_cache[0] == key:""",
        """            if False:""",
        "test_the_lattice_preview_is_not_recomputed_on_a_slider_tick",
    ),
]


def tree_is_clean() -> bool:
    r = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0 and not r.stdout.strip()


def test_passes(name: str) -> bool:
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    r = subprocess.run(
        [sys.executable, "-m", "pytest", f"{TEST}::{name}", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    return r.returncode == 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-k", dest="pattern", default=None, help="only the guards whose label contains this")
    args = ap.parse_args(argv)

    if not tree_is_clean():
        print("src/ has uncommitted changes -- refusing to edit it. Commit or stash first.")
        return 2

    cases = [b for b in BREAKS if args.pattern is None or args.pattern.lower() in b[0].lower()]
    if not cases:
        print("no guard matched")
        return 2

    missed: list[str] = []
    for label, rel, optimised, before, name in cases:
        path = ROOT / rel
        original = path.read_text(encoding="utf-8")
        if original.count(optimised) != 1:
            print(f"STALE   {label}: the anchor no longer matches {rel} -- update this script")
            missed.append(label)
            continue
        path.write_text(original.replace(optimised, before, 1), encoding="utf-8", newline="\n")
        try:
            still_passes = test_passes(name)
        finally:
            path.write_text(original, encoding="utf-8", newline="\n")
        print(f"{'MISSED ' if still_passes else 'caught '} {label}\n         -> {name}")
        if still_passes:
            missed.append(label)

    print()
    if missed:
        print(f"{len(missed)} guard(s) did not catch their regression:")
        for label in missed:
            print(f"  - {label}")
        return 1
    print(f"all {len(cases)} guards caught their regression")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
