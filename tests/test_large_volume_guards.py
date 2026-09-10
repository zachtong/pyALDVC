"""Guards on the *shape* of the work, not its speed, so they can run in CI on any runner.

The large-volume work of 2026-09 removed whole-volume passes and allocations that had no reason to
scale with the scan (see `CHANGELOG.md` and `docs/large_volume_limits.md`). Timing thresholds cannot
protect that -- a CI runner's speed varies, which is why the `perf` tests are excluded -- but two
things about it are deterministic and machine-independent: **how many bytes are allocated** and **how
many times a function is called**. Those are what this file pins.

The other guards of the same kind live with their subjects:

- `tests/test_mesh.py::test_subset_valid_fraction_allocates_nothing_volume_sized`
- `tests/test_tiling.py::test_tiling_never_hands_the_gradient_kernel_a_whole_volume`
- `tests/test_frame_cache.py` -- the browsing cache stays inside its budget

If one of these fails, the question is not "is the machine slow" but "did a whole-volume pass come
back".

`scripts/check_guards.py` reverts each optimisation in turn and checks that the matching test here
fails -- a guard that passes either way asserts nothing. Add a case there with every guard added here.
"""

from __future__ import annotations

import os
import tracemalloc

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from al_dvc.gui import mask_editor as mask_editor_module  # noqa: E402
from al_dvc.gui.app import MainWindow, create_application  # noqa: E402
from al_dvc.gui.mask_editor import MaskEditor, MaskOp  # noqa: E402
from al_dvc.io.volume_ops import build_reference_bundle, normalize_volume  # noqa: E402

SHAPE = (48, 56, 64)  # small: these assertions are about ratios, not about size
BIG = (64, 64, 64)


@pytest.fixture(scope="module")
def qapp():
    return create_application(["pytest"])


def _pump(n=15):
    for _ in range(n):
        QApplication.processEvents()


def _peak_bytes(fn) -> int:
    fn()  # warm anything lazy first, so the measurement is of the work
    tracemalloc.start()
    fn()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return int(peak)


# ----------------------------------------------------------------------------- allocation
def test_a_mask_operation_allocates_nothing_volume_sized():
    """A drawn shape is 2-D and extruded; it used to be built as a full boolean volume first."""
    n_vox = int(np.prod(BIG))
    editor = MaskEditor(BIG, base="full")
    one_slice = MaskOp("rectangle", "xy", ((4.0, 5.0), (58.0, 59.0)), depth=(30, 30), mode="add")
    whole = MaskOp("rectangle", "xy", ((4.0, 5.0), (58.0, 59.0)), mode="add")
    assert _peak_bytes(lambda: editor.apply(one_slice)) < 0.05 * n_vox
    # a whole-depth op writes the volume once through a broadcast view; it must not also build one
    assert _peak_bytes(lambda: editor.apply(whole)) < 0.5 * n_vox


def test_an_editor_over_the_whole_volume_holds_one_boolean_volume():
    """base="full" is symbolic: an ones() volume plus the copy the editor makes of it was two."""
    n_vox = int(np.prod(BIG))
    assert _peak_bytes(lambda: MaskEditor(BIG, base="full")) < 1.4 * n_vox
    editor = MaskEditor(BIG, base="full")
    assert editor.base is None and editor.mask.nbytes == n_vox


def test_a_run_without_a_mask_carries_no_mask_volume():
    """An all-ones uint8 volume is one byte per voxel of RAM, and the same again in VRAM."""
    volume = normalize_volume(np.random.default_rng(0).integers(0, 4000, SHAPE, dtype=np.uint16))
    n_vox = float(np.prod(SHAPE))
    bundle = build_reference_bundle(volume, None, "on_the_fly")
    assert not bundle.has_mask and bundle.mask.shape == (1, 1, 1)
    resident = sum(getattr(bundle, name).nbytes for name in ("f", "gx", "gy", "gz", "mask"))
    assert resident <= 4.0 * n_vox + 64  # the normalised volume, and 13 bytes of 1x1x1 placeholders
    stored = build_reference_bundle(volume, None, "stored")
    kept = sum(getattr(stored, name).nbytes for name in ("f", "gx", "gy", "gz", "mask"))
    assert kept <= 16.0 * n_vox + 64  # the volume and its three gradients, and the placeholder mask


def test_the_reference_gradients_are_not_computed_twice_per_reference(monkeypatch):
    """The bundle of the whole volume is built once and kept; a tile plan builds one per box."""
    import al_dvc.io.volume_ops as volume_ops
    from al_dvc.solver.tiling import ReferenceSource, whole_box_tile

    calls: list[tuple[int, ...]] = []
    real = volume_ops.compute_gradients
    monkeypatch.setattr(volume_ops, "compute_gradients", lambda f: (calls.append(f.shape), real(f))[1])
    volume = normalize_volume(np.random.default_rng(1).integers(0, 4000, SHAPE, dtype=np.uint16))
    source = ReferenceSource(f=volume, mask=None, gradient_mode="stored")
    box = whole_box_tile(source.shape)
    for _ in range(3):
        source.bundle_for(box)
    assert len(calls) == 1


# ----------------------------------------------------------------------------- call counts
@pytest.fixture
def window(qapp):
    w = MainWindow()
    w.state.set_volume_arrays(
        [np.random.default_rng(k).integers(0, 4000, SHAPE, dtype=np.uint16) for k in range(2)], ["ref", "def"]
    )
    _pump()
    yield w
    w.close()


def _count_box_of_mask(monkeypatch) -> list[int]:
    """Count the whole-volume bounding-box scans. The editor caches it; the callers do not."""
    seen = [0]
    real = mask_editor_module.box_of_mask

    def spy(mask):
        seen[0] += 1
        return real(mask)

    monkeypatch.setattr(mask_editor_module, "box_of_mask", spy)
    return seen


def test_one_texture_window_edit_scans_the_volume_once(window, monkeypatch):
    """The texture window asks for the region's box a dozen times per edit; only one may scan."""
    tw = window.open_texture_window()
    _pump(25)
    tw.go_to_step(tw.TAB_SWEEP)
    _pump()
    seen = _count_box_of_mask(monkeypatch)
    tw.region.apply(MaskOp("rectangle", "xy", ((8.0, 9.0), (40.0, 30.0)), mode="replace"))
    _pump()
    assert seen[0] == 1, f"{seen[0]} whole-volume bounding-box scans for one edit"
    seen[0] = 0
    tw.region.set_slices(20, 24, 28)  # a slider tick must not scan at all: nothing changed the region
    _pump()
    assert seen[0] == 0
    tw.close()


def test_one_edit_repaints_the_region_viewer_once(window, monkeypatch):
    """An edit moves the region, its bounding box and the cube overlay; that is one repaint, not three."""
    tw = window.open_texture_window()
    _pump(25)
    tw.go_to_step(tw.TAB_SWEEP)
    _pump()
    painted = [0]
    real = tw.region.canvas.draw_idle
    monkeypatch.setattr(tw.region.canvas, "draw_idle", lambda: (painted.__setitem__(0, painted[0] + 1), real())[1])
    tw.region.apply(MaskOp("rectangle", "xy", ((8.0, 9.0), (40.0, 30.0)), mode="replace"))
    _pump()
    assert painted[0] == 1, f"{painted[0]} repaints for one edit"
    painted[0] = 0
    tw.region.set_slices(20, 24, 28)  # three sliders, one repaint
    _pump()
    assert painted[0] == 1
    tw.close()


def test_the_lattice_preview_is_not_recomputed_on_a_slider_tick(window, monkeypatch):
    """plan_lattice flood-fills every element box; the mask and the parameters did not move."""
    import al_dvc.gui.panels.viewer as viewer_module

    mask = np.zeros(SHAPE, dtype=bool)
    mask[6:-6, 8:-8, 10:-10] = True
    window.state.set_mask(0, mask=mask)
    # The plan must actually succeed: the default 32/16 does not fit a volume this small, and
    # build_grid_axes raises before any volume-sized work, so the error path would cache nothing and
    # this test would pass while measuring the wrong thing.
    window.state.set_params(winsize=(16, 16, 16), winstepsize=(8, 8, 8))
    _pump()
    calls = [0]
    real = viewer_module.plan_lattice

    def spy(*args, **kwargs):
        calls[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(viewer_module, "plan_lattice", spy)
    window.viewer.redraw()
    _pump()
    first = calls[0]
    assert window.viewer._plan_cache is not None, "the plan failed, so this measures the error path"

    for index in (10, 12, 14):
        window.viewer._on_slider("z", index)
        _pump()
    assert calls[0] == first, f"{calls[0] - first} lattice previews for three slider ticks"
    window.state.set_params(winsize=(24, 24, 24))  # a parameter change must recompute it
    _pump()
    assert calls[0] > first


def test_the_grey_window_is_sampled_once_per_volume(window, monkeypatch):
    """The percentiles used to be recomputed per redraw, and for a float volume built two copies."""
    import al_dvc.gui.panels.viewer as viewer_module

    calls = [0]
    real = viewer_module.grey_limits

    def spy(volume, *args, **kwargs):
        calls[0] += 1
        return real(volume, *args, **kwargs)

    monkeypatch.setattr(viewer_module, "grey_limits", spy)
    window.state.set_current_frame(1)
    _pump()
    after_load = calls[0]
    assert after_load >= 1
    for _ in range(3):
        window.viewer.redraw()
        _pump()
    assert calls[0] == after_load, "the grey window was resampled by a redraw"
