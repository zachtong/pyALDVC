"""The window's frame cache: browsing a long sequence must not pin every frame it has shown."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from al_dvc.gui.app_state import AppState  # noqa: E402
from al_dvc.io.volume_io import save_volume  # noqa: E402

SHAPE = (8, 10, 12)


@pytest.fixture
def sequence(tmp_path, monkeypatch):
    """Six small frames on disk, with a cache limit of two so the eviction is exercised."""
    monkeypatch.setenv("PYALDVC_CACHE_FRAMES", "2")
    paths = []
    for k in range(6):
        p = tmp_path / f"frame_{k:02d}.npy"
        save_volume(p, np.full(SHAPE, k, dtype=np.uint16))
        paths.append(str(p))
    state = AppState()
    state.add_volume_paths(paths)
    return state


def test_browsing_keeps_only_the_budget(sequence):
    state = sequence
    assert state.cache_limit() == 2
    for k in range(6):
        state.volume_array(k)
    resident = [i for i, v in enumerate(state.volumes) if v.resident]
    assert len(resident) <= 2, resident
    assert 0 in resident  # the reference is never dropped
    assert state.volumes[5].resident  # nor is the frame that was just asked for


def test_the_frame_on_screen_survives(sequence):
    state = sequence
    state.volume_array(0)  # the reference, as the viewer reads it when the sequence is opened
    state.set_current_frame(3)
    state.volume_array(3)
    for k in (1, 2, 4, 5):
        state.volume_array(k)
    assert state.volumes[3].resident and state.volumes[0].resident


def test_a_frame_held_by_a_caller_stays_valid_after_release(sequence):
    """Releasing drops the cache's reference only; the array a caller holds is untouched."""
    state = sequence
    held = state.volume_array(4)
    for k in (1, 2, 5):
        state.volume_array(k)
    assert not state.volumes[4].resident
    assert held.shape == SHAPE and int(held.flat[0]) == 4  # still readable, still the right frame
    assert np.array_equal(state.volume_array(4), held)  # and re-reading gives the same thing


def test_an_in_memory_frame_is_never_released(monkeypatch):
    monkeypatch.setenv("PYALDVC_CACHE_FRAMES", "1")
    state = AppState()
    state.set_volume_arrays([np.zeros(SHAPE, dtype=np.uint16) for _ in range(4)])
    for k in range(4):
        state.volume_array(k)
    assert all(v.resident for v in state.volumes)  # there is no file to read them back from


def test_nothing_is_released_while_a_run_is_going(sequence, monkeypatch):
    state = sequence
    for k in range(6):
        state.volume_array(k)
    monkeypatch.setattr(type(state), "busy", property(lambda self: True))
    state.volumes[2].load()
    state.volumes[3].load()
    assert state.release_frames() == 0  # the worker reads through the entries
