"""Session files, format 3: a faithful round trip of everything, projects whose files moved, older formats,
and files that are damaged or hostile. The window is not needed here (see test_session_gui.py)."""

from __future__ import annotations

import builtins
import json
import os
import shutil
import zipfile
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from al_dvc.analysis import Correction, NodeFilter, Region  # noqa: E402
from al_dvc.core.config import dvcpara_default, para_to_dict  # noqa: E402
from al_dvc.core.pipeline import run_aldvc  # noqa: E402
from al_dvc.gui.app_state import AppState, RunState  # noqa: E402
from al_dvc.gui.mask_editor import MaskEditor, MaskOp  # noqa: E402
from al_dvc.gui.session import (  # noqa: E402
    FORMAT_VERSION,
    SessionError,
    apply_session,
    load_session,
    locate_missing,
    save_session,
)
from al_dvc.io import session_paths  # noqa: E402
from al_dvc.io.session_bundle import DOCUMENT, RESULTS, SessionBundle, is_bundle, read_session_document  # noqa: E402
from al_dvc.io.session_serialize import (  # noqa: E402
    SerializeError,
    decode,
    encode,
    load_object_npz,
    load_result_npz,
    save_object_npz,
    save_result_npz,
    write_npz,
)
from al_dvc.io.volume_io import save_volume  # noqa: E402
from al_dvc.strain.compute_strain import compute_strain  # noqa: E402
from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian  # noqa: E402

SHAPE = (36, 40, 44)  # (nz, ny, nx)


@pytest.fixture(scope="module")
def qapp():
    from al_dvc.gui.app import create_application

    return create_application(["pytest"])


@pytest.fixture(scope="module")
def frames():
    centre = tuple((s - 1) / 2 for s in SHAPE[::-1])
    ref = generate_speckle_volume(SHAPE, sigma=2.0, seed=21)
    out = [ref]
    for k in (1, 2):
        disp = affine_displacement(np.diag([0.004 * k, -0.002 * k, 0.003 * k]), (0.3 * k, -0.2, 0.1), centre)
        out.append(warp_volume_lagrangian(ref, disp))
    return out


@pytest.fixture(scope="module")
def run_result(frames):
    """A completed run with strain: the real records (ADMM diagnostics, local passes, flags, uncertainty)."""
    para = dvcpara_default(winsize=16, winstepsize=8, search_radius=4, admm_max_iter=2, verbose=False)
    res = run_aldvc(para, list(frames), None, compute_strain=False)
    # the strain the post-processing window computes, with its own parameters
    spara = replace(res.dvc_para, strain_method="fd", strain_type="green_lagrange", strain_plane_fit_halfwidth=(2, 2, 2))
    strains = [compute_strain(res.dvc_mesh, spara, fr.U_accum, valid=res.dvc_mesh.node_valid) for fr in res.result_disp]
    return replace(res, result_strain=strains, dvc_para=spara)


def assert_same(a, b, where: str = "result") -> None:
    """``b`` is ``a`` bit for bit: same types, same dtypes, same shapes and memory order, same bytes."""
    assert type(a) is type(b), f"{where}: {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape, f"{where}: {a.dtype}{a.shape} != {b.dtype}{b.shape}"
        # a contiguous array keeps its memory order; a strided view (a column of U) comes back as its own array
        assert not a.flags.c_contiguous or b.flags.c_contiguous, f"{where}: C order lost"
        assert not a.flags.f_contiguous or b.flags.f_contiguous, f"{where}: Fortran order lost"
        assert np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes(), f"{where}: values differ"
    elif isinstance(a, np.generic):
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), where
    elif is_dataclass(a):
        for f in fields(a):
            assert_same(getattr(a, f.name), getattr(b, f.name), f"{where}.{f.name}")
    elif isinstance(a, dict):
        assert list(a) == list(b), f"{where}: keys {list(a)} != {list(b)}"
        for k in a:
            assert_same(a[k], b[k], f"{where}[{k!r}]")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{where}[{i}]")
    elif isinstance(a, float):
        assert (a == b) or (np.isnan(a) and np.isnan(b)), f"{where}: {a} != {b}"
    else:
        assert a == b, f"{where}: {a!r} != {b!r}"


def _volumes(folder: Path, frames, names=("v0.npy", "v1.npy", "v2.npy")) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, vol in zip(names, frames):
        path = folder / name
        save_volume(path, vol)
        paths.append(path)
    return paths


def _state(qapp, paths, result=None) -> AppState:
    state = AppState()
    state.add_volume_paths([str(p) for p in paths])
    assert state.wait_for_shapes(10_000)
    if result is not None:
        state.set_results(result)
    return state


# ---------------------------------------------------------------------------- the serializer
def test_the_result_archive_is_bit_exact(run_result, tmp_path):
    res = run_result
    assert res.result_disp[0].admm is not None and res.result_disp[0].admm.beta_sweep is not None
    assert res.result_disp[0].U_std is not None and res.result_disp[0].outlier is not None and res.result_strain
    path = tmp_path / "r.npz"
    save_result_npz(path, res)
    back = load_result_npz(path)
    assert_same(res, back)
    # arrays shared by the result stay shared (one copy in the archive, one object when read)
    with np.load(path, allow_pickle=False) as npz:
        n_arrays = len(npz.files) - 1
    assert n_arrays < sum(1 for _ in _arrays(res))


def _arrays(obj):
    if isinstance(obj, np.ndarray):
        yield obj
    elif is_dataclass(obj):
        for f in fields(obj):
            yield from _arrays(getattr(obj, f.name))
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _arrays(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _arrays(v)


def test_the_serializer_keeps_types_orders_and_keys():
    obj = {
        "f32": np.float32(1.5),
        "nan": float("nan"),
        "inf": -float("inf"),
        "neg0": -0.0,
        "keys": {0.3678794411714424: "1/e", 0.1: (1, 2)},
        "slices": (slice(2, 10), slice(None, 4, 2)),
        "fortran": np.asfortranarray(np.arange(12.0).reshape(3, 4)),
        "i8": np.array([-1, 0, 7], dtype=np.int8),
        "bool": np.array([True, False]),
        "empty": np.empty((0, 3)),
        "nested": [[1, "a"], (None, True)],
    }
    arrays, root = encode(obj)
    json.dumps(root, allow_nan=False)  # strict JSON
    back = decode(root, arrays)
    assert_same(obj, back, "obj")


def test_the_texture_analysis_round_trips(frames, tmp_path):
    from al_dvc.texture import analyse_cube, sweep_concentric

    vol = frames[0]
    result = analyse_cube(vol, ((4, 36), (4, 36), (2, 34)), (1.0, 1.0, 1.0), None)
    sweep = sweep_concentric(vol, (22, 20, 18), None, start=8, step=8, count=3)
    path = tmp_path / "t.npz"
    save_object_npz(path, {"result": result, "sweep": sweep}, "texture")
    back = load_object_npz(path, "texture", dict)
    assert_same({"result": result, "sweep": sweep}, back, "texture")


@pytest.mark.parametrize(
    "root",
    [
        {"__dataclass__": "Popen", "fields": {}},  # not a record a session holds
        {"__dataclass__": "DVCMesh", "fields": {"coordinates": None, "__class__": 1}},  # a field it does not have
        {"__ndarray__": "a99999"},  # an array that is not there
        {"__npscalar__": "|O", "value": 1},  # a scalar of Python objects
        {"__slice__": ["a", None, None]},
        {"__weird__": 1},
        {"__float__": "1e999"},
    ],
)
def test_the_decoder_refuses_what_it_did_not_write(root):
    with pytest.raises(SerializeError):
        decode(root, {})
    deep = root
    for _ in range(100):
        deep = [deep]
    with pytest.raises(SerializeError):
        decode(deep, {})


class _Evil:
    def __reduce__(self):
        return (exec, ("import builtins; builtins._pyaldvc_pwned = True",))


def _bundle(path: Path, document: dict, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(DOCUMENT, json.dumps(document))
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_a_pickle_in_the_archive_is_refused_and_never_run(qapp, run_result, frames, tmp_path):
    paths = _volumes(tmp_path / "data", frames)
    good = save_session(_state(qapp, paths, run_result), tmp_path / "good.aldvc")
    doc = read_session_document(good)
    import io

    evil = io.BytesIO()
    with zipfile.ZipFile(evil, "w") as zf:
        manifest = json.dumps({"serialize_version": 1, "kind": "PipelineResult", "root": {"__ndarray__": "a00000"}})
        buf = io.BytesIO()
        np.save(buf, np.array(list(manifest.encode()), dtype=np.uint8))
        zf.writestr("__manifest__.npy", buf.getvalue())
        buf = io.BytesIO()
        np.save(buf, np.array([_Evil()], dtype=object), allow_pickle=True)
        zf.writestr("a00000.npy", buf.getvalue())
    bad = _bundle(tmp_path / "evil.aldvc", doc, {RESULTS: evil.getvalue()})
    with pytest.raises(SessionError):
        load_session(bad)
    assert not hasattr(builtins, "_pyaldvc_pwned")


def test_damaged_and_foreign_files_raise_session_errors(qapp, run_result, frames, tmp_path):
    paths = _volumes(tmp_path / "data", frames)
    good = save_session(_state(qapp, paths, run_result), tmp_path / "good.aldvc")
    doc = read_session_document(good)
    raw = good.read_bytes()
    cases = {
        "truncated.aldvc": raw[: len(raw) // 2],
        "garbage.aldvc": os.urandom(4096),
        "empty.aldvc": b"",
        "text.aldvc": b"not a session",
    }
    for name, data in cases.items():
        (tmp_path / name).write_bytes(data)
        with pytest.raises(SessionError):
            load_session(tmp_path / name)
    with zipfile.ZipFile(tmp_path / "nodoc.aldvc", "w") as zf:
        zf.writestr("results.npz", b"PK")
    with pytest.raises(SessionError, match="session.json"):
        load_session(tmp_path / "nodoc.aldvc")
    newer = _bundle(tmp_path / "newer.aldvc", {**doc, "format": FORMAT_VERSION + 1}, {})
    with pytest.raises(SessionError, match="newer"):
        load_session(newer)
    for broken in (
        {**doc, "volumes": "all"},
        {**doc, "volumes": [{"path": 3}]},
        {**doc, "para": {"winsize": -1}},
        {**doc, "display": "dark"},
        {**doc, "views": {"view3d": 1}},
        {**doc, "results": {"archive": RESULTS, "result_uids": "x"}},
        {**doc, "results": {"archive": RESULTS, "result_uids": []}},  # the archive is not in the file
        {**doc, "volumes": [{**doc["volumes"][0], "mask": {"embedded": "m42"}}]},
        {**doc, "analysis": {"regions": [{"id": 1, "shape": "cube"}]}},
    ):
        path = _bundle(tmp_path / "broken.aldvc", broken, {})
        with pytest.raises(SessionError):
            load_session(path)
    # a manifest naming a record that does not exist
    bogus = tmp_path / "bogus.npz"
    write_npz(bogus, {}, {"serialize_version": 1, "kind": "PipelineResult", "root": {"__dataclass__": "os.system", "fields": {}}})
    path = _bundle(tmp_path / "bogus.aldvc", doc, {RESULTS: bogus.read_bytes()})
    with pytest.raises(SessionError):
        load_session(path)


def test_a_failed_save_leaves_the_previous_file_untouched(qapp, run_result, frames, tmp_path, monkeypatch):
    from al_dvc.gui import session as session_module

    paths = _volumes(tmp_path / "data", frames)
    state = _state(qapp, paths, run_result)
    target = tmp_path / "proj" / "s.aldvc"
    save_session(state, target)
    before = target.read_bytes()

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(session_module, "save_result_npz", boom)
    state.set_display(colormap="magma")
    state.dirty = True
    with pytest.raises(SessionError, match="disk full"):
        save_session(state, target)
    assert target.read_bytes() == before
    assert sorted(p.name for p in target.parent.iterdir()) == ["s.aldvc"]  # no temporary file left behind
    assert state.dirty  # nothing was saved


# ---------------------------------------------------------------------------- everything comes back
def test_a_completed_session_round_trips_exactly(qapp, run_result, frames, tmp_path):
    paths = _volumes(tmp_path / "data", frames)
    state = _state(qapp, paths, run_result)
    shape = SHAPE
    drawn = np.zeros(shape, dtype=bool)
    drawn[4:30, 5:35, 6:40] = True
    state.set_mask(0, mask=drawn)
    state.set_mask(1, mask=drawn.copy())  # the same mask on two frames: stored once
    mask_file = tmp_path / "data" / "mask2.npy"
    other = np.zeros(shape, dtype=np.uint8)
    other[10:20, 10:20, 10:20] = 1
    save_volume(mask_file, other)
    assert state.set_mask(2, path=str(mask_file))
    state.volumes[1].label = "first load step"
    grip = Region(1, "grip", "sphere", {"centre": [22, 20, 18], "radius": 9.0}, color="#ff0000")
    state.set_regions([grip, Region(2, "slab", "box", {"lo": [0, 0, 0], "hi": [30, 30, 12]})])
    state.set_display_correction(Correction("rigid", NodeFilter(converged_only=True, min_zncc=0.8), fit_region=grip))
    state.analysis_settings = {"motion": "rigid", "group": "strain", "shown": "exx", "min_share": 40}
    state.set_output_dir(tmp_path / "out")
    state.results_path = str(tmp_path / "out" / "aldvc.npz")
    state.write_checkpoints = True
    state.set_current_frame(2)
    state.set_display(
        display_field="exx",
        colormap="magma",
        color_auto=False,
        color_min=-0.01,
        color_max=0.02,
        overlay_alpha=0.4,
        show_overlay=False,
        background_frame=0,
        slice_layout="row",
        slice_equal_scale=True,
        show_mesh=False,
        show_subset_window=True,
    )
    state.slice_index = {"z": 7, "y": 11, "x": 13}
    state.set_mask_display(show=True, alpha=0.6, target="all")
    views = {"main": {"tab": 1}, "view3d": {"mode": "surface", "iso_levels": 3}, "post": {"tab": 1, "frame": 1}}
    path = save_session(state, tmp_path / "proj" / "s.aldvc", views=views)
    assert is_bundle(path) and not state.dirty
    with SessionBundle(path) as bundle:
        assert {DOCUMENT, RESULTS, "masks.npz"} <= bundle.names
        _arr, meta = bundle.arrays("masks.npz")
        assert len(meta["masks"]) == 2  # frames 0 and 1 share one mask; the mask file of frame 2 is the other
    doc = read_session_document(path)
    assert doc["format"] == 3 and doc["volumes"][0]["relative"] == "../data/v0.npy"
    assert doc["volumes"][2]["mask_source"]["relative"] == "../data/mask2.npy"

    data = load_session(path)
    fresh = AppState()
    assert apply_session(data, fresh, path) == []
    assert_same(run_result, fresh.results)
    assert [v.uid for v in fresh.volumes] == [v.uid for v in state.volumes]
    assert fresh.result_uids == state.result_uids and fresh.result_frame() == state.result_frame() == 1
    assert [v.label for v in fresh.volumes] == ["", "first load step", ""]
    assert [v.header_shape for v in fresh.volumes] == [shape] * 3  # sizes known from the session: nothing re-read
    assert fresh.volumes[0].mask.dtype == np.bool_ and np.array_equal(fresh.volumes[0].mask, drawn)
    assert fresh.volumes[1].mask is fresh.volumes[0].mask  # one array for the frames that share a mask
    assert np.array_equal(fresh.volumes[2].mask, other > 0)
    assert fresh.regions == state.regions and fresh.display_correction == state.display_correction
    assert fresh.analysis_settings == state.analysis_settings
    assert fresh.output_dir == state.output_dir and fresh.results_path == state.results_path
    assert fresh.write_checkpoints is True and fresh.para == state.para
    for key in (
        "display_field",
        "colormap",
        "color_auto",
        "color_min",
        "color_max",
        "overlay_alpha",
        "show_overlay",
        "background_frame",
        "slice_layout",
        "slice_equal_scale",
        "show_mesh",
        "show_subset_window",
        "current_frame",
        "slice_index",
        "show_mask",
        "mask_alpha",
        "mask_target",
    ):
        assert getattr(fresh, key) == getattr(state, key), key
    assert fresh.ui_state == views
    assert fresh.run_state == RunState.DONE and not fresh.dirty
    assert fresh.session_path == path


def test_saving_an_opened_session_again_writes_the_same_content(qapp, run_result, frames, tmp_path):
    paths = _volumes(tmp_path / "data", frames)
    first = save_session(_state(qapp, paths, run_result), tmp_path / "a.aldvc")
    state = AppState()
    apply_session(load_session(first), state, first)
    second = save_session(state, tmp_path / "b.aldvc")
    a, b = read_session_document(first), read_session_document(second)
    for doc in (a, b):
        doc.pop("saved")
    assert a == b
    assert_same(load_session(first).results, load_session(second).results)


# ---------------------------------------------------------------------------- moved projects
def _saved_project(qapp, root: Path, frames, result, data_dir="data") -> Path:
    paths = _volumes(root / data_dir, frames)
    state = _state(qapp, paths, result)
    state.set_output_dir(root / "proj" / "results")
    return save_session(state, root / "proj" / "s.aldvc")


def test_a_project_moved_as_a_whole_opens(qapp, run_result, frames, tmp_path):
    """(a) The session and the volumes moved together; the volumes were outside the session's folder."""
    old = tmp_path / "old"
    _saved_project(qapp, old, frames, run_result)
    new = tmp_path / "elsewhere" / "renamed_project"
    shutil.move(str(old), str(new))
    data = load_session(new / "proj" / "s.aldvc")
    assert data.missing == [] and [v["how"] for v in data.volumes] == ["relative"] * 3
    state = AppState()
    apply_session(data, state, new / "proj" / "s.aldvc")
    assert [Path(v.path) for v in state.volumes] == [new / "data" / f"v{k}.npy" for k in range(3)]
    assert state.results is not None and state.run_state == RunState.DONE
    assert state.output_dir == new / "proj" / "results"  # it lived next to the session: it moved with it


def test_a_session_moved_alone_still_finds_its_volumes(qapp, run_result, frames, tmp_path):
    """(b) Only the session file moved: the saved absolute paths still hold."""
    session = _saved_project(qapp, tmp_path / "p", frames, run_result)
    moved = tmp_path / "desk" / "copy.aldvc"
    moved.parent.mkdir()
    shutil.move(str(session), str(moved))
    data = load_session(moved)
    assert data.missing == [] and [v["how"] for v in data.volumes] == ["saved"] * 3


def test_volumes_moved_next_to_the_session_are_found(qapp, run_result, frames, tmp_path):
    """(c) The volumes moved into a folder beside the session (named as their old folder), into the session's own
    folder, or with the session's folder as their new root (the relocation of one applies to the others)."""
    session = _saved_project(qapp, tmp_path / "p", frames, run_result)
    shutil.move(str(tmp_path / "p" / "data"), str(session.parent / "data"))
    data = load_session(session)
    assert data.missing == [] and {v["how"] for v in data.volumes} == {"beside"}
    for k in range(3):
        shutil.move(str(session.parent / "data" / f"v{k}.npy"), str(session.parent / f"v{k}.npy"))
    data = load_session(session)
    assert data.missing == [] and {v["how"] for v in data.volumes} == {"session_folder"}


def test_one_relocation_moves_the_others(qapp, run_result, frames, tmp_path):
    root = tmp_path / "p"
    names = ("v0.npy", "series/day1/v1.npy", "series/day2/v2.npy")
    for name in names[1:]:
        (root / "data" / Path(name).parent).mkdir(parents=True, exist_ok=True)
    paths = _volumes(root / "data", frames, names)
    session = save_session(_state(qapp, paths, run_result), root / "proj" / "s.aldvc")
    for name in names:  # the content of the data folder moves into the project folder
        target = session.parent / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(root / "data" / name), str(target))
    data = load_session(session)
    assert data.missing == []
    assert [v["how"] for v in data.volumes] == ["session_folder", "moved", "moved"]
    assert Path(data.volumes[2]["path"]) == session.parent / "series" / "day2" / "v2.npy"


def test_missing_volumes_keep_the_results(qapp, run_result, frames, tmp_path):
    """(d) The volumes are gone: everything else is restored, the frames are marked, the log says how to fix it."""
    session = _saved_project(qapp, tmp_path / "p", frames, run_result)
    shutil.rmtree(tmp_path / "p" / "data")
    data = load_session(session)
    assert len(data.missing) == 3
    state = AppState()
    seen = []
    state.log_message.connect(lambda message, level: seen.append((level, message)))
    missing = apply_session(data, state, session)
    assert missing == [str(tmp_path / "p" / "data" / f"v{k}.npy") for k in range(3)]
    assert all(v.missing for v in state.volumes) and state.results is not None and state.run_state == RunState.DONE
    assert_same(run_result, state.results)
    assert state.volume_shape() == SHAPE  # the saved size, without an error
    warning = next(m for level, m in seen if level == "warning" and "were not found" in m)
    assert "v0.npy" in warning and "choose the folder" in warning
    assert not state.dirty


def test_the_folder_the_user_points_at_rebinds_every_missing_volume(qapp, run_result, frames, tmp_path):
    """(e) Nothing is where the rules look; the user's folder is searched by name, and sibling folders follow."""
    root = tmp_path / "p"
    names = ("scan1/v0.npy", "scan1/v1.npy", "scan2/v2.npy")
    for sub in ("scan1", "scan2"):
        (root / "data" / sub).mkdir(parents=True)
    paths = _volumes(root / "data", frames, names)
    session = save_session(_state(qapp, paths, run_result), root / "proj" / "s.aldvc")
    archive = tmp_path / "archive" / "2026"
    archive.parent.mkdir()
    shutil.move(str(root / "data"), str(archive))
    data = load_session(session)
    assert len(data.missing) == 3
    asked = []

    def locate(first_missing):
        asked.append(first_missing)
        return str(archive / "scan1")

    state = AppState()
    assert apply_session(data, state, session, locate_folder_cb=locate) == []
    assert asked == [str(root / "data" / "scan1" / "v0.npy")]  # asked once, for the first missing file
    assert [Path(v.path) for v in state.volumes] == [archive / n for n in names]
    assert not any(v.missing for v in state.volumes)
    # the same through the data alone
    data = load_session(session)
    assert locate_missing(data, archive / "scan1") == []


def test_a_different_file_with_the_same_name_is_not_adopted(qapp, run_result, frames, tmp_path):
    """(f) A file that only shares the name is not the volume: it stays missing; one at the saved place that
    changed is used but reported."""
    session = _saved_project(qapp, tmp_path / "p", frames, run_result)
    shutil.rmtree(tmp_path / "p" / "data")
    save_volume(session.parent / "v0.npy", np.zeros((8, 8, 8), np.float32))  # same name, another size
    data = load_session(session)
    assert data.volumes[0]["missing"] and len(data.missing) == 3
    # a changed file at the saved place
    session = _saved_project(qapp, tmp_path / "q", frames, run_result)
    save_volume(tmp_path / "q" / "data" / "v1.npy", np.zeros((8, 8, 8), np.float32))
    data = load_session(session)
    assert data.missing == [] and "bytes instead of" in data.volumes[1]["changed"]
    assert data.volumes[1]["shape"] is None  # the saved size no longer describes it: it is read again
    state = AppState()
    seen = []
    state.log_message.connect(lambda message, level: seen.append((level, message)))
    apply_session(data, state, session)
    assert any(level == "warning" and "not the file the session was saved with" in m for level, m in seen)


def test_references_and_fingerprints():
    ref = session_paths.reference(__file__, Path(__file__).parent.parent)
    assert ref["relative"] == "tests/test_session.py" and ref["fingerprint"]["size"] == Path(__file__).stat().st_size
    assert session_paths.matches(__file__, ref)
    assert not session_paths.matches(__file__, {**ref, "fingerprint": {**ref["fingerprint"], "size": 1}})
    with pytest.raises(ValueError):
        session_paths.check_reference({"path": 5}, "x")
    with pytest.raises(ValueError):
        session_paths.check_reference({"path": "a", "fingerprint": {"size": -1}}, "x")


# ---------------------------------------------------------------------------- older formats
def _legacy_project(qapp, root: Path, frames, fmt: int) -> tuple[Path, np.ndarray]:
    """A session of format 1 or 2, as those versions wrote it (plain JSON; format 2 with the drawn masks as files in
    <name>_masks, format 1 with the drawing operations)."""
    paths = _volumes(root / "data", frames)
    mask = np.zeros(SHAPE, dtype=bool)
    mask[3:30, 4:32, 5:40] = True
    ed = MaskEditor(SHAPE)
    ed.apply(MaskOp("rectangle", "xy", ((5.0, 4.0), (39.0, 31.0)), depth=(3, 29)))
    assert np.array_equal(ed.mask, mask)
    volumes = []
    for k, p in enumerate(paths):
        entry = {"path": f"data/{p.name}", "label": f"f{k}", "mask": None, "mask_ops": None}
        if fmt == 2 and k < 2:
            (root / "s_masks").mkdir(exist_ok=True)
            save_volume(root / "s_masks" / f"mask_{k:02d}.npy", mask.astype(np.uint8))
            entry["mask"] = f"s_masks/mask_{k:02d}.npy"
        elif fmt == 1 and k < 2:
            entry["mask_ops"] = ed.to_dict()
        volumes.append(entry)
    doc = {
        "format": fmt,
        "pyaldvc": "1.1.0",
        "para_revision": 2,
        "volumes": volumes,
        "para": para_to_dict(dvcpara_default(winsize=20, winstepsize=10)),
        "output_dir": "results",
        "display": {"field": "disp_v", "frame": 1, "colormap": "viridis", "color_auto": True, "background_frame": 0},
        "results_path": None,
        "analysis": {"regions": [Region(3, "r", "sphere", {"centre": [1, 2, 3], "radius": 4.0}).as_dict()]},
    }
    path = root / "s.aldvc"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path, mask


@pytest.mark.parametrize("fmt", [1, 2])
def test_older_sessions_open_move_and_save_as_format_3(qapp, frames, tmp_path, fmt):
    session, mask = _legacy_project(qapp, tmp_path / "old", frames, fmt)
    shutil.move(str(tmp_path / "old"), str(tmp_path / "new"))  # the whole project moved, mask folder included
    session = tmp_path / "new" / "s.aldvc"
    data = load_session(session)
    assert data.format == fmt and data.missing == [] and data.results is None
    state = AppState()
    assert apply_session(data, state, session) == []
    assert [v.label for v in state.volumes] == ["f0", "f1", "f2"]
    for k in range(2):
        assert np.array_equal(state.volumes[k].load_mask(), mask)
    assert state.volumes[2].load_mask() is None
    assert state.display_field == "disp_v" and state.colormap == "viridis" and state.current_frame == 1
    assert state.para.winsize == (20, 20, 20) and len(state.regions) == 1
    assert state.output_dir == tmp_path / "new" / "results" and state.run_state == RunState.IDLE
    again = save_session(state, tmp_path / "new" / "s3.aldvc")
    assert is_bundle(again) and read_session_document(again)["format"] == 3
    back = load_session(again)
    assert np.array_equal(back.volumes[0]["mask"], mask) and back.volumes[0]["mask_ops"] is None
    assert back.volumes[2]["mask"] is None


def test_the_command_line_reads_regions_from_a_bundle(qapp, run_result, frames, tmp_path):
    from al_dvc.cli import _load_regions

    paths = _volumes(tmp_path / "data", frames)
    state = _state(qapp, paths, run_result)
    state.set_regions([Region(4, "core", "sphere", {"centre": [20, 20, 18], "radius": 6.0})])
    path = save_session(state, tmp_path / "s.aldvc")
    assert [r.name for r in _load_regions(str(path))] == ["core"]


# ---------------------------------------------------------------------------------------------- review hardening
def test_a_member_declaring_more_data_than_it_holds_is_refused(tmp_path):
    """A hand-made archive whose .npy header claims a huge array must fail cleanly, before numpy allocates it."""
    import io
    import zipfile

    from numpy.lib import format as npy_format

    from al_dvc.io.session_serialize import MANIFEST_KEY, SerializeError, read_npz

    path = tmp_path / "hostile.npz"
    with zipfile.ZipFile(path, "w") as zf:
        manifest = io.BytesIO()
        npy_format.write_array(manifest, np.frombuffer(b"{}", dtype=np.uint8))
        zf.writestr(MANIFEST_KEY + ".npy", manifest.getvalue())
        header = io.BytesIO()
        npy_format.write_array_header_1_0(header, {"descr": "<f8", "fortran_order": False, "shape": (10**7, 10**6)})
        zf.writestr("big.npy", header.getvalue() + b"\0" * 64)  # 80 TB declared, 64 bytes held
    with pytest.raises(SerializeError, match="declares"):
        read_npz(path)


def test_a_non_finite_setting_is_left_out_instead_of_failing_the_save(tmp_path):
    import json
    import zipfile

    from al_dvc.io.session_bundle import DOCUMENT, write_bundle

    target = tmp_path / "s.zip"
    doc = {"display": {"color_min": float("nan"), "color_max": 2.0, "levels": [1.0, float("inf")]}, "n": np.float32(3)}
    write_bundle(target, [], lambda: doc)
    with zipfile.ZipFile(target) as zf:
        written = json.loads(zf.read(DOCUMENT))
    assert written == {"display": {"color_max": 2.0, "levels": [1.0, None]}, "n": 3.0}


def test_the_compression_probe_takes_any_memory_order():
    from al_dvc.io.session_serialize import member_level

    a = np.random.default_rng(1).random((64, 48, 32))
    assert member_level(np.asfortranarray(a)) == member_level(a) == member_level(a[::2, ::3])
