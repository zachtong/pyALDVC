"""MaskEditor: rasterisation per plane, depth ranges, add/cut, undo/redo, persistence."""

from __future__ import annotations

import numpy as np
import pytest

from al_dvc.gui.mask_editor import MaskEditor, MaskOp, rasterise, rasterise_2d

SHAPE = (12, 16, 20)  # (nz, ny, nx)


def test_rectangle_on_each_plane_extrudes_along_the_normal():
    nz, ny, nx = SHAPE
    # XY plane: (h, v) = (x, y); normal z
    r = rasterise(MaskOp("rectangle", "xy", ((2, 3), (5, 6))), SHAPE)
    assert r.shape == SHAPE
    expected = np.zeros(SHAPE, dtype=bool)
    expected[:, 3:7, 2:6] = True
    np.testing.assert_array_equal(r, expected)
    # XZ plane: (h, v) = (x, z); normal y, depth range inclusive
    r = rasterise(MaskOp("rectangle", "xz", ((1, 4), (3, 8)), depth=(5, 7)), SHAPE)
    expected = np.zeros(SHAPE, dtype=bool)
    expected[4:9, 5:8, 1:4] = True
    np.testing.assert_array_equal(r, expected)
    # YZ plane: (h, v) = (y, z); normal x, single slice
    r = rasterise(MaskOp("rectangle", "yz", ((0, 0), (15, 11)), depth=(9, 9)), SHAPE)
    expected = np.zeros(SHAPE, dtype=bool)
    expected[:, :, 9] = True
    np.testing.assert_array_equal(r, expected)


def test_rectangle_corners_in_any_order_and_clipped():
    a = rasterise_2d(MaskOp("rectangle", "xy", ((5, 6), (2, 3))), 20, 16)
    b = rasterise_2d(MaskOp("rectangle", "xy", ((2, 3), (5, 6))), 20, 16)
    np.testing.assert_array_equal(a, b)
    c = rasterise_2d(MaskOp("rectangle", "xy", ((-5, -5), (100, 100))), 20, 16)
    assert c.all()
    d = rasterise_2d(MaskOp("rectangle", "xy", ((30, 30), (40, 40))), 20, 16)
    assert not d.any()


def test_ellipse_and_polygon_geometry():
    e = rasterise_2d(MaskOp("ellipse", "xy", ((4, 2), (16, 14))), 20, 16)
    assert e[8, 10] and e[2, 10] and e[8, 4]  # centre, top, left
    assert not e[2, 4] and not e[14, 16]  # corners of the bounding box are outside
    area = e.sum()
    assert abs(area - np.pi * 6 * 6) / (np.pi * 36) < 0.15
    tri = rasterise_2d(MaskOp("polygon", "xy", ((0, 0), (19, 0), (0, 15))), 20, 16)
    assert tri[0, 0] and tri[1, 1] and not tri[15, 19]
    assert abs(tri.sum() - 0.5 * 19 * 15) / (0.5 * 19 * 15) < 0.15


def test_brush_stroke_is_a_swept_disc():
    dot = rasterise_2d(MaskOp("brush", "xy", ((10, 8),), radius=2.0), 20, 16)
    assert dot[8, 10] and dot[8, 12] and dot[6, 10] and not dot[8, 13] and not dot[5, 10]
    line = rasterise_2d(MaskOp("brush", "xy", ((2, 8), (17, 8)), radius=1.0), 20, 16)
    assert line[8, 2:18].all() and line[7, 5] and line[9, 5] and not line[6, 5] and not line[8, 19]
    # a stroke leaving the plane is clipped, not an error
    off = rasterise_2d(MaskOp("brush", "xy", ((-3, -3), (25, 20)), radius=1.5), 20, 16)
    assert off.any()


def test_op_validation():
    with pytest.raises(ValueError):
        MaskOp("rectangle", "xy", ((0, 0),))
    with pytest.raises(ValueError):
        MaskOp("polygon", "xy", ((0, 0), (1, 1)))
    with pytest.raises(ValueError):
        MaskOp("brush", "xy", ((0, 0),), radius=0.0)
    with pytest.raises(ValueError):
        MaskOp("rectangle", "ab", ((0, 0), (1, 1)))
    with pytest.raises(ValueError):
        MaskOp("rectangle", "xy", ((0, 0), (1, 1)), mode="xor")
    with pytest.raises(ValueError):
        MaskOp("rectangle", "xy", ((0, 0), (1, 1)), depth=(5, 2))
    with pytest.raises(ValueError):
        MaskEditor((0, 3, 3))
    with pytest.raises(ValueError):
        MaskEditor(SHAPE, base=np.zeros((2, 2, 2), dtype=bool))


def test_add_cut_invert_fill_and_undo_redo():
    ed = MaskEditor(SHAPE)
    assert ed.coverage == 0.0 and not ed.can_undo
    ed.apply(MaskOp("rectangle", "xy", ((0, 0), (9, 7))))
    full_box = ed.mask.copy()
    assert ed.mask[:, :8, :10].all() and ed.coverage == pytest.approx(80 / 320)
    ed.apply(MaskOp("rectangle", "xy", ((0, 0), (4, 3)), depth=(0, 5), mode="cut"))
    assert not ed.mask[:6, :4, :5].any() and ed.mask[6:, :4, :5].all()
    assert ed.can_undo and not ed.can_redo
    assert ed.undo()
    np.testing.assert_array_equal(ed.mask, full_box)
    assert ed.can_redo
    assert ed.redo()
    assert not ed.mask[:6, :4, :5].any()
    assert ed.undo() and ed.undo() and not ed.undo()
    assert ed.coverage == 0.0
    ed.apply(MaskOp("fill"))
    assert ed.mask.all()
    ed.apply(MaskOp("invert"))
    assert not ed.mask.any()
    ed.apply(MaskOp("brush", "yz", ((3, 4), (8, 4)), radius=1.0, depth=(2, 2)))
    assert ed.mask[:, :, 2].any() and not ed.mask[:, :, 3].any()
    ed.apply(MaskOp("empty"))
    assert ed.coverage == 0.0
    # a new operation after undo discards the redo stack
    ed.undo()
    ed.apply(MaskOp("fill"))
    assert not ed.can_redo


def test_base_mask_and_reset():
    base = np.zeros(SHAPE, dtype=bool)
    base[:, :, :10] = True
    ed = MaskEditor(SHAPE, base=base)
    assert ed.coverage == 0.5
    ed.apply(MaskOp("rectangle", "xy", ((10, 0), (19, 15)), depth=(0, 0)))
    assert ed.mask[0].all() and ed.mask[1, :, 10:].sum() == 0
    ed.undo()
    np.testing.assert_array_equal(ed.mask, base)
    ed.reset()
    assert ed.coverage == 0.0 and not ed.can_undo
    ed.reset(base=np.ones(SHAPE, dtype=bool))
    assert ed.coverage == 1.0


def test_persistence_roundtrip():
    ed = MaskEditor(SHAPE)
    ed.apply(MaskOp("ellipse", "xz", ((2, 1), (15, 10)), depth=(3, 9)))
    ed.apply(MaskOp("polygon", "xy", ((1, 1), (18, 2), (10, 14)), mode="cut"))
    ed.apply(MaskOp("brush", "yz", ((2, 2), (6, 9)), radius=1.5))
    d = ed.to_dict()
    assert d["shape"] == list(SHAPE) and len(d["ops"]) == 3
    import json

    d2 = json.loads(json.dumps(d))  # JSON-clean
    ed2 = MaskEditor.from_dict(d2)
    np.testing.assert_array_equal(ed2.mask, ed.mask)
    assert ed2.ops == ed.ops


def test_many_operations_fold_into_the_base():
    from al_dvc.gui import mask_editor

    old = mask_editor.MAX_UNDO_REPLAY_OPS
    mask_editor.MAX_UNDO_REPLAY_OPS = 5
    try:
        ed = MaskEditor((4, 8, 8))
        for i in range(8):
            ed.apply(MaskOp("rectangle", "xy", ((i, 0), (i, 7))))
        assert ed.mask[:, :, :8].all()
        ed.undo()  # replay folds the oldest operations into the base
        assert ed.mask[:, :, :7].all() and not ed.mask[:, :, 7].any()
        assert len(ed.ops) <= 5 and ed.base is not None
    finally:
        mask_editor.MAX_UNDO_REPLAY_OPS = old


def _apply_through_a_full_volume(mask, op, shape):
    """What ``apply`` did before it learned to write through the slab: the equivalence oracle."""
    region = rasterise(op, shape)
    out = mask.copy()
    if op.mode == "replace":
        out[...] = region
    elif op.mode == "add":
        out |= region
    else:
        out &= ~region
    return out


@pytest.mark.parametrize("plane", ["xy", "xz", "yz"])
@pytest.mark.parametrize("mode", ["replace", "add", "cut"])
def test_writing_through_the_slab_equals_the_full_volume_combine(plane, mode):
    """Nothing volume-sized is allocated any more, so the result must be proved identical."""
    nz, ny, nx = SHAPE
    h_len, v_len = {"xy": (nx, ny), "xz": (nx, nz), "yz": (ny, nz)}[plane]
    n_len = {"xy": nz, "xz": ny, "yz": nx}[plane]
    rng = np.random.default_rng(11)
    depths = [None, (0, 0), (n_len // 2, n_len // 2), (1, n_len - 2), (-5, 2), (n_len - 1, n_len + 9), (n_len + 2, n_len + 5)]
    shapes = [
        ("rectangle", ((1.0, 1.0), (h_len - 2.0, v_len - 2.0)), {}),
        ("ellipse", ((0.5, 0.5), (h_len - 1.5, v_len - 1.5)), {}),
        ("polygon", ((1.0, 1.0), (h_len - 2.0, 2.0), (h_len / 2, v_len - 2.0)), {}),
        ("brush", tuple((2.0 + k, 3.0 + 0.5 * k) for k in range(4)), {"radius": 2.5}),
        ("rectangle", ((-9.0, -9.0), (-3.0, -3.0)), {}),  # entirely off the plane
    ]
    for depth in depths:
        for shape_name, points, extra in shapes:
            op = MaskOp(shape_name, plane=plane, points=points, depth=depth, mode=mode, **extra)
            start = rng.random(SHAPE) > 0.5
            ed = MaskEditor(SHAPE, base=start.copy())
            ed.apply(op)
            np.testing.assert_array_equal(ed.mask, _apply_through_a_full_volume(start, op, SHAPE), err_msg=f"{op}")


def test_the_full_base_is_symbolic_and_costs_one_volume():
    ed = MaskEditor(SHAPE, base="full")
    assert ed.mask.all() and ed.base is None  # nothing stored for it
    assert ed.coverage == 1.0 and ed.box() == ((0, 20), (0, 16), (0, 12))
    ed.reset(base="full")
    assert ed.mask.all() and ed.base is None
    with pytest.raises(ValueError, match="base must be an array"):
        MaskEditor(SHAPE, base="everything")


def test_the_box_and_count_caches_follow_every_edit():
    """They are read a dozen times per edit, so they are cached -- and every writer must invalidate."""
    ed = MaskEditor(SHAPE, base="full")
    assert ed.count == 12 * 16 * 20
    ed.apply(MaskOp("rectangle", "xy", ((4, 5), (9, 10)), mode="replace", depth=(2, 3)))
    assert ed.box() == ((4, 10), (5, 11), (2, 4)) and ed.count == 6 * 6 * 2
    ed.apply(MaskOp("empty"))
    assert ed.box() is None and ed.count == 0
    ed.undo()  # back to the rectangle
    assert ed.box() == ((4, 10), (5, 11), (2, 4)) and ed.count == 6 * 6 * 2
    ed.undo()  # back to the base with no operations left: _replay never reaches apply()
    assert ed.box() == ((0, 20), (0, 16), (0, 12)) and ed.count == 12 * 16 * 20
    ed.redo()
    assert ed.count == 6 * 6 * 2
    ed.reset()
    assert ed.box() is None and ed.count == 0
    assert all(isinstance(v, int) for pair in MaskEditor(SHAPE, base="full").box() for v in pair)
