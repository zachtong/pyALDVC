"""Export dialog and the run_export driver."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.name == "nt":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

pytest.importorskip("PySide6")

from al_dvc.core.config import dvcpara_default  # noqa: E402
from al_dvc.core.pipeline import run_aldvc  # noqa: E402
from al_dvc.gui.dialogs.export_dialog import ExportConfig, run_export  # noqa: E402
from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from al_dvc.gui.app import create_application

    return create_application([])


@pytest.fixture(scope="module")
def pair():
    shape = (40, 44, 48)
    centre = tuple((s - 1) / 2 for s in shape[::-1])
    ref = generate_speckle_volume(shape, sigma=2.0, seed=7)
    dfm = warp_volume_lagrangian(ref, affine_displacement(np.diag([0.01, -0.005, 0.005]), (0.4, -0.3, 0.2), centre))
    return ref, dfm


@pytest.fixture(scope="module")
def result(pair):
    para = dvcpara_default(winsize=12, winstepsize=6, search_radius=3, admm_max_iter=1, verbose=False)
    return run_aldvc(para, list(pair))


def test_run_export_writes_every_selected_format(result, pair, tmp_path):
    cfg = ExportConfig(
        out_dir=tmp_path / "out",
        npz=True,
        mat=True,
        csv=True,
        vtk=True,
        report=True,
        images=True,
        fields=["disp_u", "exx"],
        frames=[0],
        image_dpi=50,
    )
    seen = []
    paths = run_export(result, cfg, background=pair[0], progress_fn=lambda f, m: seen.append((f, m)))
    assert len(paths) == 6 and all(p.exists() for p in paths)
    assert (tmp_path / "out" / "aldvc.npz").is_file() and (tmp_path / "out" / "aldvc_report.pdf").is_file()
    assert list((tmp_path / "out" / "images").glob("*.png"))
    assert seen[-1][0] == 1.0
    with pytest.raises(ValueError):
        run_export(result, ExportConfig(out_dir=tmp_path / "none", npz=False))


def test_export_dialog_end_to_end(qapp, pair, result, tmp_path):
    from al_dvc.gui.app import MainWindow

    window = MainWindow()
    window.show()
    assert window.state.results is None
    window._on_export_requested()  # nothing to export: only a warning in the console
    assert getattr(window, "export_dialog", None) is None
    window.state.set_volume_arrays(list(pair), ["ref", "def"])
    window.state.set_results(result)
    dialog = window.open_export_dialog()
    assert dialog.isVisible() and window.open_export_dialog() is dialog
    names = [dialog.fields.item(i).data(0x0100) for i in range(dialog.fields.count())]
    assert "disp_u" in names and "exx" in names
    dialog._check_fields(lambda n: n in ("disp_u", "von_mises"))
    assert dialog.selected_fields() == ["disp_u", "von_mises"]
    dialog.folder.setText(str(tmp_path / "exp"))
    dialog.basename.setText("run1")
    dialog.checks["csv"].setChecked(True)
    dialog.checks["images"].setChecked(True)
    dialog.all_frames.setChecked(False)
    cfg = dialog.config()
    assert cfg.frames == [0] and cfg.basename == "run1" and cfg.images and cfg.image_layout == "row"
    dialog.start()
    assert dialog.wait(300_000)
    qapp.processEvents()
    assert (tmp_path / "exp" / "run1.npz").is_file()
    assert list((tmp_path / "exp" / "csv").glob("run1_*.csv"))
    assert list((tmp_path / "exp" / "images").glob("disp_u_frame_001.png"))
    assert window.state.output_dir == tmp_path / "exp"
    assert "Done" in dialog._status.text()
    dialog.close()
    window.close()
