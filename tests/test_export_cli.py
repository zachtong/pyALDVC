import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from al_dvc.cli import main
from al_dvc.core.config import dvcpara_default
from al_dvc.core.pipeline import run_aldvc
from al_dvc.export import (
    export_csv,
    export_mat,
    export_npz,
    export_params,
    export_report,
    export_run_summary,
    export_vtk,
    field_array,
    load_npz_result,
)


@pytest.fixture(scope="module")
def result(affine_pair):
    f, g, _ = affine_pair
    para = dvcpara_default(winsize=16, winstepsize=8, search_radius=5, verbose=False, admm_max_iter=2)
    return run_aldvc(para, [f, g])


def test_export_npz_roundtrip(result, tmp_path):
    p = export_npz(result, tmp_path / "r.npz")
    d = load_npz_result(p)
    assert np.allclose(d["U_1"], result.result_disp[0].U)
    assert np.allclose(d["exx_1"], result.result_strain[0].exx, equal_nan=True)
    assert d["para"]["winsize"] == [16, 16, 16]
    assert tuple(d["grid_shape"]) == result.dvc_mesh.grid_shape


def test_export_mat(result, tmp_path):
    from scipy.io import loadmat

    p = export_mat(result, tmp_path / "r.mat")
    m = loadmat(str(p))
    assert m["coordinatesFEM"].shape == (result.dvc_mesh.n_nodes, 3)
    assert m["elementsFEM"].min() >= 1
    U_int = m["ResultDisp_interleaved"][0, 0].ravel()
    assert np.allclose(U_int, result.result_disp[0].U.ravel())


def test_export_csv_vtk(result, tmp_path):
    paths = export_csv(result, tmp_path / "csv", fields=["disp_u", "exx", "von_mises"])
    assert len(paths) == 1
    lines = paths[0].read_text().splitlines()
    assert lines[0] == "x,y,z,valid,disp_u,exx,von_mises"
    assert len(lines) == result.dvc_mesh.n_nodes + 1
    vtks = export_vtk(result, tmp_path / "vtk", fields=["exx"])
    root = ET.parse(vtks[0]).getroot()
    assert root.attrib["type"] == "ImageData"
    names = {da.attrib["Name"] for da in root.iter("DataArray")}
    assert {"displacement", "exx", "zncc"} <= names
    assert (tmp_path / "vtk" / "aldvc.pvd").exists()
    with pytest.raises(ValueError):
        export_csv(result, tmp_path / "bad", fields=["nope"])


def test_export_report_and_params(result, tmp_path, affine_pair):
    from al_dvc.synthetic import evaluate_at_nodes, gradient_at_nodes

    assert result.result_disp[0].admm is not None and result.result_disp[0].admm.beta_sweep is not None
    p = export_report(result, tmp_path / "rep.pdf")
    assert p.exists() and p.stat().st_size > 10_000
    disp = affine_pair[2]
    gt = {"U": [evaluate_at_nodes(disp, result.dvc_mesh.coordinates)],
          "F": [gradient_at_nodes(disp, result.dvc_mesh.coordinates)]}
    p2 = export_report(result, tmp_path / "rep_gt.pdf", gt=gt)
    assert p2.stat().st_size > p.stat().st_size
    q = export_params(result.dvc_para, tmp_path / "p.yaml")
    assert "winsize" in q.read_text()
    s = export_run_summary(result, tmp_path / "s.json")
    js = json.loads(s.read_text())
    assert js["summary"]["n_nodes"] == result.dvc_mesh.n_nodes
    assert field_array(result, 0, "disp_magnitude").shape == (result.dvc_mesh.n_nodes,)


def test_cli_synth_run_info_plot(tmp_path):
    data = tmp_path / "data"
    assert main(["-q", "synth", str(data), "--shape", "48", "52", "56", "--mode", "stretch", "--value", "0.02",
                 "--frames", "1", "--dtype", "uint16"]) == 0
    files = sorted(data.glob("*.tif"))
    assert len(files) == 2
    assert main(["-q", "info", str(files[0])]) == 0
    out = tmp_path / "out"
    assert main(["-q", "run", "--volumes", str(data), "-o", str(out), "--winsize", "16", "--step", "8",
                 "--export", "npz", "summary", "--set", "search_radius=4", "admm_max_iter=2"]) == 0
    assert (out / "aldvc.npz").exists() and (out / "aldvc_summary.json").exists()
    assert main(["-q", "plot", str(out / "aldvc.npz"), "--field", "exx", "-o", str(tmp_path / "exx.png")]) == 0
    assert (tmp_path / "exx.png").exists()
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"volumes": str(data), "output": str(tmp_path / "out2"), "export": ["csv"],
                               "para": {"winsize": 16, "winstepsize": 8, "search_radius": 4, "use_global_step": False}}))
    assert main(["-q", "run", str(cfg)]) == 0
    assert list((tmp_path / "out2" / "csv").glob("*.csv"))
