"""The shipped example scripts run end to end on synthetic data."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_tutorial_script_runs(tmp_path):
    out = tmp_path / "tut"
    script = ROOT / "examples" / "scripting" / "tutorial_real_data.py"
    res = subprocess.run(
        [sys.executable, str(script), "--quick", "--output", str(out)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(ROOT),
    )
    assert res.returncode == 0, res.stderr[-2000:]
    assert (out / "results.npz").is_file()
    assert (out / "vtk" / "aldvc.pvd").is_file()
    assert (out / "report.pdf").stat().st_size > 10_000
    assert (out / "checkpoints" / "meta.json").is_file()
    assert "RMSE against the known deformation" in res.stdout


def test_tutorial_notebook_is_valid_and_matches_script():
    nb = json.loads((ROOT / "examples" / "tutorial_real_data.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    kinds = [c["cell_type"] for c in nb["cells"]]
    assert kinds.count("code") >= 6 and kinds.count("markdown") >= 6
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for name in ("dvcpara_default", "run_aldvc", "export_vtk", "export_report", "checkpoint_dir", "U_std"):
        assert name in code
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "<cell>", "exec")  # every code cell is valid Python
