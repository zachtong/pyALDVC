"""Batch-process several DVC samples from one YAML/JSON config.

    python examples/scripting/batch_process.py my_batch.yaml

Each entry of ``samples`` inherits the top-level ``para`` / ``export`` and
may override any key. A failing sample is logged and the batch continues.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from pathlib import Path

from al_dvc import dvcpara_default, run_aldvc, warmup
from al_dvc.export import export_csv, export_mat, export_npz, export_report, export_run_summary, export_vtk
from al_dvc.io import FileVolumeProvider, resolve_volume_paths

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
log = logging.getLogger("batch")


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text)
    return json.loads(text)


def run_sample(sample: dict, shared_para: dict, exports: list[str]) -> None:
    name = sample.get("name", Path(str(sample["volumes"])).stem)
    para = dvcpara_default(**{**shared_para, **sample.get("para", {})})
    paths = resolve_volume_paths(sample["volumes"])
    masks = sample.get("masks")
    mask_paths = None
    if masks:
        mask_paths = resolve_volume_paths(masks)
        if len(mask_paths) == 1:
            mask_paths = mask_paths * len(paths)
    provider = FileVolumeProvider(paths, para.voi, mask_paths, load_kwargs=sample.get("load_kwargs", {}))
    out = Path(sample.get("output", f"results/{name}"))
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result = run_aldvc(para, provider)
    log.info("%s: %d frames, %d nodes, %.1fs", name, result.n_frames, result.dvc_mesh.n_nodes, time.perf_counter() - t0)
    base = sample.get("basename", name)
    if "npz" in exports:
        export_npz(result, out / f"{base}.npz")
    if "mat" in exports:
        export_mat(result, out / f"{base}.mat")
    if "csv" in exports:
        export_csv(result, out / "csv", base)
    if "vtk" in exports:
        export_vtk(result, out / "vtk", base)
    if "report" in exports:
        export_report(result, out / f"{base}_report.pdf")
    if "summary" in exports:
        export_run_summary(result, out / f"{base}_summary.json")


def main(config_path: Path) -> int:
    cfg = load_config(config_path)
    samples = cfg.get("samples") or [cfg]
    shared_para = cfg.get("para", {})
    exports = cfg.get("export", ["npz", "report", "summary"])
    warmup()
    failures = 0
    for sample in samples:
        try:
            run_sample(sample, shared_para, exports)
        except Exception:  # keep going, report at the end
            failures += 1
            log.error("sample %s failed:\n%s", sample.get("name", "?"), traceback.format_exc())
    log.info("batch finished: %d/%d samples succeeded", len(samples) - failures, len(samples))
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    raise SystemExit(main(Path(sys.argv[1])))
