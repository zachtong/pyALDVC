"""Parameter / run-summary export (JSON or YAML)."""

from __future__ import annotations

import json
from pathlib import Path

from ..core.config import DVCPara, para_to_dict
from ..core.data_structures import PipelineResult
from .export_utils import ensure_dir, result_summary


def export_params(para: DVCPara, path: str | Path, summary: dict | None = None) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    payload = {"para": para_to_dict(para)}
    if summary is not None:
        payload["summary"] = summary
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml

        p.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    else:
        if p.suffix.lower() != ".json":
            p = p.with_suffix(".json")
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def export_run_summary(result: PipelineResult, path: str | Path) -> Path:
    return export_params(result.dvc_para, path, summary=result_summary(result))
