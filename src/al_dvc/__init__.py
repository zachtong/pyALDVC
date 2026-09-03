"""pyALDVC -- Augmented Lagrangian Digital Volume Correlation in Python.

Quick start::

    from al_dvc import dvcpara_default, run_aldvc, load_volume
    ref = load_volume("ref.tif")
    dfm = load_volume("def.tif")
    para = dvcpara_default(winsize=32, winstepsize=16)
    result = run_aldvc(para, [ref, dfm])
"""

from __future__ import annotations

__version__ = "0.1.0"

from .core.config import DVCPara, dvcpara_default, para_from_dict, para_to_dict
from .core.data_structures import (
    DVCMesh,
    FrameResult,
    FrameSchedule,
    PipelineResult,
    StrainResult,
    VOIRange,
)
from .core.pipeline import run_aldvc
from .io.volume_io import load_volume, load_volumes, save_volume
from .solver.warmup import warmup

__all__ = [
    "__version__",
    "DVCPara", "dvcpara_default", "para_from_dict", "para_to_dict",
    "DVCMesh", "FrameResult", "FrameSchedule", "PipelineResult", "StrainResult", "VOIRange",
    "run_aldvc", "load_volume", "load_volumes", "save_volume", "warmup",
]
