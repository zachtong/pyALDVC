"""The compute backend in words: the GPU's name, or why the CPU kernels run and how to get the GPU.

The parameter panel, the status bar and the self-test describe the backend from here, so the three say
the same thing. The portable Windows bundle is CPU-only on purpose (``packaging/pyaldvc.spec`` leaves
numba.cuda out): its users are pointed at the pip install, which is the way to the GPU.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

from .i18n import tr, tr_noop

GPU_INSTALL_COMMAND = 'pip install "al-dvc[gpu]"'
REASON_CHARS = 400  # the technical reason in a tooltip is cut here; the log keeps it whole

# the CPU cases: why the CPU kernels run
PORTABLE = "portable"  # the portable bundle, CPU-only by design
MISSING = "missing"  # the optional GPU backend (numba-cuda) is not installed
NO_DEVICE = "no_device"  # installed, but no NVIDIA device or driver
ERROR = "error"  # installed, a device is there, and the CUDA stack failed on it
GPU = "gpu"

_GPU_LABEL = tr_noop("GPU: {name}")
_CPU_LABEL = tr_noop("CPU")
# (line in the parameter panel, full sentence for the tooltips and the self-test); "{command}" is the pip
# command, never translated, and has a line of its own in the panel so that the word wrap never cuts it
_TEXTS: dict[str, tuple[str, str]] = {
    PORTABLE: (
        tr_noop("CPU (portable version). For GPU acceleration, install with pip:\n{command}"),
        tr_noop("CPU (portable version). For NVIDIA GPU acceleration, install pyALDVC with pip: {command}"),
    ),
    MISSING: (
        tr_noop("CPU. GPU acceleration is not installed:\n{command}"),
        tr_noop("CPU. GPU acceleration is not installed: {command}"),
    ),
    NO_DEVICE: (
        tr_noop("CPU. No NVIDIA GPU or driver was found."),
        tr_noop("CPU. No NVIDIA GPU or driver was found."),
    ),
    ERROR: (
        tr_noop("CPU. The GPU backend did not start."),
        tr_noop("CPU. The GPU backend did not start."),
    ),
}


@dataclass(frozen=True)
class BackendStatus:
    """What the automatic backend choice runs on, and why when it is the CPU."""

    case: str  # GPU, PORTABLE, MISSING, NO_DEVICE or ERROR
    device: str = ""  # the GPU's name (case GPU)
    reason: str = ""  # the technical reason the GPU backend is off (the CPU cases)

    @property
    def on_gpu(self) -> bool:
        return self.case == GPU


@dataclass(frozen=True)
class BackendText:
    """The status in the user's words: a short label, the panel's line and a tooltip."""

    label: str  # status bar: "CPU" or "GPU: <name>"
    line: str  # parameter panel
    tooltip: str  # the full sentence, with the technical reason when there is one


def running_frozen() -> bool:
    """True inside the PyInstaller bundle, the portable version."""
    return bool(getattr(sys, "frozen", False))


def gpu_backend_installed() -> bool:
    """Whether the optional GPU backend (numba-cuda, the ``[gpu]`` extra) is installed."""
    try:
        from importlib.metadata import version

        version("numba-cuda")
    except Exception:
        return False
    return True


def backend_status() -> BackendStatus:
    """Probe the backend (once per process) and say which case it is."""
    from al_dvc.solver.cuda_kernels import cuda_available, device_name, unavailable_kind, unavailable_reason

    if cuda_available():
        return BackendStatus(GPU, device=device_name())
    reason = unavailable_reason()
    if running_frozen():
        return BackendStatus(PORTABLE, reason=reason)
    kind = unavailable_kind()
    installed = gpu_backend_installed()
    if kind == MISSING and installed:
        kind = ERROR  # the extra is installed but numba.cuda does not import: a broken install, not a missing one
    elif kind == ERROR and not installed:
        kind = MISSING  # numba's own CUDA module failed on the device: installing the extra is the fix
    return BackendStatus(kind if kind in (MISSING, NO_DEVICE) else ERROR, reason=reason)


def describe(status: BackendStatus, translate: Callable[[str], str] | None = tr) -> BackendText:
    """The texts for ``status``, translated; ``translate=None`` gives the English (the self-test report)."""
    t = translate if translate is not None else (lambda s: s)
    if status.on_gpu:
        gpu = t(_GPU_LABEL).format(name=status.device)
        return BackendText(label=gpu, line=gpu, tooltip="")
    line, full = (t(s).format(command=GPU_INSTALL_COMMAND) for s in _TEXTS.get(status.case, _TEXTS[ERROR]))
    tooltip = full
    if status.case in (NO_DEVICE, ERROR) and status.reason:
        tooltip += "\n" + status.reason[:REASON_CHARS]
    return BackendText(label=t(_CPU_LABEL), line=line, tooltip=tooltip)
