"""Reproduce texture UI state defects using production methods and stdlib fakes.

This is a logic probe, not a Qt integration test. Run from any working directory.
Assertions describe the current faulty behavior and must change when it is fixed.
"""

import ast
from pathlib import Path
from types import SimpleNamespace


SOURCE = Path(__file__).resolve().parents[1] / "src/al_dvc/gui/texture_window.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))


class Signal:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)


def production_method(class_name, method_name, environment=None):
    cls = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    namespace = dict(environment or {})
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[method_name]


def probe_cancel_without_sweep():
    finished, cancelled = Signal(), Signal()
    worker = SimpleNamespace(
        progress=Signal(), finished_analysis=finished, cancelled=cancelled,
        failed=Signal(), _stop=True, _sweep=None, _vol=None, _mask=None,
        _spacing=None, _settings={},
    )
    run = production_method("_TextureWorker", "run", {
        "analyse_texture": lambda *args, **kwargs: "RESULT",
        "_Cancelled": type("_Cancelled", (Exception,), {}),
    })
    run(worker)
    assert finished.calls == [("RESULT", None)] and not cancelled.calls
    print("CONFIRMED: cancelled analysis without sweep still emits successful completion.")


def probe_empty_roi():
    class EmptyMask:
        shape = (24, 24, 24)

        def any(self):
            return False

    state = SimpleNamespace(
        volumes=[1], volume_array=lambda index: SimpleNamespace(shape=(24, 24, 24)),
        reference_mask=EmptyMask,
    )
    window = SimpleNamespace(_state=state, use_roi=SimpleNamespace(isChecked=lambda: True))
    reference = production_method("TextureWindow", "_reference", {
        "np": SimpleNamespace(asarray=lambda value: value),
    })
    volume, mask = reference(window)
    assert volume is not None and mask is None
    print("CONFIRMED: enabled ROI restriction with empty mask silently returns no mask.")


def probe_replaced_volume():
    enabled = []
    window = SimpleNamespace(
        _state=SimpleNamespace(volumes=["NEW_VOLUME"]), result="OLD_RESULT",
        recommendation="OLD_RECOMMENDATION",
        _btn_analyse=SimpleNamespace(setEnabled=enabled.append),
        _is_running=lambda: False, _update_status=lambda: None,
    )
    production_method("TextureWindow", "_on_volumes_changed")(window)
    assert window.result == "OLD_RESULT" and window.recommendation == "OLD_RECOMMENDATION"
    assert enabled == [True]
    print("CONFIRMED: replacing volume preserves old result and recommendation.")


if __name__ == "__main__":
    probe_cancel_without_sweep()
    probe_empty_roi()
    probe_replaced_volume()
