"""Reproduce frame-playback feedback using actual source and stdlib doubles.

Run from any directory with Python 3.10+. No Qt/VTK installation is required.
The probe extracts the production frame_at and _play_frame function bodies.
"""

import ast
import math
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace


@dataclass
class Options:
    frame: int = 0
    slice_index: object = None
    warp_scale: float = 1.0


@dataclass
class Frame:
    index: int
    time: float
    camera: object
    options: object


def extract_function(path, name, namespace, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = tree.body
    if class_name:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name).body
    function = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    function.returns = None
    for argument in function.args.args:
        argument.annotation = None
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)


def main():
    root = Path(__file__).resolve().parents[1]
    namespace = {"np": SimpleNamespace(floor=math.floor), "replace": replace, "Frame": Frame}
    extract_function(root / "src/al_dvc/gui/view3d_animation.py", "frame_at", namespace)
    extract_function(root / "src/al_dvc/gui/panels/view3d.py", "_play_frame", namespace, "View3DPanel")
    spec = SimpleNamespace(kind="frames", speed=2, direction=1, fps=20)
    state = SimpleNamespace(
        results=SimpleNamespace(result_disp=[1, 2, 3, 4], volume_shape=(20, 20, 20)),
        display_frame=0,
    )
    panel = SimpleNamespace(
        _state=state,
        _play_base=(None, Options()),
        animation_spec=lambda: spec,
        options=lambda: Options(frame=state.display_frame),
    )
    times = [0.50, 0.533, 0.566, 0.60]
    actual = []
    for time in times:
        frame = namespace["_play_frame"](panel, time)
        # Mirror _on_play_tick -> set_current_frame(frame + 1) -> display_frame.
        state.display_frame = frame.options.frame
        actual.append(state.display_frame)
    expected = [1, 1, 1, 1]
    print(f"Times: {times}; speed: 2 frames/second; result frames: 4")
    print(f"Expected: {expected}")
    print(f"Actual:   {actual}")
    assert actual != expected, "The audited feedback defect no longer reproduces; review this probe."
    print("CONFIRMED: live frame changes are fed back into the animation start frame.")


if __name__ == "__main__":
    main()
