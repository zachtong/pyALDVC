#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RVE autocorrelation analysis driver.

This script automates sweeping ROI sizes around a fixed center to identify
the Representative Volume Element (RVE) using the existing autocorrelation
utilities defined in `acf_analysis.py`.

Constraints enforced per user instructions:
    * Denoising is always disabled.
    * Downsampling is not applied.
    * Autocorrelation plots display mean curves only (no +/- std bands).

ATTRIBUTION: This code was originally developed based on the IMPPY3D framework
             (https://github.com/usnistgov/imppy3d). See IMPPY3D_README.md and
             IMPPY3D_LICENSE.txt in this directory for the original library
             documentation and license terms.

AUTHORS: Zixiang (Zach) Tong, @UT-Austin; Yujie Zhang, @UT-Austin;
         Dr. Alexander K. Landauer, NIST
DATE: 2025.05.25
NOTE: Claude AI (Anthropic) assisted with code formatting, structure optimization,
      and documentation standardization
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend for batch figure generation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# acf_analysis.py lives in the same directory as this script.
# Python adds the script's own directory to sys.path automatically when run directly,
# but we ensure it explicitly here for cases where the module is imported or run via
# a different working directory (e.g. pytest, IDE run configs).
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(CURRENT_DIR))

CONFIG_SUBDIR_NAME = "rve_configs"
CONFIG_DIR = CURRENT_DIR / CONFIG_SUBDIR_NAME
DEFAULT_CONFIG_NAME = "rve_config.json"

from acf_analysis import (  # type: ignore  # pylint: disable=wrong-import-position
    apply_roi_cropping,
    compute_autocorrelation,
    compute_radial_autocorrelation,
    load_tiff_stack,
    select_output_directory,
    select_tiff_file,
)

# Global constants
EPS_FLOAT = 1e-12
DENOISE_OPTION_DISABLED = "0"

FONT_FAMILY = "Arial"
TITLE_FONT_SIZE = 20
AXIS_FONT_SIZE = 20
TICK_FONT_SIZE = 20
LEGEND_FONT_SIZE = 20
ZOOM_TITLE_FONT_SIZE = 32
ZOOM_AXIS_FONT_SIZE = 32
ZOOM_TICK_FONT_SIZE = 32


@dataclass
class RoiSpecification:
    index: int
    width: int
    height: int
    depth: int

    def roi_label(self) -> str:
        return f"ROI_{self.index + 1:02d}"

    def size_tag(self) -> str:
        return f"W{self.width}_H{self.height}_D{self.depth}"

    def file_tag(self) -> str:
        return f"{self.roi_label()}_{self.size_tag()}"

    def effective_length(self) -> float:
        return float((self.width * self.height * self.depth) ** (1.0 / 3.0))


@dataclass
class RoiResult:
    spec: RoiSpecification
    actual_dims: Tuple[int, int, int]
    distance: np.ndarray
    corr_mean: np.ndarray
    corr_std: np.ndarray
    length_map: Dict[float, Optional[float]]
    depth_indices: Tuple[int, int]
    roi_coords: Tuple[int, int, int, int]

    def effective_length(self) -> float:
        height, width, depth = self.actual_dims
        return float((height * width * depth) ** (1.0 / 3.0))


@dataclass
class TargetSpec:
    """Specification for a correlation threshold target used in convergence testing.

    Attributes:
        name: Internal identifier (e.g. ``"feature_scale"``).
        value: Correlation threshold in (0, 1) (e.g. 0.3679 for 1/e).
        label: Display label for plots and reports.
        tier: Interpretation tier (``"feature"``, ``"mesoscale"``, ``"diagnostic"``).
        cv_threshold: Maximum coefficient of variation (std/mean) for the sliding
            window to be considered stable.  Set to ``None`` to skip CV testing.
        required: If ``True``, this target must converge for the combined RVE to
            be declared.
        diagnostic_only: If ``True``, results are reported but not used for gating.
        abs_tolerance: Maximum absolute standard deviation (in voxels) for the
            sliding window to be considered stable.  Acts as a fallback when
            characteristic lengths are small and the CV criterion becomes overly
            strict.  A window is stable when *either* ``cv <= cv_threshold``
            *or* ``std <= abs_tolerance``.  Default: 0.5 voxels.
    """

    name: str
    value: float
    label: str
    tier: str
    cv_threshold: Optional[float]
    required: bool
    diagnostic_only: bool = False
    abs_tolerance: float = 0.5

    def is_gate(self) -> bool:
        """Return True if this target participates in RVE gating decisions."""
        return self.required and not self.diagnostic_only and self.cv_threshold is not None


@dataclass
class TargetDecision:
    spec: TargetSpec
    sorted_start_index: Optional[int]
    roi_result: Optional[RoiResult]
    stable: bool
    reason: Optional[str] = None

    @property
    def effective_length(self) -> Optional[float]:
        if self.roi_result is None:
            return None
        return self.roi_result.effective_length()


DEFAULT_TARGET_SPECS: List[TargetSpec] = [
    TargetSpec(
        name="feature_scale",
        value=float(np.exp(-1.0)),
        label="1/e",
        tier="feature",
        cv_threshold=0.05,
        required=True,
        abs_tolerance=0.5,
    ),
    TargetSpec(
        name="mesoscale",
        value=0.1,
        label="0.1",
        tier="mesoscale",
        cv_threshold=0.10,
        required=False,
        abs_tolerance=0.5,
    ),
    TargetSpec(
        name="tail_diagnostic",
        value=0.01,
        label="0.01",
        tier="diagnostic",
        cv_threshold=0.20,
        required=False,
        diagnostic_only=True,
        abs_tolerance=0.5,
    ),
]

def announce_step(message: str) -> None:
    print(message)
    sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RVE autocorrelation analysis across ROI sizes.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to a JSON configuration file. If omitted, you can choose interactively.",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unrecognized arguments: {' '.join(unknown)}")
    return args


def is_within_config_dir(path: Path) -> bool:
    try:
        config_root = CONFIG_DIR.resolve(strict=False)
        path.resolve(strict=False).relative_to(config_root)
        return True
    except Exception:
        return False


def resolve_config_choice(choice: str) -> Optional[Path]:
    raw_choice = Path(choice).expanduser()
    candidates: List[Path] = []
    if raw_choice.is_absolute():
        candidates.append(raw_choice)
    else:
        if raw_choice.parts and raw_choice.parts[0] == CONFIG_SUBDIR_NAME:
            relative_choice = Path(*raw_choice.parts[1:])
        else:
            relative_choice = raw_choice
        candidates.append(CONFIG_DIR / relative_choice)
        if relative_choice.suffix.lower() != ".json":
            candidates.append((CONFIG_DIR / relative_choice).with_suffix(".json"))
        candidates.append(CURRENT_DIR / raw_choice)
        candidates.append(Path.cwd() / raw_choice)
        candidates.append(raw_choice)

    seen: set[str] = set()
    for candidate in candidates:
        try:
            key = str(candidate.resolve(strict=False))
        except Exception:
            key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate
            if is_within_config_dir(resolved):
                return resolved
    return None


def available_config_files(default_path: Path) -> List[Path]:
    if CONFIG_DIR.is_dir():
        files = sorted(p for p in CONFIG_DIR.glob("*.json") if p.is_file())
    else:
        files = []
    if default_path.is_file() and default_path not in files:
        files.insert(0, default_path)
    return files


def interactive_config_selection(default_path: Path) -> Path:
    candidates = available_config_files(default_path)
    default_exists = default_path.is_file()

    if not candidates and default_exists:
        print(f"\nUsing default configuration: {default_path}")
        return default_path

    print("\nConfiguration file selection")
    if candidates:
        try:
            default_key = default_path.resolve(strict=False)
        except Exception:
            default_key = default_path
        print(f"Available JSON files in {CONFIG_DIR}:")
        for idx, candidate in enumerate(candidates, start=1):
            try:
                candidate_key = candidate.resolve(strict=False)
            except Exception:
                candidate_key = candidate
            tag = " (default)" if default_exists and candidate_key == default_key else ""
            print(f"  [{idx}] {candidate.name}{tag}")
        instruction = "Type the number of a listed file"
        if default_exists:
            instruction += ", press Enter for the default"
        instruction += ", or provide a path to another JSON file."
        print(instruction)
    else:
        if default_exists:
            print(f"Default configuration located at: {default_path}")
        print(f"No configuration files discovered in '{CONFIG_SUBDIR_NAME}/'. Please place JSON files there or specify a path inside that folder.")

    prompt = (
        "Select config [press Enter for default]: "
        if default_exists
        else "Select config (enter index or file path): "
    )
    while True:
        choice = input(prompt).strip()
        if not choice:
            if default_exists:
                return default_path
            print("A configuration file path is required.")
            continue
        if candidates and choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(candidates):
                return candidates[idx - 1]
        resolved = resolve_config_choice(choice)
        if resolved:
            return resolved
        print("Invalid selection. Provide a listed index or a valid file path.")


def resolve_config_path(cli_value: Optional[str]) -> Path:
    default_path = CONFIG_DIR / DEFAULT_CONFIG_NAME
    if cli_value:
        resolved = resolve_config_choice(cli_value)
        if resolved is None:
            raise FileNotFoundError(
                f"Configuration file not found within '{CONFIG_SUBDIR_NAME}/': {cli_value}"
            )
        return resolved
    return interactive_config_selection(default_path)


def load_settings(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_target_specs(settings: Dict[str, Any]) -> List[TargetSpec]:
    raw_targets = settings.get("length_targets")
    if raw_targets is None:
        return list(DEFAULT_TARGET_SPECS)
    if not isinstance(raw_targets, list):
        raise TypeError("'length_targets' must be a list of objects.")

    targets: List[TargetSpec] = []
    for idx, entry in enumerate(raw_targets):
        if not isinstance(entry, dict):
            raise TypeError(f"Target specification at index {idx} must be an object.")

        try:
            value = float(entry["value"])
        except KeyError as exc:
            raise KeyError(f"Target specification at index {idx} missing 'value'.") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Target specification at index {idx} has invalid 'value'.") from exc

        if value <= 0 or value >= 1:
            raise ValueError(f"Target value at index {idx} must be between 0 and 1 (exclusive). Got {value}.")

        name = str(entry.get("name") or f"target_{idx}")
        label = str(entry.get("label") or f"{value:g}")
        tier = str(entry.get("tier") or name)
        cv_threshold_raw = entry.get("cv_threshold")
        cv_threshold: Optional[float]
        if cv_threshold_raw is None:
            cv_threshold = None
        else:
            try:
                cv_threshold = float(cv_threshold_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Target specification '{name}' has invalid 'cv_threshold' value: {cv_threshold_raw}"
                ) from exc
            if cv_threshold <= 0:
                raise ValueError(f"Target '{name}' must have cv_threshold > 0 or null.")

        required = bool(entry.get("required", False))
        diagnostic_only = bool(entry.get("diagnostic_only", False))
        if diagnostic_only and required:
            raise ValueError(
                f"Target '{name}' cannot be both 'diagnostic_only' and 'required'. Please adjust configuration."
            )

        abs_tolerance = float(entry.get("abs_tolerance", 0.5))

        targets.append(
            TargetSpec(
                name=name,
                value=float(value),
                label=label,
                tier=tier,
                cv_threshold=cv_threshold,
                required=required,
                diagnostic_only=diagnostic_only,
                abs_tolerance=abs_tolerance,
            )
        )

    gating_targets = [t for t in targets if t.is_gate()]
    if not gating_targets:
        raise ValueError("Configuration must include at least one non-diagnostic target with a CV threshold.")

    return targets


def ensure_even(value: int) -> int:
    return value if value % 2 == 0 else max(value - 1, 2)


def build_roi_specs(schedule: Dict[str, Dict[str, int]]) -> List[RoiSpecification]:
    initial = schedule.get("initial", {})
    step = schedule.get("step", {})
    count = int(schedule.get("count", 0))
    if count <= 0:
        raise ValueError("ROI schedule 'count' must be a positive integer.")

    init_w = int(initial.get("width", 0))
    init_h = int(initial.get("height", 0))
    init_d = int(initial.get("depth", 0))
    step_w = int(step.get("width", 0))
    step_h = int(step.get("height", 0))
    step_d = int(step.get("depth", 0))

    if min(init_w, init_h, init_d) <= 0:
        raise ValueError("ROI schedule initial dimensions must be positive.")

    specs: List[RoiSpecification] = []
    for idx in range(count):
        width = ensure_even(init_w + idx * step_w)
        height = ensure_even(init_h + idx * step_h)
        depth = max(2, init_d + idx * step_d)
        specs.append(RoiSpecification(index=idx, width=width, height=height, depth=depth))

    return specs


def compute_depth_indices(num_layers: int, depth: int, center: Optional[int]) -> Tuple[int, int]:
    depth = max(2, depth)
    depth = min(depth, num_layers)
    if center is None:
        center = num_layers // 2
    center = int(np.clip(center, 0, num_layers - 1))

    half = depth // 2
    start = max(center - half, 0)
    end = start + depth
    if end > num_layers:
        end = num_layers
        start = max(end - depth, 0)
    # ensure at least two layers
    if end - start < 2:
        end = min(start + 2, num_layers)
    return start, end


def compute_roi_coords(
    rows: int, cols: int, width: int, height: int, center_x: Optional[int], center_y: Optional[int]
) -> Tuple[int, int, int, int]:
    width = ensure_even(width)
    height = ensure_even(height)

    if center_x is None:
        center_x = cols // 2
    if center_y is None:
        center_y = rows // 2

    center_x = int(np.clip(center_x, 0, cols - 1))
    center_y = int(np.clip(center_y, 0, rows - 1))

    half_w = width // 2
    half_h = height // 2

    x_start = max(center_x - half_w, 0)
    x_end = min(center_x + half_w, cols)
    y_start = max(center_y - half_h, 0)
    y_end = min(center_y + half_h, rows)

    if (x_end - x_start) % 2 == 1:
        if x_end < cols:
            x_end += 1
        elif x_start > 0:
            x_start -= 1
    if (y_end - y_start) % 2 == 1:
        if y_end < rows:
            y_end += 1
        elif y_start > 0:
            y_start -= 1

    x_start = int(np.clip(x_start, 0, cols - 1))
    x_end = int(np.clip(x_end, x_start + 1, cols))
    y_start = int(np.clip(y_start, 0, rows - 1))
    y_end = int(np.clip(y_end, y_start + 1, rows))

    return y_start, y_end, x_start, x_end


def compute_nice_ticks(min_val: int, max_val: int, desired: int = 3) -> List[int]:
    min_val_f = float(min_val)
    max_val_f = float(max_val)
    if min_val_f > max_val_f:
        min_val_f, max_val_f = max_val_f, min_val_f

    span = max_val_f - min_val_f
    if np.isclose(span, 0.0):
        rounded = int(round(min_val_f))
        return [rounded]

    desired = max(2, desired)
    rough_step = span / (desired - 1)
    magnitude = 10 ** np.floor(np.log10(rough_step)) if rough_step > 0 else 1.0
    nice_multipliers = np.array([1.0, 2.0, 2.5, 5.0, 10.0])
    candidate_steps = nice_multipliers * magnitude

    best_ticks: Optional[List[float]] = None
    for step in candidate_steps:
        if step <= 0:
            continue
        start_tick = np.ceil(min_val_f / step) * step
        ticks: List[float] = []
        val = start_tick
        while val <= max_val_f + 1e-9:
            if min_val_f - 1e-9 <= val <= max_val_f + 1e-9:
                ticks.append(val)
            val += step

        if len(ticks) >= 2:
            if len(ticks) <= desired:
                best_ticks = ticks
                if len(ticks) == desired:
                    break
            else:
                indices = np.linspace(0, len(ticks) - 1, num=desired, dtype=int)
                best_ticks = [ticks[i] for i in indices]
                break

    if best_ticks is None or len(best_ticks) < 2:
        best_ticks = [min_val_f, max_val_f]

    rounded_ticks: List[int] = []
    for val in best_ticks:
        rounded = int(round(val))
        rounded = max(int(np.floor(min_val_f)), min(int(np.ceil(max_val_f)), rounded))
        if not rounded_ticks or rounded != rounded_ticks[-1]:
            rounded_ticks.append(rounded)

    while len(rounded_ticks) > desired:
        rounded_ticks = rounded_ticks[:-1]

    if len(rounded_ticks) < 2:
        start_tick = int(round(min_val_f))
        end_tick = int(round(max_val_f))
        if end_tick == start_tick:
            end_tick = start_tick + 1
        rounded_ticks = [start_tick, end_tick]

    return rounded_ticks[:desired]


def prompt_for_center(max_x: int, max_y: int, default_x: int, default_y: int) -> Tuple[int, int]:
    print("\nROI center not specified in configuration.")
    print(f"Press Enter to accept defaults. Default center -> X: {default_x}, Y: {default_y}")
    while True:
        user_x = input(f"Enter ROI center X (0-{max_x - 1}): ").strip()
        user_y = input(f"Enter ROI center Y (0-{max_y - 1}): ").strip()

        if not user_x and not user_y:
            return default_x, default_y

        try:
            x_val = int(user_x) if user_x else default_x
            y_val = int(user_y) if user_y else default_y
        except ValueError:
            print("Invalid input. Please enter integer coordinates or leave blank for defaults.")
            continue

        if x_val < 0 or y_val < 0 or x_val >= max_x or y_val >= max_y:
            print("Coordinates must be within image bounds.")
            continue

        return x_val, y_val


def sanitize_correlation(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float).flatten()
    arr[arr < EPS_FLOAT] = EPS_FLOAT
    arr[arr > 1.0] = 1.0
    return arr


def sanitize_header_label(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in label.strip())
    cleaned = "_".join(filter(None, cleaned.split("_")))
    return cleaned or "target"


def calculate_characteristic_lengths(
    distance: np.ndarray, corr_mean: np.ndarray, targets: List[TargetSpec]
) -> Dict[str, Optional[float]]:
    distance = np.asarray(distance, dtype=float).flatten()
    corr_mean = sanitize_correlation(corr_mean)

    result: Dict[str, Optional[float]] = {}
    for spec in targets:
        target = float(spec.value)
        if corr_mean[0] < target:
            result[spec.name] = None
            continue
        if target < 0:
            result[spec.name] = None
            continue

        below = np.where(corr_mean <= target)[0]
        if below.size == 0:
            result[spec.name] = None
            continue

        idx_low = below[0]
        if idx_low == 0:
            result[spec.name] = float(distance[0])
            continue

        idx_high = idx_low - 1
        y1, y2 = corr_mean[idx_high], corr_mean[idx_low]
        x1, x2 = distance[idx_high], distance[idx_low]

        if abs(y2 - y1) <= EPS_FLOAT:
            interp_x = float(x1)
        else:
            slope = (y2 - y1) / (x2 - x1)
            interp_x = float(x1 + (target - y1) / slope)
        result[spec.name] = max(interp_x, 0.0)

    return result


def determine_target_decisions(
    roi_results: List[RoiResult],
    targets: List[TargetSpec],
    window_size: int,
) -> Dict[str, TargetDecision]:
    """Evaluate convergence for each target using a sliding window over ROI results.

    For each target, the function slides a window of ``window_size`` consecutive
    ROIs (sorted by effective size) and checks whether the characteristic lengths
    within that window are stable.  A window is **stable** when either:

    - **Relative criterion**: CV = std / mean <= ``cv_threshold``, or
    - **Absolute criterion**: std <= ``abs_tolerance`` (voxels).

    The absolute criterion prevents false negatives when characteristic lengths
    are inherently small (e.g. < 5 voxels) and minor numerical noise inflates
    the CV beyond the threshold.

    Once a stable window is found, all subsequent windows must also pass
    (persistence check).  The first ROI of the earliest persistent stable
    window is reported as the convergence point.
    """
    decisions: Dict[str, TargetDecision] = {}
    indexed_results = list(enumerate(roi_results))
    indexed_results.sort(key=lambda item: (item[1].effective_length(), item[0]))
    sorted_results = [item[1] for item in indexed_results]
    window_size = max(2, window_size)
    insufficient_results = len(sorted_results) < window_size

    if not sorted_results:
        for spec in targets:
            decisions[spec.name] = TargetDecision(
                spec=spec,
                sorted_start_index=None,
                roi_result=None,
                stable=False,
                reason="No ROI results available.",
            )
        return decisions

    def values_for_window(target_name: str, end_idx: int) -> Optional[List[float]]:
        start_idx = end_idx - window_size + 1
        if start_idx < 0:
            return None
        collected: List[float] = []
        for idx in range(start_idx, end_idx + 1):
            value = sorted_results[idx].length_map.get(target_name)
            if value is None:
                return None
            collected.append(float(value))
        return collected

    for spec in targets:
        if spec.cv_threshold is None:
            decisions[spec.name] = TargetDecision(
                spec=spec,
                sorted_start_index=None,
                roi_result=None,
                stable=False,
                reason="No CV threshold configured; skipping stability test.",
            )
            continue

        if insufficient_results:
            decisions[spec.name] = TargetDecision(
                spec=spec,
                sorted_start_index=None,
                roi_result=None,
                stable=False,
                reason=f"Fewer ROI results than required window size ({window_size}).",
            )
            continue

        chosen_start: Optional[int] = None
        failure_reason = "Sliding windows failed the CV threshold."

        for end_idx in range(window_size - 1, len(sorted_results)):
            window_vals = values_for_window(spec.name, end_idx)
            if window_vals is None:
                failure_reason = "Missing characteristic length values within sliding windows."
                continue

            mean_val = float(np.mean(window_vals))
            if mean_val <= EPS_FLOAT:
                continue
            std_val = float(np.std(window_vals))
            cv_ok = std_val / mean_val <= spec.cv_threshold
            abs_ok = std_val <= spec.abs_tolerance
            if not (cv_ok or abs_ok):
                failure_reason = (
                    f"Coefficient of variation {std_val / mean_val:.3f} exceeded threshold {spec.cv_threshold:.3f} "
                    f"and std {std_val:.3f} exceeded abs_tolerance {spec.abs_tolerance:.3f}."
                )
                continue

            future_ok = True
            for future_end in range(end_idx + 1, len(sorted_results)):
                future_vals = values_for_window(spec.name, future_end)
                if future_vals is None:
                    future_ok = False
                    failure_reason = "Missing characteristic length values in future windows."
                    break
                future_mean = float(np.mean(future_vals))
                if future_mean <= EPS_FLOAT:
                    continue
                future_std = float(np.std(future_vals))
                future_cv_ok = future_std / future_mean <= spec.cv_threshold
                future_abs_ok = future_std <= spec.abs_tolerance
                if not (future_cv_ok or future_abs_ok):
                    future_ok = False
                    failure_reason = (
                        f"Future window CV {future_std / future_mean:.3f} exceeded threshold {spec.cv_threshold:.3f} "
                        f"and std {future_std:.3f} exceeded abs_tolerance {spec.abs_tolerance:.3f}."
                    )
                    break
            if future_ok:
                chosen_start = end_idx - window_size + 1
                break

        if chosen_start is None:
            decisions[spec.name] = TargetDecision(
                spec=spec,
                sorted_start_index=None,
                roi_result=None,
                stable=False,
                reason=failure_reason,
            )
        else:
            decisions[spec.name] = TargetDecision(
                spec=spec,
                sorted_start_index=chosen_start,
                roi_result=sorted_results[chosen_start],
                stable=True,
                reason=None,
            )

    return decisions


def plot_roi_overview(
    base_slice: np.ndarray,
    roi_results: List[RoiResult],
    output_path: Path,
    center_x: int,
    center_y: int,
) -> None:
    with plt.rc_context({"font.family": FONT_FAMILY, "font.sans-serif": [FONT_FAMILY]}):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(base_slice, cmap="gray")

        cmap = plt.get_cmap("tab20")
        for idx, res in enumerate(sorted(roi_results, key=lambda r: r.effective_length())):
            color = cmap(idx % cmap.N)
            y_start, y_end, x_start, x_end = res.roi_coords
            width = x_end - x_start
            height = y_end - y_start
            rect = plt.Rectangle(
                (x_start, y_start),
                width,
                height,
                linewidth=1.8,
                edgecolor=color,
                facecolor="none",
                label=res.spec.roi_label(),
            )
            ax.add_patch(rect)

        ax.scatter([center_x], [center_y], color="red", marker="+", s=120, label="ROI center")
        ax.set_title("ROI footprints on middle z-layer", fontsize=26)
        ax.set_xlabel("X (voxels)", fontsize=26)
        ax.set_ylabel("Y (voxels)", fontsize=26)
        ax.tick_params(axis="both", labelsize=26)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        fig.savefig(output_path.with_suffix(".svg"), dpi=300)
        plt.close(fig)


def plot_roi_overview_zoom(
    base_slice: np.ndarray,
    roi_results: List[RoiResult],
    output_path: Path,
    center_x: int,
    center_y: int,
    scale_factor: float = 1.3,
) -> None:
    if not roi_results:
        return

    rows, cols = base_slice.shape
    largest_roi = max(
        roi_results,
        key=lambda res: (res.roi_coords[1] - res.roi_coords[0]) * (res.roi_coords[3] - res.roi_coords[2]),
    )
    y_start, y_end, x_start, x_end = largest_roi.roi_coords
    roi_height = max(y_end - y_start, 1)
    roi_width = max(x_end - x_start, 1)

    pad_y = int(np.ceil((roi_height * scale_factor - roi_height) / 2.0))
    pad_x = int(np.ceil((roi_width * scale_factor - roi_width) / 2.0))

    zoom_y_start = max(y_start - pad_y, 0)
    zoom_y_end = min(y_end + pad_y, rows)
    zoom_x_start = max(x_start - pad_x, 0)
    zoom_x_end = min(x_end + pad_x, cols)

    if zoom_y_end <= zoom_y_start or zoom_x_end <= zoom_x_start:
        return

    zoom_slice = base_slice[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end]

    with plt.rc_context({"font.family": FONT_FAMILY, "font.sans-serif": [FONT_FAMILY]}):
        fig, ax = plt.subplots(figsize=(8, 8))
        extent = (
            float(zoom_x_start),
            float(zoom_x_end),
            float(zoom_y_end),
            float(zoom_y_start),
        )
        ax.imshow(zoom_slice, cmap="gray", extent=extent, origin="upper")

        cmap = plt.get_cmap("tab20")
        for idx, res in enumerate(sorted(roi_results, key=lambda r: r.effective_length())):
            color = cmap(idx % cmap.N)
            y0, y1, x0, x1 = res.roi_coords
            x0_clamped = max(x0, zoom_x_start)
            x1_clamped = min(x1, zoom_x_end)
            y0_clamped = max(y0, zoom_y_start)
            y1_clamped = min(y1, zoom_y_end)
            width = x1_clamped - x0_clamped
            height = y1_clamped - y0_clamped
            if width <= 0 or height <= 0:
                continue
            rect = plt.Rectangle(
                (x0_clamped, y0_clamped),
                width,
                height,
                linewidth=1.8,
                edgecolor=color,
                facecolor="none",
                label=res.spec.roi_label(),
            )
            ax.add_patch(rect)

        if zoom_x_start <= center_x <= zoom_x_end and zoom_y_start <= center_y <= zoom_y_end:
            ax.scatter([center_x], [center_y], color="red", marker="+", s=120, label="ROI center")

        ax.tick_params(axis="both", labelsize=ZOOM_TICK_FONT_SIZE)
        x_ticks = compute_nice_ticks(zoom_x_start, zoom_x_end, desired=3)
        y_ticks = compute_nice_ticks(zoom_y_start, zoom_y_end, desired=3)
        if len(x_ticks) < 3:
            x_ticks = compute_nice_ticks(zoom_x_start, zoom_x_end, desired=2)
        if len(y_ticks) < 3:
            y_ticks = compute_nice_ticks(zoom_y_start, zoom_y_end, desired=2)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels([f"{tick:d}" for tick in x_ticks], fontsize=ZOOM_TICK_FONT_SIZE)
        ax.set_yticklabels([f"{tick:d}" for tick in y_ticks], fontsize=ZOOM_TICK_FONT_SIZE)
        ax.set_xlim(zoom_x_start, zoom_x_end)
        ax.set_ylim(zoom_y_end, zoom_y_start)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        fig.savefig(output_path.with_suffix(".svg"), dpi=300)
        plt.close(fig)


def plot_roi_legend(roi_results: List[RoiResult], output_path: Path) -> None:
    if not roi_results:
        return

    sorted_results = sorted(roi_results, key=lambda r: r.effective_length())
    cmap = plt.get_cmap("tab20")
    handles: List[Line2D] = []
    labels: List[str] = []
    for idx, res in enumerate(sorted_results):
        color = cmap(idx % cmap.N)
        handle = Line2D([0], [0], color=color, linewidth=4.0)
        label = f"{res.spec.roi_label()} (W{res.actual_dims[1]}, H{res.actual_dims[0]}, D{res.actual_dims[2]})"
        handles.append(handle)
        labels.append(label)

    fig_height = max(2.0, 0.45 * len(handles))
    with plt.rc_context({"font.family": FONT_FAMILY, "font.sans-serif": [FONT_FAMILY]}):
        fig, ax = plt.subplots(figsize=(6, fig_height))
        ax.axis("off")
        ax.legend(handles, labels, loc="center", fontsize=LEGEND_FONT_SIZE, frameon=False)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".svg"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_autocorrelation_overlays(
    roi_results: List[RoiResult], targets: List[TargetSpec], output_path: Path
) -> None:
    with plt.rc_context({"font.family": FONT_FAMILY, "font.sans-serif": [FONT_FAMILY]}):
        fig, ax = plt.subplots(figsize=(8, 6))
        sorted_results = sorted(roi_results, key=lambda r: r.effective_length())
        cmap = plt.get_cmap("tab20")
        for idx, res in enumerate(sorted_results):
            color = cmap(idx % cmap.N)
            ax.plot(
                res.distance,
                res.corr_mean,
                label=f"{res.spec.roi_label()} (W{res.actual_dims[1]},H{res.actual_dims[0]},D{res.actual_dims[2]})",
                linewidth=2.0,
                color=color,
            )

        ax.axhline(float(np.exp(-1.0)), color="black", linestyle="--", linewidth=1.5, label="AC = 1/e")
        ax.axhline(0.1, color="gray", linestyle="--", linewidth=1.5, label="AC = 0.1")

        ax.set_xlabel("Autocorrelation length (voxels)", fontsize=AXIS_FONT_SIZE)
        ax.set_ylabel("Autocorrelation (AC)", fontsize=AXIS_FONT_SIZE)
        ax.set_title("AC curves with different ROIs", fontsize=TITLE_FONT_SIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        fig.savefig(output_path.with_suffix(".svg"), dpi=300)
        plt.close(fig)


def plot_length_trends(
    roi_results: List[RoiResult],
    targets: List[TargetSpec],
    decisions: Dict[str, TargetDecision],
    combined_required_sorted_idx: Optional[int],
    output_path: Path,
) -> None:
    indexed_results = list(enumerate(roi_results))
    indexed_results.sort(key=lambda item: (item[1].effective_length(), item[0]))
    sorted_results = [item[1] for item in indexed_results]
    sizes = [res.effective_length() for res in sorted_results]
    roi_size_labels = [
        f"{res.actual_dims[1]}x{res.actual_dims[0]}x{res.actual_dims[2]}" for res in sorted_results
    ]

    with plt.rc_context({"font.family": FONT_FAMILY, "font.sans-serif": [FONT_FAMILY]}):
        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = plt.get_cmap("tab10")
        for idx, spec in enumerate(targets):
            if spec.diagnostic_only:
                continue
            values = [res.length_map.get(spec.name) for res in sorted_results]
            if all(v is None for v in values):
                continue
            color = cmap(idx % cmap.N)
            plotted_values = [np.nan if v is None else float(v) for v in values]
            ax.plot(
                sizes,
                plotted_values,
                marker="o",
                label=f"AC target {spec.label}",
                color=color,
                linewidth=2.0,
            )

        added_markers: Set[str] = set()
        for decision in decisions.values():
            if not decision.stable or decision.sorted_start_index is None:
                continue
            if decision.spec.diagnostic_only:
                continue
            if np.isclose(decision.spec.value, np.exp(-1.0), atol=1e-4) or np.isclose(
                decision.spec.value, 0.1, atol=1e-4
            ):
                continue
            x_value = sizes[decision.sorted_start_index]
            label = decision.spec.label
            color = "black"
            if np.isclose(decision.spec.value, np.exp(-1.0), atol=1e-4):
                color = "black"
            elif np.isclose(decision.spec.value, 0.1, atol=1e-4):
                color = "gray"
            else:
                color = "orange" if decision.spec.required else "green"
            line_label = f"{label} RVE"
            if line_label not in added_markers:
                ax.axvline(x_value, color=color, linestyle="--", linewidth=1.8, label=line_label)
                added_markers.add(line_label)
            else:
                ax.axvline(x_value, color=color, linestyle="--", linewidth=1.8)

        ax.set_xlabel("Effective ROI size (geometric mean, voxels)", fontsize=AXIS_FONT_SIZE)
        ax.set_ylabel("Autocorrelation length (voxels)", fontsize=AXIS_FONT_SIZE)
        ax.set_title("Autocorrelation length trends with ROI", fontsize=TITLE_FONT_SIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
        ax.set_xticks(sizes)
        ax.set_xticklabels(roi_size_labels, rotation=90, ha="center", fontsize=TICK_FONT_SIZE)
        ax.legend(fontsize=LEGEND_FONT_SIZE, loc="best", frameon=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        fig.savefig(output_path.with_suffix(".svg"), dpi=300)
        plt.close(fig)


def save_roi_npz(res: RoiResult, output_path: Path, targets: List[TargetSpec]) -> None:
    target_values = np.array([spec.value for spec in targets], dtype=float)
    target_names = np.array([spec.name for spec in targets])
    target_labels = np.array([spec.label for spec in targets])
    length_values = np.array(
        [np.nan if res.length_map.get(spec.name) is None else float(res.length_map[spec.name]) for spec in targets],
        dtype=float,
    )

    np.savez(
        output_path,
        distance=res.distance,
        corr_mean=res.corr_mean,
        corr_std=res.corr_std,
        roi_width=res.actual_dims[1],
        roi_height=res.actual_dims[0],
        roi_depth=res.actual_dims[2],
        targets=target_values,
        target_names=target_names,
        target_labels=target_labels,
        lengths=length_values,
        depth_indices=np.array(res.depth_indices),
        roi_coords=np.array(res.roi_coords),
        step_index=res.spec.index,
        roi_label=res.spec.roi_label(),
)


def write_summary(
    roi_results: List[RoiResult],
    targets: List[TargetSpec],
    window_size: int,
    decisions: Dict[str, TargetDecision],
    summary_path: Path,
    report_path: Path,
    center_x: int,
    center_y: int,
    layer_index: int,
) -> None:
    indexed_results = list(enumerate(roi_results))
    indexed_results.sort(key=lambda item: (item[1].effective_length(), item[0]))
    sorted_results = [item[1] for item in indexed_results]

    headers = ["ROI_Label", "Width", "Height", "Depth", "Depth_Start", "Depth_End"]
    headers.extend([f"Length_{sanitize_header_label(spec.name)}" for spec in targets])
    headers.append("Effective_Size")

    lines = [",".join(headers)]
    for res in sorted_results:
        width = res.actual_dims[1]
        height = res.actual_dims[0]
        depth = res.actual_dims[2]
        depth_start, depth_end = res.depth_indices
        values = [
            res.spec.roi_label(),
            str(width),
            str(height),
            str(depth),
            str(depth_start),
            str(depth_end),
        ]
        for spec in targets:
            val = res.length_map.get(spec.name)
            values.append(f"{val:.4f}" if val is not None else "NA")
        values.append(f"{res.effective_length():.4f}")
        lines.append(",".join(values))

    summary_path.write_text("\n".join(lines), encoding="utf-8")

    required_decisions = [d for d in decisions.values() if d.spec.is_gate()]
    stable_required = [d for d in required_decisions if d.stable and d.roi_result is not None]
    combined_required = None
    if required_decisions and len(stable_required) == len(required_decisions):
        combined_required = max(
            stable_required,
            key=lambda d: d.effective_length if d.effective_length is not None else float("-inf"),
        )

    with report_path.open("w", encoding="utf-8") as report_file:
        report_file.write("# RVE Autocorrelation Summary\n\n")
        report_file.write(f"- Rolling window size: {window_size}\n")
        report_file.write(f"- ROI center: X={center_x}, Y={center_y}\n")
        report_file.write(f"- Visualization layer index: {layer_index}\n")
        report_file.write("\n## Target Thresholds\n")
        for spec in targets:
            role = (
                "Required"
                if spec.is_gate()
                else ("Diagnostic" if spec.diagnostic_only else "Optional")
            )
            if spec.cv_threshold is None:
                thresh_text = "not evaluated"
            else:
                thresh_text = f"CV \u2264 {spec.cv_threshold:.3f}"
            report_file.write(f"- {spec.label} (AC={spec.value:g}): {thresh_text} [{role}]\n")

        report_file.write("\n## RVE Decisions\n")
        for decision in decisions.values():
            role = (
                "Required"
                if decision.spec.is_gate()
                else ("Diagnostic" if decision.spec.diagnostic_only else "Optional")
            )
            if decision.stable and decision.roi_result is not None:
                dims = decision.roi_result.actual_dims
                report_file.write(
                    f"- {decision.spec.label} ({role}) stabilized at {decision.roi_result.spec.roi_label()} "
                    f"(dims: W{dims[1]}, H{dims[0]}, D{dims[2]}).\n"
                )
            else:
                reason = decision.reason or "No stable window identified."
                report_file.write(f"- {decision.spec.label} ({role}) not stable: {reason}\n")

        report_file.write("\n## Combined Required RVE\n")
        if combined_required is not None:
            dims = combined_required.roi_result.actual_dims  # type: ignore[union-attr]
            report_file.write(
                f"- Combined required RVE: {combined_required.roi_result.spec.roi_label()} "
                f"(dims: W{dims[1]}, H{dims[0]}, D{dims[2]}).\n"
            )
        elif required_decisions:
            report_file.write("- Combined required RVE: not achieved within evaluated ROI sizes.\n")
        else:
            report_file.write("- No required targets were configured.\n")

        report_file.write("\n## Notes\n")
        report_file.write(
            "All autocorrelation curves were computed with denoising disabled and without downsampling. "
            "Mean curves are plotted without +/- std bands, per user requirements.\n"
        )
        report_file.write("\n## Generated Figures\n")
        report_file.write("- figures/roi_overview.png: ROI footprints on the middle slice\n")
        report_file.write("- figures/roi_overview_zoomin.png: Zoomed-in ROI footprints\n")
        report_file.write("- figures/roi_legend.png: ROI color legend\n")
        report_file.write("- figures/autocorrelation_overlays.png: Mean autocorrelation curves by ROI size\n")
        report_file.write("- figures/characteristic_length_trends.png: Convergence of characteristic lengths\n")


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    settings = load_settings(config_path)
    print(f"\\nUsing configuration file: {config_path}")
    sys.stdout.flush()

    roi_schedule = settings.get("roi_schedule")
    if roi_schedule is None:
        raise KeyError("Configuration must include 'roi_schedule'.")
    roi_specs = build_roi_specs(roi_schedule)

    target_specs = parse_target_specs(settings)
    convergence_cfg = settings.get("convergence", {})
    window_size = int(convergence_cfg.get("window_size", 3))
    window_size = max(2, window_size)
    save_npz = bool(settings.get("save_npz", True))

    announce_step("\nStep 1: Select the 3D TIFF stack (a file dialog will open)...")
    tif_path, _ = select_tiff_file()
    announce_step("Step 2: Choose an output directory for analysis outputs (a folder dialog will open)...")
    output_dir = Path(select_output_directory())

    announce_step("Loading selected TIFF stack...")
    imgs = load_tiff_stack(tif_path)
    announce_step(f"Loaded stack with shape {imgs.shape}.")

    rows, cols = imgs.shape[1], imgs.shape[2]
    roi_center_cfg = settings.get("roi_center", {})
    prompt_center = bool(roi_center_cfg.get("prompt_if_null", False))

    default_center_x = cols // 2
    default_center_y = rows // 2

    cfg_center_x = roi_center_cfg.get("x")
    cfg_center_y = roi_center_cfg.get("y")

    if prompt_center and (cfg_center_x is None or cfg_center_y is None):
        center_x, center_y = prompt_for_center(cols, rows, default_center_x, default_center_y)
    else:
        center_x = default_center_x if cfg_center_x is None else int(cfg_center_x)
        center_y = default_center_y if cfg_center_y is None else int(cfg_center_y)

    center_x = int(np.clip(center_x, 0, cols - 1))
    center_y = int(np.clip(center_y, 0, rows - 1))
    layer_center_setting = settings.get("layer_center")
    if layer_center_setting is not None:
        layer_center_index = int(np.clip(int(layer_center_setting), 0, imgs.shape[0] - 1))
        depth_center = layer_center_index
    else:
        layer_center_index = imgs.shape[0] // 2
        depth_center = None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"RVE_{timestamp}"
    data_dir = run_dir / "data"
    fig_dir = run_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    roi_results: List[RoiResult] = []

    for spec in sorted(roi_specs, key=lambda s: s.effective_length()):
        depth_start, depth_end = compute_depth_indices(imgs.shape[0], spec.depth, depth_center)
        depth_slice = imgs[depth_start:depth_end, :, :]
        actual_depth = depth_slice.shape[0]

        roi_coords = compute_roi_coords(
            rows=depth_slice.shape[1],
            cols=depth_slice.shape[2],
            width=spec.width,
            height=spec.height,
            center_x=center_x,
            center_y=center_y,
        )

        roi_imgs = apply_roi_cropping(depth_slice, roi_coords, True, DENOISE_OPTION_DISABLED)
        actual_height, actual_width = roi_imgs.shape[1], roi_imgs.shape[2]

        nxcorrcoeff_vol, r = compute_autocorrelation(roi_imgs)
        distance, corr_mean, corr_std, _ = compute_radial_autocorrelation(nxcorrcoeff_vol, r)

        distance = np.asarray(distance, dtype=float).flatten()
        corr_mean = np.asarray(corr_mean, dtype=float).flatten()
        corr_std = np.asarray(corr_std, dtype=float).flatten()

        length_map = calculate_characteristic_lengths(distance, corr_mean, target_specs)

        result = RoiResult(
            spec=spec,
            actual_dims=(actual_height, actual_width, actual_depth),
            distance=distance,
            corr_mean=corr_mean,
            corr_std=corr_std,
            length_map=length_map,
            depth_indices=(depth_start, depth_end),
            roi_coords=roi_coords,
        )
        roi_results.append(result)

        if (actual_width != spec.width) or (actual_height != spec.height) or (actual_depth != spec.depth):
            print(
                f"Warning: ROI {spec.roi_label()} clipped to "
                f"W{actual_width}, H{actual_height}, D{actual_depth} due to dataset bounds."
            )

        if save_npz:
            npz_path = data_dir / f"{spec.file_tag()}_autocorr.npz"
            save_roi_npz(result, npz_path, target_specs)

    if roi_results:
        base_slice = imgs[layer_center_index, :, :]
        overview_path = fig_dir / "roi_overview.png"
        plot_roi_overview(base_slice, roi_results, overview_path, center_x, center_y)
        zoom_path = fig_dir / "roi_overview_zoomin.png"
        plot_roi_overview_zoom(base_slice, roi_results, zoom_path, center_x, center_y)
        legend_path = fig_dir / "roi_legend.png"
        plot_roi_legend(roi_results, legend_path)

    decisions = determine_target_decisions(roi_results, target_specs, window_size)
    required_decisions = [d for d in decisions.values() if d.spec.is_gate()]
    combined_required_sorted_idx: Optional[int] = None
    combined_required_decision: Optional[TargetDecision] = None
    if required_decisions and all(d.stable and d.sorted_start_index is not None for d in required_decisions):
        combined_required_sorted_idx = max(
            d.sorted_start_index for d in required_decisions if d.sorted_start_index is not None
        )
        stable_required_with_roi = [
            d for d in required_decisions if d.roi_result is not None and d.effective_length is not None
        ]
        if stable_required_with_roi:
            combined_required_decision = max(
                stable_required_with_roi,
                key=lambda d: d.effective_length if d.effective_length is not None else float("-inf"),
            )

    overlay_path = fig_dir / "autocorrelation_overlays.png"
    plot_autocorrelation_overlays(roi_results, target_specs, overlay_path)

    trend_path = fig_dir / "characteristic_length_trends.png"
    plot_length_trends(roi_results, target_specs, decisions, combined_required_sorted_idx, trend_path)

    summary_path = run_dir / "rve_summary.csv"
    report_path = run_dir / "RVE_report.md"
    write_summary(
        roi_results,
        target_specs,
        window_size,
        decisions,
        summary_path,
        report_path,
        center_x,
        center_y,
        layer_center_index,
    )

    print("\nRVE analysis complete.")
    print(f"Results stored in: {run_dir}")
    for decision in decisions.values():
        role = "required" if decision.spec.is_gate() else ("diagnostic" if decision.spec.diagnostic_only else "optional")
        if decision.stable and decision.roi_result is not None:
            dims = decision.roi_result.actual_dims
            print(
                f"{decision.spec.label} target ({role}) stabilized at {decision.roi_result.spec.roi_label()} "
                f"(W{dims[1]}, H{dims[0]}, D{dims[2]})"
            )
        else:
            reason = decision.reason or "no stable window identified"
            print(f"{decision.spec.label} target ({role}) did not stabilize: {reason}")

    if combined_required_decision is not None and combined_required_decision.roi_result is not None:
        dims = combined_required_decision.roi_result.actual_dims
        print(
            f"Combined required RVE: {combined_required_decision.roi_result.spec.roi_label()} "
            f"(W{dims[1]}, H{dims[0]}, D{dims[2]})"
        )
    elif required_decisions:
        print("Combined required RVE could not be established within evaluated ROI sizes.")


if __name__ == "__main__":
    main()
    
