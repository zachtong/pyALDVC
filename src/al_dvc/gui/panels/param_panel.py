"""Parameter editor bound to ``AppState.para`` (a ``DVCPara``).

Folding sections, pyALDIC style (fixed-width labels and inputs, wheel only when focused):
subset & search, solver, units, performance, advanced. Strain settings live in the strain
post-processing window; the analysed box follows the region of interest drawn on the slices
(``AppState.effective_voi``). Every choice shows a readable, translated name and keeps the
solver's key as item data (:mod:`al_dvc.gui.names`).
"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from al_dvc.io.volume_ops import memory_model

from ..app_state import AppState
from ..names import fill_combo, retranslate_combo, select_key
from ..widgets import COMBO_WIDTH, CollapsibleSection, dspin, form_label, guard_wheel, make_form, spin

VOXEL_FIELD_WIDTH = 72
SUBSET_FIELD_WIDTH = 52  # px, the three per-axis subset sizes


class ParamPanel(QWidget):
    """Form of the ``DVCPara`` fields, grouped in folding sections."""

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._updating = False
        self.labels: dict[str, QLabel] = {}
        self.sections: dict[str, CollapsibleSection] = {}
        self.combos: dict[QComboBox, str] = {}  # combo -> names.CHOICES group
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ---- subset & search
        # the odd voxel span 2h+1 per axis (x, y, z); stored as the even winsize = 2h. With the lock on, the x
        # box drives all three (a cubic subset); off, every axis is set on its own.
        self.winsize_axes = [spin(5, 257, 2, width=SUBSET_FIELD_WIDTH) for _ in range(3)]
        self.winsize = self.winsize_axes[0]
        self.winsize_lock = QCheckBox()
        self.winsize_lock.setChecked(True)
        winsize_widget = QWidget()
        winsize_row = QHBoxLayout(winsize_widget)
        winsize_row.setContentsMargins(0, 0, 0, 0)
        winsize_row.setSpacing(4)
        for w in self.winsize_axes:
            winsize_row.addWidget(w)
        winsize_row.addSpacing(4)
        winsize_row.addWidget(self.winsize_lock)
        self.winstepsize = spin(1, 128, 1)
        self.search_radius = spin(1, 128, 1)
        self.init_method = self._choice("init")
        self.interp = self._choice("interp")
        self._add_section(
            "subset",
            layout,
            [
                ("winsize", winsize_widget),
                ("winstepsize", self.winstepsize),
                ("search_radius", self.search_radius),
                ("init_method", self.init_method),
                ("interp", self.interp),
            ],
        )

        # ---- solver
        self.reference_mode = self._choice("tracking")
        self.solver = self._choice("solver")
        self._add_section("solver", layout, [("reference_mode", self.reference_mode), ("solver", self.solver)])

        # ---- units
        self.voxel = [dspin(1e-6, 1e6, 4, width=VOXEL_FIELD_WIDTH) for _ in range(3)]
        voxel_widget = QWidget()
        voxel_row = QHBoxLayout(voxel_widget)
        voxel_row.setContentsMargins(0, 0, 0, 0)
        voxel_row.setSpacing(4)
        for w in self.voxel:
            voxel_row.addWidget(w)
        self.units = QLineEdit()
        self.units.setFixedWidth(COMBO_WIDTH)
        self._add_section("units", layout, [("voxel", voxel_widget), ("units", self.units)])

        # ---- performance
        self.backend = self._choice("backend")
        self.backend_status = QLabel()
        self.backend_status.setObjectName("hint")
        self.backend_status.setWordWrap(True)
        self.n_threads = spin(0, 512, 1)
        self.gradient_mode = self._choice("gradient")
        perf = self._add_section(
            "performance",
            layout,
            [("backend", self.backend), ("n_threads", self.n_threads), ("gradient_mode", self.gradient_mode)],
        )
        perf.add_widget(self.backend_status)
        self._memory = QLabel()
        self._memory.setObjectName("hint")
        self._memory.setWordWrap(True)
        perf.add_widget(self._memory)

        # ---- advanced (folded)
        self.subpb2 = self._choice("discretisation")
        self.subset_stride = spin(1, 8, 1)
        self.init_coarse = spin(1, 8, 1)
        self.prefilter = dspin(0.0, 10.0, 2)
        self.mu = dspin(1e-8, 1e3, 6)
        self.beta_auto = QCheckBox()
        self.beta = dspin(1e-8, 1e6, 6)
        beta_widget = QWidget()
        beta_row = QHBoxLayout(beta_widget)
        beta_row.setContentsMargins(0, 0, 0, 0)
        beta_row.setSpacing(6)
        beta_row.addWidget(self.beta_auto)
        beta_row.addWidget(self.beta)
        self.admm_max_iter = spin(1, 50, 1)
        self.icgn_tol = dspin(1e-6, 0.5, 6)
        self.icgn_dp_tol = dspin(1e-6, 0.5, 6)
        self.icgn_max_iter = spin(1, 1000, 1)
        self.icgn_patience = spin(0, 100, 1)
        self.local_outlier = dspin(0.0, 20.0, 2)
        self.init_outlier = dspin(0.0, 20.0, 2)
        self.hessian_cond = dspin(1.0, 1e15, 0)
        self.min_valid_ratio = dspin(0.05, 1.0, 2)
        self.checkpoint = QCheckBox()
        self.checkpoint.setChecked(bool(state.write_checkpoints))
        advanced = self._add_section(
            "advanced",
            layout,
            [
                ("subpb2", self.subpb2),
                ("subset_stride", self.subset_stride),
                ("init_coarse", self.init_coarse),
                ("prefilter", self.prefilter),
                ("mu", self.mu),
                ("beta", beta_widget),
                ("admm_max_iter", self.admm_max_iter),
                ("icgn_tol", self.icgn_tol),
                ("icgn_dp_tol", self.icgn_dp_tol),
                ("icgn_max_iter", self.icgn_max_iter),
                ("icgn_patience", self.icgn_patience),
                ("local_outlier", self.local_outlier),
                ("init_outlier", self.init_outlier),
                ("hessian_cond", self.hessian_cond),
                ("min_valid_ratio", self.min_valid_ratio),
            ],
            expanded=False,
        )
        advanced.add_widget(self.checkpoint)
        layout.addStretch(1)

        guard_wheel(self)
        self._connect()
        self._state.params_changed.connect(self.refresh)
        self._state.volumes_changed.connect(self._update_memory)
        self._state.mask_changed.connect(self._update_memory)
        self.retranslate_ui()
        self.refresh()

    # ------------------------------------------------------------------ construction helpers
    def _choice(self, group: str) -> QComboBox:
        combo = QComboBox()
        combo.setMinimumWidth(COMBO_WIDTH)
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        fill_combo(combo, group)
        self.combos[combo] = group
        return combo

    def _add_section(self, key: str, layout, rows, expanded: bool = True) -> CollapsibleSection:
        section = CollapsibleSection(expanded=expanded)
        form = make_form()
        for name, widget in rows:
            label = form_label()
            self.labels[name] = label
            form.addRow(label, widget)
        section.add_layout(form)
        layout.addWidget(section)
        self.sections[key] = section
        return section

    # ------------------------------------------------------------------ binding
    def _connect(self) -> None:
        for i, w in enumerate(self.winsize_axes):
            w.valueChanged.connect(lambda v, i=i: self._on_winsize(i, v))
        self.winsize_lock.toggled.connect(self._on_winsize_lock)
        self.winstepsize.valueChanged.connect(lambda v: self._set("winstepsize", int(v)))
        self.search_radius.valueChanged.connect(lambda v: self._set("search_radius", int(v)))
        self.interp.currentIndexChanged.connect(lambda _i: self._set("interp_method", self.interp.currentData()))
        self.init_method.currentIndexChanged.connect(lambda _i: self._set("init_guess_method", self.init_method.currentData()))
        self.reference_mode.currentIndexChanged.connect(lambda _i: self._set("reference_mode", self.reference_mode.currentData()))
        self.solver.currentIndexChanged.connect(lambda _i: self._set("use_global_step", self.solver.currentData() == "aldvc"))
        self.subpb2.currentIndexChanged.connect(lambda _i: self._set("subpb2_method", self.subpb2.currentData()))
        for w in self.voxel:
            w.valueChanged.connect(lambda _v: self._set("voxel_size", tuple(x.value() for x in self.voxel)))
        self.units.editingFinished.connect(lambda: self._set("units", self.units.text() or "voxel"))
        self.prefilter.valueChanged.connect(lambda v: self._set("prefilter_sigma", float(v)))
        self.gradient_mode.currentIndexChanged.connect(lambda _i: self._set("gradient_mode", self.gradient_mode.currentData()))
        self.n_threads.valueChanged.connect(lambda v: self._set("n_threads", int(v)))
        self.mu.valueChanged.connect(lambda v: self._set("mu", float(v)))
        self.beta_auto.toggled.connect(self._on_beta_auto)
        self.beta.valueChanged.connect(lambda v: None if self.beta_auto.isChecked() else self._set("beta", float(v)))
        self.admm_max_iter.valueChanged.connect(lambda v: self._set("admm_max_iter", int(v)))
        self.icgn_tol.valueChanged.connect(lambda v: self._set("icgn_tol", float(v)))
        self.icgn_dp_tol.valueChanged.connect(lambda v: self._set("icgn_dp_tol", float(v)))
        self.icgn_max_iter.valueChanged.connect(lambda v: self._set("icgn_max_iter", int(v)))
        self.icgn_patience.valueChanged.connect(lambda v: self._set("icgn_patience", int(v)))
        self.subset_stride.valueChanged.connect(lambda v: self._set("subset_stride", int(v)))
        self.init_coarse.valueChanged.connect(lambda v: self._set("init_coarse_factor", int(v)))
        self.backend.currentIndexChanged.connect(lambda _i: self._set("backend", self.backend.currentData()))
        self.local_outlier.valueChanged.connect(lambda v: self._set("local_outlier_threshold", float(v)))
        self.init_outlier.valueChanged.connect(lambda v: self._set("init_outlier_threshold", float(v)))
        self.hessian_cond.valueChanged.connect(lambda v: self._set("hessian_cond_max", float(v)))
        self.min_valid_ratio.valueChanged.connect(lambda v: self._set("min_valid_ratio", float(v)))
        self.checkpoint.toggled.connect(lambda v: setattr(self._state, "write_checkpoints", bool(v)))

    def _set(self, name: str, value: Any) -> None:
        if self._updating or (value is None and name != "beta"):
            return
        try:
            self._state.set_param(name, value)
        except (ValueError, TypeError) as exc:
            self._state.log(f"{name}: {exc}", "warning")
            self.refresh()

    def _on_beta_auto(self, auto: bool) -> None:
        self.beta.setEnabled(not auto)
        self._set("beta", None if auto else float(self.beta.value()))

    def _on_winsize(self, axis: int, value: int) -> None:
        """The spin boxes show the odd span 2h+1; an even value typed by hand rounds up to the next odd one."""
        v = int(value)
        if v % 2 == 0:
            self.winsize_axes[axis].setValue(v + 1)  # re-enters with the odd value
            return
        if self._updating:
            return
        if self.winsize_lock.isChecked():
            for j, w in enumerate(self.winsize_axes):
                if j != axis and w.value() != v:
                    w.blockSignals(True)
                    w.setValue(v)
                    w.blockSignals(False)
        self._set("winsize", tuple(int(w.value()) - 1 for w in self.winsize_axes))

    def _on_winsize_lock(self, locked: bool) -> None:
        """Locking makes the subset cubic again, from the x size."""
        if locked and not self._updating:
            self._on_winsize(0, self.winsize_axes[0].value())

    # ------------------------------------------------------------------ view
    def refresh(self) -> None:
        p = self._state.para
        self._updating = True
        try:
            for w, v in zip(self.winsize_axes, p.winsize):
                w.setValue(int(v) + 1)
            if len(set(int(v) for v in p.winsize)) > 1:
                self.winsize_lock.setChecked(False)  # a non-cubic subset (session, script) unlocks the axes
            self.winstepsize.setValue(int(p.winstepsize[0]))
            self.search_radius.setValue(int(p.search_radius[0]))
            select_key(self.interp, p.interp_method)
            select_key(self.init_method, p.init_guess_method)
            select_key(self.reference_mode, p.reference_mode)
            select_key(self.solver, "aldvc" if p.use_global_step else "local")
            select_key(self.subpb2, p.subpb2_method)
            for w, v in zip(self.voxel, p.voxel_size):
                w.setValue(float(v))
            self.units.setText(p.units)
            self.prefilter.setValue(float(p.prefilter_sigma))
            select_key(self.gradient_mode, p.gradient_mode)
            self.n_threads.setValue(int(p.n_threads))
            self.mu.setValue(float(p.mu))
            self.beta_auto.setChecked(p.beta is None)
            self.beta.setEnabled(p.beta is not None)
            if p.beta is not None:
                self.beta.setValue(float(p.beta))
            self.admm_max_iter.setValue(int(p.admm_max_iter))
            self.icgn_tol.setValue(float(p.icgn_tol))
            self.icgn_dp_tol.setValue(float(p.icgn_dp_tol))
            self.icgn_max_iter.setValue(int(p.icgn_max_iter))
            self.icgn_patience.setValue(int(p.icgn_patience))
            self.subset_stride.setValue(int(p.subset_stride))
            self.init_coarse.setValue(int(p.init_coarse_factor))
            if not select_key(self.backend, p.backend):
                self.backend.setCurrentIndex(0)
            self.local_outlier.setValue(float(p.local_outlier_threshold))
            self.init_outlier.setValue(float(p.init_outlier_threshold))
            self.hessian_cond.setValue(float(p.hessian_cond_max))
            self.min_valid_ratio.setValue(float(p.min_valid_ratio))
            self.checkpoint.setChecked(bool(self._state.write_checkpoints))
        finally:
            self._updating = False
        self._update_memory()

    def _update_memory(self) -> None:
        shape = self._state.volume_shape() if self._state.volumes and self._state.volumes[0].array is not None else None
        if shape is None:
            self._memory.setText(self.tr("Memory estimate: load a volume first"))
            return
        p = self._state.para
        has_mask = any(v.mask_path or v.mask is not None for v in self._state.volumes) or self._state.mask_editor is not None
        voi = self._state.effective_voi()
        box_shape = tuple(int(e) for e in voi.clamp(shape).extent) if voi is not None else shape
        mem = memory_model(box_shape, p.gradient_mode, p.interp_method, has_mask)
        text = self.tr("Memory for a frame pair: {bpv:.0f} bytes/voxel, {gb:.2f} GB").format(
            bpv=mem["bytes_per_voxel"], gb=mem["total_gb"]
        )
        if voi is not None:
            frac = 100.0 * float(box_shape[0] * box_shape[1] * box_shape[2]) / float(shape[0] * shape[1] * shape[2])
            text += "\n" + self.tr("Analysed box from the region of interest: {pct:.0f}% of the volume").format(pct=frac)
        self._memory.setText(text)

    def refresh_backend_status(self) -> None:
        """Describe the compute backend that the automatic choice would pick (CUDA device name or CPU)."""
        from al_dvc.solver.cuda_kernels import cuda_available, device_name, unavailable_reason

        if cuda_available():
            self.backend_status.setText(self.tr("GPU: {name}").format(name=device_name()))
        else:
            self.backend_status.setText(self.tr("CPU only ({reason})").format(reason=unavailable_reason()[:80]))

    def retranslate_ui(self) -> None:
        self.winsize_lock.setText(self.tr("Cube"))
        self.winsize_lock.setToolTip(self.tr("Keep the subset cubic: one size for x, y and z"))
        texts = {
            "winsize": self.tr("Subset size [voxel]"),
            "winstepsize": self.tr("Subset step [voxel]"),
            "search_radius": self.tr("Search range [voxel]"),
            "init_method": self.tr("Initial guess"),
            "interp": self.tr("Interpolation"),
            "reference_mode": self.tr("Tracking mode"),
            "solver": self.tr("Solver"),
            "voxel": self.tr("Voxel size (x, y, z)"),
            "units": self.tr("Length unit"),
            "backend": self.tr("Compute backend"),
            "n_threads": self.tr("CPU threads (0 = all)"),
            "gradient_mode": self.tr("Gradient memory"),
            "subpb2": self.tr("Global step discretisation"),
            "subset_stride": self.tr("Subset sampling stride"),
            "init_coarse": self.tr("Coarse initial-guess lattice"),
            "prefilter": self.tr("Pre-smoothing sigma [voxel]"),
            "mu": self.tr("ADMM penalty mu"),
            "beta": self.tr("Regularisation beta"),
            "admm_max_iter": self.tr("ADMM iterations"),
            "icgn_tol": self.tr("IC-GN gradient tolerance"),
            "icgn_dp_tol": self.tr("IC-GN increment tolerance"),
            "icgn_max_iter": self.tr("IC-GN max iterations"),
            "icgn_patience": self.tr("IC-GN patience"),
            "local_outlier": self.tr("Local outlier threshold"),
            "init_outlier": self.tr("Initial-guess outlier threshold"),
            "hessian_cond": self.tr("Max Hessian condition"),
            "min_valid_ratio": self.tr("Min valid subset fraction"),
        }
        for key, label in self.labels.items():
            label.setText(texts[key])
        tips = {
            "winsize": self.tr(
                "Subset edge along x, y and z in voxels (odd: 2h+1 centred on the node). "
                "Lock it to keep the subset cubic; unlock for a flat or elongated subset, e.g. on anisotropic voxels."
            ),
            "winstepsize": self.tr("Distance between neighbouring nodes in voxels."),
            "search_radius": self.tr("Largest displacement the initial guess can find, in voxels."),
            "init_method": self.tr(
                "Pyramid: coarse-to-fine correlation search (robust to large motion). Single-level: one search at "
                "full resolution. Zero: start from no displacement. Previous frame: reuse the last frame's result."
            ),
            "interp": self.tr(
                "Sub-voxel interpolation of the deformed volume: cubic (default), B-spline (smoother), linear (fastest)."
            ),
            "reference_mode": self.tr(
                "Accumulative: every frame is correlated with the first (total displacement). "
                "Incremental: with the previous frame, then summed (large deformations)."
            ),
            "solver": self.tr(
                "Local DVC: independent subsets. AL-DVC: the subsets are coupled through a global finite-element "
                "step (ADMM), which smooths the field and improves accuracy at the cost of a few extra passes."
            ),
            "voxel": self.tr("Physical size of one voxel along x, y, z; displacements and strains are reported in these units."),
            "units": self.tr("Name of the length unit used in exports and labels (e.g. mm, um)."),
            "backend": self.tr("Automatic: GPU when numba-cuda and an NVIDIA device are present, otherwise CPU."),
            "n_threads": self.tr("CPU threads for the Numba kernels; 0 uses every core."),
            "gradient_mode": self.tr(
                "Precomputed: the three gradient volumes are stored once (fast, 21-25 bytes per voxel). "
                "On the fly: gradients are recomputed for every subset (slower, 9 bytes per voxel), for large scans."
            ),
            "subpb2": self.tr(
                "How the global step of AL-DVC is discretised: finite elements (hexahedral mesh, default) or finite "
                "differences on the node grid. pyALDIC uses finite elements only."
            ),
            "subset_stride": self.tr(
                "Sample every k-th voxel of the subset (k^3 times faster, slightly noisier). 1 = every voxel."
            ),
            "init_coarse": self.tr(
                "Search the initial guess on every k-th node only and interpolate to the others (faster on dense grids)."
            ),
            "prefilter": self.tr("Gaussian smoothing applied to every volume before the analysis (0 = off); reduces noise."),
            "mu": self.tr("Penalty weight of the ADMM coupling between the local subsets and the global field."),
            "beta": self.tr("Weight of the global regularisation; automatic picks it with an L-curve, a value fixes it."),
            "admm_max_iter": self.tr("Maximum number of ADMM passes (local + global) per frame."),
            "icgn_tol": self.tr("Stop the subset iterations when the gradient norm falls below this value."),
            "icgn_dp_tol": self.tr("Stop the subset iterations when the parameter update falls below this value (voxels)."),
            "icgn_max_iter": self.tr("Maximum number of Gauss-Newton iterations per subset."),
            "icgn_patience": self.tr("Iterations without improvement before a subset is declared stalled."),
            "local_outlier": self.tr(
                "Median-test threshold for the subset results; nodes above it are replaced by their neighbours (0 = off)."
            ),
            "init_outlier": self.tr("Median-test threshold for the initial guess (0 = off)."),
            "hessian_cond": self.tr("Subsets whose Hessian is worse conditioned than this are skipped (no texture)."),
            "min_valid_ratio": self.tr("Minimum fraction of a subset inside the region of interest for the node to be solved."),
        }
        for key, tip in tips.items():
            self.labels[key].setToolTip(tip)
        for combo, group in self.combos.items():
            retranslate_combo(combo, group)
        for key, title in {
            "subset": self.tr("Subset & search"),
            "solver": self.tr("Solver"),
            "units": self.tr("Units"),
            "performance": self.tr("Performance"),
            "advanced": self.tr("Advanced"),
        }.items():
            self.sections[key].set_title(title)
        self.beta_auto.setText(self.tr("Auto"))
        self.checkpoint.setText(self.tr("Keep checkpoints (resume interrupted runs)"))
        self.checkpoint.setToolTip(
            self.tr("Write every finished frame to disk so an interrupted run can continue; usually unnecessary.")
        )
        self._update_memory()
