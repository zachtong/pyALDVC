"""Parameter editor bound to ``AppState.para`` (a ``DVCPara``).

Five folding sections, pyALDIC style (fixed-width labels and inputs, wheel only when focused):
subset & search, solver, strain & units, performance, advanced. The analysed box is not a
parameter: it follows the region of interest drawn on the slices (see ``AppState.effective_voi``).
"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from al_dvc.io.volume_ops import memory_model

from ..app_state import AppState
from ..widgets import COMBO_WIDTH, CollapsibleSection, combo, dspin, form_label, guard_wheel, make_form, spin

INTERP_METHODS = ["cubic", "bspline", "linear"]
INIT_METHODS = ["pyramid", "ncc", "zero", "previous"]
STRAIN_METHODS = ["plane_fit", "fem", "fd", "direct"]
STRAIN_TYPES = ["infinitesimal", "green_lagrange", "euler_almansi", "hencky"]
REFERENCE_MODES = ["accumulative", "incremental"]
SUBPB2_METHODS = ["fem", "fd"]
GRADIENT_MODES = ["stored", "on_the_fly"]
BACKENDS = ["auto", "cuda", "numba"]
VOXEL_FIELD_WIDTH = 72


class ParamPanel(QWidget):
    """Form of the ``DVCPara`` fields, grouped in folding sections."""

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._updating = False
        self.labels: dict[str, QLabel] = {}
        self.sections: dict[str, CollapsibleSection] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ---- subset & search
        self.winsize = spin(5, 257, 2)  # the odd voxel span 2h+1; stored as the even winsize = 2h
        self.winstepsize = spin(1, 128, 1)
        self.search_radius = spin(1, 128, 1)
        self.init_method = combo(INIT_METHODS)
        self.interp = combo(INTERP_METHODS)
        self._add_section(
            "subset",
            layout,
            [
                ("winsize", self.winsize),
                ("winstepsize", self.winstepsize),
                ("search_radius", self.search_radius),
                ("init_method", self.init_method),
                ("interp", self.interp),
            ],
        )

        # ---- solver
        self.reference_mode = combo(REFERENCE_MODES)
        self.use_global = QCheckBox()
        self.subpb2 = combo(SUBPB2_METHODS)
        self._add_section(
            "solver",
            layout,
            [("reference_mode", self.reference_mode), ("use_global", self.use_global), ("subpb2", self.subpb2)],
        )

        # ---- strain & units
        self.strain_method = combo(STRAIN_METHODS)
        self.strain_type = combo(STRAIN_TYPES)
        self.voxel = [dspin(1e-6, 1e6, 4, width=VOXEL_FIELD_WIDTH) for _ in range(3)]
        voxel_widget = QWidget()
        voxel_row = QHBoxLayout(voxel_widget)
        voxel_row.setContentsMargins(0, 0, 0, 0)
        voxel_row.setSpacing(4)
        for w in self.voxel:
            voxel_row.addWidget(w)
        self.units = QLineEdit()
        self.units.setFixedWidth(COMBO_WIDTH)
        self._add_section(
            "strain",
            layout,
            [
                ("strain_method", self.strain_method),
                ("strain_type", self.strain_type),
                ("voxel", voxel_widget),
                ("units", self.units),
            ],
        )

        # ---- performance
        self.backend = combo(BACKENDS)
        self.backend.clear()
        for key in BACKENDS:
            self.backend.addItem(key, key)
        self.backend_status = QLabel()
        self.backend_status.setObjectName("hint")
        self.backend_status.setWordWrap(True)
        self.n_threads = spin(0, 512, 1)
        self.gradient_mode = combo(GRADIENT_MODES)
        self.subset_stride = spin(1, 8, 1)
        self.init_coarse = spin(1, 8, 1)
        self.prefilter = dspin(0.0, 10.0, 2)
        perf = self._add_section(
            "performance",
            layout,
            [
                ("backend", self.backend),
                ("n_threads", self.n_threads),
                ("gradient_mode", self.gradient_mode),
                ("subset_stride", self.subset_stride),
                ("init_coarse", self.init_coarse),
                ("prefilter", self.prefilter),
            ],
        )
        perf.add_widget(self.backend_status)
        self._memory = QLabel()
        self._memory.setObjectName("hint")
        self._memory.setWordWrap(True)
        perf.add_widget(self._memory)

        # ---- advanced (folded)
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
        self.winsize.valueChanged.connect(self._on_winsize)
        self.winstepsize.valueChanged.connect(lambda v: self._set("winstepsize", int(v)))
        self.search_radius.valueChanged.connect(lambda v: self._set("search_radius", int(v)))
        self.interp.currentTextChanged.connect(lambda v: self._set("interp_method", v))
        self.init_method.currentTextChanged.connect(lambda v: self._set("init_guess_method", v))
        self.use_global.toggled.connect(lambda v: self._set("use_global_step", bool(v)))
        self.subpb2.currentTextChanged.connect(lambda v: self._set("subpb2_method", v))
        self.reference_mode.currentTextChanged.connect(lambda v: self._set("reference_mode", v))
        self.strain_method.currentTextChanged.connect(lambda v: self._set("strain_method", v))
        self.strain_type.currentTextChanged.connect(lambda v: self._set("strain_type", v))
        for w in self.voxel:
            w.valueChanged.connect(lambda _v: self._set("voxel_size", tuple(x.value() for x in self.voxel)))
        self.units.editingFinished.connect(lambda: self._set("units", self.units.text() or "voxel"))
        self.prefilter.valueChanged.connect(lambda v: self._set("prefilter_sigma", float(v)))
        self.gradient_mode.currentTextChanged.connect(lambda v: self._set("gradient_mode", v))
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
        self.backend.currentIndexChanged.connect(lambda i: self._set("backend", str(self.backend.itemData(i))))
        self.local_outlier.valueChanged.connect(lambda v: self._set("local_outlier_threshold", float(v)))
        self.init_outlier.valueChanged.connect(lambda v: self._set("init_outlier_threshold", float(v)))
        self.hessian_cond.valueChanged.connect(lambda v: self._set("hessian_cond_max", float(v)))
        self.min_valid_ratio.valueChanged.connect(lambda v: self._set("min_valid_ratio", float(v)))
        self.checkpoint.toggled.connect(lambda v: setattr(self._state, "write_checkpoints", bool(v)))

    def _set(self, name: str, value: Any) -> None:
        if self._updating:
            return
        try:
            self._state.set_param(name, value)
        except (ValueError, TypeError) as exc:
            self._state.log(f"{name}: {exc}", "warning")
            self.refresh()

    def _on_beta_auto(self, auto: bool) -> None:
        self.beta.setEnabled(not auto)
        self._set("beta", None if auto else float(self.beta.value()))

    def _on_winsize(self, value: int) -> None:
        """The spin box shows the odd span 2h+1; an even value typed by hand rounds up to the next odd one."""
        v = int(value)
        if v % 2 == 0:
            self.winsize.setValue(v + 1)  # re-enters with the odd value
            return
        self._set("winsize", v - 1)

    # ------------------------------------------------------------------ view
    def refresh(self) -> None:
        p = self._state.para
        self._updating = True
        try:
            self.winsize.setValue(int(p.winsize[0]) + 1)
            self.winstepsize.setValue(int(p.winstepsize[0]))
            self.search_radius.setValue(int(p.search_radius[0]))
            self.interp.setCurrentText(p.interp_method)
            self.init_method.setCurrentText(p.init_guess_method)
            self.use_global.setChecked(bool(p.use_global_step))
            self.subpb2.setCurrentText(p.subpb2_method)
            self.reference_mode.setCurrentText(p.reference_mode)
            self.strain_method.setCurrentText(p.strain_method)
            self.strain_type.setCurrentText(p.strain_type)
            for w, v in zip(self.voxel, p.voxel_size):
                w.setValue(float(v))
            self.units.setText(p.units)
            self.prefilter.setValue(float(p.prefilter_sigma))
            self.gradient_mode.setCurrentText(p.gradient_mode)
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
            i = self.backend.findData(p.backend)
            self.backend.setCurrentIndex(i if i >= 0 else 0)
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
        """Describe the compute backend that ``backend=auto`` would pick (CUDA device name or CPU)."""
        from al_dvc.solver.cuda_kernels import cuda_available, device_name, unavailable_reason

        if cuda_available():
            self.backend_status.setText(self.tr("GPU: {name}").format(name=device_name()))
        else:
            self.backend_status.setText(self.tr("CPU only ({reason})").format(reason=unavailable_reason()[:80]))

    def retranslate_ui(self) -> None:
        texts = {
            "winsize": self.tr("Subset size (odd) [voxel]"),
            "winstepsize": self.tr("Subset step [voxel]"),
            "search_radius": self.tr("Search range [voxel]"),
            "init_method": self.tr("Initial guess"),
            "interp": self.tr("Interpolation"),
            "reference_mode": self.tr("Tracking mode"),
            "use_global": self.tr("Global step (ADMM)"),
            "subpb2": self.tr("Global discretisation"),
            "strain_method": self.tr("Strain method"),
            "strain_type": self.tr("Strain measure"),
            "voxel": self.tr("Voxel size (x, y, z)"),
            "units": self.tr("Length unit"),
            "backend": self.tr("Compute backend"),
            "n_threads": self.tr("CPU threads (0 = all)"),
            "gradient_mode": self.tr("Gradient storage"),
            "subset_stride": self.tr("Subset sampling stride"),
            "init_coarse": self.tr("Coarse initial-guess lattice"),
            "prefilter": self.tr("Pre-smoothing sigma"),
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
            "winsize": self.tr("Edge of the cubic subset in voxels (odd: 2h+1 centred on the node)."),
            "winstepsize": self.tr("Distance between neighbouring nodes in voxels."),
            "search_radius": self.tr("Largest displacement the initial guess can find, in voxels."),
            "subset_stride": self.tr("Sample every k-th voxel of the subset (k^3 times faster, slightly noisier)."),
            "init_coarse": self.tr("Search the initial guess on every k-th node and interpolate."),
            "backend": self.tr("auto: GPU when numba-cuda and an NVIDIA device are present, otherwise CPU."),
        }
        for key, tip in tips.items():
            self.labels[key].setToolTip(tip)
        for key, title in {
            "subset": self.tr("Subset & search"),
            "solver": self.tr("Solver"),
            "strain": self.tr("Strain & units"),
            "performance": self.tr("Performance"),
            "advanced": self.tr("Advanced"),
        }.items():
            self.sections[key].set_title(title)
        self.beta_auto.setText(self.tr("auto"))
        self.checkpoint.setText(self.tr("Keep checkpoints (resume interrupted runs)"))
        self._update_memory()
