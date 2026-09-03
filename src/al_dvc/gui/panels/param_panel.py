"""Parameter editor bound to ``AppState.para`` (a ``DVCPara``)."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from al_dvc.core.data_structures import VOIRange
from al_dvc.io.volume_ops import memory_model

from ..app_state import AppState

INTERP_METHODS = ["cubic", "bspline", "linear"]
INIT_METHODS = ["pyramid", "ncc", "zero", "previous"]
STRAIN_METHODS = ["plane_fit", "fem", "fd", "direct"]
STRAIN_TYPES = ["infinitesimal", "green_lagrange", "euler_almansi", "hencky"]
REFERENCE_MODES = ["accumulative", "incremental"]
SUBPB2_METHODS = ["fem", "fd"]
GRADIENT_MODES = ["stored", "on_the_fly"]


class ParamPanel(QWidget):
    """Form of the most useful ``DVCPara`` fields plus an advanced section."""

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._updating = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # ---- main form
        self._form = QFormLayout()
        self._form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.winsize = self._spin(8, 256, 2)
        self.winstepsize = self._spin(1, 128, 1)
        self.search_radius = self._spin(1, 128, 1)
        self.interp = self._combo(INTERP_METHODS)
        self.init_method = self._combo(INIT_METHODS)
        self.use_global = QCheckBox()
        self.subpb2 = self._combo(SUBPB2_METHODS)
        self.reference_mode = self._combo(REFERENCE_MODES)
        self.strain_method = self._combo(STRAIN_METHODS)
        self.strain_type = self._combo(STRAIN_TYPES)
        self.voxel = [self._dspin(1e-6, 1e6, 4) for _ in range(3)]
        self.units = QLineEdit()
        self.prefilter = self._dspin(0.0, 10.0, 2)
        self.gradient_mode = self._combo(GRADIENT_MODES)
        self.n_threads = self._spin(0, 512, 1)
        self.labels: dict[str, QLabel] = {}
        voxel_row = QHBoxLayout()
        for w in self.voxel:
            voxel_row.addWidget(w)
        voxel_widget = QWidget()
        voxel_widget.setLayout(voxel_row)
        voxel_row.setContentsMargins(0, 0, 0, 0)
        for key, widget in [
            ("winsize", self.winsize),
            ("winstepsize", self.winstepsize),
            ("search_radius", self.search_radius),
            ("interp", self.interp),
            ("init_method", self.init_method),
            ("use_global", self.use_global),
            ("subpb2", self.subpb2),
            ("reference_mode", self.reference_mode),
            ("strain_method", self.strain_method),
            ("strain_type", self.strain_type),
            ("voxel", voxel_widget),
            ("units", self.units),
            ("prefilter", self.prefilter),
            ("gradient_mode", self.gradient_mode),
            ("n_threads", self.n_threads),
        ]:
            label = QLabel()
            self.labels[key] = label
            self._form.addRow(label, widget)
        layout.addLayout(self._form)

        # ---- VOI
        self._voi_group = QGroupBox()
        voi_layout = QFormLayout(self._voi_group)
        self.voi_whole = QCheckBox()
        self.voi_whole.setChecked(True)
        voi_layout.addRow(self.voi_whole)
        self.voi = {}
        for axis in ("x", "y", "z"):
            lo, hi = self._spin(0, 100000, 1), self._spin(0, 100000, 1)
            row = QHBoxLayout()
            row.addWidget(lo)
            row.addWidget(QLabel("-"))
            row.addWidget(hi)
            w = QWidget()
            w.setLayout(row)
            row.setContentsMargins(0, 0, 0, 0)
            self.voi[axis] = (lo, hi)
            label = QLabel(axis)
            voi_layout.addRow(label, w)
        layout.addWidget(self._voi_group)

        # ---- advanced
        self._adv_group = QGroupBox()
        self._adv_group.setCheckable(True)
        self._adv_group.setChecked(False)
        adv = QFormLayout(self._adv_group)
        self.mu = self._dspin(1e-8, 1e3, 6)
        self.beta_auto = QCheckBox()
        self.beta = self._dspin(1e-8, 1e6, 6)
        self.admm_max_iter = self._spin(1, 50, 1)
        self.icgn_tol = self._dspin(1e-6, 0.5, 6)
        self.icgn_dp_tol = self._dspin(1e-6, 0.5, 6)
        self.icgn_max_iter = self._spin(1, 1000, 1)
        self.icgn_patience = self._spin(0, 100, 1)
        self.subset_stride = self._spin(1, 8, 1)
        self.init_coarse = self._spin(1, 8, 1)
        self.local_outlier = self._dspin(0.0, 20.0, 2)
        self.init_outlier = self._dspin(0.0, 20.0, 2)
        self.hessian_cond = self._dspin(1.0, 1e15, 0)
        self.min_valid_ratio = self._dspin(0.05, 1.0, 2)
        beta_row = QHBoxLayout()
        beta_row.addWidget(self.beta_auto)
        beta_row.addWidget(self.beta)
        beta_widget = QWidget()
        beta_widget.setLayout(beta_row)
        beta_row.setContentsMargins(0, 0, 0, 0)
        for key, widget in [
            ("mu", self.mu),
            ("beta", beta_widget),
            ("admm_max_iter", self.admm_max_iter),
            ("icgn_tol", self.icgn_tol),
            ("icgn_dp_tol", self.icgn_dp_tol),
            ("icgn_max_iter", self.icgn_max_iter),
            ("icgn_patience", self.icgn_patience),
            ("subset_stride", self.subset_stride),
            ("init_coarse", self.init_coarse),
            ("local_outlier", self.local_outlier),
            ("init_outlier", self.init_outlier),
            ("hessian_cond", self.hessian_cond),
            ("min_valid_ratio", self.min_valid_ratio),
        ]:
            label = QLabel()
            self.labels[key] = label
            adv.addRow(label, widget)
        layout.addWidget(self._adv_group)

        # ---- output
        out_row = QHBoxLayout()
        self.output_dir = QLineEdit()
        self._btn_output = QPushButton()
        out_row.addWidget(self.output_dir)
        out_row.addWidget(self._btn_output)
        self._output_label = QLabel()
        layout.addWidget(self._output_label)
        layout.addLayout(out_row)
        self.checkpoint = QCheckBox()
        self.checkpoint.setChecked(state.write_checkpoints)
        self.checkpoint.toggled.connect(lambda v: setattr(self._state, "write_checkpoints", bool(v)))
        layout.addWidget(self.checkpoint)
        self._memory = QLabel()
        self._memory.setObjectName("hint")
        self._memory.setWordWrap(True)
        layout.addWidget(self._memory)

        self._connect()
        self._state.params_changed.connect(self.refresh)
        self._state.volumes_changed.connect(self._update_memory)
        self._state.output_dir_changed.connect(lambda p: self.output_dir.setText(p))
        self.retranslate_ui()
        self.refresh()

    # ------------------------------------------------------------------ widgets
    @staticmethod
    def _spin(lo: int, hi: int, step: int) -> QSpinBox:
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        return s

    @staticmethod
    def _dspin(lo: float, hi: float, decimals: int) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setSingleStep(10 ** (-decimals) if decimals else 1.0)
        return s

    @staticmethod
    def _combo(items: list[str]) -> QComboBox:
        c = QComboBox()
        c.addItems(items)
        return c

    # ------------------------------------------------------------------ binding
    def _connect(self) -> None:
        self.winsize.valueChanged.connect(lambda v: self._set("winsize", int(v)))
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
        self.local_outlier.valueChanged.connect(lambda v: self._set("local_outlier_threshold", float(v)))
        self.init_outlier.valueChanged.connect(lambda v: self._set("init_outlier_threshold", float(v)))
        self.hessian_cond.valueChanged.connect(lambda v: self._set("hessian_cond_max", float(v)))
        self.min_valid_ratio.valueChanged.connect(lambda v: self._set("min_valid_ratio", float(v)))
        self.voi_whole.toggled.connect(self._on_voi_changed)
        for lo, hi in self.voi.values():
            lo.valueChanged.connect(self._on_voi_changed)
            hi.valueChanged.connect(self._on_voi_changed)
        self.output_dir.editingFinished.connect(lambda: self._state.set_output_dir(self.output_dir.text() or "aldvc_results"))
        self._btn_output.clicked.connect(self._on_choose_output)

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

    def _on_voi_changed(self, *_args) -> None:
        if self._updating:
            return
        whole = self.voi_whole.isChecked()
        for lo, hi in self.voi.values():
            lo.setEnabled(not whole)
            hi.setEnabled(not whole)
        if whole:
            self._set("voi", None)
        else:
            self._set("voi", VOIRange(**{ax: (int(lo.value()), int(hi.value())) for ax, (lo, hi) in self.voi.items()}))

    def _on_choose_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, self.tr("Output folder"), self.output_dir.text())
        if folder:
            self._state.set_output_dir(folder)

    # ------------------------------------------------------------------ view
    def refresh(self) -> None:
        p = self._state.para
        self._updating = True
        try:
            self.winsize.setValue(int(p.winsize[0]))
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
            self.local_outlier.setValue(float(p.local_outlier_threshold))
            self.init_outlier.setValue(float(p.init_outlier_threshold))
            self.hessian_cond.setValue(float(p.hessian_cond_max))
            self.min_valid_ratio.setValue(float(p.min_valid_ratio))
            whole = p.voi is None
            self.voi_whole.setChecked(whole)
            for ax, (lo, hi) in self.voi.items():
                lo.setEnabled(not whole)
                hi.setEnabled(not whole)
                if p.voi is not None:
                    rng = getattr(p.voi, ax)
                    lo.setValue(int(rng[0]))
                    hi.setValue(int(rng[1]))
            self.output_dir.setText(str(self._state.output_dir))
        finally:
            self._updating = False
        self._update_memory()

    def _update_memory(self) -> None:
        shape = self._state.volume_shape() if self._state.volumes and self._state.volumes[0].array is not None else None
        if shape is None:
            self._memory.setText(self.tr("Memory estimate: load a volume first"))
            return
        p = self._state.para
        mem = memory_model(
            shape, p.gradient_mode, p.interp_method, any(v.mask_path or v.mask is not None for v in self._state.volumes)
        )
        self._memory.setText(
            self.tr("Resident memory for a frame pair: {bpv:.0f} bytes/voxel, {gb:.2f} GB").format(
                bpv=mem["bytes_per_voxel"], gb=mem["total_gb"]
            )
        )

    def retranslate_ui(self) -> None:
        texts = {
            "winsize": self.tr("Subset size [voxel]"),
            "winstepsize": self.tr("Node spacing [voxel]"),
            "search_radius": self.tr("Search radius [voxel]"),
            "interp": self.tr("Interpolation"),
            "init_method": self.tr("Initial guess"),
            "use_global": self.tr("Global step (AL-DVC)"),
            "subpb2": self.tr("Global discretisation"),
            "reference_mode": self.tr("Reference mode"),
            "strain_method": self.tr("Strain method"),
            "strain_type": self.tr("Strain measure"),
            "voxel": self.tr("Voxel size (x, y, z)"),
            "units": self.tr("Units"),
            "prefilter": self.tr("Pre-smoothing sigma"),
            "gradient_mode": self.tr("Gradient storage"),
            "n_threads": self.tr("Threads (0 = all)"),
            "mu": self.tr("mu"),
            "beta": self.tr("beta (auto / value)"),
            "admm_max_iter": self.tr("ADMM iterations"),
            "icgn_tol": self.tr("IC-GN gradient tolerance"),
            "icgn_dp_tol": self.tr("IC-GN increment tolerance"),
            "icgn_max_iter": self.tr("IC-GN max iterations"),
            "icgn_patience": self.tr("IC-GN patience"),
            "subset_stride": self.tr("Subset sampling stride"),
            "init_coarse": self.tr("Coarse initial-guess lattice"),
            "local_outlier": self.tr("Local outlier threshold"),
            "init_outlier": self.tr("Initial-guess outlier threshold"),
            "hessian_cond": self.tr("Max Hessian condition"),
            "min_valid_ratio": self.tr("Min valid subset fraction"),
        }
        for key, label in self.labels.items():
            label.setText(texts[key])
        self._voi_group.setTitle(self.tr("Volume of interest"))
        self.voi_whole.setText(self.tr("Whole volume"))
        self._adv_group.setTitle(self.tr("Advanced"))
        self.beta_auto.setText(self.tr("auto"))
        self._output_label.setText(self.tr("Output folder"))
        self._btn_output.setText(self.tr("Browse..."))
        self.checkpoint.setText(self.tr("Write checkpoints (resume interrupted runs)"))
        self._update_memory()
