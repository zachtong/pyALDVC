# Changelog

All notable changes to pyALDVC are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- 3-D view tab in the GUI (`al_dvc.gui.view3d_scene`, `panels/view3d.py`,
  extra `gui3d` = pyvista + pyvistaqt): field slices, node points, iso-surface,
  warped lattice, displacement arrows, volume slices, camera presets and PNG
  screenshots; interactive pyvistaqt widget with an off-screen fallback;
  `scripts/make_view3d_report.py` (`reports/view3d.pdf`).

## [0.3.0] - 2026-09-03

"Usable without code": a standalone graphical application and a portable
Windows bundle that needs no Python installation.

### Added
- Graphical application `al-dvc-gui` (`al-dvc gui`, `pip install al-dvc[gui]`):
  PySide6 window with volume/mask list, parameter form (memory estimate,
  VOI, advanced ADMM/IC-GN settings), background pipeline worker with progress,
  stop and log, three-plane slice viewer with displacement / uncertainty /
  strain overlays, result summary, exports (npz, mat, csv, vti, PDF), session
  files (`.aldvc`), English / Simplified Chinese, background kernel warm-up,
  self-test; offscreen tests and `scripts/make_gui_report.py` (`reports/gui.pdf`).
- Portable Windows bundle: `packaging/pyaldvc.spec` + `tools/build_exe.py`
  (PyInstaller onedir, `pyALDVC.exe` and `pyALDVC-console.exe --self-test`),
  `tests/test_frozen_bundle.py` driving the built executable, and
  `.github/workflows/build-exe.yml` attaching `pyALDVC-<version>-win64.zip`
  to every `v*` release.

## [0.2.0] - 2026-09-03

"Real-scan ready": validated against the MATLAB code on a micro-CT scan,
with masks on the deformed frame, per-node uncertainty, checkpoints and a
large-volume mode.

### Added
- Large-volume mode `gradient_mode="on_the_fly"`: the kernels evaluate the
  7-point stencil on the reference at the subset voxels instead of reading three
  stored gradient volumes; resident memory drops from 21 to 9 bytes per voxel
  (a 1500^3 scan fits in 32 GB) for about 15-20 % more local-step time. The
  pipeline logs the memory model (`memory_model`) at start;
  `scripts/make_large_volume_report.py` measures both modes.
- Per-frame checkpoints: `run_aldvc(..., checkpoint_dir=DIR)` writes one
  `frame_<k>.npz` per finished frame pair (plus `meta.json`) and reuses them on
  a later call; a directory written with other parameters, volumes, schedule
  or grid is rejected (`CheckpointMismatch`) unless `resume=False`. CLI:
  `al-dvc run --checkpoint DIR [--restart]`. `scripts/make_checkpoint_report.py`.
- Deformed-frame masks: a frame's mask now also applies when the frame is the
  deformed one. Masked voxels are NaN in the sampled volume, subset voxels whose
  interpolation stencil touches them drop out of the node's correlation (the
  subset statistics are recomputed on the remaining voxels), nodes that keep less
  than half their voxels are reported `invalid_subset`, and the NCC search treats
  masked voxels as featureless. `scripts/make_mask_report.py` shows the effect
  (`reports/deformed_mask.pdf`).
- `FrameResult.U_std`: per-node standard deviation of u, v, w from the IC-GN
  normal equations (noise-corrected Hessian, see `al_dvc.solver.uncertainty`),
  exported as `disp_std_u/v/w`, `disp_std` (npz `U_std`, vti `displacement_std`,
  mat `ResultDispStd`) and shown in the PDF report; `scripts/make_uncertainty_report.py`
  calibrates it against synthetic noise (`reports/uncertainty.pdf`).
- `al_dvc.io.matlab_results`: reader for the MATLAB ALDVC `results_ws*_st*.mat`
  files (0-based coordinates, `(N, 3)` / `(N, 3, 3)` layouts) and node matching.
- `scripts/compare_matlab.py`: node-wise cross-validation against the MATLAB
  results shipped with the reference code, with a solver-equivalence check
  (both codes' local solutions refined by the same kernel) and a ZNCC
  objective comparison; writes `reports/matlab_crossval_<tag>.pdf`.
  On the micro-CT example both codes' local solutions coincide to 0.001
  voxel once refined by the same kernel and the final fields agree to
  0.005 / 0.006 / 0.02 voxel (median, u / v / w); on the diverged `eyes`
  example pyALDVC reports the failure through status codes instead of
  returning an 86-voxel field.
- `icgn_dp_tol`: separate IC-GN parameter-increment tolerance (default 1e-3
  voxel); `icgn_tol` keeps the MATLAB relative gradient-norm meaning.
- `icgn_patience` and status code `stalled` (7): IC-GN gives up on a node after
  five iterations without objective improvement instead of running to the
  100-iteration cap; textureless regions no longer dominate the run time.
- The IC-GN kernels walk the active nodes in a block-cyclic order, so spatial
  clusters of skipped or hard nodes (masks, inpainted nodes, textureless
  layers, node subsets) no longer leave most threads idle (79k-node scan:
  local step 115 -> ~350 nodes/s together with the stall rule).
- Numba kernels for volume normalisation and the 7-point gradient;
  `compute_gradients_np` and `voi_mean_std` expose the NumPy reference and
  the VOI statistics.

### Changed
- The automatic `beta` selection uses the MATLAB L-curve score
  `|u-u_hat| + h^2 |F-grad u_hat|` by default (`beta_criterion="matlab"`); the
  previous z-normalised score remains available as `"normalized"`.
- IC-GN stops on the increment criterion at 1e-3 voxel instead of 1e-2. On
  real CT data with weak z-texture the looser value left a 0.03-0.05 voxel
  unconverged residual in `w`; the cost is about twice the local iterations.
- Pre-processing of a 1024x1024x306 scan (321 M voxels) drops from about
  30 s to 1.1 s (parallel Numba normalisation 0.2 s and gradients 0.9 s; the
  SciPy gradient alone took 9 s).

## [0.1.0] - 2026-09-02

Initial release: a complete Python port of the MATLAB ALDVC pipeline with
the pyALDIC architecture.

### Added
- `DVCPara` parameter set with validation, scalar-to-(x,y,z) broadcasting,
  JSON/YAML round trip; no interactive prompts anywhere.
- Volume I/O: TIFF stacks, slice folders, MATLAB `.mat` (v5/v7.3 with axis
  permutation), NumPy; streaming `FileVolumeProvider` with a bounded cache.
- Uniform hex8 node grid with VOI/mask trimming and subset-coverage tests.
- Numba kernels: tricubic (Keys), cubic B-spline and trilinear sampling;
  12-DOF and 3-DOF IC-GN with in-place subset reads, per-node Cholesky
  factors and status codes; NumPy reference implementations for testing.
- Initial guess: Hann-windowed phase-correlation global shift, texture-aware
  coarse-to-fine NCC pyramid with a Numba spatial-domain ZNCC kernel (FFT
  engine for large search windows), node-wise search-radius expansion,
  sub-voxel quadratic peaks, PCE quality factor, universal median test and
  harmonic (spring) inpainting.
- Global step: FEM (hex8, 2x2x2 Gauss) and finite-difference operator sets
  assembled once per mesh; Jacobi-PCG multi-RHS solver (direct LU for small
  meshes); lumped-mass nodal gradient; MATLAB-compatible L-curve `beta`
  auto-tuning; scaled ADMM with `accumulate` or `reset` dual updates.
- Strain: masked 3D Savitzky-Golay plane fit, finite differences, FEM nodal
  gradient, direct ADMM gradient; four strain measures and derived
  quantities in physical units; edge-trim validity flags.
- Multi-frame tracking with `FrameSchedule` and cubic cumulative composition.
- Exports: `.npz`, `.mat` (Python and MATLAB layouts), CSV, VTK `.vti` +
  `.pvd`, PDF report, parameter/summary JSON/YAML.
- CLI `al-dvc run|synth|info|plot`; synthetic data generator with exact
  Lagrangian warps; validation and benchmark scripts producing PDF reports.
- 110 pytest tests (kernel-vs-reference, operators, search, strain, full
  pipeline against analytic ground truth, exports, CLI).
