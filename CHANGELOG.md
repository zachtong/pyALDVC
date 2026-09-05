# Changelog

All notable changes to pyALDVC are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- Starting a second GUI run with checkpoints enabled failed with
  `CheckpointMismatch`; the GUI now uses `resume="auto"`.
- Unchecking "Whole volume" left a 0..0 VOI and the run failed inside the
  worker; parameters are validated before a run starts (readable message).
- numba-cuda driver messages (`cuMemFree` at INFO) and occupancy warnings no
  longer flood the log; the CUDA probe is serialised across threads and the
  run log names the compute backend (`Compute backend: cuda (...)`).

### Changed
- Local IC-GN kernels are about 3x faster: the tricubic sampler allocated three
  `np.empty(4)` weight arrays per sampled voxel (a heap allocation each in
  Numba, 137 -> 52 ns per sample with scalar weights), the ZNCC numerator is
  accumulated in the gradient pass, and per-voxel divisions became
  multiplications. Results unchanged to 3e-14 with identical iteration counts;
  256^3 / subset 32 / step 8: local step 14.9 -> 4.6 s, ADMM local steps 5.6 ->
  1.6 s; the 1024x1024x306 micro-CT example (79,200 nodes) runs in 3.6 min
  (3.2 min with `init_coarse_factor=2`) instead of 12.5 min with unchanged
  agreement to the MATLAB code.
- The NCC pyramid refines the finer levels with radius 2 instead of 4
  (`pyramid_fine_radius`, auto-expand still covers clipped peaks): initial
  guess 4.7 -> 2.2 s at 19,683 nodes with the same error.
- `ListVolumeProvider` normalises frames on demand (LRU of three) instead of
  holding a float32 copy of every frame of a sequence.

### Added
- The 2 x 2 arrangement (XY / XZ left, YZ top-right) is the default layout of
  the slice viewer, the strain window and the exported slice images.
- Region of interest: an icon toolbar (vector icons rendered from inline SVG in
  the theme colours) replaces the combos and the four rows of text buttons;
  shape tools are toggles, modes are Replace / Add / Cut, the edit actions are
  one row of icons. "Automatic mask" segments the material in one click (Otsu
  threshold, holes filled, largest connected component) as a replayable
  `threshold` operation that sessions restore. `MaskOp` gained the `replace`
  mode.
- 3-D view: three slice sliders under the view (shared with the Slices tab);
  the "Warped grid" mode is now "Deformed lattice": only cells whose nodes are
  valid are warped and they are drawn with their edges, so a region of interest
  no longer produces an empty or two-faced picture.
### Added
- Languages: Traditional Chinese, Japanese, German, French and Spanish join
  English and Simplified Chinese (`View > Language`; the system locale picks
  the closest shipped language, e.g. `de_AT` -> German, `zh_HK` -> Traditional
  Chinese). `al_dvc.gui.i18n_tools` extracts the `tr()` strings from the code
  and audits every table; `tools/i18n_extract.py` reports coverage, lists
  missing strings or adds them to a table; a test keeps every shipped language
  complete.
- Strain post-processing window (`Analysis > Strain post-processing...`,
  Ctrl+T, or the button of the results panel): strain method, measure,
  plane-fit window and smoothing chosen after the run, computed on a worker
  thread with cancel, shown on a private three-plane canvas with its own frame
  navigation, colour range and layout; the result is written back so the main
  viewer and the exports see it. The GUI run no longer computes strain inline
  (`compute_strain=False`), like pyALDIC.
- Export dialog (`Analysis > Export results...`, Ctrl+E): destination and base
  name, formats (npz, mat, CSV, ParaView, PDF report, slice images), field and
  frame selection, image layout / colormap / DPI, progress on a worker thread,
  "Open folder". `al_dvc.export.slice_plots` draws the three planes for the
  canvases and the PNG export alike.
- Menu shortcuts: F5 run, Esc stop, Ctrl+N / Ctrl+O / Ctrl+S / Ctrl+Shift+S
  sessions.
- "Same scale" option for the three planes (slice viewer, strain window,
  image export): one voxels-per-pixel scale for XY, XZ and YZ, each pane shrunk
  to its slice and centred in its cell (`slice_plots.apply_equal_scale`);
  remembered in the session.
- Volume formats: HDF5 (`.h5` / `.hdf5`, first 3-D dataset or `mat_key`, also
  written by `save_volume`), NIfTI (`.nii`, `.nii.gz`; needs `nibabel`), NRRD
  (needs `pynrrd`), DICOM folders (needs `pydicom`, stacked by InstanceNumber)
  with a clear message naming the missing optional package; colour slices are
  converted to luminance instead of keeping the red channel; folder resolution
  and the file dialog know the new extensions.
- Left column: section titles stay pinned at the top while scrolling (stacked,
  click to jump back); the volume table shows a thumbnail of the middle slice;
  the batch dialog uses the same groups, primary button and console style as
  the main window.
- Volumes panel: a table (frame, name, shape, region) showing which frame
  carries the region of interest or its own mask, frame reordering (Up / Down,
  context menu), drag and drop of files or folders, a placeholder in the empty
  list, a hint line telling whether a region of interest crops the analysis.
- The mask tools moved from the canvas toolbar into a "Region of interest"
  section of the left column (pyALDIC's sidebar layout); the window fits
  1200 x 700.
- Window layout is remembered between sessions (geometry and column widths,
  QSettings); `View` menu toggles the data and results columns (Ctrl+1 / Ctrl+2)
  and resets the layout; minimum window size 1100 x 680. Canvas fonts follow
  the theme. Field lists show readable names (u, v, w, |u|, exx, von Mises...)
  with frame previous / next buttons; the run status shows the elapsed time and
  an estimate of the time left. Tooltips on tracking mode, global step,
  gradient storage, initial guess, interpolation and threads.
- `reports/postprocessing.pdf` (`scripts/make_postprocessing_report.py`):
  strain window and export dialog screenshots, strain timings per method,
  export timings.

### Changed
- Initial guess: only nodes with a usable reference subset are correlated
  (the others are inpainted), the coarsest pyramid level searches the
  requested radius scaled to its voxels instead of the full radius in coarse
  voxels, the sub-voxel peak neighbourhoods are gathered without a Python
  loop, and on CUDA the direct kernel is used for any offset count when the
  template fits in shared memory (no FFT fallback). Micro-CT example at step 8
  with a partial mask on the RTX 5090: initial guess 95 s -> about 40 s.
- Slice viewer: the three slices can be arranged as a row, a column or a
  2 x 2 grid (XY / XZ left, YZ top-right; remembered in the session); the
  colorbar has its own axes, so changing a slice no longer shrinks the images.
  Displacement fields are NaN outside the valid nodes like strain (the
  inpainted values outside the region of interest are not shown or exported
  with `trimmed=True`). Default colormap `turbo`. The Slices / 3-D view
  switch is a prominent segmented control; the mask toolbar's target reads
  "Mask for: This frame / All frames" with an explanation.
- 3-D view: controls follow the mode (slice positions shared with the Slices
  tab, iso level, warp scale; arrow settings only with arrows), a background
  selector (dark / black / grey / white) with contrast-aware text, and a slim
  centred scalar bar in a plain sans-serif font.
- GUI layout after pyALDIC: run controls, results, exports and the console on
  the right, folding parameter sections with fixed-width inputs on the left
  (`Subset & search`, `Solver`, `Strain & units`, `Performance`, `Advanced`),
  inputs that react to the mouse wheel only when focused, the subset size shown
  as the odd voxel span. Results stay in memory and exports ask for their
  destination; the output folder and the VOI spin boxes are gone from the
  panel: the analysed box follows the region of interest drawn on the slices
  (`voi_from_mask`). Checkpoints are an advanced option, off by default.
- Two install flavours only: `pip install al-dvc` is the complete CPU
  application (PySide6, pyvista and pyvistaqt are regular dependencies now),
  `pip install al-dvc[gpu]` adds the CUDA backend. The `gui`, `gui3d`, `viz`
  and `dev` extras are gone.

### Added
- CUDA backend (`al_dvc.solver.cuda_kernels`, extra `gpu` = `numba-cuda[cu12]`):
  the Hessian precompute, the 12-DOF IC-GN and the 3-DOF ADMM kernels as
  numba-cuda kernels, one thread block per node, float32 sampling and
  reductions with float64 solves, masks / NaN voxels / stride / noise
  correction / look-ahead stop identical to the CPU kernels (same statuses and
  iteration counts, displacements within ~1e-5 voxel). `backend="auto"` (new
  default) uses the GPU when numba-cuda and a CUDA device are present and
  falls back to the CPU kernels otherwise; `cuda` / `numba` / `numpy` force a
  backend; the GUI has a backend selector with the detected device. RTX 5090
  vs 24-core CPU: 12-DOF kernel 28x, 3-DOF 39x; the micro-CT example runs in
  23 s instead of 190 s with the same agreement to MATLAB; `reports/gpu.pdf`
  (`scripts/make_gpu_report.py`), `tests/test_cuda_backend.py` (skipped without
  a GPU). The portable Windows bundle stays CPU-only.
- `icgn_predictive_stop` (default on): the IC-GN kernels apply the current step
  and stop when the steps contract by at least 2x and the predicted next step
  `dp_k^2 / dp_{k-1}` is below `icgn_dp_tol`, instead of spending one more
  sampling pass to confirm convergence: 13 % (12-DOF) and 31 % (3-DOF) fewer
  iterations on smooth synthetic fields, 3-DOF ADMM passes on the micro-CT
  example 4.1 / 4.0 / 3.9 -> 3.6 / 3.4 / 3.3 iterations, same solution within
  `icgn_dp_tol`. `tests/test_predictive_stop.py`.
- `init_coarse_factor` (`al_dvc.solver.coarse_init`): the NCC pyramid and a
  12-DOF IC-GN run on every k-th node per axis; displacement and gradient are
  interpolated trilinearly to all nodes as the initial guess of the full pass
  (pyALDIC's seed-propagation idea without the sequential wave). Also a GUI
  advanced parameter; `tests/test_coarse_init.py`.
- `icgn_noise_hessian` (default on): the IC-GN kernels subtract the expected
  reference-gradient noise inflation `c s^2 (I3 (x) M)` from the stored Hessian
  once a node's step is below half a voxel (`s^2` from the current ZNCC, the
  model of `uncertainty.py`), capped at half of the translation diagonal. The
  fixed point is unchanged; noisy synthetic data converge in 2x fewer iterations
  (SNR ~ 5: 16 -> 8 iterations per node), clean data are untouched, and the
  ADMM local passes on the micro-CT example need 4 instead of 7 iterations per
  node with the same agreement to MATLAB. `tests/test_noise_hessian.py`.
- `subset_stride`: sample every k-th subset voxel per axis (k^3 fewer samples
  per IC-GN iteration; the Hessian, the statistics and the uncertainty model
  use the sampled set); 4.7x faster local steps at k = 2 with subset 32.
  Also in the GUI's advanced parameters.
- `scripts/make_optimization_report.py` (`reports/optimization.pdf`): before /
  after stage timings, thread scaling, stride trade-off, initial-guess variants
  and the rejected experiments (fastmath on the search kernel, FFT correlation
  at the fine pyramid level, a trilinear-start IC-GN, skipping the finest
  pyramid level).

## [0.3.1] - 2026-09-03

GUI follow-ups: a 3-D view, mask drawing on the slices, batch runs.

### Added
- 3-D view tab in the GUI (`al_dvc.gui.view3d_scene`, `panels/view3d.py`,
  pyvista + pyvistaqt): field slices, node points, iso-surface,
  warped lattice, displacement arrows, volume slices, camera presets and PNG
  screenshots; interactive pyvistaqt widget with an off-screen fallback;
  `scripts/make_view3d_report.py` (`reports/view3d.pdf`).
- Mask drawing on the slice viewer (`al_dvc.gui.mask_editor`,
  `panels/mask_tools.py`): rectangle, ellipse, polygon and brush on any of the
  three slices, extruded through all slices / the current slice / a range, add
  or cut, invert / fill / clear, undo / redo, apply to the current or all
  frames, save as a mask volume; sessions store the drawing operations;
  `scripts/make_mask_tools_report.py` (`reports/mask_tools.pdf`).
- Batch runs: `al_dvc.gui.batch` (`run_session_file`, `BatchRunner`), the
  `File > Batch run...` dialog (job table, progress, log, stop, open a finished
  session) and the CLI `al-dvc batch a.aldvc b.aldvc --export npz summary`;
  `scripts/make_batch_report.py` (`reports/batch.pdf`).

### Fixed
- Windows bundle: the VTK modules pyvista loads lazily are collected by a
  build-time probe (a static analysis found 19 of them and the frozen 3-D view
  reported pyvista as missing); the self-test names the import failure.

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
