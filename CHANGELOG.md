# Changelog

All notable changes to pyALDVC are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

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
