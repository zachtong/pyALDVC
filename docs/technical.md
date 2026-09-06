# pyALDVC technical notes

The algorithm, the accuracy and throughput figures and the comparison with the MATLAB code.
The application, its parameters, the command line and the Python API are described in the
[user guide](user_guide.md); the conventions and design decisions are in [`design.md`](design.md).

## How it works

For a reference volume `f` and a deformed volume `g` the pipeline solves,
on a regular grid of nodes:

1. **Initial guess** -- global rigid shift by phase correlation, then a
   coarse-to-fine normalised cross-correlation pyramid per node
   (sub-voxel peak fit, quality factors, outlier removal, inpainting).
2. **Local step (subproblem 1)** -- at every node an affine subset warp
   (9 gradient + 3 translation parameters) is fitted by IC-GN on the ZNSSD
   criterion. Later ADMM passes hold the gradient at the global estimate and
   fit only the 3 translations with a proximal penalty.
3. **Global step (subproblem 2)** -- a kinematically compatible field
   `u_hat` minimises `beta/2 |grad(u_hat) - (F - W)|^2 + mu/2 |u_hat - (u - v)|^2`
   over the hex8 FEM (or finite-difference) discretisation. `beta` is tuned
   automatically by an L-curve sweep on the first frame of each reference.
4. **ADMM iterations** alternate 2 and 3 with scaled dual updates until the
   RMS displacement update drops below `admm_tol` (typically 2-4 steps).
5. **Strain** from the cumulative displacement in physical units.

Every convention (axis order, node ordering, DOF layout, warp parameters)
is fixed in [`docs/design.md`](docs/design.md), which also records the
design decisions and the differences from the MATLAB code.




## Accuracy and performance

Synthetic ground-truth validation (`scripts/make_validation_report.py`,
speckle volumes with exact Lagrangian warps) and the throughput benchmark
(`scripts/benchmark_performance.py`) regenerate the PDFs in `reports/`.
Numbers from a 112x120x128 speckle volume (feature size ~4 voxels), subset
16, step 8, interior nodes:

| case | displacement RMSE (voxel) | gradient RMSE (local / AL-DVC) |
|---|---|---|
| translation (3.4, -2.6, 1.2) | 0.003 - 0.006 | 1.6e-4 / 0.5e-4 |
| affine 2 % strain | 0.004 | 3.0e-4 / 2.3e-4 |
| rotation 5 deg, large motion 12 voxels | 0.001 - 0.006 | 2e-4 / 1e-4 |
| affine 2 % with `interp_method="bspline"` | 0.0002 | 0.5e-4 / 0.1e-4 |
| affine 2 %, noise sd 0.01 (SNR ~ 6) | 0.012 (0.011 with `prefilter_sigma=0.8`) | 0.8e-3 |
| affine 2 %, noise sd 0.03 (SNR ~ 2) | 0.045 (0.032 with pre-smoothing) | 2.8e-3 |
| cylinder mask, 3-frame accumulative / incremental | 0.004 / 0.009 | -- |
| sinusoid, amplitude 1 voxel, wavelength 80 (subset 16) | 0.05 (first-order subset bias) | -- |

Throughput on a 24-core workstation (Intel Core Ultra 9 285K, Numba warm, ADMM with
2-4 steps, increment tolerance 1e-3 voxel):

| volume | subset / step | nodes | total | of which local IC-GN |
|---|---|---|---|---|
| 128^3 | 16 / 8 | 2 197 | 0.6 s | 0.1 s |
| 256^3 | 32 / 16 | 2 744 | 1.9 s | 0.6 s |
| 256^3 | 32 / 8 | 19 683 | 9.0 s (3.8 s with `subset_stride=2`) | 4.2 s (0.9 s) |
| 256^3 | 48 / 16 | 2 197 | 3.1 s | 1.5 s |
| 384^3 | 32 / 16 | 10 648 | 6.7 s | 2.8 s |
| 1024 x 1024 x 306 micro-CT (MATLAB example) | 32 / 8 | 79 200 | CPU 3.6 min, 3.2 min with `init_coarse_factor=2` (12.5 min before the 0.3.2 work); **RTX 5090: 23 s** | CPU 1.2 min + 1.5 min for three 3-DOF passes, initial guess 0.9 min (0.3 min coarse); GPU 2.6 s + 6.2 s, initial guess 9.9 s |

See `reports/validation_synthetic.pdf` and `reports/performance.pdf` for the
complete tables, noise sweep, spatial-resolution study and stage timings, and
`reports/optimization.pdf` for the optimisation study (before / after, thread
scaling, the stride trade-off with and without noise, rejected experiments).

### Agreement with the MATLAB ALDVC code

`scripts/compare_matlab.py` reruns the micro-CT example shipped with the
MATLAB code (`20190504_cut`, 1024x1024x306, 79,200 nodes at subset 32 / step
8) on the MATLAB node positions and compares node by node
(`reports/matlab_crossval_ws32_st8.pdf`). On the interior nodes converged in
both codes:

| quantity | median difference (u, v, w) [voxel] |
|---|---|
| local IC-GN solution | 0.0044, 0.0049, 0.041 |
| final AL-DVC displacement | 0.0048, 0.0055, 0.020 |
| both local solutions refined by the same kernel | 0.0001, 0.0001, 0.0012 |

The two IC-GN implementations minimise the same functional; the stored
solutions differ along z because the scan has 7x less texture in that
direction and both codes stop early there. The automatic `beta` equals the
MATLAB value, and the ZNCC of the pyALDVC fields is never below MATLAB's.
The run takes 3.6 min on a 24-core workstation (3.2 min with `init_coarse_factor=2`) and 23 s on an RTX 5090. See `docs/design.md`,
section 10.

## Tutorial

`examples/tutorial_real_data.ipynb` (and the equivalent script
`examples/scripting/tutorial_real_data.py`) walks through a complete run:
loading volumes, parameters and memory, running with a checkpoint directory,
reading status codes, ZNCC and uncertainty, slice plots, exports. Without
your own files it runs on a synthetic pair with a known deformation.

## Project layout

```
src/al_dvc/
  core/     DVCPara (config.py), data structures, run_aldvc (pipeline.py)
  io/       volume loading / streaming, normalisation, gradients
  mesh/     hex8 node grid, mask trimming, shape functions
  solver/   Numba IC-GN kernels, NCC pyramid, global operators, ADMM pieces
  strain/   gradient estimators, strain measures
  utils/    outlier test, inpainting, grid interpolation, validation
  export/   npz / mat / csv / vtk / pdf report
  viz/      matplotlib slice views
  cli.py    al-dvc command line
tests/      pytest suite (unit + synthetic integration)
scripts/    report and benchmark generators
examples/   scripted workflows and config templates
docs/       design document
```

## Testing

```bash
pytest                 # ~30 s, 100+ tests
pytest -m "not slow"   # skip the heaviest synthetic cases
PYALDVC_FROZEN_EXE=dist-exe/pyALDVC/pyALDVC-console.exe pytest tests/test_frozen_bundle.py   # after tools/build_exe.py
```
