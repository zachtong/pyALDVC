# pyALDVC design document

pyALDVC is a Python re-implementation of the MATLAB **Augmented Lagrangian
Digital Volume Correlation (ALDVC)** code (Yang, Hazlett, Landauer, Franck,
*Exp. Mech.* 2020), built on the architecture of **pyALDIC** (the 2D sibling).
This document records the decisions that were made *before* code was written
and the contracts that every module must respect.

Sections 1-3 describe the algorithm as it exists in MATLAB and what changes.
Sections 4-8 are the Python contracts. Section 9 is the roadmap, section 10 the cross-validation against the MATLAB code.

---

## 1. What ALDVC computes

Given a reference volume `f(X)` and a deformed volume `g(x)`, ALDVC solves
for a nodal displacement field `u` and a nodal displacement-gradient field
`F = grad(u)` on a regular grid of nodes by alternating two subproblems with
the Alternating Direction Method of Multipliers (ADMM):

* **Subproblem 1 (local)** -- at every node an affine subset warp is fitted
  with the inverse-compositional Gauss-Newton (IC-GN) method and a
  zero-normalised sum of squared differences (ZNSSD) criterion.
  The very first pass fits all 12 affine parameters (9 gradient + 3
  translation). Later ADMM passes hold `F` fixed at the global estimate and
  fit only the 3 translations, with a proximal penalty
  `mu/2 * |u - (u_hat + v)|^2` pulling the node toward the global solution.
* **Subproblem 2 (global)** -- a globally kinematically compatible field
  `u_hat` is found by minimising
  `beta/2 * |grad(u_hat) - (F - W)|^2 + mu/2 * |u_hat - (u - v)|^2`
  over the whole grid (finite-element or finite-difference discretisation).
* **Dual update** -- `W <- W + (grad(u_hat) - F)`, `v <- v + (u_hat - u)`.

Only 3-4 ADMM iterations are needed in practice. The output is `u_hat`
(smooth, compatible) and `grad(u_hat)` (the strain input).

## 2. MATLAB pipeline, section by section

| MATLAB section | What it does | Python module |
|---|---|---|
| S2 `ReadImageLarge3`, `funNormalizeImg3` | load `.mat` volumes, normalise by VOI mean/std | `io.volume_io`, `io.volume_ops` |
| S3 `IntegerSearch3Multigrid`, `funIntegerSearch3(Multigrid)` | FFT cross-correlation initial guess (`xcorr`, `phasecorr`, multigrid `bigxcorr`, windowed `bigxcorrUni`), q-factors (PCE, PPE), 27-point quadratic sub-voxel fit | `solver.integer_search` |
| S3 `RemoveOutliers3` | universal median test (Westerweel & Scarano), `inpaint_nans3` | `solver.init_disp`, `utils.outlier_detection`, `utils.inpaint` |
| S3 `MeshSetUp3`, `Init3` | 8-node hexahedral mesh on the grid, interleave `U0` | `mesh.grid_mesh` |
| S4 `LocalICGN3`, `funICGN3` | 12-DOF IC-GN per node, `stencil7` gradients, `ba_interp3` tricubic sampling | `solver.numba_kernels`, `solver.local_icgn` |
| S5 `Subpb23` / FD operator `funDerivativeOp3` + L-curve beta sweep | global step, beta auto-tuning | `solver.subpb2_solver`, `solver.global_operators`, `solver.beta_tuning` |
| S6 `Subpb13`, `funICGN_Subpb13` | 3-DOF IC-GN with fixed F and proximal penalty | `solver.subpb1_solver` |
| S7 `interp3(...,'makima')` composition | cumulative displacement in incremental mode | `core.pipeline._compose_cumulative` |
| S8 `ComputeStrain3`, `funGlobal_NodalStrainAvg3`, `funPlaneFit3`, `funDerivativeOp3` | strain from `u_hat`: direct / FD / plane fit / FEM; infinitesimal, Green-Lagrange, Euler-Almansi, Hencky; unit conversion | `strain.*` |
| `PlotFiles/*` | figures | `viz.*`, `export.export_report` |

### 2.1 Things the MATLAB code does that pyALDVC deliberately changes

1. **No interactive prompts.** Every `input()` becomes a validated parameter
   with a documented default (`DVCPara`). Runs are reproducible and scriptable.
2. **Per-node subset caching is impossible in 3D.** pyALDIC pre-extracts every
   subset (`(N, S, S)` arrays). In 3D that is `N * S^3` voxels -- for
   `N = 3e4, S = 33` that is 1.1e9 voxels per array, ~35 GB in float32. The
   kernels therefore read the reference volume, its three gradient volumes
   and the mask **in place**; only the per-node 12x12 Hessian, subset mean and
   normalisation are cached (`N * 147` doubles).
3. **Global step solved with PCG, not a direct solve.** The global operator is
   `beta * L + mu * M` with `beta ~ 1e-2..3e-1 * h^2 * mu`, so the system is
   `mu * (M + O(1) * L~)` and extremely well conditioned (kappa <~ 10).
   Jacobi-preconditioned CG converges in ~10-20 iterations. This avoids the
   O(N^2) fill-in of a 3D sparse LU. The three displacement components share
   one scalar operator, so they are solved together as a 3-column RHS.
4. **Operators assembled once, beta enters as a scalar.** The stiffness-like
   `Kg = int grad(N)^T grad(N)`, mass `M = int N^T N` and the three
   gradient-mass matrices `G_j = int (dN/dx_j)^T N` are assembled once per
   mesh. The beta sweep (L-curve tuning) and every ADMM iteration reuse them.
   The FD path uses the same interface with `Kg = sum_j D_j^T D_j`, `M = I`,
   `G_j = D_j^T` (`D_j` = central-difference operators), so FEM and FD are two
   operator sets behind one solver.
5. **Dual variables accumulate by default** (standard scaled ADMM). MATLAB's
   FD path accumulates, its FEM path resets `W = grad(u_hat) - F` every
   iteration; pyALDIC resets too. Both are available
   (`dual_update = "accumulate" | "reset"`); the synthetic report compares them.
6. **Initial guess.** `bigxcorr` multigrid is replaced by an explicit
   coarse-to-fine pyramid: an optional global rigid pre-shift by phase
   correlation (Hann-windowed, on a downsampled VOI), then per-node
   normalised cross-correlation on block-averaged levels. The number of
   coarse levels is chosen automatically so that (a) the search window fits
   and (b) the downsampled volume keeps at least 35 % of the full-resolution
   contrast -- block averaging washes out speckle whose correlation length is
   smaller than the block, and correlating such a level only yields random
   peaks. The template is `min(winsize, 16)` per axis unless `init_subset`
   is given (the NCC only has to find the integer peak; IC-GN refines with
   the full subset). When more than 5 % of the peaks sit on the search
   boundary, *only those nodes* are re-searched with a doubled radius; a peak
   on a side where the window was clamped to the volume boundary is not
   counted as clipped. Two interchangeable engines compute the NCC maps:
   a Numba spatial-domain kernel (one multiply-add per template voxel and
   offset, window sums from a per-node summed-area table, parallel over
   nodes) and an FFT engine (zero-padded to fast 2,3,5-smooth lengths,
   denominators from a Numba summed-area table on the valid region). The
   direct kernel is used while `offsets x template voxels <= 3e7` (it is
   3-5x faster there); both agree to 1e-7.
7. **Masks (VOI volumes).** Nodes and voxels outside a boolean mask are
   excluded from the ZNSSD sums (masked subsets), from the FEM assembly
   (elements touching an invalid node are dropped) and from the strain
   neighbourhoods. The MATLAB code has no mask support.
8. **Explicit node status codes** replace the `stepwithinwhile > MaxIterNum`
   convention: `0` converged, `1` max-iter, `2` out of bounds, `3` invalid
   subset (mask / low texture), `4` singular Hessian, `5` NaN.
9. **Strain on a grid is a filter.** The MATLAB plane fit (`funPlaneFit3`)
   solves a least-squares problem per node in a triple loop. On a regular
   grid the weighted plane fit is a separable convolution (3D
   Savitzky-Golay); with a validity mask it is 14 convolutions plus a batched
   4x4 solve. Same numbers, orders of magnitude faster.
10. **Frame schedule** (accumulative / incremental / custom reference tree)
    is taken from pyALDIC, replacing `trackingMode` + `newFFTSearch`.
11. **IC-GN stopping rule.** MATLAB stops when the gradient norm of the ZNSSD
   functional has dropped below `tol` times its *initial* value. From an
   integer initial guess that takes ~28 iterations and reaches ~0.04 voxel in
   the weakly textured z direction of a CT scan; from a sub-voxel initial
   guess the same rule is far stricter. pyALDVC keeps that criterion
   (`icgn_tol`) and adds the DIC-standard parameter-increment criterion
   `icgn_dp_tol` (1e-3 voxel), which is the one that normally fires.
   `scripts/compare_matlab.py` shows that when both codes' local solutions
   are refined by the same kernel they coincide to 1e-3 voxel, i.e. the two
   IC-GN implementations minimise the same functional; the stored solutions
   differ only by how early each code stopped and by the outlier rules.

## 3. Numerical choices and why

| Choice | pyALDVC | Rationale |
|---|---|---|
| Image gradient | 7-point central stencil (`[-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60]`), zero in a 3-voxel border | identical to MATLAB `stencil7` and pyALDIC; O(h^6) |
| Pre-smoothing | optional Gaussian `prefilter_sigma` applied to every normalised volume before gradients/correlation | the stencil amplifies white noise ~1.1x; at SNR < 5 a 0.8-voxel blur lowers displacement RMSE by ~1.5-2x (validation report) |
| Sub-voxel interpolation | `cubic` = Keys cubic convolution (a = -0.5, what `ba_interp3` does), `bspline` = cubic B-spline on pre-filtered coefficients (most accurate, SciPy `spline_filter`), `linear` = trilinear (fastest) | user-selectable; default `cubic` for MATLAB parity |
| ZNSSD statistics | over mask-valid voxels of the subset; `bottom = sqrt(sum (f - mean)^2)` (= MATLAB `sqrt((n-1)*var)` with the sample variance) | MATLAB convention, keeps `b` scaling identical and ZNCC <= 1 |
| IC-GN convergence | relative gradient norm `< icgn_tol` (MATLAB criterion, 1e-2) OR absolute `< 1e-5` OR parameter increment `|dP| < icgn_dp_tol` (1e-3 voxel, gradient terms scaled by `winsize/2`) | the MATLAB relative criterion depends on the starting point (from a sub-voxel initial guess it is far stricter than from an integer one), so the increment criterion is the operative one. At 1e-2 (the pyALDIC value) it leaves 0.03-0.05 voxel unconverged along weakly textured directions of real CT data; 1e-3 costs about twice the local iterations (7 -> 13 on the `20190504_cut` scan) and is the DIC-literature value |
| IC-GN stall detection | a 12-DOF node is abandoned (status `stalled`, best-ZNCC iterate returned) after `icgn_patience` (5) iterations that do not raise the ZNCC by 1e-4; a 3-DOF node after that many iterations that do not shrink the step norm | on the `20190504_cut` scan the top six node layers lie outside the specimen: there MATLAB and pyALDVC both produced garbage and pyALDVC spent 30-40 iterations per node (max 100), more than doubling the run time. The 3-DOF rule cannot use the ZNCC because the proximal term legitimately lowers it |
| Parallel scheduling | the IC-GN kernels walk the list of active nodes in a block-cyclic order over 64 stripes (`SCHEDULE_LANES`), so every static `prange` chunk touches all regions of the grid while still processing runs of consecutive nodes | Numba's static chunks are contiguous index ranges, i.e. grid slabs: on the `20190504_cut` scan the six textureless top layers made their threads 4-6x slower than the rest and set the wall time (115 instead of ~450 nodes/s); masks, inpainted nodes skipped in the 3-DOF passes and node subsets cluster the same way. `numba.set_parallel_chunksize` had no measurable effect with the OpenMP layer, and querying the thread count inside a kernel disables on-disk caching, hence the thread-independent schedule. The remaining imbalance is the P-core/E-core speed ratio of hybrid CPUs |
| Warp composition | 4x4 homogeneous, `W(P) <- W(P) W(dP)^-1`, inverse via 3x3 cofactors | closed form, no `np.linalg` in kernels |
| Hessian solve | in-kernel Cholesky (12x12 SPD) with fallback to pivoted Gaussian elimination | no allocations in the hot loop |
| Global step | hex8 FEM with 2x2x2 Gauss (exact for trilinear on rectangular cells) or FD; scalar operator, PCG (Jacobi) reused for u, v, w | see 2.3, 2.4 |
| beta tuning | MATLAB list `[sqrt(1e-5), 1e-2, sqrt(1e-3), 1e-1, sqrt(1e-1)] * mean(h)^2 * mu`; score `|u-u_hat| + h^2 |F-grad u_hat|` with the discrete minimum (`beta_criterion="matlab"`, default) or a z-normalised sum with log-quadratic refinement (`"normalized"`) | parity with MATLAB. On the `20190504_cut` scan the normalised score picked 0.0054 where MATLAB picked 0.0202, and the final field's ZNCC was lower with the smaller beta (less weight on the gradient term makes `grad(u_hat)` a worse subset warp) |
| Nodal gradient of `u_hat` | lumped-mass L2 projection `F_ij = M_L^-1 G_j^T u_i` (FEM) or `D_j u_i` (FD) | exact for linear fields, O(h^2), one sparse mat-vec per component |
| Cumulative composition | cubic spline on the node grid after odd-reflection padding (10 nodes) | reproduces linear fields to 1e-5 up to the grid edge; MATLAB uses `makima` |
| Strain plane fit | weighted 3-D Savitzky-Golay via 14 correlations + batched 4x4 solves | identical numbers to the MATLAB per-node loop, vectorised; masks enter as zero weights |
| Outlier test | universal median test (3x3x3, eps = 0.1 voxel), threshold default 2.0 | Westerweel & Scarano 2005 |
| NaN filling | spring model: sparse Laplacian solve with known nodes as Dirichlet data (= `inpaint_nans3` method 0/1) | exact analogue, vectorised |
| Displacement uncertainty | `Cov(P) = 2 s^2 H0^-1 + c s^4 H0^-1 (I3 (x) M) H0^-1` with `s^2 = (1 - ZNCC) bottomf^2 / n` from the residual, `H0 = H - c s^2 (I3 (x) M)` the noise-corrected Hessian, `c` the stencil noise gain and `M` the subset moment matrix; translation block -> `U_std` | zero extra cost (the Hessians are stored anyway); the plain `2 s^2 H^-1` estimate is 2-3x too small at SNR 3 because the noisy reference gradient inflates `H` and the gradient-noise x deformed-noise product adds variance. Calibrated 20-35 % below the empirical error for SNR >= 3 (`reports/uncertainty.pdf`) |
| Deformed-frame masks | masked voxels of the deformed frame are NaN in the sampled array; a subset voxel whose interpolation stencil touches one is dropped from that node's ZNSSD and the subset statistics are recomputed on the remaining voxels every iteration; a node keeping fewer than 27 voxels or less than half of its reference-valid voxels is `invalid_subset`; the NCC search sees masked voxels as featureless (0) | no kernel signature change, no extra memory unless a mask is given; the valid-fraction rule keeps the optimiser from sliding subsets into the mask (fewer voxels would otherwise lower the normalised residual for free). Nodes next to a corrupted region recover the clean-data accuracy (`reports/deformed_mask.pdf`) |
| Precision | volumes float32 in memory, all kernel accumulations float64 | halves RAM, no accuracy loss (verified in tests) |
| Pre-processing | normalisation statistics and the 7-point gradient in parallel Numba kernels (`voi_mean_std`, `_gradient_stencil7`), NumPy/SciPy references kept for tests | the SciPy `correlate1d` path took 30 s on a 1024x1024x306 scan, more than the whole local step for 80k nodes |

## 4. Coordinate and layout contracts

These are the rules every module follows. Violations are bugs.

* **Volumes** are `(nz, ny, nx)` NumPy arrays indexed `vol[z, y, x]`. This
  is what `tifffile` returns for a page stack and what `napari`/`pyvista`
  expect. MATLAB ALDVC stores `vol(x, y, z)`; `load_volume_mat` permutes
  axes so that the *same physical voxel* is addressed by `vol[z, y, x]`.
* **Node coordinates** are `(N, 3)` float64 with columns `[x, y, z]`, in
  voxels, 0-based. The physical position is `coords * voxel_size`.
* **Grid fields** are `(nz, ny, nx)`, the same axis order as volumes, so a
  displacement grid can be overlaid on the volume with no permutation.
  Node index `n = iz * ny * nx + iy * nx + ix`, i.e. `field.reshape(nz, ny, nx)`.
* **Displacement** `U` is `(N, 3)` with columns `[u, v, w]` = displacements
  along `x`, `y`, `z`. `U.ravel()` is the interleaved DOF vector
  `[u0, v0, w0, u1, ...]` used by the global solver.
* **Displacement gradient** `F` is `(N, 3, 3)` with `F[n, i, j] = du_i/dx_j`
  (deformation gradient minus identity). `F.ravel()` is row-major
  `[F11, F12, F13, F21, ...]`. (MATLAB interleaves column-major
  `[F11, F21, F31, F12, ...]`; the `.mat` exporter writes both.)
* **Warp parameters** `P` is `(12,)`: `P[0:9] = F.ravel()`, `P[9:12] = U`.
  The warp of a subset point with offset `X = x - x0` from the node is
  `x_def = x0 + P[9:12] + (I + F) X`.
* **Subset** `winsize = (wx, wy, wz)` (even integers); the subset spans
  `x0 - wx/2 .. x0 + wx/2` inclusive, i.e. `wx + 1` voxels per axis.
* **Elements** are hex8 `(E, 8)` int64 with the standard ordering
  `n0=(ix,iy,iz), n1=(ix+1,iy,iz), n2=(ix+1,iy+1,iz), n3=(ix,iy+1,iz)`,
  `n4..n7` the same with `iz+1`. `-1` marks a dropped element row.
* **Masks** are boolean `(nz, ny, nx)`, `True` = valid material.

## 5. Package layout

```
src/al_dvc/
  __init__.py, __main__.py, _numba_compat.py, cli.py
  core/     config.py (DVCPara, defaults, validation)
            data_structures.py (VOIRange, FrameSchedule, DVCMesh, FrameResult,
                                StrainResult, PipelineResult, VolumeProvider)
            pipeline.py (run_aldvc)
  io/       volume_io.py (tiff / mat / npy / slice folders; lazy providers)
            volume_ops.py (normalise, gradients, VOI clamp, prefilter)
  mesh/     grid_mesh.py (uniform hex8 grid, mask trimming, neighbours)
            hex8.py (shape functions, Gauss points)
  solver/   interp_kernels.py (numba tricubic / bspline / trilinear)
            numba_kernels.py (12-DOF + 3-DOF IC-GN, Hessian precompute)
            reference_kernels.py (NumPy reference of the above, for tests)
            local_icgn.py (S4 dispatcher), subpb1_solver.py (S6 local)
            global_operators.py (FEM / FD operator sets), subpb2_solver.py
            beta_tuning.py, integer_search.py (NCC, global shift, pyramid)
            init_disp.py (outlier removal + inpainting of the initial guess)
  strain/   compute_strain.py, gradient_methods.py, strain_types.py
  utils/    outlier_detection.py, inpaint.py, grid_interp.py, validation.py
  export/   export_npz.py, export_mat.py, export_csv.py, export_vtk.py,
            export_params.py, export_report.py
  viz/      slices.py (matplotlib orthogonal slices), volume.py (pyvista, optional)
```

## 6. Memory and performance model

Let `V = nz*ny*nx` (voxels), `N` nodes, `S = prod(winsize + 1)` subset voxels.

| Resident array | dtype | size |
|---|---|---|
| reference volume (normalised) | float32 | 4V |
| reference gradients gx, gy, gz (`gradient_mode="stored"`; 0 with `"on_the_fly"`, where the kernels evaluate the stencil on f) | float32 | 12V |
| deformed volume (normalised, optionally B-spline prefiltered) | float32 | 4V |
| masks (optional) | uint8 | 2V |
| per-node cache: H (12x12), meanf, bottomf, n_valid | float64 | 147N * 8 |
| per-node results U, F, status, iter, zncc | float64 | ~13N * 8 |
| global operators Kg, M, G_x, G_y, G_z | CSR float64 | ~27N nnz each |

Total 21 V bytes (9 V with `gradient_mode="on_the_fly"`, at 15-20 % more
local-step time). A 512^3 volume pair costs 2.8 GB (1.2 GB); 1024^3 costs
22.5 GB (9.7 GB); 1536^3 costs 76 GB (33 GB). `memory_model` computes these
numbers and the pipeline logs them at start. Frames stream from disk through
`VolumeProvider` so only one reference and one deformed volume are resident.

Kernel cost per IC-GN iteration: `N * S` samples, each a 64-tap tricubic
read plus ~30 flops. Measured numbers are in `reports/performance.pdf`
(regenerate with `scripts/benchmark_performance.py`). The global step is
negligible (PCG on a well-conditioned scalar system); the NCC initial guess
costs ~0.3-0.6 ms per node and level.

Parallelism: numba `prange` over nodes (no GIL, no per-node allocation in the
loop except one `S`-sized scratch buffer per node), `scipy.fft` with
`workers=-1` for the NCC search, sparse BLAS for the global step.

## 7. Robustness features

* Input validation at the boundary (`validate_dvcpara`, `validate_volumes`):
  shapes, dtypes, NaN/Inf, VOI in bounds, `winsize` vs volume size,
  `winstepsize` vs `winsize`, mask coverage.
* Node validity from the mask and subset coverage (`min_valid_ratio`);
  low-texture subsets rejected by a Cholesky check on the Hessian.
* Out-of-bounds warps, singular updates and NaNs terminate a node with a
  status code instead of an exception; bad nodes are inpainted from good
  neighbours (`fill_nan_spring`) and reported.
* Universal median test on the NCC initial guess and on the local IC-GN field.
* Search-radius auto-expansion when NCC peaks sit on the search boundary;
  optional global rigid pre-shift for large motions; multi-level pyramid.
* Bounded caches (reference bundle, precompute cache) so long incremental
  sequences run in flat memory; partial results are returned on cancel.
* Deterministic: no random numbers in the pipeline.

## 8. Public API

```python
from al_dvc import dvcpara_default, run_aldvc, load_volumes, warmup
warmup()                                   # optional: compile the Numba kernels up front
para = dvcpara_default(winsize=32, winstepsize=16, voxel_size=(1, 1, 1))
result = run_aldvc(para, volumes, masks=None, progress_fn=None, stop_fn=None, compute_strain=True)
result.result_disp[k].U        # (N, 3) voxels, pair k
result.result_disp[k].U_accum  # (N, 3) cumulative from frame 0
result.result_disp[k].admm     # beta, ADMM residuals, per-pass IC-GN diagnostics
result.result_strain[k].exx    # (N,) etc.; .field("exx") NaNs out unreliable nodes
result.dvc_mesh.coordinates    # (N, 3)
result.dvc_mesh.grid_shape     # (nz, ny, nx) of the node grid
```

`DVCPara.__post_init__` broadcasts scalars to `(x, y, z)` triples and
validates, so `dataclasses.replace(para, winsize=24)` is as safe as
`dvcpara_default(winsize=24)`.

CLI: `al-dvc run config.yaml`, `al-dvc synth ...`, `al-dvc info volume.tif`,
`al-dvc plot results.npz --field exx --slice z`.

## 9. Roadmap

| Phase | Content |
|---|---|
| **0.1 (this)** | core library, numba CPU backend, FEM/FD global step, pyramid NCC, masks, strain, exports (npz/mat/csv/vtk/pdf), CLI, synthetic validation reports |
| 0.1.x | real-data hardening: node-wise cross-validation against the MATLAB ALDVC results shipped with the reference code (`scripts/compare_matlab.py`, `reports/matlab_crossval.pdf`), memory / throughput profile on a full 1024x1024x306 micro-CT pair, robustness on the `eyes` data set (where the MATLAB IC-GN does not converge), GitHub repository + CI |
| 0.2 (released 2026-09-03) | "real-scan ready": deformed-frame masks in the kernel, per-node displacement uncertainty from the stored Hessian factors, per-frame checkpoints and resume, large-volume mode with on-the-fly gradients, tutorial notebook, PyPI publishing workflow (trusted publishing; the first upload needs the one-time PyPI setup described in `.github/workflows/publish.yml`). A coarse-to-fine IC-GN was considered and deferred: the current pipeline already converges at 100 % of the nodes for 30 degree rotations and 20 % strains on synthetic speckle, so the remaining hard cases (`eyes`) need larger subsets and a deformation-aware initial guess rather than resolution levels |
| 0.3 (in progress) | standalone PySide6 GUI in `al_dvc.gui` (see section 11): `AppState` + panels + worker thread + sessions + JSON-dictionary i18n + kernel warm-up + self-test, three-plane slice viewer with node-grid overlays. Still open: portable Windows build (PyInstaller, copied from pyALDIC), pyvista 3-D view, mask drawing |
| 0.4 | GPU backend (numba.cuda: tricubic sampling + per-node IC-GN, one block per node; global step stays on the CPU). A 2-3 day prototype measures the real speed-up first; Blackwell (sm_120) support in numba-cuda must be confirmed |
| later | adaptive octree mesh (3D analogue of pyALDIC's quadtree) with hanging-node hex elements; second-order (30-DOF) subset shape functions to remove the first-order curvature bias; subset splitting at discontinuities |

Decisions recorded 2026-09-02: GUI before GPU (the CPU code already handles the
82,800-node reference case in minutes, whereas nobody has used the code on real
scans yet); the GUI is a standalone pyALDVC application, not a mode inside
pyALDIC.


## 10. Cross-validation against the MATLAB code

`scripts/compare_matlab.py` runs pyALDVC on the data set shipped with the
MATLAB code (`DVC_images/20190504_cut_01/02.mat`, micro-CT, 1024x1024x306
uint16) with the parameters of `results_ws32_st8.mat` (subset 32, step 8,
finite-difference global step, cubic interpolation, `dual_update="reset"`,
`beta` from the L-curve) on exactly the MATLAB node positions, and compares
node by node. The lowest MATLAB node layer touches the volume border and is
dropped by the margin rule; 79,200 of the 82,800 nodes remain. The report is
`reports/matlab_crossval_ws32_st8.pdf`.

Two comparisons are independent of either code's stopping rule and outlier
handling and therefore decisive:

* **Solver equivalence.** Both codes' local IC-GN solutions are refined by the
  pyALDVC kernel to an increment tolerance of 1e-4. On 15,769 interior nodes
  the refined solutions differ by a median of 0.0001 / 0.0001 / 0.0012 voxel
  (u / v / w) and the gradients by 4e-5. The two implementations minimise the
  same functional. 7 % of the nodes end in distinct optima, almost all in
  `w`, because the scan has 7x less gradient energy along z than in-plane
  (mean squared gradient 0.57 / 0.50 / 0.08) and the ZNSSD surface is flat
  along z (Hessian uncertainty proxy `sqrt(H^-1)` 2.5x larger for `w`).
* **Objective values.** The ZNCC of every stored solution, evaluated with the
  pyALDVC kernel on the 61,699 nodes converged in both codes: local solutions
  0.9243 (pyALDVC) vs 0.9228 (MATLAB), refined 0.9269 vs 0.9267, final AL-DVC
  fields 0.9210 vs 0.9200. pyALDVC is never below MATLAB.

Node-wise differences pyALDVC - MATLAB on the 54,401 interior nodes converged
in both codes (voxel; gradients dimensionless):

| quantity | median abs (u, v, w) | rms (u, v, w) | > 0.1 voxel |
|---|---|---|---|
| local IC-GN | 0.0044, 0.0049, 0.041 | 0.080, 0.089, 0.224 | 23 % |
| final AL-DVC | 0.0048, 0.0055, 0.020 | 0.066, 0.070, 0.195 | 13 % |
| final gradient F | 8.8e-4 (mean of 9) | 9.5e-3 | 22 % (> 1e-2) |

The in-plane agreement (0.005 voxel) is at the level of the convergence
tolerances. The `w` differences come from three sources, all understood:
both codes stop early along the flat z direction (MATLAB's relative-gradient
rule moves by 0.04 voxel when refined, pyALDVC's increment rule at 1e-2 by
0.03, which is why `icgn_dp_tol` is now 1e-3); the outlier rules differ
(MATLAB's `RemoveOutliers3` and pyALDVC's universal median test both flag
10-15 % of the nodes on this scan, not the same ones); and the distinct optima
above.

Other findings from the same run:

* The automatic `beta` equals the MATLAB value (0.02024) with the MATLAB
  L-curve score; the z-normalised score picked 0.0054 and gave a lower final
  ZNCC, hence the change of default.
* The six upper node layers of the MATLAB VOI lie outside the specimen. MATLAB
  reports |U| > 10 voxel at 60-80 % of those nodes; pyALDVC marks them
  `stalled` / `out_of_bounds` (6,191 + 910 of 79,200) after a few iterations
  instead of iterating to the cap. pyALDVC converges 72,027 nodes in the first
  pass with 8.2 mean iterations, MATLAB 62,256 with 27.9.
* MATLAB's finite-difference global step spreads an outlier node into a
  cross-shaped artefact along the grid axes; the pyALDVC field is smooth
  there.
* Wall time on the 24-core workstation: 12.5 min for the 79,200 nodes
  (initial guess 58 s, 12-DOF pass 218 s, three 3-DOF passes 471 s, global
  steps < 1 s) plus 3.9 min for the equivalence check on 18,108 sampled
  nodes. Before the stall rule and the block-cyclic schedule the 12-DOF pass
  alone took 688 s.

The second MATLAB result file, `results_ws30_st30.mat` (`eyes_0/1`, OCT
volumes of an optic nerve head with a large, non-affine deformation; subset
30, step 30, 1,155 nodes), is a diverged run: every MATLAB node hit the
100-iteration cap and the stored displacement has a median magnitude of 86
voxel (up to 527) in a 496-voxel-wide volume. It cannot serve as a reference
and is used as a robustness case (`reports/matlab_crossval_ws30_st30.pdf`):
pyALDVC does not solve it either with these parameters (134 of 1,155 nodes
converge, 511 reach the iteration cap, 510 stall, the median test flags
almost every node), but it says so through the status codes and returns a
smooth field close to the phase-correlation shift instead of a diverged
one. The ZNCC of the pyALDVC fields is 0.30 against 0.09-0.12 for the MATLAB
fields. Solving this case would need a coarse-to-fine strategy for the
whole IC-GN (not only the initial guess) and larger subsets; it is on the
roadmap as a real-scan feature.

## 11. Graphical application

`al_dvc.gui` follows the pyALDIC architecture: one observable `AppState`
(volumes and masks, `DVCPara`, run state, results, display settings) with Qt
signals; panels that only read and write the state (`VolumePanel`,
`ParamPanel`, `RunPanel`, `SliceViewer`, `ResultsPanel`); a `PipelineWorker`
`QThread` that calls `run_aldvc` with `progress_fn` / `stop_fn` and returns the
`PipelineResult` (partial on stop); `session.py` for `.aldvc` JSON sessions with
paths relative to the file; `KernelWarmup` compiling the kernels on a daemon
thread shortly after the window opens; `self_test.py` for installation checks
(`al-dvc-gui --self-test`); the pyALDIC dark theme (`theme.py`, copied) and
Windows title-bar helpers.

Differences from pyALDIC, all consequences of the data being 3-D:

* the viewer draws the XY, XZ and YZ mid-planes of the current volume with
  sliders, and node-grid fields as blocks on the node layer nearest to each
  plane (`mesh.to_grid`, extent from the node axes), with a shared colorbar;
* masks are loaded as volumes rather than drawn; ROI drawing on slices is a
  later feature;
* translations are JSON dictionaries answered by a `QTranslator` subclass,
  which removes the Qt Linguist tool chain (pyALDIC ships `.qm` catalogs);
* dialogs go through `MainWindow._message`, which logs instead of blocking
  under the offscreen platform, so the whole application is testable headless
  (`tests/test_gui.py`, `QT_QPA_PLATFORM=offscreen`; on Windows the offscreen
  platform needs `QT_QPA_FONTDIR=C:\Windows\Fonts` to render text).
