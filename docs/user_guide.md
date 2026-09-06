# pyALDVC user guide

## 1. Data preparation

* Volumes must share one shape `(nz, ny, nx)`; any integer or float dtype
  works (normalisation is per frame over the VOI).
* Supported inputs: multi-page TIFF, a folder of 2-D slices, MATLAB `.mat`
  (`vol` variable, MATLAB `(x, y, z)` order is permuted automatically),
  `.npy`/`.npz`.
* A boolean mask volume (`True` = material) per frame or one shared mask
  restricts the correlation to the specimen. Nodes whose subset covers less
  than `min_valid_ratio` of valid voxels are skipped and later inpainted;
  elements touching them are dropped from the global step.
* Keep the volume of interest (`voi`) at least `winsize/2 + 5` voxels away
  from the borders you care about: the gradient stencil and the tricubic
  sampler need a 5-voxel margin.

## 2. Choosing parameters

| situation | suggestion |
|---|---|
| speckle / pore size `d` voxels | `winsize >= 4-5 d` (at least ~2-3 features per axis in a subset) |
| smooth strain fields | `winstepsize = winsize / 2`; larger subsets lower noise |
| strain gradients / localisation | smaller `winsize` (16-24) and `winstepsize = winsize / 4`; expect the first-order subset bias when the field curvature is large (see the validation report) |
| noisy scans (SNR < 5) | `prefilter_sigma = 0.6-1.0`, larger `winsize` |
| large motion (> 10 voxels) | leave `init_guess_method="pyramid"`, `global_shift=True`; raise `search_radius` only if the log reports many clipped peaks |
| very large motion between frames | `reference_mode="incremental"` |
| many frames, small increments | `init_guess_method="previous"` |
| anisotropic voxels | give `winsize` / `winstepsize` per axis, e.g. `(32, 32, 16)`, and `voxel_size` |
| classical local DVC | `use_global_step=False` |

`beta` is tuned automatically per reference frame. `mu = 1e-3` rarely
needs changing; if the ADMM updates in the report do not decrease,
increase `admm_max_iter`.

## 3. Reading the results

`result.result_disp[k]` holds, for frame pair `k`:

* `U` -- displacement of the pair in voxels, `(N, 3)` = `[u, v, w]`.
* `U_accum` -- cumulative displacement from frame 0 (equal to `U` in
  accumulative mode).
* `F` -- displacement gradient `(N, 3, 3)`, `F[n, i, j] = du_i/dx_j`.
* `zncc`, `status`, `n_iter` (in `admm.local_info`) -- per-node quality.
* `U_local`, `F_local` -- the local IC-GN result before the global step.

`result.result_strain[k]` holds the strain in physical units with a
`strain_valid` flag; `StrainResult.field("exx")` returns the array with
unreliable nodes set to NaN. `result.dvc_mesh.to_grid(array)` reshapes any
per-node array to the `(nz, ny, nx)` node grid.

Status codes: 0 converged, 1 max iterations, 2 warped subset left the
volume, 3 invalid subset (mask / texture), 4 singular update, 5 NaN,
6 skipped.

## 4. Diagnosing problems

* **Many nodes with status 2** -- the deformed subset leaves the volume:
  shrink the VOI, or the initial guess is wrong (check the global shift in
  the log and the `U0` field in the report).
* **Low median ZNCC (< 0.7)** -- noise, decorrelation or a wrong initial
  guess; try `prefilter_sigma`, a larger subset, or incremental tracking.
* **Speckled strain maps** -- increase `strain_plane_fit_halfwidth` or
  `strain_smoothing`; consider `winsize` larger.
* **Slow first frame** -- Numba compiles on first use; call
  `al_dvc.warmup()` at start-up (results are cached on disk).
* **Memory** -- the pipeline holds one reference volume, its three gradient
  volumes and one deformed volume in float32 (~22 bytes per voxel); use
  `FileVolumeProvider` to stream long sequences.

## 5. Exports

| function | output |
|---|---|
| `export_npz` | one archive with mesh, all frames, all fields, parameters |
| `export_mat` | MATLAB struct with Python and MATLAB (interleaved) layouts |
| `export_vtk` | one `.vti` per frame plus a `.pvd` time series for ParaView |
| `export_csv` | per-node table per frame |
| `export_report` | multi-page PDF with parameters, timings, convergence and field slices |
| `export_run_summary` | JSON with the parameters and per-frame statistics |

## 6. Parameter reference

`dvcpara_default(**overrides)` returns a validated, immutable `DVCPara`.
The most useful fields:

| field | default | meaning |
|---|---|---|
| `winsize`, `winstepsize` | 32, 16 | subset size and node spacing (voxels; scalar or `(x, y, z)`) |
| `voi` | whole volume | `VOIRange(x=(lo, hi), y=..., z=...)` inclusive voxel ranges |
| `voxel_size`, `units` | 1, "voxel" | physical scale for exported displacements and strains |
| `init_guess_method` | `"pyramid"` | `pyramid` / `ncc` / `zero` / `previous` |
| `search_radius` | 8 | NCC search half-width (coarsest pyramid level); expands automatically for clipped peaks |
| `init_subset` | None (= min(winsize, 16)) | NCC template size |
| `interp_method` | `"cubic"` | `cubic` (Keys, = MATLAB `ba_interp3`) / `bspline` / `linear` |
| `icgn_tol`, `icgn_max_iter` | 1e-2, 100 | IC-GN relative gradient-norm tolerance (MATLAB criterion) and iteration cap |
| `icgn_dp_tol` | 1e-3 | IC-GN parameter-increment tolerance in voxels (gradient terms scaled by `winsize/2`) |
| `icgn_patience` | 5 | give up on a node after this many iterations without improvement (status `stalled`); 0 disables |
| `use_global_step` | True | False = plain local subset DVC |
| `mu`, `beta` | 1e-3, auto | ADMM penalties (`beta=None` triggers the L-curve sweep) |
| `beta_criterion` | `"matlab"` | L-curve score: `matlab` (`|u-u_hat| + h^2 |F-grad u_hat|`, MATLAB rule) or `normalized` |
| `admm_max_iter`, `admm_tol` | 4, 1e-2 | ADMM iterations / stopping (voxels) |
| `subpb2_method` | `"fem"` | `fem` or `fd` global discretisation |
| `dual_update` | `"accumulate"` | standard scaled ADMM; `reset` reproduces the MATLAB FEM path |
| `prefilter_sigma` | 0 | Gaussian pre-smoothing of every volume (helps low-SNR data) |
| `strain_method` | `"plane_fit"` | `plane_fit` / `fem` / `fd` / `direct` |
| `strain_plane_fit_halfwidth` | 1 | plane-fit window half-width in nodes |
| `strain_type` | `"infinitesimal"` | `green_lagrange`, `euler_almansi`, `hencky` |
| `reference_mode` | `"accumulative"` | or `"incremental"`; `frame_schedule` for custom trees |
| `subset_stride` | 1 | sample every k-th subset voxel per axis: k^3 fewer voxels per IC-GN iteration (local steps 5x faster at k = 2 with subset 32), the same result on clean data, about 3x the noise-induced error (a subset with k^3 fewer voxels), the smoothing bias of the full span; 2 is a good choice for subsets of 32 and more on data with a decent SNR |
| `init_coarse_factor` | 1 | > 1: the NCC search and a 12-DOF IC-GN run on every k-th node per axis only (k^3 fewer nodes) and the displacement *and* gradient are interpolated to all nodes as the initial guess of the full pass, which then starts within ~0.1 voxel with the local gradient in place; 2 is a good choice for smooth fields on dense grids (micro-CT example: initial guess 50 -> 17 s, run 233 -> 206 s, same result; on clean synthetic data the full pass also needs fewer iterations) |
| `icgn_noise_hessian` | True | Gauss-Newton steps with the noise-corrected Hessian once a node is in its fine-convergence phase (capped at half of the diagonal): the stored Hessian is inflated by the reference-gradient noise, which makes the plain steps too short; same fixed point, about 2x fewer iterations on noisy data (real micro-CT: 4 instead of 7 per ADMM pass), no change on clean data |
| `icgn_predictive_stop` | True | one-step look-ahead: when the IC-GN steps contract and the predicted next step is below `icgn_dp_tol`, apply the current step and stop instead of spending one more sampling pass to confirm (13-30 % fewer iterations on smooth fields, same solution within `icgn_dp_tol`) |
| `backend` | `auto` | `auto` runs the local solvers on an NVIDIA GPU when `numba-cuda` and a CUDA device are present and falls back to the CPU otherwise; `cuda` insists on the GPU (error when unusable); `numba` / `numpy` force the CPU |
| `n_threads` | 0 (all) | Numba thread count (CPU backend) |
| `gradient_mode` | `"stored"` | `on_the_fly` drops the three gradient volumes (21 -> 9 bytes per voxel) for scans that do not fit otherwise; about 15-20 % slower local step |

## 7. GPU acceleration

```bash
pip install "al-dvc[gpu]"          # numba-cuda with the CUDA 12 wheels; needs an NVIDIA driver
```

With an NVIDIA GPU the local solvers (Hessian precompute, 12-DOF IC-GN, the
3-DOF ADMM passes) run as CUDA kernels compiled at first use for the installed
card (any compute capability the CUDA 12 toolchain supports, Maxwell and
newer; the RTX 5090 included). Results agree with the CPU kernels to about
1e-5 voxel (float32 sampling, float64 solves); the local steps are 25-40x
faster on an RTX 5090 than on a 24-core CPU. Installations without the `gpu` flavour,
without a driver or without a usable device are unaffected: `backend="auto"`
(the default) probes CUDA once and uses the CPU kernels otherwise, and the GUI
shows which backend it picked. The portable Windows bundle is CPU-only.

## 8. Command line

The same analysis without the window, for scripts and clusters:

```bash
al-dvc synth data/synth --shape 96 96 96 --mode stretch --value 0.02   # synthetic test pair
al-dvc run --volumes data/synth -o results --winsize 24 --step 12 --export npz vtk report
al-dvc run config.yaml                                                 # see examples/scripting/
al-dvc plot results/aldvc.npz --field exx --frame 1
al-dvc batch study/*.aldvc --export npz summary report              # sessions saved by the GUI, one after another
al-dvc texture scan/ref.tif                                            # correlation lengths and a subset suggestion
al-dvc info scan/*.tif
```

## 9. Scripting

Everything the application does is also available from Python (`al_dvc.run_aldvc`) for
automated studies; see `examples/tutorial_real_data.ipynb`.
