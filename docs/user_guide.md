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
