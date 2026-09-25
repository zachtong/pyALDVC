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

### A synthetic case to try

`python scripts/make_synthetic_case.py` writes `datasets/synthetic_crack/` (ignored by git): a
speckled block with a crack that stops at a front,
two pores and a per-frame region of interest, plus the exact displacement of every frame. Load the
four files of `volumes/` and the matching masks, run with subset 32 and step 8, and compare a run
with "Split at boundaries" on and off: the displacement next to the crack changes, the far field does
not. The folder's own `README.md` says how to evaluate the truth.

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
6 skipped, 7 stalled. `outlier` marks the nodes that converged but were
rejected by the median test and replaced: their value is not a measurement.

### The 3-D view

The **3-D view** tab draws the field on the node grid in one of four modes:
orthogonal **slices** at the slice viewer's positions, the node **points**,
**iso-surfaces**, or the **deformed lattice** (the valid cells moved by the
displacement times *Warp scale*). In the iso-surface mode, *Surfaces* sets how
many are drawn: one lies at the *Iso level* (a fraction of the colour range);
several are spread evenly over the colour range (0 to 120 with 5 surfaces: 20,
40, 60, 80, 100), each in the colour of its value on the colour bar, and the
status line lists the levels. *Cut away a quarter* removes the quarter of the
surfaces that faces the camera, split at the centre of the node grid in x and
y, so the inner surfaces show; the quarter follows the camera row (and a mouse
turn, once the mouse is released), and an orbit keeps it. The scene takes the
*Overlay opacity* of the results panel: at 1, the iso-surfaces, the points and
the deformed lattice are opaque. The same pictures from a script:

```python
from al_dvc.gui.view3d_scene import CameraSpec, SceneOptions, facing_quadrant, render_png

camera = CameraSpec(preset="iso", azimuth=-95.0, elevation=-5.0)
opts = SceneOptions(field="disp_magnitude", frame=len(result.result_disp) - 1, mode="surface",
                    iso_levels=5, iso_cutaway=True, cutaway_quadrant=facing_quadrant(camera),
                    clim=(0.0, 120.0), background="#ffffff")
render_png(result, "iso_surfaces.png", opts, camera=camera)
```

### Statistics of the results

*Analysis > Post-processing...* (Ctrl+T, or *Post-processing...* in the results
panel) opens the post-processing window, which holds the strain and the statistics
of the result, one tab each; it opens on the tab last shown (the **Strain** tab the
first time). The statistics are on its **Analysis** tab:

* **Statistics of** -- displacement, its uncertainty, the strain tensor,
  principal or equivalent strains, the local rotation, det F or ZNCC. The
  table gives, per component, the nodes used, mean, standard deviation
  (population, divided by N, as the DVC Challenge 2.0 paper), median, robust
  standard deviation (1.4826 x MAD), 5th/95th percentiles, min, max, RMS, and
  the **95 % confidence interval of the mean** with the effective number of
  independent nodes (see below). **Statistics over** restricts all of it to a
  region.
* **Regions** -- add a box, sphere, cylinder or slab (typed corners, centre,
  radius, axis), or draw a rectangle, ellipse or polygon on one of the slices:
  a drawn outline goes through the whole node grid along the slice's normal,
  and its depth can be edited. Regions are in voxels of the reference volume,
  bounds included; each has a name, a colour and a node count, and is outlined
  on the slices (the selected one thicker). Esc cancels a drawing.
* **Nodes used** -- by default only measured nodes: converged, and not
  rejected by the median test. A ZNCC floor, edge layers and cut subsets can
  be left out too; the line below says how many nodes each rule removed.
* **Rigid-body motion** -- remove the translation (the mean displacement,
  as the DVC Challenge 2.0 noise floor does), the rigid motion (translation
  and rotation, fitted exactly in physical units) or the affine part
  (leaving the non-affine displacement). The fitted translation, rotation
  and residual are shown. Removing a rotation recomputes the strain from
  `R^T (I + H) - I`: the Green-Lagrange strain does not change, the
  infinitesimal strain loses the rotation's `cos(theta) - 1`. The stored
  result never changes. **Fitted over** a region -- a grip, an undeformed
  part -- fits the motion there and removes it from the whole field. **Also in
  the main window and exports** shows the corrected field on the slices and in
  the 3-D view and writes it to CSV, ParaView, the images and the report, with
  a badge in the results panel (*As measured* turns it off) and a
  `<basename>_correction.json` next to the files; npz and mat keep the result
  as measured.
* **Over frames** computes the statistics of every frame (mean ± std, mean
  ± 95 % CI, median ± robust std, ...), of one region or of every region side
  by side (*Compare regions*); a frame using fewer than the chosen share of a
  region's nodes is left as a gap. **Regions** compares the field on the
  slices across the regions for the current frame (mean with its interval).
  **Profile** gives the mean and std of each node layer along x, y or z, for
  the current frame or every frame. **Line** samples the field between two
  points (typed, or picked on the slices) and, over all frames, gives the
  field at the two ends and a **virtual extensometer**: the length between
  the two material points, strain `L/L0 - 1`, which no rigid motion changes.
  **Noise floor** gives, for a static or known-translation
  pair, the bias and noise floor per component, `u_rms`, the same after a
  rigid fit, the strain mean and standard deviation, MAER/SDER, the virtual
  strain gauge and, with several static frames, the spatial and temporal
  standard deviations of the iDICs Good Practices Guide; **Homogeneous**
  gives the affine fit of the region (F, rotation, infinitesimal and
  Green-Lagrange strain) beside the mean of the nodal strains.
* *Save as CSV* writes the data of the tab on screen (the series, the
  regions, the profile, the line and extensometer); the summary goes to JSON,
  a chart to PNG, and the table to the clipboard. Every file records the node
  rules, the region, the motion removed, where it was fitted, and the
  definitions. Regions, the correction and these settings are saved with the
  session.

Node values are correlated (subsets overlap; the global step and the strain
fit smooth), so `std/sqrt(N)` is far too small to be the uncertainty of a
mean. The interval shown uses the effective number of independent nodes,
estimated from the autocorrelation of the field after a plane is removed (a
real gradient across the region is the field, not uncertainty). On simulated
correlated noise it covers the true mean 91-95 % of the time, 85-90 % when the
region spans only two or three correlation lengths: read it as a lower bound
then. The same statistics from a script:

```python
from al_dvc.analysis import NodeFilter, frame_stats, noise_floor

fs = frame_stats(result, 0, ("disp_u", "exx"), NodeFilter(min_zncc=0.8), motion="rigid")
print(fs.stats["exx"].mean, fs.fit.rotation_deg)
print(noise_floor(result, 0).displacement.noise)

from al_dvc.analysis import Region, extensometer, mean_confidence

grip = Region(1, "grip", "box", {"lo": [0, 0, 0], "hi": [40, 500, 500]})
fs = frame_stats(result, 0, ("exx",), motion="rigid", fit_region=grip.node_mask(result), with_ci=True)
print(fs.stats["exx"].mean, "+-", fs.stats["exx"].ci95)
print(extensometer(result, (40, 60, 70), (120, 60, 70)).strain)
```

From the command line, `al-dvc stats run.npz --regions session.aldvc --region gauge
--fit-region grip --motion rigid --ci` (or `--compare` for every region side by side).

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
| `subset_split` | True | a subset that touches a hole, the region edge or a crack keeps only the part connected to its centre (6-connected, decided at full resolution), and the node grid is cut at the same boundaries, so the global step, the inpainting, the median test and the strain fit stay on one side of it; `min_valid_ratio` then applies to the kept part and the kept share of every subset is exported as `split_fraction`. The integer search still correlates the whole template, so a subset that lost more than 10 % of its voxels takes its starting displacement from its own side of the boundary instead. Available on every backend, GPU included. Without a region of interest nothing is split and the run is identical to one with the option off; it is also skipped, with a warning, when the packed rows would need more than 512 MB |
| `init_coarse_factor` | 1 | > 1: the NCC search and a 12-DOF IC-GN run on every k-th node per axis only (k^3 fewer nodes) and the displacement *and* gradient are interpolated to all nodes as the initial guess of the full pass, which then starts within ~0.1 voxel with the local gradient in place; 2 is a good choice for smooth fields on dense grids (confocal example: initial guess 50 -> 17 s, run 233 -> 206 s, same result; on clean synthetic data the full pass also needs fewer iterations) |
| `icgn_noise_hessian` | False | *Advanced > Noise-corrected steps.* An acceleration, not in the MATLAB ALDVC: Gauss-Newton steps with the noise-corrected Hessian once a node is in its fine-convergence phase (capped at half of the diagonal). The stored Hessian is inflated by the reference-gradient noise, which makes the plain steps too short; same fixed point, about 2x fewer iterations on noisy data (real confocal scan: 4 instead of 7 per ADMM pass), no change on clean data. **Use it only when both scans carry comparable noise** (repeated scans with the same settings). The noise is estimated from the residual, which cannot tell the two scans apart: with a clearly cleaner reference -- an averaged or longer reference scan, or synthetic noise added to a copy of the reference, as in the DVC Challenge noise-floor test -- it over-corrects and many subsets stop without converging (static test, 2 % noise in the deformed volume only: 52 % of the nodes converged with it under AL-DVC and 10 % with Local DVC alone, 100 % without) |
| `icgn_predictive_stop` | True | one-step look-ahead: when the IC-GN steps contract and the predicted next step is below `icgn_dp_tol`, apply the current step and stop instead of spending one more sampling pass to confirm (13-30 % fewer iterations on smooth fields, same solution within `icgn_dp_tol`) |
| `backend` | `auto` | `auto` runs the local solvers on an NVIDIA GPU when `numba-cuda` and a CUDA device are present and falls back to the CPU otherwise; `cuda` insists on the GPU (error when unusable); `numba` / `numpy` force the CPU |
| `n_threads` | 0 (all) | Numba thread count (CPU backend) |
| `gradient_mode` | `"stored"` | `on_the_fly` drops the three gradient volumes (21 -> 9 bytes per voxel) for scans that do not fit otherwise; about 15-20 % slower local step |

## 7. GPU acceleration

```bash
pip install "al-dvc[gpu]"          # numba-cuda with the CUDA 12 wheels; needs an NVIDIA driver
pip install --upgrade "al-dvc[gpu]"   # later: update, GPU packages included
```

With an NVIDIA GPU the local solvers (Hessian precompute, 12-DOF IC-GN, the
3-DOF ADMM passes) run as CUDA kernels compiled at first use for the installed
card (any compute capability the CUDA 12 toolchain supports, Maxwell and
newer; the RTX 5090 included). Results agree with the CPU kernels to about
1e-5 voxel (float32 sampling, float64 solves); the local steps are 25-40x
faster on an RTX 5090 than on a 24-core CPU. Installations without the `gpu` flavour,
without a driver or without a usable device are unaffected: `backend="auto"`
(the default) probes CUDA once and uses the CPU kernels otherwise, and the GUI
shows which backend it picked: the GPU's name, or why the CPU runs, under
*Performance* in the parameters and in the status bar (the technical reason of
a GPU backend that did not start is in the tooltip and the console). The
portable Windows bundle is CPU-only: for GPU acceleration, install pyALDVC with
pip as above.

## 8. Command line

The same analysis without the window, for scripts and clusters:

```bash
al-dvc synth data/synth --shape 96 96 96 --mode stretch --value 0.02   # synthetic test pair
al-dvc run --volumes data/synth -o results --winsize 24 --step 12 --export npz vtk report
al-dvc run config.yaml                                                 # see examples/scripting/
al-dvc plot results/aldvc.npz --field exx --frame 1
al-dvc batch study/*.aldvc --export npz summary report              # sessions saved by the GUI, one after another
al-dvc texture scan/ref.tif                                            # correlation lengths and a subset suggestion
al-dvc texture scan/ref.tif --sweep --rve-criterion cv-window          # RVE sweep with the DVC Challenge 2.0 test
al-dvc texture scan/ref.tif --sweep --estimator window                 # the raw window estimator (geometric decay kept)
al-dvc info scan/*.tif
al-dvc stats results/aldvc.npz --motion rigid --noise-floor -o stats     # statistics per frame (CSV + JSON)
```

## 9. Scripting

Everything the application does is also available from Python (`al_dvc.run_aldvc`) for
automated studies; see `examples/tutorial_real_data.ipynb`.

## 10. Settings

The **Settings** menu, right after *View*, holds what the application remembers from one start to the
next:

* **Theme** -- *Dark* (the default) or *Light*: a white window with light grey side columns, and white
  slice and chart canvases with black ticks, labels and titles. The switch is immediate, no restart,
  in every open window: the post-processing, texture, guide, export and batch windows follow, and so do
  the icons, the console and every canvas. The 3-D view and the texture plots switch their background
  with the theme (dark or white) until you pick one in their own *Background* list; your choice then
  stays. The demonstrations in the texture analysis guide are pre-rendered and keep their dark frame.
* **Language** -- seven languages, switched immediately.
* **Notify when a task finishes** -- also shows a message box when a task finishes (a run, the strain,
  a texture analysis, an export, a 3-D recording); the taskbar entry flashes either way.

The theme is saved in the application settings (`ui/theme`: `dark` or `light`) and applied before the
window opens at the next start. The environment variable `PYALDVC_THEME=light` (or `dark`) starts in that
theme whatever was saved, e.g. for screenshots in a fixed look.
