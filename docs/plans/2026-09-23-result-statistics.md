# Statistics of DVC results -- development plan

**Date** 2026-09-23 · **Status** decisions taken (section 9), phases 1 and 2 implemented (phase 2 without the imported
load axis, left out by the owner) · **Scope** displacement and strain
fields after a run: summary statistics, rigid-body motion, regions, series over frames, profiles,
noise floor, exports.

This plan rests on three surveys made for it: the pyALDVC code as it is today, the sibling and
original codes (pyALDIC, MATLAB ALDIC/ALDVC) together with the DVC Challenge 2.0 paper, and the
practice of established DIC/DVC software and the literature. Section 1 condenses what they found;
sources are linked where a claim depends on them.

---

## 1. What exists, and what practice expects

### 1.1 pyALDVC today

- **Data.** `PipelineResult.result_disp[k-1]` (`FrameResult`) holds per node: `U`, `U_accum`
  (cumulative from frame 0, voxels), `F`, `zncc`, `U_std` (voxels, NaN except at converged nodes),
  `status` (0 converged, 1 max_iter, 2 out of bounds, 3 invalid subset, 4 singular, 5 NaN,
  6 skipped, 7 stalled; the last local pass), `split_fraction`. `StrainResult` (physical units):
  `exx..eyz` (tensorial shear), `principal` (values only), `max_shear`, `von_mises`, `volumetric`,
  `det_F`, `rotation_deg`, `strain_valid`. `DVCMesh`: `coordinates` `[x,y,z]` in voxels,
  `grid_shape`, `spacing`, `node_valid`, `boundary_nodes`.
- **One accessor** feeds every display and export: `export_utils.field_array(result, frame, name,
  trimmed)`. Trimming uses `node_valid` for displacement and `strain_valid` for strain -- **never
  `status`**.
- **Statistics that exist:** the run summary (median ZNCC, converged fraction per frame), the
  results panel summary text, the PDF report's ZNCC/error histograms with mean and sd in the
  title, and 1-99 % colour ranges. **No** rigid-body removal, region statistics, series over
  frames, profiles, or statistics of the result fields themselves.
- **UI templates to reuse:** the texture window (step strip, `RegionViewer` with drawing tools,
  stale-source tagging, CSV/JSON export), the strain window (collapsible sidebar, `QThread`
  worker, publishing back through `AppState.set_results`), `FieldSliceCanvas`, `ExportDialog`,
  the CLI subparsers. matplotlib QtAgg is already embedded and bundled.

### 1.2 Sibling codes and the paper

- **MATLAB ALDVC** (`main_ALDVC.m` L870-1028): "strain statistics for uniform deformations" --
  after cropping the plane-fit half-width plus 5/5/3 more nodes, mean and std (N-1) of the normal
  strains and the symmetrised shears, det F, Poisson ratios; `errorbar` mean ± std against frame.
  Error against ground truth: vector RMS and x-profiles of mean ± std (`PlotDispErr3.m`). **No
  rigid-body removal anywhere.**
- **pyALDIC** `main`: min/max/mean/std per frame in the HTML report only. Branch
  `feat/post-processing-analysis` (unmerged, design agreed 2026-08-26): **probe-based analysis** --
  point, line and area probes in frame-0 coordinates, NaN-aware reductions (mean, median, max,
  min, std, valid_fraction; strain and COD on lines), `TimeSeries` with a valid-fraction threshold
  and per-frame flags, one-row-per-frame CSV with a self-describing header, an **"Analysis" tab in
  the strain window (canvas left, chart right)**, embedded matplotlib, probes saved in the session.
  pyALDVC should follow this design so the two applications work the same way.
- **DVC Challenge 2.0** (Tong et al., section 2.3): bias `mu_j = (1/N) sum (u_DVC - u_nominal)`
  (Eq. 1), noise floor `sigma_j` = population std (1/N) of the error after removing `mu_j` -- i.e.
  after removing **rigid translation only** (Eq. 2; footnote 1 notes rotation is not removed),
  `u_rms = sqrt(|mu|^2 + |sigma|^2)` (SI Eq. S10), per-component histograms annotated with mean and
  std. Residual rigid motion between repeat scans showed up as biases up to 5.7 voxel. pyALDVC's
  noise-floor numbers must reproduce these definitions exactly.

### 1.3 Established software and standards

- **VIC-3D** ([manual 11.2](https://downloads.correlatedsolutions.com/VIC-3D-11.2-manual.pdf)):
  inspector tools on the contour plot (point, line, polygon, circle, rectangle, extensometer);
  *Extract* turns them into series over frames or against imported load, and with no items plots
  the whole-AOI average; line slices over all frames; statistics per item per frame (min, max,
  mean, median, std); **Remove rigid motion** with three references -- average transformation,
  one point fixed, three points or regions fixed -- and the note that strains are insensitive to
  it; a rigid fit reporting rotation, translation and residual std.
- **GOM/ZEISS Correlate**: rigid-body compensation by aligning to a chosen component (3-2-1,
  local or global best-fit); virtual extensometers; statistics per surface component.
- **iDVC** (CCPi, [results](https://tomographicimaging.github.io/iDVC/results.html)): histograms of
  u, v, w with mean, std and a Gaussian fit; displacement "relative to reference point 0".
- **SPAM** ([deformation](https://www.spam-project.dev/docs/spam.deformation.html)): decomposition of
  F into translation, rotation, zoom, stretch, strain; bad points selected by return status and
  correlation.
- **iDICs Good Practices Guide** (ed. 2, 2025): noise floor from static or rigidly moved images
  with the test's own settings, as the **spatial** std (over the ROI, averaged over images) and
  the **temporal** std (over time, averaged over points); bias = mean; report the value *and the
  method*; VSG size `L_VSG = (L_window - 1) L_step + L_subset` (Eq. 7.3); points are not
  independent once subsets overlap by more than about a third.
- **DVC precision literature**: MAER and SDER (Liu & Morgan 2007) -- mean and std over points of
  the average absolute strain component -- are the bone-DVC standard; repeated zero-strain scans
  per specimen (Palanca 2016); power laws of precision against sub-volume size (Dall'Ara 2017).

### 1.4 Four facts the design must respect

1. **Not every value on the grid was measured.** A node that did not converge carries the global
   step's value or an inpainted one; median-test outliers are replaced and **their flag is not
   stored**. Statistics must say which nodes they use, and by default use measured ones only.
2. **Node values are strongly correlated.** With subset 32 and step 16, neighbours share half
   their voxels; the ADMM step and the strain plane fit correlate them further. The std of the
   field is meaningful; `std/sqrt(N)` as the uncertainty of a mean is not (it can be 8-20x too
   small). An honest confidence interval needs an effective sample size.
3. **Rigid rotation changes some quantities and not others.** Translation changes nothing but
   `U`. A rotation `R` maps `F -> R F`: the Green-Lagrange strain, principal stretches and every
   objective invariant are unchanged; the **infinitesimal strain is not** (a pure 2 deg rotation
   gives a spurious normal strain of about 6e-4, comparable to a DVC noise floor). So a rotation
   must be fitted exactly (Kabsch), not linearised, and strain must be recomputed from the
   corrected displacement, not edited.
4. **Nonlinear averages carry a noise bias.** The mean of |u|, von Mises, principal-magnitude or
   Green-Lagrange strain is positive even in a zero-strain test (this is why MAER is never zero).
   The strain of an affine fit of the displacement has no such bias. For a linear strain component
   the affine fit and the mean of nodal strains agree and scatter alike (measured in phase 1,
   reports/statistics.pdf page 4; the lower-variance claim of the first draft did not hold).

---

## 2. Features

Grouped by the question a user asks. Phases are in section 7.

| # | Question | Feature | Phase |
|---|---|---|---|
| F1 | How big and how spread is this field? | Summary table per field and frame: N used, valid fraction, mean, std, median, robust std (1.4826 MAD), p5/p95, min, max, RMS | 1 |
| F2 | Which nodes are in it? | Node filter: measured only (converged, not an outlier), ZNCC floor, strain-valid, drop edge nodes (k layers), drop cut subsets; live counts per reason | 1 |
| F3 | What is the rigid motion, and what is left without it? | Rigid-body removal: none / translation / rigid (translation + finite rotation) / affine; readout per frame (t, rotation axis-angle and Euler angles, residual RMS, nodes used); fit over the whole selection or a chosen anchor region | 1 |
| F4 | How noisy is my measurement? | Noise-floor preset for a static or known-translation pair: DVC Challenge 2.0 bias and noise floor per component, `u_rms`, strain mean and std, MAER/SDER, VSG size; with several static frames, spatial and temporal std (iDICs GPG) | 1 |
| F5 | What is the average deformation? | Homogeneous deformation of a region by affine fit: mean F, polar decomposition (rotation, stretch), Green-Lagrange and infinitesimal strain of the fit, next to the mean of the nodal strains; non-affine residual as a field | 1 |
| F6 | How does it evolve with load? | Series over frames of any reduction, for the whole selection and every region; mean ± std band; gaps where the valid fraction falls below a threshold | 1 (whole) / 2 (regions) |
| F7 | How is it distributed? | Histogram per component with mean, std and a normal curve; later Q-Q plot, box/violin per frame | 1 / 3 |
| F8 | What happens in this part of the sample? | Regions: box, sphere, cylinder, slab, a region drawn on the slices, the DVC region of interest; several named, coloured regions compared in one table and chart | 2 |
| F9 | How does it vary along the sample? | Profiles: slab average along x, y or z (mean ± std per node layer); a line between two points sampled over all frames (current frame highlighted, others grey) | 2 |
| F10 | What would an extensometer read? | Virtual extensometer: `(L - L0)/L0` between two points tracked by `U_accum` (rotation-invariant) | 2 |
| F11 | Is this mean significant? | Confidence interval of a mean from an effective sample size (autocorrelation of the field, or thinning to one node per correlation length); block bootstrap as a check | 2 |
| F12 | Against the applied load? | Import a per-frame table (load, time, crosshead) as the x axis of series; measured against nominal strain | 2 |
| F13 | Show me the corrected field | Corrected displacement and recomputed strain in the main viewer, 3-D view and exports, with a visible "rigid motion removed" badge | 2 |
| F14 | Advanced | Robust fits (IRLS Huber/Tukey, RANSAC), inverse-variance weights from `U_std`, anchor on three regions (fixture posts), per-region rotation, precision against subset size with a power-law fit | 3 |

What is deliberately left out: statistics on the voxel volume itself (the texture window does that),
spatial resolution measurement (needs dedicated star/sine data), and anything that edits the stored
result.

---

## 3. Definitions (the contract)

Written once in `al_dvc.analysis` and quoted in the exports, so every number can be reproduced.

**Nodes used.** `used = node_valid AND status == 0 AND not outlier AND zncc >= zncc_min AND
in_region AND not_edge(k) [AND split_fraction == 1] [AND strain_valid for strain fields]`, each term
switchable, the default being the first three (plus `strain_valid` for strain). The count removed by
each term is reported.

**Scalar statistics** of the `n` used values `x_i`:

| Name | Definition | Note |
|---|---|---|
| mean | `sum x_i / n` | |
| std | `sqrt(sum (x_i - mean)^2 / n)` | population std (ddof 0), as DVC Challenge 2.0 Eq. 2; MATLAB ALDVC used n-1 -- negligible at these n, but stated in every export |
| median, p5, p95, min, max | NaN-aware | |
| robust std | `1.4826 * median(|x_i - median|)` | insensitive to the few wild nodes that dominate a std |
| RMS | `sqrt(sum x_i^2 / n)` | |
| valid fraction | `n / n_candidates` in the region | |

**Vector (displacement) statistics** per component `j`: bias `mu_j` = mean of `u_j - u_nominal,j`
(Eq. 1), noise floor `sigma_j` = std of the same (Eq. 2), `u_rms = sqrt(|mu|^2 + |sigma|^2)` (S10).
`u_nominal` is zero (static pair), a typed translation, or a loaded ground-truth field.

**Strain precision** (zero-strain pair): mean and std per component; `MAER = mean_k(e_k)`,
`SDER = std_k(e_k)` with `e_k = (1/6) sum_c |eps_c,k|` over the six tensor components (tensorial
shear, stated).

**Effective sample size.** `n_eff = n / sum_lags rho(lag)` with `rho` the autocorrelation of the
mean-removed field on the node grid, normalised by the autocorrelation of the selection mask (FFT);
`SEM = std / sqrt(n_eff)`, `CI95 = mean ± 1.96 SEM`. Shown with the rule of thumb
`n (step / L)^3` (`L` = subset size for displacement, `L_VSG` for strain) as a sanity check.

**Units.** Displacement in the physical unit (`voxel_size`, `units`); coordinates of regions and
probes are stored in voxels and converted at read-out, so changing the voxel size never invalidates
a saved region. Strain dimensionless, with a display option for %, or microstrain.

---

## 4. Backend

New Qt-free package **`al_dvc.analysis`** (mirrors pyALDIC's `al_dic.analysis`), NumPy-vectorised
throughout: a run has up to about 2 million nodes per frame, and none of the operations below is
worse than `O(N log N)`.

| Module | Contents |
|---|---|
| `fields.py` | The one field registry (name, label, unit kind, vector or scalar, needs strain), replacing the three copies in the results panel, strain window and export dialog; `field_values(result, frame, name, correction=None)` wrapping `field_array` |
| `selection.py` | `NodeFilter` (frozen dataclass of the switches above), `Region` (`whole`, `roi`, `box`, `sphere`, `cylinder`, `slab`, `mask` from a drawn voxel region sampled at the node positions, `nodes`), `select_nodes(result, frame, region, node_filter, field) -> Selection` (boolean mask plus the removed count per reason) |
| `stats.py` | `summarize(values) -> FieldStats`; `vector_stats(U, nominal) -> VectorStats` (Eq. 1, 2, S10); `strain_precision(strain) -> MAER/SDER`; `effective_sample_size(grid_values, grid_mask)`; `spatial_temporal_std(frames)` |
| `motion.py` | `fit_translation`, `fit_rigid` (weighted Kabsch/Umeyama with the reflection guard `d = sign det(V U^T)`; optional IRLS), `fit_affine` (least squares on `[1, X - Xc]`, then `scipy.linalg.polar`); a frozen `MotionFit` (kind, `R`, `t`, centre, rotation vector and Euler angles, residual RMS, nodes used, anchor region); `remove_motion(coords, U_accum, fit) -> U'` with `U' = R^T (X + U - t) - X` |
| `series.py` | `extract_series(result, region, field, reduction, node_filter, correction)` over frames; `TimeSeries(frames, values, valid_fraction, flags, unit)` as in pyALDIC; `contiguous_runs` so charts leave gaps |
| `profiles.py` | Slab profiles along an axis (mean, std, n per node layer); line sampling by trilinear interpolation on the node grid with NaN propagation (no clamped edge values); virtual extensometer from two points moved by interpolated `U_accum` |
| `corrected.py` | A corrected view of a result: `U'` per frame, strain recomputed by `compute_strain` with the settings recorded in `dvc_para` (exact: the plane fit is linear, so `grad U' = R^T F - I`). Cached by (result identity, frame, fit), never written into `PipelineResult` |
| `export_stats.py` | CSV one row per frame (pyALDIC layout: `frame, <region>_<field>_<reduction>, ..._valid_fraction, ..._flag`, empty cell for NaN, a header recording filters, regions, motion fits, definitions, version); JSON of the summary; report pages |

Around it:

- **Record the median-test outliers.** `FrameResult` gains an `outlier` boolean array (set where the
  local and 3-DOF passes replace a node); npz/mat export it. Without it "measured nodes only" still
  counts replaced values as measured. Small pipeline change, done first.
- **Load a result back.** `load_npz_result` returns a dict today; `result_from_npz` rebuilds a
  `PipelineResult`, so the CLI and scripts can analyse an exported run.
- **CLI.** `al-dvc stats results.npz --field disp exx --frames all --region box X0 X1 Y0 Y1 Z0 Z1
  --motion rigid --noise-floor [--nominal DX DY DZ] -o stats/` writing the CSV and JSON.
- **Rigid fit weights.** Default: equal weights over the used nodes. Option: inverse variance from
  `U_std` (which is calibrated 20-35 % low, so it is a relative weight only).
- **Which frames a fit belongs to.** One fit per frame on `U_accum` (Lagrangian, frame-0 nodes). In
  incremental mode the per-pair `status` and `zncc` live in the pair's own reference; the filter
  then uses the frame-0 validity and says so.

---

## 5. User interface

### 5.1 Where it lives

Following pyALDIC's agreed design, the strain window becomes the **post-processing window** with
two tabs: **Strain** (today's content, unchanged) and **Analysis** (new). The results panel's
buttons open it on the right tab; the Analysis menu gets *Statistics...* (Ctrl+Shift+T). One window
keeps one frame slider, one field canvas and one place where post-processing happens.

### 5.2 Analysis tab

```
+-- Post-processing --------------------------------------------------------------------+
| [ Strain ] [ Analysis ]                                                                |
+--------------------------------------------+-------------------------------------------+
| tools: [whole] [box] [sphere] [slab] [draw] | > Field and frames   field(s), frames      |
|        [point] [line] [extensometer]        | > Nodes used    61 204 / 79 200            |
| +---------+ +---------+                     |     converged only  [x]  ZNCC >= [0.80]    |
| |   XY    | |   XZ    |   field + regions   |     edge layers [1]  cut subsets [ ]       |
| +---------+ +---------+   (coloured)        | > Regions   (table: name, colour, shape,   |
| +---------+                                 |              N, show, delete)             |
| |   YZ    |   frame  < ======|====== >      | > Rigid motion  [rigid v]  fit over [all v]|
| +---------+                                 |     t = (1.21, -0.40, 80.02) vx            |
|                                             |     rotation 0.31 deg about (0.1,0.9,0.4)  |
|                                             |     residual RMS 0.043 vx                  |
+--------------------------------------------+-------------------------------------------+
| [Summary] [Histogram] [Over frames] [Profile] [Noise floor] [Homogeneous]   copy  export |
|  table or chart (matplotlib, same figure for screen and PNG/PDF)                      |
+----------------------------------------------------------------------------------------+
```

- **Canvas (left).** Three orthogonal slices of the chosen field (the existing `FieldSliceCanvas`),
  regions drawn as coloured outlines on every slice they cross, probes as markers. Drawing tools
  are the texture window's (rectangle, ellipse, polygon, brush; depth: all slices, current, range)
  plus parametric shapes whose numbers can be typed. Background is the reference volume while
  drawing, since regions live in frame-0 coordinates.
- **Sidebar (right, collapsible sections)** in the order a user decides: what (field, frames),
  which nodes (filter with live counts), where (regions), rigid motion (mode, anchor, readout per
  frame), export.
- **Results (bottom tabs).** *Summary* table (rows = regions x fields, columns = statistics; the
  current frame, or all frames with a frame column); *Histogram* (per component, mean and std lines,
  normal curve, current frame); *Over frames* (mean ± std band or any reduction, several regions
  overlaid, gaps where the valid fraction is below the threshold, x axis = frame, or an imported
  load/time); *Profile* (slab average along an axis with a band, or a line over all frames);
  *Noise floor* (the paper's table plus histograms, filled when the run is a static or
  known-translation pair); *Homogeneous* (fitted F, rotation, stretch and strains of the region
  against the mean nodal strain, with the non-affine residual available as a canvas field).

### 5.3 Interaction

- **Live for the current frame.** Any change of field, filter, region or motion recomputes the
  current frame's numbers after a short debounce (about 150 ms); this is milliseconds even at
  2 million nodes. **All frames** (series, noise floor over several frames, corrected strain)
  run on a worker thread with progress and Cancel, like the strain window.
- **Stale marks.** Every result is tagged with its source (result identity, field, frame, filter,
  region revision, motion fit), as the texture window does; a table or chart that no longer matches
  the controls is greyed with "out of date -- recompute".
- **Counts always visible.** "61 204 of 79 200 nodes: 12 540 not converged, 3 110 outliers, 2 346
  edge" -- the user sees what the numbers stand on, and a region whose valid fraction falls below
  the threshold says so rather than reporting a mean of a few nodes.
- **Rigid motion is a view, never an edit.** The stored result stays as computed. The correction
  applies inside the Analysis tab; a separate switch applies it to the main viewer, the 3-D view
  and exports, which then carry a visible "rigid motion removed (rigid, fitted over ...)" badge
  and record the fit.
- **Selection and linking.** Clicking a region row highlights it on the slices; clicking a chart
  series selects its region; the frame slider, the table and the charts follow one frame.
- **Copy and export.** Copy the table (tab-separated, pastes into Excel), export CSV/JSON, save any
  chart as PNG/PDF, add the analysis pages to the PDF report.
- **Persistence.** Regions, probes, filter and motion settings are saved in the session (format
  bump, older sessions load with none).
- **House rules.** Every string through `tr()` in the six tables, locale-safe spin boxes, wheel
  guard, compact fixed-width inputs, no required file choice before computing.

---

## 6. Visualisation

| View | Used for | Built from |
|---|---|---|
| Summary table | F1, F2 | Qt table model; tab-separated copy |
| Histogram per component with mean/std lines and a normal curve | F1, F4, F7 | matplotlib; bins by Freedman-Diaconis, shared across components |
| Mean ± std band over frames, several regions | F6, F12 | matplotlib; NaN gaps kept |
| Slab profile with band; line over all frames | F9 | node-grid means per layer; trilinear line sampling |
| Corrected field and non-affine residual on the slices | F3, F5, F13 | the existing slice canvas, a new "corrected" field source |
| Noise-floor table and histograms (paper layout) | F4 | same as the paper's Fig. 2 and tables |
| Q-Q plot, box/violin per frame | F7 (phase 3) | `scipy.stats.probplot`, matplotlib |
| Displacement against coordinate with the affine line | F5 | scatter (decimated) + fit |

Dark theme colours as the other windows; every chart has the unit in its axis label and the
filter/region/motion in its title or a footnote, so an exported PNG is self-explaining.

---

## 7. Phases

### Phase 1 -- the numbers people report (backend first, minimal UI)

1. `FrameResult.outlier` recorded and exported; `result_from_npz`.
2. `analysis.fields`, `selection`, `stats`, `motion`, `series` (whole selection), `corrected`
   (current frame, displacement and recomputed strain).
3. Analysis tab: field/frames, node filter with counts, rigid motion with readout, Summary,
   Histogram, Over frames (whole selection), Noise floor, Homogeneous; CSV/JSON/PNG export.
4. `al-dvc stats` CLI.
5. Tests and `reports/statistics.pdf` (section 8).

### Phase 2 -- where and along what

Regions and probes (drawn and parametric, several at once), region series, slab and line profiles,
virtual extensometer, effective sample size and confidence intervals, imported load/time axis,
corrected fields in the main viewer / 3-D view / exports with the badge, session persistence.

*Done (2026-09-23), except the imported load/time axis (not wanted).* Notes from the implementation:
the probes are the two ends of the line (the field there over every frame); the session keeps an
optional `analysis` entry instead of a format bump, so older versions still open newer sessions. The
effective sample size needed two corrections found by Monte Carlo: removing the mean (or a plane) biases
every estimated correlation low, and in 3-D a quarter of the correlation sum lies beyond the 0.05 cut --
without them the 95 % interval covered 76-88 %; with them, 91-95 % (85-90 % for a region only two or
three correlation lengths wide). A plane is removed before the estimate so that a real gradient is not
taken for uncertainty.

### Phase 3 -- advanced

Robust (IRLS, RANSAC) and uncertainty-weighted fits, three-region anchors, per-region rotations,
Q-Q and box/violin plots, precision-versus-subset-size study (power-law fit) across several runs.

Each phase is released on its own; phase 1 alone already answers "how noisy", "what rigid motion",
"what mean strain" and "how does it evolve".

---

## 8. Validation

| Test | Expectation |
|---|---|
| Rigid recovery on `synthetic.rotation_displacement` + translation, 0.1-30 deg | `R`, `t` exact to 1e-10 without noise; against `scipy.spatial.transform.Rotation.align_vectors` as an oracle with noise |
| Reflection guard | a near-planar node set never returns `det R = -1` |
| Invariance | after rigid removal Green-Lagrange strain unchanged to 1e-12, infinitesimal strain changed by the predicted `cos(theta) - 1` |
| Affine fit on `affine_displacement` | fitted F exact; polar decomposition matches the imposed rotation and stretch |
| Paper definitions | bias, noise floor and `u_rms` equal to Eq. 1, 2, S10 evaluated by hand on a small field (ddof 0) |
| MAER/SDER | hand-computed on a small field |
| Filters | each switch removes exactly its nodes; counts add up; an all-invalid frame gives NaN without warnings |
| Effective sample size | Monte Carlo on correlated synthetic fields: the 95 % interval covers the true mean 93-97 % of the time, the naive one far less |
| Series and gaps | a region losing nodes below the threshold becomes NaN from the right frame, flagged |
| GUI (offscreen) | live update, stale marks, worker cancel, correction switch reaches the main viewer, session round-trip |
| CLI | `al-dvc stats` on an exported npz reproduces the GUI table |

`scripts/make_statistics_report.py` -> `reports/statistics.pdf`: definitions and a worked noise-floor
example in the paper's layout; rigid recovery against angle and noise; spurious infinitesimal strain
from rotation with and without removal; nodal-mean versus affine mean strain under noise (the noise
bias); confidence-interval coverage; screenshots of the tab; limitations.

---

## 9. Decisions (confirmed by the owner, 2026-09-23: all seven as recommended)

| # | Decision | Recommendation |
|---|---|---|
| 1 | Where the UI lives | A tab in the strain window, renamed post-processing window (pyALDIC's agreed design) -- rather than a separate statistics window |
| 2 | Default node filter | Measured nodes only (converged, not an outlier) plus `strain_valid` for strain, with the switch visible |
| 3 | Standard deviation | Population (1/N), as the DVC Challenge 2.0 paper; stated in exports |
| 4 | Scope of rigid removal | A view inside the Analysis tab; an explicit switch carries it to the main viewer and exports, with a badge |
| 5 | Rotation fit | Finite (Kabsch), not linearised; translation-only kept as the paper's noise-floor definition |
| 6 | Mean strain | Show both the mean of nodal strains and the affine-fit strain, labelled |
| 7 | Phase 1 content | As in section 7, including the noise-floor preset (it reproduces the paper's tables) |

## 10. Issues found during the survey (separate small fixes)

- The strain window lists `det_F` and `rotation_deg` twice in its field list (`strain_window.py`,
  `_available_fields`).
- The 3-D view may mix voxel and physical units when `voxel_size != 1` (the image data use voxels,
  slice positions and warp vectors physical); untested, to be checked.
- `units` is a free label not tied to `voxel_size`; a voxel size with the label "voxel" mislabels
  every displacement.
- The run summary's converged fraction divides by all nodes, invalid ones included.
