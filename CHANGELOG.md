# Changelog

All notable changes to pyALDVC are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.8.0] - 2026-09-11

### Added
- **Guards that keep the large-volume work from rotting.** `tests/test_large_volume_guards.py` pins the
  *shape* of the optimisations rather than their speed: how many bytes an operation allocates and how
  many times a function is called. Both are deterministic, so unlike a timing threshold they can run
  on a CI runner of any speed -- which is why the `perf` tests are excluded and these are not. Eight
  guards: a depth-limited mask op allocates nothing volume-sized, an editor over the whole volume holds
  one boolean volume and not two, a run without a mask carries a 1x1x1 placeholder rather than an
  all-ones volume, the whole-volume reference bundle is built once per reference, one texture-window
  edit scans the volume for its bounding box once (it used to be twelve times), one edit repaints the
  region viewer once, the lattice preview is not recomputed on a slider tick, and the grey window is
  sampled once per volume rather than once per redraw.
  `scripts/check_guards.py` reverts each optimisation in the source, checks that the matching guard
  fails, and puts the source back -- a guard that passes with and without the change it protects reads
  as coverage while asserting nothing. All eight catch their regression today; the script refuses to
  run unless `src/` is clean, so an interrupted check costs one `git checkout`.
- **The background image under a field has its own frame.** The node grid never leaves the reference
  configuration, so a field value is drawn where its subset started; drawn over the selected
  deformed frame the two are a displacement apart, because the material that was at that position
  has moved. The slice tab has a `Background` combo -- *Selected frame* (the default, unchanged
  behaviour), *Reference (frame 0)*, or any frame -- and the 3-D tab's volume slices follow the same
  setting, so the two tabs cannot disagree. A hint next to it says which configuration is on screen.
  The choice is saved in the session, and a pinned frame that the sequence no longer has is
  forgotten rather than re-applied later. The strain window and the image export already paired the
  field with the reference volume; this makes the main window able to do the same.
  `docs/field_configuration.md` records where each display draws its field and why option 2 of three
  was the one taken.
- **`para.tile_local`: solve the local steps over sub-boxes of the volume.** Every local kernel
  addresses `f`, `gx`, `gy`, `gz`, `mask` and `g` relative to a node centre and none of them reduces
  across nodes, so a block of nodes can be solved against a crop of the volumes with the crop's
  origin subtracted from the coordinates -- no kernel signature changes, in any of the three
  backends. `tile_local` is the target box edge in voxels; 0, the default, is a single whole-volume
  box, which is the untiled path by construction.
  - What it buys: the reference gradients, the reference mask and the box of the deformed frame are
    bounded by the box instead of the scan, and on the GPU that is the whole upload. Measured on a
    256^3 pair with subset 32 and step 16: the precompute peak falls from 12.8 to 8.0 bytes per voxel
    at `tile_local=192` (27 boxes) and to 3.3 at 128 (343 boxes), and the GPU's resident set falls in
    the same proportion -- 21 bytes per voxel of the box rather than of the scan, so 1024^3 needs
    about 1.2 GB of VRAM instead of 22.5 GB.
  - What it costs: a box holds the node span plus the halo, and small boxes are mostly halo, so their
    voxels are loaded several times over. The same measurement: 2.1x the wall clock at 192, 10x at
    128. It is a lever for a scan that does not fit, not a free win, which is why it is off unless
    asked for.
  - The halo is derived, not guessed: `winsize/2 + 3` for the reference (the gradient stencil), and
    for the deformed side the displacement actually present in the initial guess, the subset's own
    stretch (`tile_strain_margin`), the search radius, the interpolation margin and
    `tile_disp_margin`. A subset that leaves its *box* rather than the volume is a short halo, not an
    out-of-bounds node, and the two are told apart by which faces the box cut: those nodes are
    re-solved on the whole volume and logged.
  - The answer does not change. The precompute is bit-identical tiled and untiled -- `H`, `L`, the
    means, the valid mask and the subset-splitting rows -- across both gradient modes, with and
    without a mask. The solve is identical in status, iteration count and ZNCC, and the displacement
    moves by at most 1e-14 voxels against a convergence tolerance of 1e-3, because a tile-local `x0`
    rounds `x0 + P[9]` differently. A whole run reproduces the same field to 1e-14.
  - The B-spline coefficients stay global and the box is cut out of the coefficients, not out of the
    grey values, so `interp_method="bspline"` is exact under tiling rather than approximate.
  - Not solved by this: the provider still serves whole normalised frames, so the host floor is
    unchanged at about 13 bytes per voxel and 2048^3 is still out of reach. That needs a provider
    that serves boxes. `docs/large_volume_limits.md` records that and everything else the
    optimisation work left behind, with file references and what a fix would take.

### Changed
- **Large volumes: the application stops doing whole-volume work for local changes.** Measured on a
  384^3 volume (a 3 GB scan is about fifteen times that): drawing a region rectangle in the texture
  window 320 -> 131 ms, moving a slice slider 262 -> 50 ms, undo 144 -> 26 ms, and the mask operation
  behind a one-slice brush stroke 12 -> 0.008 ms -- what is left of a drawing gesture is the redraw,
  not the mask. A masked run's peak volume memory falls from 53 to 14 GB at 1024^3 and from 422 to
  112 GB at 2048^3. `scripts/bench_large_volume.py` measures it and
  `scripts/make_large_volume_report.py` draws `reports/large_volume.pdf`.
  - `subset_valid_fraction` counted each node's valid voxels through a full `(nz+1, ny+1, nx+1)`
    int64 summed-area table -- a measured 24.1 bytes per voxel, 26 GB at 1024^3, and it runs twice
    per reference since subset splitting became the default. A z-slab sweep over 2-D integral images
    gives the same integers with `O(ny * nx)` memory: 1753 -> 110 ms and 24.06 -> 0.06 bytes/voxel.
  - A mask operation is written through the slices it spans instead of building a full boolean
    volume and combining it: a one-slice rectangle 12.00 -> 0.01 ms, a brush stroke 12.91 -> 0.73 ms.
    The bounding box and the voxel count are cached on the editor instead of being recomputed by
    each of the twelve callers per edit, and an all-True base is symbolic, so an editor over the
    whole volume holds one boolean volume instead of two.
  - Both slice viewers keep their matplotlib artists and update the data, and draw slices at the
    pane's own resolution: one canvas draw of three 2048^2 slices with the mask tint and outline is
    2.69 s at full resolution and 0.14 s decimated. The region reduction is conservative -- a sample
    counts as inside only when every voxel it covers is -- so thin exclusions survive.
  - `effective_voi` and the lattice preview are cached on `mask_revision` instead of running four
    and one whole-volume passes per slider tick, and the preview stopped copying the boolean mask to
    uint8 (57 -> 29 ms before caching). The grey window samples evenly spaced indices, which a fixed
    stride could alias onto a single voxel column, and the strain window samples it once per volume.
  - The window no longer pins every frame it has shown: a run over files streams them through
    `FileVolumeProvider` (with `masks=` for the ones drawn in the application), browsing keeps a
    bounded number of frames (`PYALDVC_CACHE_FRAMES` overrides it), and the finished worker releases
    its inputs instead of holding them for the life of the window.
  - A run without a mask carries a 1x1x1 mask placeholder instead of an all-ones volume, in RAM and
    in VRAM; the frame loop releases the previous reference before building the next; the masked
    deformed volume is copied once instead of twice; and the pyramid's reference standard deviation
    is accumulated per slice instead of materialising `f - mean`.
- **`gradient_mode` defaults to `"auto"`**: the three gradient volumes (12 bytes per voxel, 12.9 GB
  at 1024^3) are kept until they would need more than 8 GB -- about 880^3 -- and dropped for the
  in-kernel stencil above that. Below the threshold nothing changes. It is resolved once, where the
  volume shape is known, so bundles, the memory model and checkpoint metadata never see `"auto"`.
  `memory_model` also stops charging a byte per voxel for a mask that does not exist.
- **Texture analysis: one region, one centre, concentric cubes.** The window that slid inside a
  larger range is no longer the analysis the application runs. A cube is now compared with a copy
  of itself shifted by each lag, and every lag is divided by the number of voxel pairs that still
  overlap -- the correction the original DVC Challenge scripts lacked, and which is exactly the
  factor `prod_j (1 - |h_j| / N_j)` that made a small region read a shorter correlation length.
  Nothing outside the analysed cube is read, so each size of the RVE sweep rests on its own voxels
  alone and the sweep can be read as a convergence study. Lags are reported to a quarter of the
  cube edge, and lags keeping less than 30 % of the pairs are dropped (`min_overlap`, was 50 %),
  which leaves the reported lag cube complete in every direction.
- The three steps changed with it: step 1 (the region) is now optional -- it says where the
  analysis may look, defaults to the whole volume and bounds every cube; step 2 takes a centre
  point picked on the slices and sweeps concentric cubes around it, drawing them on the three
  slices; step 3 analyses the cube step 2 settled on, and no longer has a window of its own. The
  window-size input and "use this size" button are gone. New API: `analyse_cube`,
  `sweep_concentric(vol, centre, bounds, start, step, count)`, `concentric_sizes`,
  `concentric_boxes`, `cube_box`, `cube_limits`, `max_lag_for`, `box_centre`. `analyse_range` and
  `sliding_autocorrelation` stay in the library as the reference the report compares against.
- CLI: `al-dvc texture` takes `--centre X Y Z`, `--size`, `--region` (was `--range`) and
  `--sweep-count`; `--window` is gone. The exported JSON reports `centre`, `size`, `box`,
  `max_lag`, `min_overlap` and `fill` instead of `range`/`window`.
- The synthetic case generator stretches the grey values of the material over the whole range
  (one window for every frame), so the speckle has about three times the contrast it had.

### Fixed
- **The texture analysis no longer lets the region move under a running job.** The sweep was handed
  the mask editor's own array while region editing stayed enabled, so a shape added mid-sweep changed
  the input of the cube sizes that had not been analysed yet -- and the source tag could detect that
  afterwards but never reconstruct what had actually been correlated. `MaskEditor.snapshot()` now
  returns a read-only view and the next edit rebinds the editor to a copy, so the worker keeps the
  region it was given and nothing is copied unless an edit really happens. A boolean volume of a
  large scan is a byte per voxel, which is why this is copy-on-write rather than a copy at dispatch.
- **A drag on the slices of another step no longer redraws the region.** The drawing tools live on
  step 1's page, but the canvas kept the last tool and mode, so a navigation-like drag on step 2 with
  *Replace* selected silently threw away a carefully drawn region. The viewer has an explicit
  editable state: step 1 draws, the other steps browse and (in step 2) pick a centre.
- **Whole volume and a typed bounding box can be undone.** Both went through `MaskEditor.reset`,
  which drops the operations and the redo stack, so the Undo button beside them could not bring back
  a region that had taken real work to draw. They are ordinary operations now -- a `fill`, and a
  rectangle extruded through z -- which is also exactly the box, verified over three hundred random
  boxes. Copying an arbitrary DVC region of interest still replaces the base and says so: a boolean
  volume cannot be carried in a `MaskOp`.
- **An unusable typed bounding box is reported instead of ignored.** `normalise_box` rejects a width
  below two voxels and the handler swallowed that and returned, so the spin boxes showed one range
  while the region -- and therefore the analysis -- was still the previous one. The bounds are three
  pairs of spin boxes that report every keystroke, so snapping them back would fight the typist:
  instead the message says what the rule is and what is still in use, and both analyses are disabled
  until the bounds are a box again.
- **An RVE-only result can be exported.** The exports were enabled only when an autocorrelation
  result existed, so completing step 2 and wanting its plot or its numbers meant running an unrelated
  step 3 first. The PNG now follows the plot on the tab you are looking at, the JSON summary accepts a
  sweep with no autocorrelation, and the profiles CSV stays what its name says -- the
  autocorrelation's own export.
- **The exported summary says what each analysis came from.** `save_json` always wrote the retained
  autocorrelation, sweep and recommendation together with no record of their inputs, so an
  autocorrelation of one volume beside a sweep of another -- which the window allows, and warns about
  on screen -- was indistinguishable once saved, and two volumes of the same geometry exported
  identically. There is now a `provenance` block per analysis: volume name and uid, region box and
  revision, centre, spacing, the captured units, the analysis box or sweep settings, whether it still
  describes the current input, and a `same_input` flag when both are present.
- **The cube info names every capped axis.** The reduction notice tested the largest axis, so a cube
  clipped on one axis only -- 64 x 64 x 24 against a scalar control reading 64 -- said nothing at all.
- **A finished RVE sweep no longer overwrites a cube size it does not describe.** `_on_sweep_finished`
  wrote the stable size into step 3 whenever one existed, before any staleness check, and forced the
  view to the RVE tab -- so a sweep of a region the user had already replaced quietly became the
  analysed cube, labelled as coming from the RVE analysis. It now writes only when the sweep still
  describes the current input *and* the user has not typed a size since dispatch; otherwise the
  sweep is kept, the log says which case it was, and the controls and the current tab are left
  alone. The provenance label is also cleared when the reference volume changes, so it cannot credit
  the analysis of another volume. The manual *Use size* action already had this check.
- **The texture window's buttons settle whatever order the worker's signals arrive in.** The result,
  cancel and failure signals are emitted from inside `QThread.run`, so a slot can reach the UI while
  the thread is still alive -- and the refresh declines to act then, which could leave *Apply* and
  *Use size* disabled until some later input change happened to refresh them. The native `finished`
  signal now performs the final refresh, with the worker bound into the connection so a termination
  from a job that has since been replaced is ignored.
- **The subset-split byte budget is decided for the whole reference, not per tile.** When the packed
  keep rows exceeded `MAX_SPLIT_BYTES` part-way through a tiled plan, splitting was switched off from
  that tile onward and the keep rows were dropped for the whole reference -- but the tiles that had
  already run kept the Hessian, mean and ZNCC denominator they had built over their *kept component
  alone*. Those nodes then correlated the full subset window against a normalisation built from a
  subset of it, while the nodes of the later tiles used the full window throughout. Nothing raised
  and nothing was logged past the one warning; the numbers were simply wrong, and inconsistently so
  between tiles. `plan_split` now sizes the keep rows for the whole reference before any tile runs,
  so the decision is one decision, and `split_rows` takes the per-node in-mask fraction it computed
  rather than sweeping the mask again. Reachable on a masked run large enough to be tiled whose
  candidate subsets need more than 512 MB of keep rows.

- The node grid drawn on the slices and the deformed lattice in the 3-D view joined nodes that
  the mask separates, so a crack or a hole looked bridged even though the solver had cut the
  mesh there. Both now stop at the boundary: the slice preview marks the cut edges with the
  same bridging test the solver uses (also before a run, while the subset and step are being
  chosen), and the 3-D lattice draws only the elements the mesh kept. Whether a node column
  happened to land inside the crack decided how it looked before, which is why it seemed to
  depend on the subset size and the step.

## [0.7.0] - 2026-09-07

### Added
- Subset splitting at boundaries (`subset_split`, on by default): a subset whose window
  contains masked voxels keeps only the 6-connected in-mask component around its centre, so
  a subset next to a hole, the region edge or a crack no longer mixes the two sides. Same
  rule for every boundary, computed once per reference at full resolution and stored as
  packed bits for the affected nodes only; `min_valid_ratio` then applies to the kept part.
  Results carry `split_fraction` per node (kept share, exported to NumPy, MATLAB and
  ParaView and checkpointed like ZNCC). The node grid is cut at the same boundaries:
  every hex8 element whose corners the mask separates inside its box is dropped (the
  3-D port of pyALDIC's `mark_bridging`), and the cut edges keep the finite-difference
  operators, the bad-node inpainting, the median test and the strain plane fit on one
  side, so the global step no longer smooths a jump away. On two rigid bodies
  separated by a masked wall the displacement error next to the wall drops from 0.38
  to 0.02 voxel (`reports/subset_split.pdf`). Check box "Split at boundaries" in the
  advanced parameters. All three backends (numba, NumPy reference, CUDA) share the same
  gate, and the initial guess of a cut subset is taken from its own side of the boundary
  instead of the integer search, whose template still spans it. A run without a region of
  interest is unchanged; a run with one now honours its boundaries by default, so its
  displacement next to a hole, an edge or a crack differs from 0.6.0. The split is skipped
  with a warning when the packed keep rows would need more than 512 MB.

## [0.6.0] - 2026-09-07

### Added
- Built-in texture analysis guide (Help menu, and "How it works" in the texture window):
  short text, the autocorrelation and subset formulas, and pre-rendered demonstrations of
  one analysis inside a region, the RVE sweep and the step from the 1/e length to the
  subset. Regenerated by `scripts/make_guide_animations.py`.

### Changed
- Subset suggestion: 4 times the 1/e length per axis by default (was 2.5), adjustable in
  step 3 of the texture window; the window and the guide say it is a recommended start, not
  a guarantee.
- Strain, texture and guide windows are independent windows: they no longer stay pinned
  above the main window.
- RVE result shown as a headline (the window) and one stable length per threshold instead
  of a paragraph; the RVE plots keep their legends outside the axes.
- Texture analysis window rebuilt around three steps, each with its own tab and parameter page
  and a strip at the top showing what every step produced: 1. Region, drawn inside the window
  on its own slice viewer (rectangle, ellipse, polygon, brush; orange outside the region, so it
  cannot be mistaken for the red DVC mask) or copied with "Same as DVC ROI"; 2. Representative
  volume element (RVE), with the radial autocorrelation of every window size and the length
  against the size in one tab, and one click to make the stable size the window of step 3;
  3. Autocorrelation. The autocorrelation lengths and the subset suggestion stay in view on
  every step. Drawing the range on the main window's slices is gone.
- Node grid on the slices drawn thicker and brighter.

- Texture analysis rebuilt around an analysis range and a sliding window. The range is a
  box of the volume (whole volume, the region of interest's bounding box, or drawn on the
  slices as dashed rectangles); a window centred in it is compared with its shifted copies,
  the shifts reaching (range - window) / 2 per axis with a constant number of voxel pairs,
  so no estimator choice, maximum lag or minimum overlap is needed any more. The window
  size analysis comes first: concentric windows of growing size, the stable size written
  into the window with one click. `al-dvc texture` takes `--range` and `--window`.
- Texture analysis window redesigned: the autocorrelation analysis and the window size
  analysis are two parallel analyses with their own parameters, buttons and progress;
  the window size analysis reports the size from which the correlation length is stable
  and writes it into the analysis window with one click. Plots: larger fonts, dark /
  white / grey background, linear or log scale, each curve and the ± 1 std band can be
  switched off (the band is in the legend), threshold lines are labelled, zoom and pan
  with a reset. The correlation lengths and the subset suggestion are highlighted boxes;
  the table says "not reached", "no profile" or "plateau" instead of symbols; every
  parameter has a tooltip that says what it does.
- Strain window: the fit window is chosen as its full size per axis (3 x 3 x 3,
  5 x 5 x 3, ...) with a Cube lock; smoothing steps are half a node; every parameter
  has a tooltip.
- 3-D view: the Frames animation plays the reference state (no displacement) followed
  by the result frames, and a "Smooth" option interpolates the displacement and the
  field between consecutive frames, so the deformed lattice moves like the real
  deformation; it replaces the separate "Deformed lattice" animation. Speed is in
  frames per second. Each of the three slices (XY, XZ, YZ) has its own check box in
  the Slices mode (also for the volume slices). A slice sweep is only offered when
  the Slices mode or the volume slices are on; the mode is never switched behind the
  user's back.
- Volume import: a "Natural order (1, 2, ..., 10)" check box (remembered) chooses
  between numeric and character order, like pyALDIC, and re-sorts the list; an
  import keeps one volume type only (the most numerous one) and reports the files
  it skipped; Add folder (or a dropped folder) replaces the sequence instead of
  appending to it.
- Subset step is set per axis like the subset size (three boxes and a "Same"
  lock), so an anisotropic step from the texture analysis is visible and
  survives edits.
- Mask tool hints say what Fill, Clear and Remove mask mean for the analysis.

### Fixed
- 3-D view went blank after an animation was paused on the reference state or when the
  reference volume was selected; it now shows the paused frame, or the undeformed lattice
  for the reference volume, instead of a hint.
- 3-D view crashed the application (access violation inside VTK) on large scans when
  the volume slices were on and the scene was rebuilt, for example by the frames
  animation or a mode change. The orientation axes widget was destroyed and
  recreated on every rebuild; it is now created once per window. Volume slices are
  drawn as textured quads (four vertices and one image per plane) instead of
  million-point surfaces, with a shared grey window and smooth interpolation.
  A native crash now writes the Python stack of every thread to
  `pyaldvc_crash.log` next to the application log.
- The left column is at least as wide as its widest row, so the Add folder button,
  the region column of the volume table and the mask hint are no longer cut off.
- Region of interest persistence: a drawn mask is saved next to the session as a
  composed mask file, so a region drawn before switching frames is no longer
  lost, and a mask saved with "Save mask" is no longer inverted or doubled when
  the session is reopened. Undo history stays in memory; sessions of older
  versions are still replayed.
- "Mask for: All frames" no longer overwrites the other frames' masks the moment
  it is selected; it only says where the next drawing operations go. A new
  "Copy to all frames" button does the copy explicitly, and the undo button
  reverses it.
- Results keep the identity of the volumes they were computed from: removing,
  adding or reordering volumes after a run no longer pairs a field with the
  wrong image; a volume without a result says so (slices, 3-D view, results
  panel). The volume list is locked while a run is active.
- A run, a strain computation or a texture analysis that finishes after its
  session, result or sequence was replaced is discarded with a message instead
  of being published into the new context. Opening a session during a run is
  refused.
- Strain: the metadata of a computed strain comes from the parameters the
  worker used, not from controls edited meanwhile (which are disabled during
  the computation); a cancel that arrives during the last frame is honoured;
  the buttons settle on every outcome; the settings start from the result's
  own parameters.
- Texture analysis: the result is tagged with its input (reference volume,
  region revision, calibration, settings); a suggestion from a previous input
  cannot be applied, exports say so, and units come from the analysis time.
  An empty or mismatching region stops with a message instead of silently
  taking the whole volume; a cancel without the sweep is honoured; the sweep
  plot states each threshold's verdict; file errors are reported.
- 3-D view: the frames animation advances by elapsed time (it advanced
  cumulatively at the tick rate); changing the animation kind or the result
  while paused starts afresh; a hidden view pauses; a mouse drag, a camera
  preset, Turn / Tilt / Zoom, speed, direction or axis changes continue the
  running animation from what is on screen; screenshots of a paused animation
  show the paused frame; a frames recording draws every field on its own
  volume; MP4 recordings stream to disk and every format has a frame limit
  shown in the dialog; Play is refused while recording, the Record button
  cancels a recording, and the preview resumes afterwards.
- Export dialog: the job's settings are frozen while it runs and the completion
  message names the folder actually written; no fields selected is rejected;
  the frame range says which formats it applies to; base names must be file
  names; existing files are replaced only after a question; a failure after
  some formats were written lists what was written.
- Batch: CSV and VTK outputs carry the session name (two sessions sharing a
  folder no longer overwrite each other); threshold masks replay; the buttons
  settle on the thread's own end; queueing the current session says the saved
  file is used.
- Sessions: written atomically, validated completely before the document is
  replaced, newer formats refused with a message; New / Open / Exit ask before
  unsaved edits are lost; the session remembers the results archive the export
  dialog wrote.
- Application exit cancels and waits for every worker (strain, texture,
  recording, export, batch, run) while the window stays responsive, and stays
  open when a thread does not stop.
- Display: a loaded session restores the colour range, alpha, overlay and
  lattice toggles into the controls; reversed colour limits are corrected;
  the field label uses the result's units; drawing gestures commit with the
  slice, depth, mode and tool they started with; files are added in natural
  order (frame2 before frame10).

## [0.5.0] - 2026-09-05

### Added
- Completion notifications: a finished run, strain, texture analysis, export or
  recording flashes the taskbar and opens a non-modal message (View menu,
  "Notify when a task finishes", remembered).
- 3-D view: Space plays / pauses, Home resets the camera; a mouse drag or wheel
  is written back into the Turn / Tilt / Zoom boxes, and animations, recordings
  and screenshots start from the camera on screen.

### Fixed
- 3-D view: the colour bar has its own narrow renderer beside the scene, so it
  never overlaps the volume; its title wraps to the bar's width.
- 3-D view animations read the controls on every tick: changing the mode, the
  arrows, the volume slices or the field while an animation plays takes effect
  at once (the old playback kept the options it started with, so a frames
  animation stayed on the deformed lattice). The frames animation moves the
  application's current frame, so the Slices tab and the Frame box follow, and
  stop returns to the starting frame. Rebuilding the scene no longer resets a
  camera the user has turned; changing the mode keeps the camera too.
- 3-D view: the record control is a labelled button; while frames are rendered
  the status says the view resumes afterwards.

### Changed
- Typography: three levels, sentence case throughout. Column and box titles
  (Volumes, Region of interest, Parameters, Run, Post-processing, Display,
  Export, Summary) are 12 px bold in the primary colour; the folding
  sub-sections (Subset & search, Solver, ...) are 11 px bold in the secondary
  colour; field labels are regular; hints are 11 px muted. The upper-case
  sub-section titles and letter spacing are gone.
- 3-D view: the control rows share a label column, so Mode, Background and
  Animate line up.
- 3-D view: a camera row (turn, tilt, zoom, reset on top of the presets) and an
  animation row. Orbit about x, y or z, the result frames in sequence, a slice
  sweep along an axis, or the deformed lattice growing to its warp scale and
  back; direction and speed are set per kind, play / pause / stop run live in
  the interactive and the static backend. "Record" writes the same sequence
  off-screen as GIF (always), MP4 (with imageio-ffmpeg) or PNG frames at the
  view's size, 1280 x 960 or 1920 x 1440, with frame rate, duration and loop
  settings, a progress bar and cancel (`gui/view3d_animation.py`).
- The results column opens with a "Post-processing" box holding the texture
  analysis and strain buttons (texture needs only a volume); the summary is two
  lines with the per-frame details folded away, so the buttons stay in view.
- Texture analysis (`al_dvc.texture`, the "Texture analysis" window under
  Analysis / Ctrl+X, and `al-dvc texture`): the autocorrelation of the
  reference volume inside the region of interest, profiles along x, y, z and
  over spherical shells with physical distances, correlation lengths at 1/e,
  0.1 and 0.01 with a status per threshold, the noise floor, a periodicity
  check, a size sweep that analyses several positions per sub-volume size and
  decides a plateau against the largest sizes, and a subset / step suggestion
  per axis with "Apply to parameters". The estimator divides every lag by its
  overlap count (the finite-window estimate of the DVC Challenge scripts is
  kept for comparison), the inverse FFT is shifted before the lags are cut
  (the scripts cut first and mislabelled the negative lags whenever the FFT
  length exceeded 2N-1), negative correlations are kept, and a Boolean sphere
  model with its closed-form correlation validates the lengths to about 2 %
  (`reports/texture.pdf`, `docs/texture_analysis_plan.md`).

### Changed
- 3-D view: the colour bar is drawn with the interface font file (Segoe UI or
  Arial on Windows, DejaVu Sans elsewhere) instead of VTK's built-in "arial".
- Slices: the node grid follows pyALDIC's toggles. "Show grid" (on by default)
  draws a thin, pale grid instead of the thick yellow one: before a run the
  lattice the parameters would place inside the region of interest, after a
  run the mesh the run used, drawn over the field. "Show subset" (off by
  default) outlines the subset of the crosshair node, its neighbour dashed,
  and the subset of the node under the pointer; it works without the grid.
- The volume selected in the list is the displayed frame: its result (frame
  k = deformed volume k) is overlaid, the reference volume shows no field, and
  the Frame box of the results panel selects the volume. A finished run selects
  the first deformed volume and hides the mask tint (the mask checkbox brings
  it back).
- The slice colour bar is drawn from its own opaque mappable; it no longer
  inherits the overlay's transparency (it looked faded).
- 3-D view: the colour bar moves left as its title gets longer, so the title is
  never clipped at the viewport edge; the deformed lattice keeps only cells
  whose 8 nodes are valid (a cell with one NaN corner was drawn transparent)
  and, when no such cell exists, draws the valid nodes and says so; switching
  the mode re-frames the camera.
- The README hero band is the designed `assets/banner-master.png` (volume,
  node lattice, displacement, logo); `scripts/make_branding.py` copies it
  and renders its own banner only when the master is missing.

## [0.4.1] - 2026-09-05

### Fixed
- The Windows bundle failed its self-test: PyInstaller left out numba's
  compiled helpers `numba._devicearray` and `numba.mviewbuf`, so the frozen
  application could not import numba and every kernel fell back to NumPy. They
  are now listed as hidden imports, and the self-test reports the underlying
  import error instead of "numba is not installed".

## [0.4.0] - 2026-09-05

The application matured: readable parameters in seven languages, a node-lattice
preview and per-axis subsets, strain and export windows, the 3-D view, branding.

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
- Node-lattice preview on the slices: once a region of interest is drawn and
  "Node lattice" is checked, the slice viewer draws the lattice the run will
  place as a dark-yellow grid (the layer nearest to each slice, dimmed when
  the layer is off the slice, stopping at the region's edge), outlines the
  subset of the node at the crosshair with its
  neighbour's subset dashed to show the overlap, and outlines the subset of
  the node under the pointer. A line above the slices gives the grid size, the
  node count, the subset edges and the overlap, or the pipeline's message when
  the subset does not fit. The preview updates with every parameter, region or
  slice change and is hidden while a result is overlaid
  (`gui/lattice_preview.py`, `reports/subset.pdf`).
- Non-cubic subsets in the application: the subset size is three boxes
  (x, y, z) with a "Cube" lock, on by default, so one value still sets a cubic
  subset; unlocked, each axis is set on its own (flat or elongated subsets,
  e.g. for anisotropic voxels). The solver kernels already took per-axis
  half-widths; tests now cover them (Numba against the NumPy reference, the
  pipeline on the affine field with 13 x 17 x 25 and 25 x 17 x 13 subsets).
- Branding: application icon (`assets/icon`, SVG / PNG / ICO, shown in the window
  title bar and used by the Windows bundle), a README hero banner, screenshots
  and a workflow GIF, all rendered offscreen from synthetic data by
  `scripts/make_branding.py`; the README opens with the banner, badges, the
  language list, "Why pyALDVC?" and the key features. The icon comes from the
  hand-made master `assets/icon/pyALDVC-master.png` (the designed "A + cube"
  mark, which is the official icon; the built-in SVG is only a fallback); the demo data is an open-cell foam with a
  localised vortex under compression, whose displacement magnitude is a torus
  (rings on the slices, a doughnut iso-surface in the 3-D view); the 3-D view's
  colour bar is titled with the readable field name.
- Readable names everywhere (`gui/names.py`): every choice shows a translated
  label and keeps the solver's key as item data (Cubic / B-spline / Linear,
  Pyramid search / Single-level search / Zero displacement / Previous frame,
  Accumulative / Incremental, Local DVC / AL-DVC, Precomputed / On the fly,
  Automatic / GPU (CUDA) / CPU, Plane fitting / Finite elements / Finite
  differences / Solver gradient, Infinitesimal / Green-Lagrange / Euler-Almansi /
  Hencky); result fields are named in words (Displacement magnitude, Von Mises
  strain...) and node statuses too. The parameter panel is regrouped: Subset &
  search, Solver (tracking mode, solver Local DVC / AL-DVC), Units, Performance
  (compute backend, CPU threads, gradient memory) and Advanced (global step
  discretisation, sampling stride, coarse lattice, pre-smoothing, ADMM and
  IC-GN settings), every row with a tooltip; strain settings live in the strain
  window only. Combo items of every panel, the 3-D view included, follow a
  language switch. `PYALDVC_LANGUAGE` pins the language; tests keep their
  settings in a temporary folder.
- Drawing a shape may leave the image: the pointer is clamped to the slice
  edge, so a rectangle that hugs the border no longer needs the last voxel
  to be hit exactly.
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
