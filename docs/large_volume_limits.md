# Large volumes: what is solved, what is not

Status: 2026-09-09, after the optimisation work recorded in `CHANGELOG.md` under *Unreleased*.
Measurements come from `scripts/bench_large_volume.py` and are drawn in `reports/large_volume.pdf`;
regenerate both rather than trusting the numbers below if the code has moved.

This file exists so the unsolved half is traceable. Every entry names the file and line, what it
costs, and what a fix would take. Nothing here is a bug report -- these are known, deliberate limits.

## Where the ceiling now is

A masked run's resident volume memory is about **13 bytes per voxel** (`memory_model`,
`src/al_dvc/io/volume_ops.py`), so:

| scan | uint16 on disk | resident | fits |
|---|---|---|---|
| 512^3 | 0.3 GB | 3 GB | anything |
| 1024^3 | 2.1 GB | 14 GB | a 32 GB workstation |
| 1536^3 | 7.2 GB | 47 GB | a 64 GB workstation |
| 2048^3 | 17.2 GB | 112 GB | nothing normal |

The 13 bytes are: two normalised float32 frames in the provider (8), the deformed frame's
interpolation preparation (4) and, on a masked run, the mask (1) plus the copy the NCC search reads
(4, transient). `para.tile_local` bounds the *reference* side and the GPU upload by the tile box, but
not these, because they are produced before any tiling.

## 1. The provider serves whole frames

`FileVolumeProvider.get_normalized` (`src/al_dvc/io/volume_io.py:447`) calls
`normalize_volume(raw, self._voi)` and returns a **full-shape** array: the VOI selects the voxels the
mean and standard deviation are computed over, not the voxels that are resident. So the existing VOI
plumbing reduces no memory at all, and neither does tiling the kernels.

Getting past 1536^3 needs `get_raw_box(idx, box)` / `get_normalized_box(idx, box)` on the provider
protocol, with the normalisation still computed once globally by streaming (`_voi_moments` in
`volume_ops.py` already does this per slice and works over a memmap unchanged). `al_dvc.solver.tiling`
is the consumer that is already waiting for it: `ReferenceSource.bundle_for(box)` would ask the
provider for the box instead of cropping a resident array.

Estimated 3-4 weeks. It is the only route to 2048^3 and it is a project of its own.

## 2. Nothing is memory-mapped

`src/al_dvc/io/volume_io.py:302` reads `.npy` with `np.load(str(p), mmap_mode=None)` -- mapping is
explicitly disabled. Enabling `mmap_mode="r"` would let a `.npy` reference stream, but two things
must be handled first:

- `np.ascontiguousarray(arr)` in `normalize_volume` (`volume_ops.py`) faults the whole file in.
- **This repository lives under a OneDrive path.** Mapping a OneDrive placeholder hydrates the whole
  file on first touch, and a UNC share behaves the same way. Any out-of-core path must refuse, or at
  least warn, when the volume is on a reparse point or a network share.

No other format can be mapped as it stands: TIFF is read whole (`_read_tiff`), a slice folder is read
per file into a list and then stacked (peak 2x the volume), and `_read_mat` materialises **every**
array in the file before picking one (scipy branch `loadmat`, HDF5 branch the `candidates` loop) --
a `.mat` holding five volumes costs five volumes of RAM.

## 3. Per-node memory is not bounded

- `H_all` + `L_all` are 2304 bytes per node (float64 12x12, twice) per reference context, held in the
  `LocalContext`: 4.5 GB at 2 M nodes, and `_REF_CACHE_SIZE = 2` can hold two of them. `L_all` is a
  dense lower-triangular factor (half of it is zeros) and `H_all` is symmetric (the CUDA precompute
  already computes only its 78 upper-triangle entries before expanding).
- `store_local_result` (`src/al_dvc/core/config.py:141`) defaults to True and keeps `U_local` and
  `F_local` per frame for every frame: about 270 bytes per node per frame, so 54 GB at 2 M nodes and
  100 frames. Setting it False is the fix; it is not the default because the local field is what a
  user compares against the ADMM one.
- `cuda_kernels.py` re-uploads `H_all` and `L_all` on every solver call rather than caching them:
  2 x 2304 bytes per node of PCIe traffic per ADMM iteration. Caching them raises resident VRAM, so
  it is a throughput change, not a memory one, and it needs its own measurement.

## 4. Work still on the Qt thread

- The automatic mask (`threshold_region`, `src/al_dvc/gui/mask_editor.py`) runs `vol > level`, then
  `binary_fill_holes` (several full-volume temporaries) and `ndimage.label` (int32, 4 bytes per
  voxel), synchronously. The Otsu histogram already subsamples; nothing after it does. Past ~512^3
  this freezes or exhausts memory. It belongs on a worker thread with progress.
- `AppState.save_mask` converts the mask to uint8 and writes the file on the UI thread, with no
  progress and no cancel.
- `gui/batch.py` `load_session_inputs` materialises every volume and every mask of a session before
  the run starts -- the streaming provider added for the interactive run is not used there.
- `panels/view3d.py` `_volumes_per_result_frame` loads the whole sequence into a list for a "frames"
  recording when volume slices are on.
- `view3d_scene._grey_window` recomputes its 200 k-sample grey window inside every `build_scene`,
  i.e. on every animation tick, and `_volume_for_scene` re-fetches the volume each tick.

## 5. Deliberate approximations

- **`gradient_mode="auto"`** switches to the in-kernel stencil above 8 GB of gradients (about 880^3).
  Where it switches, nodes within 3 voxels of a volume face are *refused* instead of being given the
  zero gradient `compute_gradients` writes into its border. That is a different answer near the
  faces, and it is the better one, but it is a difference. Below the threshold nothing changes.
- **Display decimation** (`display_stride` / `decimate_region` in `src/al_dvc/export/slice_plots.py`)
  reduces a region slice conservatively: a sample counts as inside only when every voxel it covers
  is. So a one-voxel *exclusion* survives and is drawn thicker than it is, while a one-voxel
  *inclusion* is drawn thinner or disappears. The analysis never sees the reduced array.
- **Tiled local steps** move the displacement by about 1e-14 voxels against a convergence tolerance
  of 1e-3, because a tile-local `x0` rounds `x0 + P[9]` differently. The precompute is bit-identical.

## 6. Not surfaced in the GUI

`para.tile_local`, `tile_disp_margin` and `tile_strain_margin` are settable from a config file or a
script but have no control in the parameter panel; the first merge kept the blast radius small. The
combo would be about fifteen lines plus its six translations, and the panel's memory estimate line
already resolves `gradient_mode="auto"`, so it can show the tiled figure the same way.

## 7. The configuration mismatch in the field displays

Separate from memory, and recorded here because it was found while measuring: the field displays draw
a **reference-configuration** field. The slice tab and the 3-D volume slices now let the background
frame be chosen so the two can be paired (`AppState.background_frame`), but the field itself is still
never warped outside the 3-D `warped` mode. See `docs/field_configuration.md`.
