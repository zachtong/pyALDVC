# Which configuration the field displays draw in

Status: 2026-09-09, recorded from a question about the main window's slice and 3-D tabs. Nothing here
is fixed yet; this file states what the code does today so the choice can be made deliberately.

## What the code does

**The field is always drawn at reference-configuration positions**, with one exception.

| display | where the field sits | file |
|---|---|---|
| main window, slice tab | reference node grid | `gui/panels/viewer.py` (the overlay extent is built from `mesh.x0/y0/z0` +- spacing/2) |
| strain window | reference node grid | `export/slice_plots.py::draw_field_planes`, same extent |
| exported field images | reference node grid | the same function |
| 3-D, `slices` / `points` / `surface` | reference node grid | `gui/view3d_scene.py::node_grid` builds `pv.ImageData(origin=(mesh.x0[0], mesh.y0[0], mesh.z0[0]))` |
| 3-D, `warped` | **deformed**: the lattice moved by the cumulative displacement x `warp_scale` | `gui/view3d_scene.py`, the `warped` branch, `warp_by_vector(DISPLACEMENT_ARRAY, ...)` |

The node grid is built once on the reference (`mesh_setup` from `build_grid_axes`) and never moved.
`node_grid` attaches the displacement as a point-data vector but only the `warped` mode applies it.
The displacement it applies is the **cumulative** one (`displacement_physical` uses `U_accum` when
present), so `warped` shows frame k relative to the reference, not relative to frame k-1.

**The background image is always the selected frame**, and in the main window the selected frame also
chooses the field:

- `gui/panels/viewer.py::_on_volumes_changed` loads `state.volume_array(state.current_frame)`.
- `AppState.result_frame()` maps the selected volume to result index `k - 1` and returns `None` for
  `k < 1`. So selecting the reference volume turns the overlay **off** entirely, and whenever a field
  is visible in the slice tab the image under it is a deformed frame.
- The 3-D tab's grey slice planes come from `panels/view3d.py::_volume_for_scene` -> the current
  frame as well, drawn at fixed voxel positions (an image cannot be warped).
- The **strain window is the opposite**: `strain_window.py::_load_data` always loads
  `volume_array(0)`, and its checkbox says so ("Show the reference volume under the field"). It is
  self-consistent: a reference field over the reference image.

## The consequence

In the main window's slice tab, and in the 3-D tab whenever volume slices are on, a
reference-configuration field is drawn over a deformed-configuration image. The material that was at
reference position `X` is at `X + U` in frame k, so the value drawn at `X` sits over the material
that arrived there from `X - U`. **The mismatch is exactly the displacement field.**

That is invisible at 0.1 voxel and obvious at 5. It matters most where it is most likely to be
looked at: a crack that has opened, the edge of a specimen that has moved, a strain concentration
next to a feature the user is trying to line the field up with.

The strain window does not have the problem. The 3-D `warped` mode does not have it for the lattice,
but its grey planes stay where the voxels are, so a warped lattice over volume slices is still two
configurations in one picture.

## What could be done

Three independent options, in increasing size:

1. **Say which configuration is on screen.** A line in the slice tab ("field: reference positions,
   image: frame 3") costs nothing and removes the ambiguity. It does not fix the mismatch.
2. **Let the background frame be chosen.** Split the background frame from the result frame so the
   field of frame k can be drawn over frame 0's image, which is the self-consistent pairing the
   strain window already uses. `AppState.volume_for_result` already exists to map the other way. The
   work is in the viewer's data path and the panel's controls, and `result_frame()` would no longer
   be derived from `current_frame`.
3. **Draw the field warped in the slice tab.** Consistent with the deformed image, and what an
   Eulerian reading of the field wants, but a warped node grid is no longer a regular lattice, so the
   slice overlay stops being an `imshow` of a grid and becomes a scattered or triangulated draw. The
   3-D `warped` mode already pays that cost with `warp_by_vector`; the slice view would need its own
   resampling, and "which slice does a warped node fall on" has no single answer.

Option 2 is the one that makes the picture correct without changing how the field is drawn.
