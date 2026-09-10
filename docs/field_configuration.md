# Which configuration the field displays draw in

Status: 2026-09-09. Recorded from a question about the main window's slice and 3-D tabs; option 2
below has since been implemented, and this file is kept as the record of why.

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

**The background image is chosen separately** (see "What was done" below); before that change it was
always the selected frame, which is still the default. The selected frame also chooses the field:

- `AppState.result_frame()` maps the selected volume to result index `k - 1` and returns `None` for
  `k < 1`. So selecting the reference volume turns the overlay **off** entirely: the field is only
  ever visible while a deformed volume is selected.
- Before the change, `gui/panels/viewer.py` loaded `volume_array(state.current_frame)`, so the image
  under a visible field was always a deformed frame, and the 3-D tab's grey planes followed the
  current frame too.
- The **strain window was already the other way**: `strain_window.py::_load_data` loads
  `volume_array(0)`, and its checkbox says so ("Show the reference volume under the field"). It is
  self-consistent: a reference field over the reference image. So is the image export
  (`dialogs/export_dialog.py`, `volume_array(0)`).

## The consequence

With the background following the selected frame -- the default, and the only behaviour before this
was addressed -- the main window's slice tab and the 3-D tab's volume slices draw a
reference-configuration field over a deformed-configuration image. The material that was at reference
position `X` is at `X + U` in frame k, so the value drawn at `X` sits over the material that arrived
there from `X - U`. **The mismatch is exactly the displacement field.**

That is invisible at 0.1 voxel and obvious at 5. It matters most where it is most likely to be
looked at: a crack that has opened, the edge of a specimen that has moved, a strain concentration
next to a feature the user is trying to line the field up with.

The strain window does not have the problem. The 3-D `warped` mode does not have it for the lattice,
but its grey planes stay where the voxels are, so a warped lattice over volume slices is still two
configurations in one picture.

## What was done: option 2

`AppState.background_frame` (`None` = follow the selected frame, an index = pin that frame) and
`AppState.background_index()`, which resolves it and falls back to the selected frame when the
sequence no longer has the pinned one. The slice tab has a `Background` combo -- *Selected frame*,
*Reference (frame 0)*, *Frame k* -- and the 3-D tab's volume slices read the same setting, so the two
tabs never disagree. The setting is saved in the session; a pin that no longer points at a frame is
forgotten rather than left to re-apply itself when frames are added back.

Next to the combo, a hint says which configuration is on screen: *(both in the reference
configuration)* when the background is frame 0, *(field at reference positions, image deformed)*
otherwise. That is option 1 from the list below, which came along for free.

The default is unchanged -- the background still follows the selected frame -- because that is the
pairing that answers "what does frame 3 look like"; the reference pairing answers "where did this
value come from", and now both are available.

Not changed: `result_frame()` is still derived from `current_frame`, so *which* result is displayed
is still chosen by selecting a volume. Only the image underneath became independent.

## The other options considered

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

Option 2 was chosen: it makes the picture correct without changing how the field is drawn. Option 3
remains open and would be the honest answer for an Eulerian reading of the field.
