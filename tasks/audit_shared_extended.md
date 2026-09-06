# Shared state, volume, ROI, and display review

Static source analysis only. D = deterministic source path; R = conditional risk; U = usability issue.

## G01 — P1/D: Changing the volume sequence leaves old results attached

Scenario: finish a run, remove or reorder a deformed volume, then select that row. `AppState.remove_volume` (app_state.py:131), `move_volume` (:137), and `add_volume_paths` (:115) do not invalidate results. `result_frame` (:327) maps the current row by index, with no source identity. The slice viewer obtains its background from the new list (:197) and its field from the old result (:245). Reordering equal-shaped volumes can silently pair the wrong field and image. Adding volumes or selecting an uncomputed frame after a partial run clamps to the last computed result instead of showing unavailable. Fix: immutable volume/result identity mapping and explicit unavailable/stale states; never substitute another frame's result.

## G02 — P1/D: Main run completion can publish into a different session

Scenario: Run A, then File > Open B, or change the volume list while A runs. MainWindow.open_session_path (app.py:326) does not guard active work; _on_run_state_changed (:501) only updates the title. Volume actions also remain enabled. RunPanel.start (run_panel.py:77) captures entries and parameters, but _on_finished (:142) unconditionally publishes into current AppState. Fix: operation generation tokens, immutable input snapshots, and coordinated session switching. Preserve A's result separately or reject its stale completion.

## G03 — P1/R: Editing ROI while a worker holds the same mask can mutate its input

VolumeEntry.load_mask returns its cached array (app_state.py:57); _push_mask assigns the live editor mask directly (:232); MaskEditor.apply changes it in place (mask_editor.py:261). RunPanel.start passes loaded masks without copying (run_panel.py:83,109); ROI actions only check volume presence (mask_tools.py:243). With an existing editor, subsequent edits can mutate a worker-owned input. The exact computational impact depends on when downstream code reads or copies it; no solver internals were audited. Fix at the UI/worker boundary with read-only snapshots or explicit edit exclusion.

## G04 — P1/D: Changing mask target immediately overwrites masks on other frames

Draw on one frame, then change Mask for to All frames. set_mask_display immediately calls _push_mask (app_state.py:294), replacing each frame's mask with the editor mask. This is a data-changing action hidden behind a target selector. Returning to This frame does not restore previous masks; undo tracks editor operations, not the replaced per-frame masks. Fix: separate target selection from an explicit Copy mask to frames action with a reversible transaction and visible scope.

## G05 — P2/D: In-progress polygon commits using settings changed after drawing began

Start a polygon at one slice with Add, then change the slice, depth, mode, or drawing tool and finish with Enter. _on_press stores plane/shape/points only (viewer.py:482); _commit_gesture reads current settings and depth (:587). Tool changes do not cancel the gesture (mask_tools.py:199). The final region can be applied to a different slice or with Cut/Replace despite its original context. Fix: snapshot gesture context and cancel or explicitly rebase on relevant context changes. Volume/frame changes and layout changes already cancel gestures.

## G06 — P2/D: Imported display settings do not match visible controls

Session application sets color_auto, limits, and alpha (session.py:198), but ResultsPanel.refresh restores only field/frame/colormap (results_panel.py:215). It neither restores auto_range/vmin/vmax/alpha/show_overlay nor subscribes to display_changed. SliceViewer.sync_from_state (viewer.py:171) restores layout/equal scale but not show_mesh/show_subset checkboxes. A saved manual scale can render while Auto is still checked. Fix: one complete guarded state-to-controls synchronization path.

## G07 — P2/D: Editing units can relabel an existing displacement without converting it

field_array converts displacement using result.dvc_para.voxel_size (export/export_utils.py:74), while SliceViewer.redraw labels it with current state.para.units (viewer.py:330). Change units after a run and the old numeric result gets a new unit label. Fix: render result metadata from its immutable provenance; distinguish next-run calibration from result calibration.

## G08 — P2/D: Reversed color limits lack validation

ResultsPanel directly writes either limit (results_panel.py:171–172,197–209); AppState.set_display assigns without range validation (:380). Entering min > max reaches rendering and normalization with invalid limits. Fix: validate the pair, visibly reject/correct it, and preserve the last valid rendered range. Do not log success or silently leave an outdated image.

## G09 — P2/U: File order and ROI terminology invite mistakes

VolumePanel uses lexicographic sorted(files) (volume_panel.py:176,185), so frame1/frame10/frame2 is possible. Show and validate sequence order and provide natural sort. Fill sets every voxel true (mask_editor.py:265) although the tooltip is only Fill (mask_tools.py:319); label Select entire volume. Clear creates an empty mask while Remove mask means unrestricted volume; explicitly explain these opposite analysis consequences.

## G10 — P2/R: Loading and ROI operations block the UI before background work starts

RunPanel.start loads all volumes/masks synchronously before RUNNING is set (run_panel.py:77–86). SliceViewer selection loads the volume and computes percentiles on the GUI thread (:197–223). Automatic ROI runs synchronously (mask_tools.py:215). Large data may make the app appear frozen with Stop unavailable. Fix: cancellable loading/preprocessing stage with progress and duplicate-action protection; keep completed state until success.

## Existing protections

- RunPanel.start guards an already running worker; Run/Stop enabled state follows RunState.
- Run start catches volume loading and parameter validation errors before clearing prior results.
- Parameter edits replace the configuration rather than mutating that configuration object in place.
- Frame selection is bounds checked; volume loads and automatic mask errors are surfaced in logs.
- Gesture coordinates clamp to image edges; Escape cancels; short polygons are rejected; frame/volume/layout changes cancel gestures.
- WheelGuard prevents an unfocused spin box or combo from consuming scroll events. A focused control can still change while scrolling, which is expected under this policy but deserves clear focus styling.
- Console messages escape HTML, and the console has a bounded block count.
