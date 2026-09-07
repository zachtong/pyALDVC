# pyALDVC UI, UX, and interaction safety audit

Date: 2026-09-05

## Executive assessment

The application has useful local safeguards, but its cross-window interaction contract is incomplete. The greatest risks are plausible-looking results attached to the wrong inputs, ROI persistence errors, and background tasks publishing into a changed session. These outrank visual polish. 3D View additionally needs a coherent playback state machine: visible scene, paused state, controls, screenshots, and recording currently do not consistently represent the same state.

This review considers both ordinary workflows and adversarial user behavior: repeated clicks, changing controls mid-operation, hiding a playing view, changing sessions while workers run, cancelling at the last moment, editing masks during computation, reopening windows, conflicting output paths, and malformed or inaccessible files. These are foreseeable actions to handle, not user fault.

## Scope, method, and limits

- Static source inspection only for this expanded review. No application launches, GUI automation, tests, or reproduction scripts were run for this extension, as requested by the user.
- Reviewed the main window/menus, shared state, volume list/import, ROI tools/editor and persistence, parameters, run/stop, slice and result display, 3D scene/playback/recording, Texture Analysis, Strain, export, batch, session lifecycle, and shared control safeguards.
- Numerical solver correctness, algorithm validity, and backend performance benchmarking are excluded. Source reads outside the GUI are restricted to the data, persistence, and export contracts needed to understand user-visible behavior.
- Visual judgments such as actual clipping at a DPI setting, contrast on a physical display, keyboard focus traversal, screen-reader output, and graphics-driver stability require separate UI inspection. They are not certified here.
- Earlier isolated source-function probes exist in tasks but are not GUI tests. This document does not use them to claim a complete runtime acceptance test.
- **Definite / D** means a behavior follows from inspected source under the stated preconditions. **Risk / R** means severity or occurrence also depends on timing, data size, filesystem behavior, or worker/backend internals. **UX / U** means the contract or feedback is confusing even when the code executes as written.
- **P1**: prioritize before trusting saved/exported analyses or broad user release; data integrity, data loss, or serious lifecycle/resource exposure. **P2**: incorrect interaction, recoverability, or material confusion. No issue is called a guaranteed crash solely because a worker or renderer exists.
- Source references use repository-relative paths and one-based lines. Repeated descriptions of a shared root cause in different module sections are cross-impact evidence, not separate defect counts. The report covers the inspected feature surface, not every mathematically possible action sequence.

## Reading map

1. Priority and interaction contract below.
2. Shared state, volume, ROI, and display findings (G01–G10).
3. ROI persistence findings (M01–M02).
4. Texture and Strain findings (A1–A9 and bounded UX observations).
5. 3D interleaving findings (E1–E7 and supporting first-pass findings).
6. Session, batch, export, and shutdown findings (numbered items).
7. Future acceptance matrix and remediation sequence.

## Priority map

| Priority family | User-visible consequence | Main evidence |
|---|---|---|
| Bind every worker completion to its source generation | Run A or strain A gets published into session/result B | G02, A1, A2 |
| Preserve ROI exactly through save/open | Region changes silently after switching frames or saving a processed mask | M01, M02 |
| Preserve result-to-volume identity | Old fields appear on reordered, added, or uncomputed volumes | G01, E4 |
| Define texture result validity | An obsolete recommendation or whole-volume result is accepted as current ROI advice | A3, Texture ROI observation |
| Coordinate shared mutable inputs | An ROI edit can modify a mask reference held by an active worker | G03 |
| Prevent output collision and unsaved-state loss | Batch outputs overwrite one another; session edits disappear on New/Open/Quit | IO items 2, 4, 8, 11 |
| Bound recording and settle worker lifetimes | Long recording has no real cancel; quitting does not settle all jobs | E7, IO item 3, A6 |
| Repair playback state transitions | Hidden views drive visible state; paused sessions resume with obsolete baselines | E1–E3, E6 and Frames feedback finding |

## Required interaction contract

These are proposed product rules, not claims that they are implemented.

| State | Allowed user actions | Required behavior for conflicting actions |
|---|---|---|
| No data / no result | Import, edit next-run defaults, open session | Disable result-specific actions with a short reason |
| Loading / preparing data | Cancel loading, navigate unrelated controls | Do not accept duplicate loading/run requests; preserve previous usable state until commit |
| DVC running / stopping | Inspect captured inputs and progress | Data/session changes require explicit transition policy; settings edits must be labeled next-run-only; old completion cannot target a new session |
| Texture/Strain running | Inspect last successful output, request cancellation | Show captured source/settings; reject stale completion; never relabel output with edited settings |
| 3D playing | Change supported live display properties | Camera edits either rebase predictably or are disabled; leaving the tab pauses by default; global frame ownership is explicit |
| 3D paused | Inspect and screenshot the current rendered frame | Changing animation kind/result starts a new animation context; manual camera edits survive resume or explicitly restart it |
| Recording / exporting | View job source, destination, progress; cancel where supported | Separate frozen job settings from next-job controls; Stop preview is not Cancel recording; prevent conflicting writers |
| ROI gesture active | Complete or Escape-cancel the gesture | Snapshot plane/slice/mode/depth/target; changing context cancels or explicitly rebases it |
| Session dirty | Continue editing or save | New/Open/Quit offers Save/Discard/Cancel; cancelled transitions leave everything unchanged |
| Application closing | Review active jobs and unsaved state | Ask before side effects; cooperatively stop jobs and close only after actual termination; timeout is not completion |

## Detailed findings



---

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

Session application sets color_auto, limits, and alpha (session.py:182), but ResultsPanel.refresh restores only field/frame/colormap (results_panel.py:215). It neither restores auto_range/vmin/vmax/alpha/show_overlay nor subscribes to display_changed. SliceViewer.sync_from_state (viewer.py:171) restores layout/equal scale but not show_mesh/show_subset checkboxes. A saved manual scale can render while Auto is still checked. Fix: one complete guarded state-to-controls synchronization path.

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


---

# ROI persistence findings

## M01 — P1/Definite: Switching frames and continuing to draw loses part of the saved ROI

Scenario: draw region A on frame 0; switch to another frame and back; add region B; save and reopen the session. `AppState.set_current_frame` drops the editor (`src/al_dvc/gui/app_state.py:334-338`). `ensure_mask_editor` creates a new editor using the composed current mask as its base, without restoring the previous operation history (`:202-226`). `_push_mask` replaces the stored mask_ops with only the new editor's operations (`:232-240`). `MaskEditor.to_dict` explicitly omits the base (`src/al_dvc/gui/mask_editor.py:342-344`). Session restore rebuilds only those operations over a file base or an empty base (`src/al_dvc/gui/session.py:140-160`). Thus earlier in-memory region A is absent after reopening, even though A+B was visible at save time. A Replace operation may hide the loss; additive/cut workflows expose it.

The same missing-base persistence contract affects history folding in `_replay` (`mask_editor.py:304-314`): older operations can be merged into an unserialized base. The full-base programmatic reset also illustrates the contract gap, but is not claimed as a separate visible button workflow.

Fix: serialize an exact base plus operations, or save the composed mask as the authoritative snapshot. Keep undo history separate from persistence requirements. Changing frames must not make an otherwise saved ROI irrecoverable. Acceptance: saved/reopened masks are bitwise identical after multi-frame editing and long undo histories.

## M02 — P1/Definite: Saving an edited mask and reopening a session can apply operations twice

Scenario: edit/invert a mask, click Save mask, save the session, reopen it. `AppState.save_mask` writes the already composed current mask and assigns its file path, but retains mask_ops (`app_state.py:278-290`). Session restore loads that composed file as the base and replays the retained operations (`session.py:149-159`). Invert is non-idempotent: the reopened ROI can become its opposite. Add/Replace can conceal this defect, so an ordinary rectangle-only save path does not establish safety.

Fix: when saving a composed mask as the base, reset/rebase persisted operations consistently; or persist the original unmodified base separately. All-frame mask propagation must preserve each target's exact state and reversibility. A successful file write is not sufficient proof that Save/Open preserves the ROI.



---

# Extended Texture and Strain interaction audit

## Scope and evidence

Static source review only for this extension. No tests or new probes were run. Numerical algorithm correctness is excluded. References are repository-relative paths with one-based source lines. The earlier texture probe is separate evidence and is not new validation of the findings below.

## Highest-priority asynchronous integrity findings

### A1 — P1: Strain completion can attach old strain to a different displacement result

- Scenario: Start strain computation for result A. While it runs, load another result B, or start another DVC run that completes before the old strain worker. The old worker finishes after B becomes current.
- Evidence: `src/al_dvc/gui/strain_window.py:314` captures A in the worker; `:344-351` retrieves the current `state.results` and replaces its strain with the worker output, without checking result identity, mesh, frame count, or generation. `:368` publishes the mixed result to all viewers and exports. `:399-400` responds to result replacement only by refreshing the display. `src/al_dvc/gui/panels/run_panel.py:119` clears results when a new DVC run starts, and `:146` later publishes its result.
- Impact: B's displacement/mesh and A's strain become a single result. Matching shapes can make this look plausible; different shapes or frame counts can break downstream display/export. The splice is certain from control flow; an actual downstream crash was not reproduced.
- Safe path: Keep the displacement result unchanged until strain completion. A normal completion on the same result avoids this cross-result splice.
- Required design: Capture source result identity/generation and reject obsolete completion. Publish only against the captured source if it is still current. Coordinate new runs, result import, and session reset with auxiliary workers.

### A2 — P1: Editing strain controls during computation falsifies result metadata

- Scenario: Start with plane fitting and infinitesimal strain; change method, measure, smoothing, or fitting width before completion.
- Evidence: `strain_window.py:310-314` snapshots parameters for computation; the controls remain enabled at `:319-320`. Completion calls `strain_para()` again at `:348`, copies the newly selected settings into result metadata at `:352-361`, and unconditionally clears stale state at `:364`. The worker computed with its original `_para` at `:86`.
- Impact: Exportable `dvc_para` describes settings that did not produce the output. The window also claims it is no longer stale. Individual `StrainResult` method/measure can disagree with the enclosing result metadata.
- Safe path: Do not edit controls during computation. Editing after completion does call `_mark_stale()` and displays a previous-settings warning (`:293-297`, `:475-476`).
- Required design: Commit the worker's captured parameter snapshot; compare current controls with it before clearing stale state. Disabling computational controls while busy is an additional UX option, not a substitute for generation validation.

### A3 — P1: Texture results lack source validity across dataset, ROI, and settings changes

- Scenario: Analyse A; replace reference volume or ROI with B; apply or export. Alternatively replace the data while A is running, then accept A's completion as if it described B. Starting a second analysis leaves the previous Apply/export actions usable throughout the new run.
- Evidence: `src/al_dvc/gui/texture_window.py:270-271` connects volume changes to a button/status refresh and mask changes only to status. `:391-393` never invalidates result/recommendation. `:322-328` starts a worker without disabling Apply/export or clearing the previous result. `:353-362` accepts every completion. `:396-400` applies any retained recommendation to current parameters. No settings-change invalidation is wired.
- Impact: A recommendation can silently be applied to unrelated current data; a failed or cancelled retry leaves previous export/apply actions available without source provenance.
- Safe path: Stable reference, ROI, and calibration; use a completed result from that unchanged input. Existing export helpers serialize the stored result rather than recomputing from current widgets, which preserves its values but does not identify it clearly to the user.
- Required design: Tag results with input identity, mask revision, parameter snapshot, and units; show stale/previous-result status and prevent applying obsolete recommendations.

## Cancellation, failure, and lifecycle

### A4 — P2: Cancel can still publish successful output

- Texture: `_TextureWorker.run` at `texture_window.py:90-111` checks `_stop` only inside the optional sweep branch. With the default unchecked sweep, Cancel is ignored. With sweep enabled, checks before and after the sweep do provide cancellation boundaries.
- Strain: `strain_window.py:80-89` checks only before each frame. Cancel during the final frame, including every single-frame computation, is followed by successful publication at `:97` because there is no final check.
- Impact: The user explicitly abandons a calculation but sees it committed. In Strain this also updates shared results automatically.
- Safe path: Strain cancellation before a subsequent frame is observed; Texture sweep cancellation is observed at its existing boundaries. Neither worker forcibly interrupts a numerical kernel.
- Required design: Check cancellation immediately before publishing output, and expose a pending cancellation status. Keep cooperative cancellation semantics explicit.

### A5 — P2: Clearing results during strain work leaves the busy controls inconsistent

- Scenario: Start strain; start a new DVC run or reset the session; old strain completes while `state.results` is still None.
- Evidence: `strain_window.py:345-347` returns before restoring Cancel/Compute/progress. Results-change refresh at `:427` may disable Compute while the worker is running; no QThread `finished` connection performs final UI cleanup.
- Impact: Cancel can remain enabled after the worker has exited; Compute can stay disabled until another refresh. No source-aware discarded-result message explains the event.
- Safe path: If a later result change occurs after worker exit, `_load_data()` recalculates Compute availability. This does not fix the missed cleanup itself.
- Required design: Centralize terminal cleanup for success, failure, cancellation, and obsolete completion; run it regardless of result availability.

### A6 — P2: Auxiliary workers have no explicit close/shutdown contract

- Evidence: Neither analysis window implements `closeEvent`. Both workers are QObject children of their windows (`texture_window.py:322`, `strain_window.py:314`). `src/al_dvc/gui/app.py:567-580` only handles batch and main DVC workers during shutdown. Auxiliary windows are reused on reopen (`app.py:410-439`).
- Scenario: Close a processing window during a long calculation, then change session or exit the application.
- Impact: Closing merely hides the usual QMainWindow, so work continues and can still mutate state. Application shutdown has no explicit cancel/wait for those threads. A QThread destruction/exit failure is a risk, not an observed crash in this static extension.
- Safe path: Complete or cancel-and-wait before exit. Close/reopen while idle retains the window and its settings by design.
- Required design: Define whether close means hide or cancel and communicate it; always settle all worker lifetimes on application shutdown.

## Cross-panel settings, units, and export

### A7 — P2: Texture Apply during a main DVC run changes future parameters, not that run

- Evidence: `texture_window.py:396-400` has no run-state guard. `src/al_dvc/gui/panels/run_panel.py:91,111-113` passes a parameter snapshot into the pipeline before work starts. `src/al_dvc/gui/app_state.py:348-350` replaces `state.para` rather than mutating that snapshot.
- Scenario: Start DVC, then apply a texture suggestion while it is still running.
- Impact: The main parameter panel now shows new subset/step while the active run uses the old values, without a next-run-only message. This is a presentation/provenance issue, not evidence that an active solver's parameters mutate mid-run.
- Safe path: Apply before starting DVC, or explicitly treat changes as next-run settings.
- Related mismatch: `panels/param_panel.py:263` displays only the x step from the anisotropic tuple written by Texture; `:190` editing writes a scalar, normalizing all axes. Provide three-axis step visibility.

### A8 — P2: Analysis parameter and calibration provenance does not follow result lifecycle

- Texture evidence: `texture_window.py:410-417` decides physical display and unit label using current state parameters, while physical lengths came from analysis-time spacing. There is no `params_changed` binding. Changing units during a run can attach current units to an earlier spacing's values at completion. Changing units after completion leaves the table unchanged.
- Strain evidence: `_load_params()` runs only during construction (`strain_window.py:263,267-278`), not on result/session changes (`:399-400`). `strain_para()` starts from current `state.para` (`:280-291`), while completed displacement results retain their own `dvc_para`. A reused Strain window may show settings from a previous session, and current calibration can differ from the result's calibration. Whether an individual numerical method uses each such parameter is outside this review.
- Safe path: Consistent session/result parameters and unchanged calibration. The existing Strain status reads method/measure from the stored strain (`:478-483`), which is a useful truth source, but does not validate all current controls.
- Required design: Distinguish stored-result settings from editable next-computation settings; derive post-processing base metadata from the selected displacement result; snapshot and display units with texture outputs.

### A9 — P2: Export remains available for old or stale output without a clear export contract

- Evidence: Texture failure/cancellation at `texture_window.py:378-389` changes status/buttons but preserves results and enabled export/apply. Strain export is enabled solely by existing strain (`strain_window.py:426`) and is not gated or annotated for stale settings (`:293-297`) or recomputation. Both PNG save paths let file errors escape their slots (`texture_window.py:554-583`, `strain_window.py:487-495`).
- Scenario: Recompute with changed settings, encounter failure or cancellation, then export believing the revised analysis is represented.
- Impact: Old data is legitimately still present, but users cannot reliably distinguish export of the last successful calculation from the currently selected settings. A save-permission failure lacks an actionable in-window message. Strain's canvas helper creates parent directories (`field_canvas.py:207-213`); Texture's writers do not.
- Safe path: Export a known completed result without intervening changes. Retaining previous output on failure is useful if clearly labeled with its source/settings and completion time.
- Additional navigation defect: `app.py:418` requests strain preselection, but `app.py:447-450` passes `preselect_strain` only on first creation of the reused ExportDialog. Opening general export first, then entering through Strain, does not reapply that requested preselection.

## Additional bounded UX observations

- Empty/wrong-shape ROI silently becomes full volume (`texture_window.py:284-286`), contradicting the checked restriction. Stop with an actionable message instead.
- Toggling Texture's ROI checkbox has no status-refresh connection; Ready text can continue describing the previous scope until another event.
- Texture sweep convergence reasons exist in JSON (`texture_window.py:702-710`) but are not explained in the displayed sweep plot (`:523-553`).
- Strain's status update has no busy guard (`strain_window.py:469-483`): changing controls during work can overwrite Computing text with stale/no-strain text until the next progress event.
- Strain manual color limits are passed through without checking min < max (`strain_window.py:454`; `export/slice_plots.py:188`). Reversed limits can cause Matplotlib rendering errors rather than a field-level validation message. No rendering exception was executed in this extension.

## Recommended order

1. Prevent Strain cross-result commits and record worker-snapshot metadata.
2. Introduce source generations and stale-result contracts for both windows.
3. Unify cancellation and terminal UI cleanup, including shutdown.
4. Make next-run settings, anisotropic step, calibration provenance, and last-successful export explicit.

No product changes were made as part of this report.


---

# Extended 3D interaction audit

## Scope and evidence

Static source review only for this extension. No runtime actions, GUI probes, or new execution tests were performed. Product code was not changed. Paths below are repository-relative; line numbers refer to the source reviewed. "Definite" means the control flow follows directly from source, not that a GUI session reproduced it. "Risk" means the outcome additionally depends on timing, object destruction, rendering cost, or dataset size. Numerical algorithms are outside scope.

## High-priority interleaved scenarios

### E1. Hidden 3D playback keeps driving the visible application (P2, definite)

Scenario: select Frames, press Play, then switch to Slices to inspect a particular volume or edit its mask.

The timer continues running because the panel has a showEvent but no hideEvent or tab-deactivation handler. Each tick can still call state.set_current_frame, changing the globally selected volume and clearing its mask editor. Consequently the user cannot hold the visible Slices view on the desired frame while the hidden animation plays. Orbit/Slice/Warp also continue rendering hidden scenes, with possible responsiveness cost.

Evidence: `src/al_dvc/gui/panels/view3d.py:625-628`, `784-785`, `816-835`; `src/al_dvc/gui/app.py:87-91`; `src/al_dvc/gui/app_state.py:334-339`. The performance magnitude is a risk; continued timer execution and frame mutation are definite.

Recommendation: pause on deactivation, retaining the frame and explicit resume state. If background playback is intentional, expose a persistent global playback indicator and stop control.

### E2. Pausing, changing animation type, and resuming reuses the old clock and camera (P2, definite)

Scenario: play Orbit for several seconds, pause, choose Slice sweep or Frames, then press Play. Alternatively pause Orbit, rotate the camera manually, then resume.

The kind-change handler calls stop_animation only when `_playing` is true. Paused playback retains `_play_base` and `_play_offset`. toggle_play reuses them rather than creating a new base. The newly selected animation therefore starts at the old animation time instead of zero. A manually adjusted camera while paused updates `_live_state`, but resume still uses the previous `_play_base` camera, discarding that new view on the next tick.

Evidence: `src/al_dvc/gui/panels/view3d.py:610-623`, `752-765`, `767-771`, `773-785`.

Related case: changing kind while Frames is actively playing first changes the combo's selected kind, then invokes stop_animation. Restoration at `801-802` checks the new kind rather than the kind being stopped, so switching from Frames to Orbit does not restore the original frame.

Recommendation: explicitly store playback kind and distinguish pause/resume from a new animation session. Rebase or terminate paused playback when the animation type or camera changes.

### E3. Replacing results while paused leaves an old animation baseline attached to new data (P2, definite)

Scenario: pause an animation, load a different session/result or start and finish a new analysis, then press Play or Stop.

`_on_results_changed` clears playback only when currently playing, not when paused. It clears `_live_state` but leaves `_play_base` and `_play_offset`. Resuming can apply the old dataset's CameraState to a differently positioned/sized new dataset; stopping Frames can select the old start frame in the new dataset if that index remains valid. Out-of-range frame requests are safely ignored by AppState, but stale baseline reuse remains a definite defect.

Evidence: `src/al_dvc/gui/panels/view3d.py:568-574`, `782-785`, `799-805`; `src/al_dvc/gui/session.py:171-189`; `src/al_dvc/gui/app_state.py:334-339`, `366-372`.

Recommendation: clear every active or paused animation session whenever result identity changes, independently of `_playing`.

### E4. Volume deletion/reordering can pair old results with the wrong 3D image (P1/P2, definite state mismatch)

Scenario: complete an analysis, turn on Volume slices, then reorder or delete a volume. The volume UI allows those operations after results exist.

AppState removes/reorders volume entries without clearing or remapping `results`. The 3D panel resolves the field by current numerical frame index and the image by the current entry in the mutated volume list. The original result-to-volume identity is no longer guaranteed. Reordering the reference is especially misleading: existing displacement fields still describe the original run's reference. During Frames animation, deleted trailing entries additionally make some set_current_frame requests silently fail while the tick renders its computed result frame.

Evidence: `src/al_dvc/gui/app_state.py:131-155`, `319-339`; `src/al_dvc/gui/panels/volume_panel.py:214-219`, `234-249`; `src/al_dvc/gui/panels/view3d.py:485-503`, `521-525`, `555-557`, `823-832`.

Recommendation: invalidate results on sequence identity/order changes, or preserve a stable result-to-volume mapping and display the original analysis inputs. This is an application-state issue affecting 3D, not a numerical-core finding.

### E5. Recording completion updates a different/new session's controls (P2, definite)

Scenario: start recording, then use New session or load another session before the recording finishes.

The worker intentionally retains its original result and volume arguments, so replacing state alone does not invalidate those Python object references. However the callbacks are not scoped to the originating result/session. They unconditionally enable Record even when the new state has no results, and can restore `_last_info`/log a completed export into the new session context. The now-enabled Record button safely returns without action because `_on_record` checks results, but the UI communicates an available action that does nothing.

Evidence: `src/al_dvc/gui/panels/view3d.py:850-855`, `875-889`, `891-893`, `936`; `src/al_dvc/gui/app.py:314-318`; `src/al_dvc/gui/app_state.py:396-408`.

Recommendation: recompute enabled state in callbacks; associate recording notifications with the captured job/session. Define whether changing sessions cancels or explicitly backgrounds the export.

### E6. Camera interaction during playback is accepted visually but overwritten (P2, definite)

Scenario: play Orbit, drag the camera, choose a camera preset, adjust Turn/Tilt/Zoom, or press Home/Reset.

EndInteractionEvent returns immediately while `_playing`, so mouse camera movement is not retained. Numeric controls request a camera reset through invalidate, but invalidate returns during playback and the tick does not apply pending resets. The next tick reapplies the animation's old base camera. Controls remain enabled, suggesting these actions work.

Evidence: `src/al_dvc/gui/panels/view3d.py:563-564`, `587-608`, `610-613`, `769-771`, `828`, `920-934`.

Recommendation: either rebase animation on the new camera or temporarily disable conflicting controls with an explanation. Do not silently accept and discard input.

### E7. Recording can run concurrently with restarted preview; there is no recording stop action (P2 definite UI behavior, concurrency risk)

Scenario: start recording, press Play again, change modes/volume slices, or press Stop expecting recording to stop.

record() pauses existing playback, but disables only the Record button. Play remains available and toggle_play does not check `_recorder`. Stop only stops the preview timer and clears playback state. `_RecordWorker.cancel` exists but has no UI caller. Preview and offscreen export may then render concurrently. The export continues using captured settings while the live controls can show different settings; this is a snapshot export, not necessarily incorrect, but the UI does not clearly separate those states.

Evidence: `src/al_dvc/gui/panels/view3d.py:109-110`, `773-805`, `843-858`, `918-936`; `src/al_dvc/gui/view3d_animation.py:164-168`.

The concurrent OpenGL/VTK rendering safety and resource impact are risks, not established crashes. The absence of cancellation and continued playback availability are definite.

Recommendation: provide a dedicated Cancel recording action; explicitly label export as a snapshot of settings captured at start. Gate concurrent rendering if the backend cannot safely support it.

## Qualifications and supporting findings from the first pass

- Frames cumulative advancement remains a definite logic defect: `_play_frame` uses live options at `view3d.py:771`, while `824` writes the output back to state. The previously saved independent source-function probe is separate evidence from the earlier pass; it was not rerun for this static extension.
- Screenshot mismatch applies specifically to animated camera/slice/warp quantities. At `view3d.py:718`, export reconstructs the view from static controls/base_camera, while ticks render their frame-specific values at `828-835`. An ordinary non-animated screenshot is not implicated by this finding. Frames screenshot generally follows current frame because playback writes it back to state.
- Frames recording freezes the volume image: `view3d.py:851` captures one volume, and `view3d_animation.py:168` uses it across all result frames. This matters only with Volume slices enabled and different source volumes; overlays disabled or identical images do not expose it.
- Noncompatible mode/animation combinations are definite silent no-ops, not crashes: `view3d.py:752-765` does not enforce mode compatibility, and `view3d_scene.py:388-415` determines whether slice positions or warp_scale affect visible geometry.
- Recording shutdown is a risk: `app.py:567-581` omits recording cancellation/waiting. A QThread destruction abort depends on object teardown timing, so it should not be reported as an executed/inevitable crash. Incomplete recording on application exit is the direct lifecycle concern.
- Large GIF/MP4 exports retain every frame (`view3d_animation.py:171`, `177-182`). The UI admits up to 36,000 full-resolution frames; roughly 299 GB of raw RGB storage is possible before conversion overhead. Memory exhaustion depends on chosen settings and system resources. No such export was attempted.
- Record failure removes progress and enables the button but leaves the status text saying recording is underway (`view3d.py:859-860`, `886-889`); the log contains the actual error. Completion also never resumes preview despite the status promise at `860`.
- Dynamic speed/direction/axis changes do not rebase elapsed time: `frame_at` multiplies the new speed by total elapsed time (`view3d_animation.py:105-123`). This causes discontinuous jumps rather than changes beginning at the present position. It is definite behavior, but severity depends on whether such edits are promised to be continuous.

## Covered paths that have explicit protections

1. Missing results/backend: toggle_play returns early (`view3d.py:780-781`), screenshot returns None (`710-711`), and record rejects the request (`843-844`). A stale enabled button after recording does not by itself dereference missing results.
2. Results replaced during active playback: `_on_results_changed` stops it (`568-570`). The gap is the paused state, not the actively playing state.
3. Ordinary rendering errors: refresh catches and reports them (`656-659`); tick rendering catches errors, logs them, and stops (`836-838`). Frame construction is outside the tick's try block (`821`), so this protection should not be described as universal.
4. Recording exceptions: `_RecordWorker.run` catches ordinary exceptions and emits failed (`112-119`). It does not protect against process-level memory termination, a Qt teardown abort, or all graphics-driver failures.
5. Duplicate recording: record checks a running worker (`843`), preventing simultaneous jobs through repeated button invocation in normal operation.
6. Language changes preserve combo identity: `names.py:161-164` updates item labels rather than item data/current index. No evidence that translation selects another animation/mode or resets playback. Existing rendered text/status can retain its earlier language because retranslate_ui does not redraw the scene; this is cosmetic.
7. Fast Play/Pause/Stop calls are serialized on the Qt UI thread and share one QTimer (`view3d.py:773-805`). No source evidence of multiple preview timers being spawned. The known defects concern baseline/session transitions, not a demonstrated timer race.
8. Changing mode/arrows/outline/background while playing is intentionally read by the next tick (`view3d.py:563-564`, `767-771`). Rebuild-on-options-change at `826-827` supports these updates. Camera controls are the exception because the base camera is captured separately.
9. Out-of-range current-frame requests are ignored (`app_state.py:335`). This avoids an index-write error, but can hide the result/volume mismatch after deleting frames.
10. A recording keeps strong references to its original result/volume in `_args` (`view3d.py:106`, `850-851`). Merely assigning new state.results is not evidence of a use-after-free. In-place mutation by another component would require separate proof.

## Recommended validation scenarios for a future authorized GUI pass

These were not executed in this extension: (a) Frames -> switch to Slices -> select a frame; (b) Orbit -> Pause -> drag -> Play; (c) Frames -> Pause -> change kind -> Play; (d) Pause -> load differently sized result -> Stop/Play; (e) record -> New session -> wait for completion; (f) record -> Play -> Stop and verify export state; (g) reorder/delete a volume after analysis and compare volume identity; (h) close during a short recording in an isolated process. The last scenario should preserve user data and use disposable output.


---

# Static UI audit: sessions, batch, export, shutdown

Scope: source inspection only; no application execution, numerical validation, or product edits. P1 = data integrity/loss or serious lifecycle risk; P2 = incorrect workflow or materially misleading feedback. Source references are repository-relative, with one-based lines. Runtime consequences are stated as risks where timing matters.

## Findings

1. **P1: Opening a session during an active main run replaces its context.** `app.py:326` has no running guard, including recent-session and batch Open in window entry points. `session.py:165-186` replaces volumes, parameters, and results. `panels/run_panel.py:143-146` later installs the previous run's result without checking a session generation. Scenario: start A, open B, wait for A; B's UI receives A's results. Reject context replacement during dependent jobs or discard completion from an obsolete generation.

2. **P1: Batch CSV/VTK jobs sharing an output directory overwrite each other.** `batch.py:105,109` omits the supplied session basename, so exporters use `aldvc` for every job (`export/export_csv.py:17,36`; `export/export_vtk.py:54,84,87`). Different session names protect NPZ but not CSV/VTK; fewer frames leave older trailing files. Main export can target the same paths concurrently. Use per-job output folders, pass basename consistently, preflight collisions, and coordinate writers.

3. **P1: Shutdown does not coordinate export, and ignores worker wait timeouts.** Export workers have dialog parents (`dialogs/export_dialog.py:422`), no cancellation/close protocol; `app.py:567-580` waits only batch/main workers and accepts closing even if 60-second waits return false. Batch dialog also blocks its GUI thread then closes without checking wait success (`dialogs/batch_dialog.py:312-316`). Scenario: export a large report then quit, or close during a long noninterruptible batch step. Risk: unfinished output and live-thread destruction on teardown; not runtime-confirmed crash. Use a common asynchronous shutdown coordinator; keep the window alive until workers actually finish. Ordinary export-dialog Close only hides the dialog by default, so it should not be described as guaranteed thread destruction.

4. **P1: New/Open/Quit lose unsaved session edits without a dirty-state prompt.** `app.py:315-344,567-580` resets/replaces/closes without asking whether to save edited parameters or ROI drawing. A run-stop question is not a save question. Track document dirty state and offer Save/Discard/Cancel on context destruction.

5. **P2: Batch cannot replay saved threshold masks.** `batch.py:77` calls `MaskEditor.from_dict` without volume intensities, whereas GUI restore supplies them (`session.py:150-156`). `mask_editor.py:270-272` explicitly rejects threshold operations without a volume. Scenario: segment ROI by threshold, save, add to batch; otherwise valid GUI session fails in batch. Pass the loaded volume into mask reconstruction.

6. **P2: Saving and reopening a session does not restore results.** `session.py:171` clears results; `session.py:193-199` only reads an NPZ to log its array count. `app.py:358-363` only associates a hardcoded `aldvc.npz`, missing custom and batch basenames and potentially pointing at an unrelated older file. Users expect Save/Open or batch Open in window to reopen their analysis. Provide explicit result restoration/import, persist the actual exported artifact identity, and make configuration-only saving explicit until supported.

7. **P2: Invalid session data can partially replace the current document.** `load_session` validates only top-level keys/parameter conversion (`session.py:109-138`), while `apply_session` starts mutating state before parsing display numbers (`165-184`). A file with an invalid display frame/color value raises after volumes/results are replaced; `app.py:330` catches only SessionError. Validate all schema fields and build candidate state before committing. Also report unsupported format versions, currently written but unchecked.

8. **P2: Session write failures lack a controlled error path and may damage an existing save.** `session.py:103-104` directly writes the destination, without atomic replacement or wrapping OSError, but caller handles only SessionError (`app.py:366`). Disk full, permission denial, or interrupted save can yield an unhandled error or truncated file. Write a temporary sibling, replace atomically, and translate IO failures into actionable UI feedback.

9. **P2: Export controls remain editable and completion reports the wrong folder.** `dialogs/export_dialog.py:405-430` snapshots config and only disables Export. User changes destination during work; worker writes old destination but completion and log read current `folder.text()` (`439-443`), and Open folder also reads it (`299`). Disable job-specific controls or distinguish next-job settings; completion must display captured job destination and result identity.

10. **P2: Export selection does not consistently mean what it displays.** Clicking None still exports displacement because of fallback (`export_dialog.py:85-87`). Frame restriction affects CSV/VTK/images but not NPZ/MAT/PDF (`90-106`); NPZ's tooltip says everything, but the shared Frames control does not clearly label scope and report ignores it. Reopening refresh also resets frame_to to the last frame (`342-343`). Require fields where relevant, label full-archive/report behavior, and preserve valid frame choices.

11. **P2: Export base name accepts paths and all outputs overwrite without a collision workflow.** `export_dialog.py:377` accepts arbitrary trimmed input, then joins it directly (`90-100`); a pasted absolute path or `../name` can escape the selected folder. This is a local usability/data-loss issue, not a remote security claim. Validate a leaf filename, preview actual output paths, and ask overwrite/choose another name when files exist.

12. **P2: Partial export failure does not enumerate completed outputs.** `run_export` writes formats sequentially and exits on first exception (`export_dialog.py:88-118`), while `_on_failed` reports only generic failure (`445-449`). Batch assigns outputs only on complete return (`batch.py:145`), likewise losing written-path metadata on a later export failure. Users retry and overwrite earlier files without knowing what succeeded. Keep per-format outcome records and retry only failed outputs.

13. **P2: Batch Add current session actually queues the saved disk version.** `batch_dialog.py:202-204` adds session_path; `batch.py:127` later reloads that path. Unsaved parameter/ROI edits are ignored, and files edited while waiting affect queued jobs. Explicitly offer a saved snapshot or label disk-version behavior. Options also remain editable while a batch runs but worker configuration was captured at `batch_dialog.py:233`; disable them or label next-run settings.

14. **P2: Batch controls may remain disabled after completion.** `finished_all` is emitted inside QThread.run (`batch_dialog.py:59-61`); its UI slot calls `_update_buttons` (`282-294`), which asks `isRunning` (`251-252`). Delivery before run returns leaves Start disabled and Stop enabled, with no native QThread.finished connection to update again. Timing-dependent static risk. Connect final UI enablement to native thread termination.

## Existing safeguards and bounded safe paths

- Export start and batch start explicitly reject a running worker, limiting double-click duplication (`export_dialog.py:405-408`; `batch_dialog.py:215-217`).
- Batch locks add/remove/clear/open queue buttons during work (`batch_dialog.py:304-310`); normal UI clicks do not mutate indexed rows mid-run.
- Export captures result/config/background references before dispatch, so replacing state.results does not by itself redirect its worker to the new result. Its displayed status and destination can still be misleading.
- Batch records ordinary per-session failures and continues subsequent jobs (`batch.py:152-156,190-215`); stop skips later jobs. This does not solve partial-file accounting.
- New session is guarded during main RUNNING/STOPPING, but Open/recent Open lack the same guard.
- Different batch output folders avoid the demonstrated cross-session filename collision; independent jobs are not intrinsically unsafe merely because they overlap.

## Additional scenario coverage

Reviewed repeat Start/Export/Stop, closing child versus main window, changing destination/settings while exporting, opening sessions through all visible entry points, selecting no fields/formats, reversed frame ranges, saved versus unsaved batch inputs, duplicate jobs, malformed/missing files, shared output folders, and partial write failures. Empty format selection is rejected in interactive export; reversed frame bounds are sorted intentionally. Batch allows no exports (compute-only), which should be explicitly described because results then have no retained export artifact.


---

# Future acceptance matrix

The following checks are specified for future implementation/acceptance, not executed in this static audit. Use disposable inputs and output folders when testing overwrite and shutdown cases. Cover one-result and multiple-result datasets, equal-shaped but different images, different shapes, partial results, and anisotropic calibration.

| ID | Action sequence | Required observable outcome |
|---|---|---|
| T01 | Rapidly click Run/Run/Stop/Run | One worker; stopping state prevents restart until termination; partial result clearly identified |
| T02 | Run A; open B through File, Recent, or batch Open | Transition blocked/coordinated, or A completion discarded/retained separately; never A results in B |
| T03 | Complete A; reorder/delete/add volumes | Preserve stable result mapping or mark results stale/unavailable; no substitution by row index |
| T04 | Stop after one result; select a later uncomputed frame | Show no result for that frame; do not repeat the last computed field |
| T05 | Run with a drawn mask; edit/invert/clear it | Active run's captured mask is immutable, or editing is explicitly unavailable |
| T06 | Start Strain A; new DVC finishes as B; old Strain finishes | B remains uncontaminated; old completion has an explicit obsolete outcome |
| T07 | Compute Strain; change method/measure/smoothing | Stored metadata matches actual computation snapshot; new controls remain marked stale |
| T08 | Cancel Texture without sweep; cancel during final Strain frame | No success publication after accepted cancellation; terminal buttons and progress settle |
| T09 | Analyse Texture; change ROI/data/spacing; Apply | Recommendation is blocked as stale with an explanation |
| T10 | Analyse Texture successfully; retry with failure; export | Clearly export the last successful result and its provenance, or require a fresh result |
| T11 | Restrict Texture to empty/wrong-shape/missing-file ROI | Actionable input error; no silent whole-volume fallback |
| T12 | Apply step (4,4,10); inspect/edit step | All axes visible; edits preserve intended anisotropy or explicitly link axes |
| T13 | Frames Play; switch to Slices; select/draw a frame | Hidden animation cannot silently change the selected frame or cancel the ROI gesture |
| T14 | Orbit Play; drag/preset/zoom/reset | Predictable rebase or explicit disabled action; no accepted-then-overwritten input |
| T15 | Play; Pause; change animation kind; Play | New kind starts with a valid new baseline/time |
| T16 | Pause; rotate camera; Play | Resume contract preserves/rebases the edited view |
| T17 | Pause A; replace result with differently sized B; Play/Stop | No old camera/time/frame baseline survives into B |
| T18 | Frames Play at a chosen fps over several UI ticks | Result frame depends on elapsed time from fixed start, not cumulative refresh count |
| T19 | Pause Orbit/Slice/Warp; Screenshot | Saved view matches the actual paused camera/geometry/slice state |
| T20 | Record Frames with different Volume slices | Each exported field uses its matching source image |
| T21 | Choose Warp animation in Slices; Slice animation in Points | Compatible mode selected or explanatory no-op prevention |
| T22 | Record; Play again; Stop preview; Cancel recording | Separate preview/record states and controls; recording cancellation is available and bounded |
| T23 | Request long high-resolution GIF/MP4 | Bounded memory/streaming, clear resource estimate, responsive cancellation |
| T24 | Record/export; New session; job completes/fails | Notification identifies original job; current controls reflect current eligibility |
| T25 | Start polygon; change slice/mode/depth/target/tool; Enter | Gesture cancelled or committed with explicit captured context |
| T26 | Draw A; switch away/back; add B; save/open | Exact ROI preserved, including A |
| T27 | Invert ROI; Save mask; Save session; Open | No double application; mask bitwise identical |
| T28 | Different masks on frames; change target to All | No hidden irreversible overwrite; explicit copy transaction is reversible |
| T29 | Load saved manual color range/alpha/lattice toggles | Widgets and rendered state agree immediately |
| T30 | Set min >= max; change result units/calibration | Inline validation and preserved valid view; numeric values and labels use same provenance |
| T31 | Export; change folder/fields/frame range mid-job | Output and completion use captured config; next-job edits clearly distinguished |
| T32 | Export None fields; select subset of frames for each format | No silent fallback; unsupported selection scope explicitly labeled |
| T33 | Batch two sessions to same folder; export concurrently | No filename collision or mixed stale frame files |
| T34 | Auto-threshold ROI; save; batch-run same session | Batch reconstructs the same mask as GUI loading |
| T35 | Add current session with unsaved edits to batch | Explicit snapshot or clear saved-version warning; no silent ignored edits |
| T36 | Save/Open after completed run and custom export basename | Exact result restoration or explicit configuration-only contract |
| T37 | Open malformed session; save to unwritable/full disk | Old document intact; no truncated previous save; actionable failure |
| T38 | Multi-format export fails after one successful format | Completed/failed paths enumerated; retry scope clear |
| T39 | Edit ROI/parameters; New/Open/Quit; cancel prompt | Original document remains unchanged and available |
| T40 | Close child analysis/export window; reopen | Clearly defined hide/continue policy; progress and ownership remain visible |
| T41 | Quit with DVC/Strain/Texture/export/record/batch active | All jobs settled asynchronously; wait timeout cannot authorize unsafe teardown |
| T42 | Finish batch near thread-return boundary | Buttons reach idle on native thread completion, not an earlier custom signal |
| T43 | Change language while playing or working | Stable selected keys, job state, and source identity; translated feedback refreshes coherently |
| T44 | Resize/hide/reopen panels; scroll over numeric controls | No unintended unfocused value edits; actions remain discoverable; actual layout requires a future visual check |

## Remediation sequence

1. **Establish data ownership.** Introduce session/input/result revisions and immutable job snapshots. Every completion validates its originating revision before publishing. Result fields, units, volumes, ROI, and analysis settings travel together. Never use a current widget value to describe an older computation.
2. **Repair persistence and file ownership.** Preserve exact ROI masks, make session save atomic, validate a full candidate session before replacing state, and use actual result artifact identities. Isolate output namespaces and enumerate partial success.
3. **Centralize job lifecycle.** Model idle/running/cancelling/succeeded/failed/obsolete states. Use native worker termination for cleanup. Define close-as-hide versus close-as-cancel and coordinate all workers on application shutdown. Do not block the GUI for a fixed long wait.
4. **Unify animation state.** Store animation kind, source generation, start frame/camera, elapsed time, and last rendered frame. Clear/rebase on context changes; pause hidden playback by default. Screenshot the current rendered state; record against a captured sequence of matching inputs.
5. **Make editing contracts visible.** Separate active-job settings from next-job settings, expose all anisotropic axes, show stale/source labels, validate paired limits, and distinguish Empty ROI from No ROI. Include clear destination and collision previews for exports.
6. **Add targeted acceptance coverage when implementation begins.** Prioritize the matrix above over isolated button happy paths. Existing tests are useful but do not prove these cross-window sequences; this review did not run them.

## Completion record

The requested expanded static audit and documentation are complete. Product source and numerical code were not modified. The extension produced review notes and this consolidated document only. Remaining runtime/visual acceptance is explicitly outside the user's requested scope, not a claim that all reported risks have been observed on a running application.

