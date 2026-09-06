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
