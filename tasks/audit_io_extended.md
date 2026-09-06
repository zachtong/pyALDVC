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
