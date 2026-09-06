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
