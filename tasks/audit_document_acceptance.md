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
