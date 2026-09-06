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

