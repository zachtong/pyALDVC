# ROI persistence findings

## M01 — P1/Definite: Switching frames and continuing to draw loses part of the saved ROI

Scenario: draw region A on frame 0; switch to another frame and back; add region B; save and reopen the session. `AppState.set_current_frame` drops the editor (`src/al_dvc/gui/app_state.py:334-338`). `ensure_mask_editor` creates a new editor using the composed current mask as its base, without restoring the previous operation history (`:202-226`). `_push_mask` replaces the stored mask_ops with only the new editor's operations (`:232-240`). `MaskEditor.to_dict` explicitly omits the base (`src/al_dvc/gui/mask_editor.py:342-344`). Session restore rebuilds only those operations over a file base or an empty base (`src/al_dvc/gui/session.py:140-160`). Thus earlier in-memory region A is absent after reopening, even though A+B was visible at save time. A Replace operation may hide the loss; additive/cut workflows expose it.

The same missing-base persistence contract affects history folding in `_replay` (`mask_editor.py:304-314`): older operations can be merged into an unserialized base. The full-base programmatic reset also illustrates the contract gap, but is not claimed as a separate visible button workflow.

Fix: serialize an exact base plus operations, or save the composed mask as the authoritative snapshot. Keep undo history separate from persistence requirements. Changing frames must not make an otherwise saved ROI irrecoverable. Acceptance: saved/reopened masks are bitwise identical after multi-frame editing and long undo histories.

## M02 — P1/Definite: Saving an edited mask and reopening a session can apply operations twice

Scenario: edit/invert a mask, click Save mask, save the session, reopen it. `AppState.save_mask` writes the already composed current mask and assigns its file path, but retains mask_ops (`app_state.py:278-290`). Session restore loads that composed file as the base and replays the retained operations (`session.py:149-159`). Invert is non-idempotent: the reopened ROI can become its opposite. Add/Replace can conceal this defect, so an ordinary rectangle-only save path does not establish safety.

Fix: when saving a composed mask as the base, reset/rebase persisted operations consistently; or persist the original unmodified base separately. All-frame mask propagation must preserve each target's exact state and reversibility. A successful file write is not sufficient proof that Save/Open preserves the ROI.

