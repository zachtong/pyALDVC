# Discontinuity diagnostics: the design after it was measured

Status: **designed, not implemented.** 2026-09-09. This is the record of two designs I proposed and
four independent critiques that read the real code and measured it. Both designs survived in
substance; three of the four details of each did not. Do not implement from my original sketch --
implement from the "corrected design" sections below.

Full critique transcripts (four lenses plus a synthesis) are in the workflow run
`wf_9d46997c-0a3` under `.claude/projects/.../subagents/workflows/`. Probe scripts the critics wrote
are in that session's scratchpad and were not kept.

## A. Flagging subsets that straddle a discontinuity

### What I proposed, and why it is wrong

Flag a node when a mask boundary is inside its subset window but splitting removed nothing:
`unsplit = (C < Vc) & (split_fraction == 1.0)`, with `C` the in-mask voxel count and `Vc` the
clipped window volume.

Two measured objections, raised independently by three critics:

1. **`Vc` is dead arithmetic.** `build_grid_axes` (`src/al_dvc/mesh/grid_mesh.py:43-45`) places
   nodes at `border = GRADIENT_BORDER(3) + INTERP_MARGIN(2) = 5` voxels from every face, so no
   window this pipeline can build ever overhangs the volume. `C < Vc` is therefore *identically*
   `frac < 1.0`, which is already recorded as `split_index >= 0`. Computing it would cost a second
   full sweep of the mask -- the exact cost the z-slab rewrite of `subset_valid_fraction` exists to
   avoid. The same border argument kills both confounders I worried about: window clipping and the
   3-voxel on-the-fly gradient `margin` are dead paths for pipeline meshes (verified by identical
   counts for `stored` and `on_the_fly`).

2. **What remains is a surface detector, not a discontinuity detector.** "A boundary is in the
   window and the in-mask voxels are one connected piece" is the *definition* of an ordinary
   specimen-surface node. Measured on the real `subset_valid_fraction` / `split_rows` path:

   | mask | valid nodes | flagged | comment |
   |---|---|---|---|
   | plain cylindrical specimen, no crack | 865 | 490 (57 %) | all false positives |
   | convex ROI box | 785 | 380 (48 %) | all false positives |
   | isolated pore r=4 | -- | 124 | false positives |
   | excluded needle r=2 | -- | 120 | grey: non-affine but continuous |
   | cylinder + one internal crack front | -- | 520, of which 80 straddle | precision **0.15** |

   A second critic measured 336/672 (50 %) on its own cylinder case. The two agree.

   The reason is structural: **window-local connectivity can never see a crack front, because the
   front is exactly where the two sides are still joined.**

3. **`split_fraction == 1.0` is a built-in false negative.** A node that was partly cut but whose
   kept component still straddles a discontinuity (a branched crack, a second front on the kept
   side, a front plus the specimen surface in one window) can never be flagged.

### Corrected design

Ship a **continuous straddle measure**, computed on the **kept** voxel set, independent of
`split_fraction`:

- `B` = out-of-mask voxels 6-adjacent to in-mask voxels inside the window.
- PCA of `B` -> eigenvalues `l1>=l2>=l3`; sheet normal = smallest eigenvector, thickness
  `sqrt(l3)`, planarity `l3/l2`.
- Split the kept voxels (already packed in `split_keep`) by the sign of their signed distance to
  that sheet; `straddle_fraction` = the minority-side fraction, 0..0.5.

Measured by the critic that proposed it, on the same masks: **0 flags** for the specimen surface,
the ROI box, the pore, the needle and a window merely touching a crack face; **60 flags with zero
false positives and 0.75 recall** on the crack front. Cost is one extra pass plus a 3x3 `eigh` per
candidate -- negligible beside IC-GN. A cheaper corroborating signal ("flagged and within one node
step of a node with `split_fraction < 1`") was also measured: 0 false positives but recall only
0.25, so it is a fallback, not the primary.

A geodesic-detour test (BFS distance minus L1 inside the window) was measured and **does not work**:
6-connected free space makes the excess exactly 0 whenever a monotone staircase path exists, so
front nodes score 0. Do not implement it.

Required properties:

- Computed **after** the tile loop, in `precompute_local_context`, with whole-volume coordinates;
  per-tile numbers carried back in `split_parts` the way `n_keep` / `n_inmask` already are.
- **`None` when splitting did not run** (subset_split off, or any tile over `MAX_SPLIT_BYTES`) and
  `NaN` at invalid nodes. A `0 flagged` summary must never be printable when the answer is unknown.
- Documented as a **mask-geometry diagnostic, not a data diagnostic**: it can only see
  discontinuities the segmentation already resolved. Invisible to it: a crack whose faces are in
  contact, an unsegmented crack, a debonded interface, slip on a grain boundary, a shear band, and a
  crack thinner than the segmentation. Pair it with the per-node ZNCC and `U_std` for those.
- It is a property of the **reference geometry**, not of the frame, so under the default accumulative
  schedule it cannot see a front that appears after frame 0.
- The report page must include the smooth-surface control (which must flag ~0) and must **stratify by
  distance to the front**: an averaged flagged-vs-unflagged comparison at precision 0.15 would show
  no difference and appear to disprove a real effect.

The risk a critic named and I agree with: a boolean called `unsplit` that lights up half the surface
of every masked specimen will be ignored, or worse used downstream as "bad node" and drive people to
mask away good data. Ship the continuous measure with the measured true/false-positive table above.

## B. Crack-opening displacement at cut edges

### What I proposed, and what is wrong with it

`COD = (U[hi] - U[lo]) - 0.5*(F[lo]+F[hi]) @ (x_hi - x_lo)` over every edge with `edge_ok == False`
whose both endpoints are valid, decomposed into edge-parallel (mode I) and perpendicular (II/III).

1. **"Both endpoints valid" is a tautology.** `mesh_cut.py:119` returns
   `~((n_total > 0) & (n_bridge == n_total))`, so an edge in no live element is reported **True**
   (uncut) -- my brief had this backwards. And elements with any invalid corner are dropped
   *before* `cut_mesh` (`grid_mesh.py:221-226`), so `edge_ok == False` already implies both
   endpoints valid. The design therefore shipped with **no test that the void actually separates
   these two nodes**. What `edge_ok == False` really means is element-level: every all-corners-valid
   element containing the edge is *bridging* -- and an element can be bridging because two other
   corners are separated while this edge is solid material.
2. **The bulk term is the `t=1/2` special case** of the first-order expression and charges bulk
   stretch to the void. Residual bias `(F_hi-F_lo)·dx·(1/2-t)` plus `|F|·(t2-t1)|dx|`; with the
   repo's own fixture (8-voxel node step, 2-voxel wall) the gap term alone is ~0.04 voxels at 2 %
   strain against a ~0.03-voxel noise floor, and it dominates near a tip. The orientation `F @ dx`
   is correct (`einsum('nij,nj->ni')`, no transpose).
3. **Edge-direction decomposition is not mode I/II/III.** The edge axis is a grid artefact; the
   crack normal is not. II and III cannot be separated at all without the crack-front tangent.
4. **Inpainted displacements are indistinguishable from measured ones**, and they cluster exactly at
   the crack: `local_icgn` overwrites bad nodes by spring inpainting and `status` stays
   `STATUS_CONVERGED` for a median-test outlier; `subpb1` replaces bad nodes' `U` with `U_hat` and
   the caller discards its `bad` (`pipeline.py:323`). `FrameResult` has no `filled` flag.
5. **The per-edge set is empty exactly where the opening is largest**: a crack wide enough to
   invalidate a node column drops every element touching it, so those edges are never "cut".

### Corrected design

One cheap addition fixes items 1, 2 and part of 5: **walk the `h+1` voxel line between the two
rounded node centres** (`segment_crosses_mask`, planned in
`docs/plans/2026-09-07-subset-splitting.md:189` and never written). That single walk

- proves the void separates *these two nodes* (rejecting the element-level false positives),
- yields `t1` / `t2`, so `COD = dU - F_lo @ (t1*dx) - F_hi @ ((1-t2)*dx)`, falling back to
  `t1=t2=0.5` only when the walk finds no out-of-mask voxel (and then flagging the edge),
- gives the absolute reference gap.

Report the disagreement count between `~edge_ok` and the segment test rather than assuming it is
zero.

Also required:

- Ship the **COD vector in [x,y,z] plus its magnitude** as the primary output: basis-free and honest.
  A mode-I number needs an estimated normal (PCA of the out-of-mask voxels near the edge midpoint,
  with `l3/l2` as a planarity score, sign fixed by `n·dx>0`, converted as a covector
  `n_i ∝ n_vox,i/s_i` then renormalised). Never claim II vs III.
- Read `U` and `F` from **`FrameResult`** (voxel units), not `StrainResult` (whose `F` is already
  scaled by `s_i/s_j`, so mixing it with a voxel `dx` is wrong under anisotropic `voxel_size`) and
  not `U_accum`. Warn when `para.disp_smoothing > 0`: `smooth_grid_field` takes no `edge_ok` and
  smooths straight across the cut, i.e. across the jump being measured.
- Add a per-node `filled` flag to `FrameResult` and gate every COD on `~filled` at both ends plus
  finite `U_std`. Do **not** gate on the plane fit's `complete`: it is False at every cut node by
  construction and would delete the whole field.
- `gradient_fd` can leave a whole column of `F` as NaN (a node cut on both its `-j` and `+j` edge),
  so emit `dU` and `cod` separately with a per-edge `bulk_corrected` flag; an uncorrected COD is
  still usable, a NaN is not.
- `edge_ok` is `(N,3)` node-indexed; `edges_ok_grid()` returns **None** unless something was cut. Use
  `mesh.edges_ok()` or handle None with a logged reason, never crash or emit zeros.
- The exported result carries only frame 0's cut topology (`pipeline.py:433`), so a crack that grows
  during the series has no COD on its new edges. Compute per frame against that frame's own mesh
  where `mesh_by_frame` has one, and log which topology was used.

## Also found, not part of either design

- **An existing defect**: when one tile blows `MAX_SPLIT_BYTES`, splitting is dropped for the whole
  frame and `split_fraction` becomes `None`, with `split_used = False` set once. Worth fixing in the
  same change as A, since A must handle the same path.
- `local_split` / `merge_split` do **not** need to learn about a new per-node array; the tiled merge
  is correct only because the halo guarantees a node's window never straddles a box face, and the
  new statistic must be computed against the tile's own mask shape and tile-local coordinates.
- `clean_initial_guess` already routes cut nodes around the boundary and skips exactly the nodes A
  would flag -- the one actionable in-solver consumer, missing from my plan.
- Making either quantity displayable in the GUI is a six-file, six-language change, and a boolean
  cannot be shipped as a displayable field.
