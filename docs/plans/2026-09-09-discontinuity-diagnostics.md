# Discontinuity diagnostics: the design after it was measured

Status: **designed, not implemented.** 2026-09-09.

This is the record of two designs I sketched, four independent critiques that read the real code and
measured it, and a synthesis that re-verified the critiques' own claims before merging them. Both
designs survived in substance; most of the details did not. **Do not implement from my original
sketch** -- implement from §A′ and §B′.

The critique transcripts are in workflow run `wf_9d46997c-0a3` under
`.claude/projects/.../subagents/workflows/`. Probe scripts were written to the session scratchpad and
not kept; the measured tables below are what survives, and every one of them should be reproduced by
the report pages in §5 rather than trusted from here.

## Verified before use

The synthesis re-ran each load-bearing claim rather than accepting it. Results, including two that
contradict the critiques and two that contradict an earlier version of this file:

| claim | verdict |
|---|---|
| `C < Vc` is equivalent to `frac < 1.0` | **True.** `build_grid_axes` (`grid_mesh.py:43-45`) uses `border = GRADIENT_BORDER(3) + INTERP_MARGIN(2)`; over six masks at 80x160x160, win 16, `clipped_windows = 0` in every case. `Vc` is dead arithmetic. |
| the naive flag fires on defect-free masks | **True.** Plain cylinder 855 of 2790 valid nodes (31 %); ROI box 1464 of 3487 (42 %); needle 927; pore 916 -- with zero splits anywhere in those masks. |
| the `margin` lowers `split_fraction` | **False.** `numba_kernels.py:247-260` builds `sub` *with* the margin and `:272-273` counts `n_inmask` from that same `sub`, so it cancels. (An earlier version of this file said the margin was merely "a dead path"; the real reason is cancellation, and the band is unreachable as well.) |
| `edge_ok == False` covers "no element here at all" | **False.** `mesh_cut.py:119` returns `~((n_total>0) & (n_bridge==n_total))`, so `n_total==0` gives **True**, and `grid_mesh.py:221` drops invalid-corner elements *before* `cut_mesh`, so every `edge_ok==False` edge already has both endpoints valid. My "both endpoints valid" gate was a tautology. |
| `edge_ok` is empty where the crack is open | **True, and worse than the critiques said.** At 64x96x96, win 16, step 8: a crack lying **on** a node plane gives **0** cut edges; thickness 7 gives **0**; a 45-degree one-voxel crack gives **0**. |
| `split_fraction == 1.0` can hide a real disconnection | **True.** A one-voxel wall at the window's inner edge, half 8: full-resolution kept/in-mask = 4335/4624 = **0.9375**, while the sampled `split_fraction` reads 0.8889 at stride 2 and **1.0000 at stride 3**. `split_fraction` is a sampled-lattice ratio and is not a topology statement. |
| the split byte-budget bail-out corrupts already-split tiles | **True, and it is a correctness defect, not a reporting one.** `local_icgn.py:212-218` sets `split_used=False` *after* earlier tiles' `H_all`/`meanf`/`bottomf`/`n_valid` were already built with `split_kw`, and `:258-259` then drops `split_index`/`split_keep` for the whole reference -- so those nodes correlate a voxel set their normalisation was not built from. Pre-existing; fix it first and alone. |
| `F @ dx` orientation | **Correct as written.** `gradient_methods.py:149` sets `F[...,c,j] = coef[1+j,c]`, i.e. `du_c/dx_j`. |

## Why my original sketch fails

**Design A (proposed).** Flag a node when a mask boundary is inside its subset window but splitting
removed nothing: `unsplit = (C < Vc) & (split_fraction == 1.0)`.

1. `Vc` is dead arithmetic (above), so the first half reduces to the candidate test the code already
   has (`split_index >= 0`) -- and computing it would cost a second full sweep of the mask, the exact
   cost the z-slab rewrite of `subset_valid_fraction` exists to avoid.
2. What remains is a **specimen-surface detector**: "a boundary crosses the window and the in-mask
   voxels are one connected piece" is the definition of an ordinary surface node. 31-42 % of valid
   nodes on masks with no discontinuity anywhere. The reason is structural -- **window-local
   connectivity can never see a crack front, because the front is exactly where the two sides are
   still joined.**
3. `split_fraction == 1.0` is also a built-in false negative: a node that was partly cut but whose
   kept component still straddles a discontinuity (a branched crack, a second front on the kept side)
   can never be flagged.

**Design B (proposed).** `COD = dU - 0.5*(F[lo]+F[hi]) @ d` over every edge with `edge_ok == False`
whose both endpoints are valid, decomposed along the edge into mode I and modes II/III.

1. "Both endpoints valid" is a tautology (above), so the design shipped with **no test that the void
   separates these two nodes at all**.
2. `edge_ok` is *empty* in three common geometries (above), i.e. the per-edge set is missing exactly
   where the opening is largest.
3. The symmetric bulk term is the `t1 = t2 = 1/2` special case and charges bulk stretch to the void.
   Measured `t1` on real meshes: **0.375 and 0.500** for the same crack moved one voxel.
4. Edge-direction decomposition is not a mode I/II/III split: the edge axis is a lattice artefact and
   the crack normal is not -- a 45-degree crack cuts no edges at all.
5. Inpainted displacements are indistinguishable from measured ones and cluster at the crack:
   `local_icgn.py:431` computes `bad` and throws it away before inpainting at `:442`; `FrameResult`
   has no `filled` flag.

## A′. Ship `straddle_fraction`, not `unsplit`

Candidates are unchanged (`node_valid & (frac < 1.0)`, `local_icgn.py:154-155`) -- **no second mask
sweep**. For each candidate, inside the full-resolution window `W = [c-h, c+h]^3` (out-of-volume
voxels count as out-of-mask, as `_build_split_rows_jit` already does):

```
sub    = in-mask indicator on W                      (already built)
keep   = 6-connected component of sub at the centre  (already built, full resolution)
B      = { v in W : sub[v]==0 and v is 6-adjacent to some u in W with sub[u]==1 }
if |B| < 8:                          straddle = 0
mu     = mean(B offsets, voxels);  C = cov(B offsets)
l1<=l2<=l3, e1 = eigenvector of l1   (host-side batched np.linalg.eigh over (M,3,3))
t      = sqrt(l1)                    # sheet thickness
a      = sqrt(l2)                    # sheet extent
if t > 1.5 or a < 3.0*max(t, 0.5):   straddle = 0    # thick slab / shell / rod, not a sheet
s(v)   = (v - mu) . e1  for every v with keep[v] != 0
n_plus = #{ s > +0.5 };  n_minus = #{ s < -0.5 }
straddle_fraction[n] = min(n_plus, n_minus) / n_keep_fullres          # in [0, 0.5]
```

Elsewhere: `0.0` at valid non-candidate nodes; `NaN` at nodes the precompute rejected (matching
`split_fraction[~valid] = NaN`); the **whole array `None`** when splitting did not run (no mask,
`subset_split` off, byte budget exceeded), logged as "not measured" and **never as 0**.

Thresholds are module constants (`STRADDLE_B_MIN=8`, `STRADDLE_T_MAX=1.5`, `STRADDLE_PLANAR_MIN=3.0`,
`STRADDLE_BAND=0.5`), **not** `DVCPara` fields -- see out-of-scope item 7.

### Measured separation (repo code, 80x160x160, win 16, step 6)

| mask | valid | candidates | split (sf<1) | naive `unsplit` | `straddle > 0.08` |
|---|---|---|---|---|---|
| plain cylinder | 2790 | 855 | 0 | **855 (31 %)** | **0** |
| ROI box | 3487 | 1464 | 0 | **1464 (42 %)** | **0** |
| cylinder + pore r=5 | 2789 | 917 | 1 | 916 | **0** (max 0.050) |
| cylinder + needle r=1 | 2781 | 927 | 0 | 927 | **0** |
| cylinder + through crack | 2592 | 1116 | 342 | 774 | **0** |
| cylinder + crack front x=80 | 2700 | 1035 | 144 | 891 | **18** (x = 73...79) |

Front-case quantiles of the non-zero straddle: median 0.096, p90 0.131. The pore's 0.050 peak sits
just under the shipped threshold, which is thin -- page 4 of the report sweeps it.

### What it separates, and what it does not

Separates *"a thin planar boundary crosses the window and correlated voxels lie on both sides of it"*
from *"the window merely touches a boundary"*. Computed on the **kept** set independently of
`split_fraction`, so a partly-cut subset that still straddles is caught.

Does **not** separate a crack front from any other terminating thin sheet -- a notch tip, a
delamination edge, an unresolved lamella. All are equally non-affine, so lumping them is right, but
**the field must not be named `crack_front`**. It says nothing about which side is which, nor about
the size of the jump.

False positives: concave planar grooves thinner than 1.5 voxels with material on both sides of the
fitted plane (0 in the cases probed, not provably 0); a real but perfectly bonded thin interface; an
oblate pore flat enough to pass the sheet gate.

False negatives: anything with **no mask boundary** -- a crack whose faces touch, an unsegmented
crack, a debonded interface, a shear band (candidates are `frac < 1.0`, so a solid mask yields zero
candidates); boundaries thicker than 1.5 voxels or shorter than `3*t` inside the window; the outer
ring where the minority side falls below threshold (36 nodes non-zero in the front case, 18 above
0.08); and any run where splitting did not execute -- reported as unknown.

## B′. `crack_opening`

### Pair search, replacing `edge_ok == False` as the key

For each valid node `lo` and axis `a`, walk `+a` to the nearest valid node `hi` within
`COD_MAX_STEPS = 2` node steps, requiring every skipped intermediate node to be **invalid** so no
pair is double-counted. Accept when the two rounded centres are **not 6-connected** inside the box
spanning them padded by `half[a]` voxels -- the same `flood_fill_from` primitive `bridging_elements`
already uses. `edge_ok` becomes a cross-check, not the key.

| mask (64x96x96, win 16, step 8) | invalid | `edge_ok` cuts | line-walk only | connectivity |
|---|---|---|---|---|
| crack between node planes, t=2 | 0 | 45 | 45 | **45** |
| crack **on** a node plane, t=2 | 45 | **0** | 45 | **45** |
| wide crack, t=7 | 45 | **0** | 45 | **45** |
| 45-degree crack, 1 voxel | 45 | **0** | 70 | **70** |
| crack front, t=2 | 0 | 20 | 25 | **20** |
| flat surface + deep notch | 60 | 0 | 0 | **0** |
| cylinder, concave grooves | 60 | 0 | 10 | **5** at pad 3, **0** at pad >= 6 |

The connectivity test reproduces `edge_ok` exactly where `edge_ok` exists and recovers the three
geometries where it is empty. A line walk alone over-reports (25 vs 20 at the front, 10 vs 5 on
grooves), so it is not the acceptance test -- it supplies `t1`/`t2`/`gap_ref` only. Pad sweep: 3
gives 5 groove false positives, 6 and 8 give 0, 12 drops front pairs 20 to 15 as the test retreats
from the tip. **Ship `pad = half[axis]`.**

### The opening

Voxel units, `U` and `F` from **`FrameResult`** -- never `StrainResult` (`disp_smoothing` /
`strain_smoothing` go through `smooth_grid_field`, which takes no `edge_ok` and smooths straight
across the cut) and never `U_accum` (`interp_grid_field`, also not edge-aware).

```
d  = x_hi - x_lo                        (axis-aligned, |d| = L voxels)
walk the L+1 voxel line; i1 = first out-of-mask index, i2 = last
t1 = i1/L,  t2 = i2/L,  gap_ref = count of out-of-mask voxels on the line
COD = (U[hi] - U[lo]) - F[lo] @ (t1*d) - F[hi] @ ((1-t2)*d)        # einsum('ij,j->i')
```

**When `F` is unavailable or incomplete** (`gradient_fd` can NaN a whole column `F[...,:,j]`;
`plane_fit` NaNs below 5 reachable nodes; `strain_method="direct"` falls back silently): per
component `j`, use whichever endpoint's column is finite; if both are NaN, drop that column's
correction and set `bulk_corrected[pair, j] = False`. **Never propagate NaN into `COD`** -- an
uncorrected COD is off by about strain x node spacing, which is usable; a NaN is not. If the line
finds no out-of-mask voxel (disconnected around a corner), fall back to `t1 = t2 = 0.5`,
`gap_ref = 0`, flag `line_gap = False`.

Endpoint gate: post-precompute `mesh.node_valid` (`pipeline.py:211` overwrites it with `ctx.valid`)
**and** `~filled` (new `FrameResult.filled`, the `bad` mask `local_icgn.py:431` computes and discards)
**and** finite `U_std` at both ends. Do **not** gate on the plane fit's `complete`: it is False at
every cut node by construction and would delete the whole field.

Physical units: `COD_phys[i] = voxel_size[i]*COD[i]`; position `0.5*(x_lo+x_hi)*voxel_size`;
`sigma[i] = sqrt(U_std[lo,i]^2 + U_std[hi,i]^2)*voxel_size[i]`; report `|opening|/sigma`.

### Basis of the decomposition

**The estimated boundary normal, not the edge axis.** Reuse A′'s sheet fit on the out-of-mask
boundary voxels of the pair box already flood-filled: `n` is the smallest eigenvector, signed so
`n . d > 0`; planarity `p = sqrt(l2)/sqrt(l1)`. Under anisotropic voxels transform as a covector
(`n_i` proportional to `n_vox,i / s_i`, then renormalise). Then

- **primary product**: the `cod` vector in `[x, y, z]` and `|cod|` -- basis-free;
- `opening = cod . n`, `sliding = |cod - (cod . n) n|`, only where `p >= 3.0`, NaN otherwise.

### Must not be claimed

- **Not mode I/II/III.** `opening` is the normal component and `sliding` the tangential magnitude;
  separating II from III needs the crack-front tangent, which is not estimated.
- **Not absolute crack opening.** It is the *change* in face separation since the reference scan
  (`edge_ok` and the mask are the reference frame's; `U` is reference to deformed). `gap_ref` ships
  alongside so the reference gap is visible; a crack already open in the reference contributes
  nothing.
- **Not a crack-tip measurement.** Resolution along the crack is one node spacing; the endpoint
  displacements are themselves one-sided affine extrapolations from split subsets; accuracy degrades
  toward the front, where the opening goes to zero and `|grad F|` grows.
- **Absent, not zero**, where the crack is unsegmented, where the gap exceeds
  `COD_MAX_STEPS * spacing` (fully separated bodies), and, for post-hoc computation, on frames whose
  reference is not frame 0.

## Files to change

### A′, in this order

| path | change |
|---|---|
| `solver/local_icgn.py` | ~~**First, alone, with its own test:** fix the budget defect.~~ **Done** -- `plan_split` sizes the keep rows for the whole reference before the tile loop, `split_rows` takes the fraction instead of re-sweeping, and `tests/test_subset_split.py::test_the_split_budget_is_decided_for_the_reference_not_per_tile` fails against the old per-tile budget. Size the whole-reference keep-row need *before* the tile loop (per-tile candidate counts come from the `subset_valid_fraction` call `split_rows` already makes) and disable splitting for the reference before any tile runs, instead of at `:212-218` after some tiles built split Hessians. |
| `solver/numba_kernels.py` | Two-pass extension of `_build_split_rows_jit`. Pass 1 (same window loop) collects `B` and returns per-candidate centroid (3), covariance (6 upper terms) and `n_b`; the host does one batched `eigh` over `(M,3,3)` plus the sheet gate, which **keeps LAPACK out of `prange`**. Pass 2 (`_straddle_sides_jit`) re-walks only gated windows and returns `n_plus`, `n_minus`, `n_keep_fullres`. `build_split_rows` returns a 6-tuple. Constants live here. |
| `solver/reference_kernels.py` | `build_split_rows_np` + `straddle_from_window(sub, keep)` -- the NumPy oracle CLAUDE.md requires (`build_split_rows` has none today) and the numpy-backend path. |
| `solver/local_icgn.py` | `split_rows` to a 6-tuple (grow the `cand.size == 0` early return too); `LocalContext.straddle_fraction`; `precompute_local_context` appends `straddle` to `split_parts` and scatters it in the existing loop at `:262-268` (`zeros(N, float32)`, scatter at `taken`, NaN at `~valid`, `None` when `split_used` is False); the existing log line gains the straddle count and median. |
| `core/data_structures.py` | `FrameResult.straddle_fraction: NDArray[np.float32] \| None = None`. |
| `core/pipeline.py` | `straddle_fraction=ctx.straddle_fraction` at the `FrameResult(...)` call; a comment at `coarse_init.py:100` that the coarse context's field is discarded (different `N`, no `edge_ok`). |
| `core/checkpoint.py` | the name at `:97`, the `d.get(...)` kwarg at `:180`. |
| `export/export_npz.py`, `export_vtk.py`, `export_mat.py` | mirror the three existing `split_fraction` blocks. |
| `gui/panels/results_panel.py` | extend the existing split summary; **one** new `tr()` string. |
| `gui/translations/*.json` (six) | that one string (`test_i18n.py` demands no missing, empty or obsolete keys). |
| `docs/user_guide.md`, `docs/design.md`, `CHANGELOG.md` | the field, the constants, and the measured false-positive/negative table. |

Untouched deliberately: `tiling.py` (`merge_split`/`local_split` handle only `split_index` and
`split_keep`; the per-node merge is the three lines above), `cuda_kernels.py` (host-side statistic),
the whole `field_array` display path, `core/config.py`.

### B′

| path | change |
|---|---|
| `mesh/mesh_cut.py` | `pair_disconnected(mask, a, b, pad)` plus its `njit` kernel next to `_bridging_jit`; NumPy reference in the test. |
| `strain/opening.py` **(new)** | `find_cut_pairs(mesh, mask, half, max_steps)` returning `(lo, hi, axis, step, t1, t2, gap_ref, normal, planarity)`; `crack_opening(mesh, U, F, mask, para, U_std=None, filled=None)` returning `OpeningResult`; `opening_table(result, voxel_size)`; constants `COD_MAX_STEPS=2`, `COD_PLANAR_MIN=3.0`. Empty result plus a logged reason, never an exception. |
| `strain/__init__.py` | export `crack_opening`, `find_cut_pairs`, `OpeningResult`. |
| `solver/local_icgn.py`, `core/pipeline.py` | thread the final local pass's `bad` into a new `FrameResult.filled`. |
| `core/data_structures.py` | `FrameResult.filled`, plus four node-indexed arrays laid out like `edge_ok`: `cod (N,3,3) float32` (NaN where no pair), `cod_ok (N,3) bool`, `cod_gap_ref (N,3) float32`, `cod_step (N,3) int8`. Indexed by the **lower** node and axis, so every exporter and the checkpoint are one line each; normal, planarity, opening and sliding are derived on demand and do not ride the checkpoint. |
| `core/pipeline.py` | compute inside the frame loop, where the mesh and the reference mask are live (`PipelineResult` keeps only `done_meshes[0]`, and each reference re-cuts its own mesh). No new `DVCPara` flag -- the cost is one flood fill per candidate pair, 45-70 pairs on a 96^3 case. |
| `core/checkpoint.py` | four names, four `d.get(...)` kwargs, plus `filled`. |
| `export/export_npz.py`, `export_mat.py` | the four arrays. |
| `export/export_vtk.py` | three vector point-data fields (`write_vti` has no cell or edge path). |
| `export/export_csv.py` | a separate `<basename>_opening_<frame>.csv` from `opening_table` -- **not** through `field_array`, which is per-node only. |
| `gui/panels/results_panel.py` + six translations | one summary line, one new `tr()` string. |
| `docs/design.md`, `docs/user_guide.md`, `README.md`, `CHANGELOG.md` | the definition, the basis caveat, the four must-nots. |

Untouched deliberately: `compute_strain.py`, `gradient_methods.py`, `global_operators.py`, all CUDA
kernels, the `field_array`/`names.py`/`export_dialog.py`/3-D display path.

## Tests

### A′ (`tests/test_subset_split.py` plus two existing files)

- `test_straddle_numba_matches_reference` -- random masks, `build_split_rows` against
  `build_split_rows_np`, exact to 1e-12 on straddle, thickness and planarity.
- **`test_straddle_is_zero_on_smooth_boundaries` (the falsifier)** -- cylinder, ROI box, needle,
  isolated pore, through crack: `straddle_fraction == 0` at *every* valid node while
  `split_index >= 0` at 855 / 1464 / 927 / 917 / 1116 of them. **If this cannot pass, the field is a
  surface detector and must not ship.**
- `test_straddle_fires_at_a_crack_front` -- flagged set non-empty, and every flagged centre within
  one subset half-width of the front line *and* geometrically straddling, i.e. precision 1.0.
- `test_straddle_unknown_when_split_did_not_run` -- `subset_split=False`, no mask, and a forced tiny
  `MAX_SPLIT_BYTES`: `None` in all three.
- `test_straddle_stride_independent` -- straddle equal at `subset_stride` 1 and 2 while
  `split_fraction` differs, documenting 0.9375 / 0.8889 / 1.0000.
- `tests/test_tiling.py` -- tiled equals whole for `straddle_fraction`.
- `tests/test_checkpoint.py` -- round trip, plus the budget-defect regression: a forced small budget
  over a multi-tile plan must give byte-identical `H_all` and `n_valid` to a run with splitting
  disabled from the start.
- **Effect test, which decides whether A′ is worth shipping** -- speckle volume, crack front,
  `two_body_displacement` behind the front and a smooth affine field ahead: median `|U - U_gt|` at
  `straddle_fraction > 0.08` must be at least 3x the median at `straddle_fraction == 0` candidates
  **matched on `frac`**. A ratio near 1 falsifies A′.

### B′ (`tests/test_opening.py`, new)

- **(a) sign convention** -- the existing `wall_pair` geometry (analytic jump `(-1.1, 0.7, -0.5)`):
  `cod` within 0.02 voxels on every accepted pair, no pair elsewhere. Note in the test that `F = 0`
  on both sides, so this case cannot validate the bulk term.
- **(b) bulk term, which falsifies the `t1`/`t2` complexity** -- two bodies with different affine `F`
  (+2 % / -1 % xx) and the wall off-centre between node planes (`t1 = 0.375`, measured achievable)
  with a 2-voxel gap: `|cod - jump| <= 0.02` voxels, *and* the symmetric-0.5 form's error must exceed
  `|F_hi - F_lo| * L * |0.5 - t1|`. If the two forms are indistinguishable, drop `t1`/`t2`.
- **(c) coverage** -- the six probe masks: connectivity pairs equal `edge_ok` cuts where `edge_ok` is
  non-empty (45 == 45, 20 == 20), and equal 45 / 45 / 70 where `edge_ok` is 0. If `edge_ok` already
  covered those, the module is unjustified.
- **(d) no false positives on concave surfaces** -- grooved cylinder and notched slab: 0 pairs at
  `pad = half` (measured 0 at pad 6 and 8, 5 at pad 3).
- **(e) missing `F`** -- NaN one endpoint or one column: `cod` finite, `bulk_corrected` False in that
  column, residual bounded by `|F| * L`.
- **(f) silent at the front** -- accepted pairs stop within one node spacing of the front;
  cross-reference the nodes carrying `straddle_fraction > 0`, which is the region where no COD exists
  and the strain is untrustworthy.
- **(g)** `gap_ref` equals the segmented thickness for t = 1, 2, 3, 5 (one-step pairs) and 7 (two-step).
- **(h)** checkpoint round trip and tiled-versus-whole equality of the four arrays.

## Report pages

**Extend `scripts/make_subset_split_report.py`** (it already plots `split_fraction`):

1. **Flag-population map** -- one mask slice per case, node markers coloured by `straddle_fraction`,
   with the naive `unsplit` set as open circles behind. The whole argument is visual: 855 open circles
   around an uncracked cylinder against 0 filled markers.
2. **Counts table** -- the measured table of §A′, naive percentage beside the straddle count.
3. **Does the flag mean anything?** -- synthetic front with ground truth: `|U - U_gt|`, ZNCC and
   `U_std` binned by distance to the front with `straddle_fraction` on a twin axis, plus box plots of
   `straddle > 0.08` against `straddle == 0` candidates matched on `frac`.
4. **Threshold and limitation sweep** -- straddle against the two sheet-gate constants across all six
   masks (pore 0.050 against front 0.096 / 0.131), and the stride table.

Not worth shipping if the flagged population's error, ZNCC and `U_std` distributions are
indistinguishable from matched unflagged candidates (ratio below 1.5), or if the smooth-boundary
controls flag more than a handful of nodes at the shipped constants.

**New `scripts/make_opening_report.py` -> `reports/opening.pdf`:**

1. **Accuracy** -- cases (a) and (b): measured COD components against the analytic jump per pair, a
   residual histogram, with the symmetric-0.5 form overplotted so the bulk-term bias is visible.
2. **Coverage** -- pairs found against `edge_ok` cuts for crack on and off a node plane, thickness
   1/2/3/5/7, 45 degrees, front. The three zero columns of `edge_ok` are the justification.
3. **False positives** -- grooved cylinder and notched slab at pad 3 / `half` / `2*half`: false
   positives against pad, with the front-pair count at each pad (20 / 20 / 15) showing the trade.
4. **Field view** -- COD magnitude at pair midpoints over a slice; `opening` and `sliding` against the
   estimated normal; planarity as marker size and `|opening|/sigma` as colour, so unresolved pairs are
   visible.
5. **Limitations** -- zero pairs when the gap exceeds `COD_MAX_STEPS * spacing`; nothing where the
   crack is unsegmented; increment-not-absolute semantics; one-node-spacing resolution.

Not worth shipping if page 1 shows a residual comparable to the analytic jump (bulk term or sign
wrong), or page 3 cannot reach zero concave-surface false positives without also killing the front
pairs.

## Deliberately out of scope

1. **The `C` / `Vc` clipped-window arithmetic** -- 0 clipped windows measured in every case. Replace
   with one assertion in `split_rows`.
2. **The boolean `unsplit`, under any name** -- 31-42 % of valid nodes on defect-free masks. Not even
   as a derived convenience: it will be read as "bad node" and drive users to mask away good data.
3. **The `margin` caveat** -- it cancels between numerator and denominator, and the band is
   unreachable (border 5, tile halo at least `half+3`, margin 3). One assertion, no code.
4. **A geodesic-detour discriminator** -- 6-connected free-space distance equals L1, so a monotone
   staircase around a front gives excess 0; measured recall 0.25 against 0.75 for the sheet test.
5. **Mode II/III separation, and "mode I" as a name** -- needs the crack-front tangent curve.
6. **A normal from a smoothed-mask gradient** -- vanishes by symmetry at the mid-plane of a thin
   symmetric gap. PCA of the boundary voxels only.
7. **Any new `DVCPara` parameter for either feature** -- `_meta_for` embeds all of `para_to_dict` and
   `_meta_diff` reports `para.<new_key>` as a `CheckpointMismatch`, which the GUI's `resume="auto"`
   catches by silently restarting and discarding completed frames. Constants in code; the report
   pages sweep them.
8. **Making either quantity a displayable `field_array` field** -- `split_fraction` is not one today;
   adding one means `export_utils` + `names.py` + `results_panel` + `export_dialog` +
   `field_canvas`/`viewer`/`view3d` + six translation tables under a 100 %-coverage test, for a
   diagnostic only meaningful next to the mask.
9. **Post-hoc COD from `PipelineResult`** -- `dvc_mesh = done_meshes[0]` only, and each reference
   re-cuts its own mesh; compute in the frame loop.
10. **`cut_edge_nodes`'s `lo + strides[axis]` wrap at the last x column** -- latent only (those
    entries are True by construction, since no element has its low-x corner there). Noted, not fixed
    here.
11. **Discontinuities the segmentation does not resolve** -- contacting faces, debonded interfaces,
    shear bands. Both features are *mask-geometry* diagnostics; the data-side signal is the existing
    ZNCC, `U_std` and residual. Keep them separate in the exports and say so in the docs.

## Also found, not part of either design

- The **budget defect** in the verification table above: a correctness bug, pre-existing, and the
  first thing to fix because A′ touches the same path. **Fixed** -- see `CHANGELOG.md` under Fixed.
- `local_split` / `merge_split` do **not** need to learn about a new per-node array; the tiled merge
  is correct only because the halo guarantees a node's window never straddles a box face.
- `clean_initial_guess` already routes cut nodes around the boundary and skips exactly the nodes A′
  would flag -- the one actionable in-solver consumer, missing from my original plan.
