# Subset splitting at boundaries (window splitting) — implementation plan

Status: implemented 2026-09-07 (phases 0-4; the CUDA gate and the same-side initial guess included).
Port of pyALDIC's masked-subset IC-GN to 3-D. Measured effect and cost: `reports/subset_split.pdf`.

## 1. Definition

When the correlation subset of a node contains masked voxels, label the connected components of
the *in-mask* voxels inside the subset window and keep only the component that contains the
subset centre. The kept voxels are the only ones that enter the reference statistics (mean,
variance), the Hessian, the residual and the ZNCC. Every zero in the mask is a boundary: the outer
edge of the region of interest, internal holes, cracks, and (optionally) the faces of the volume.
Nothing distinguishes a "geometric" boundary from a "physical" one, and nothing needs to.

Connectivity is 6 (the 3-D analogue of 4 in 2-D): a continuous masked surface one voxel thick then
separates the two sides; 26-connectivity would leak through a one-voxel diagonal crack. Components
are found inside the window only, with paths restricted to the window, so two regions that are
connected only outside the window count as separate (conservative for thin ligaments).

What splitting does not do: a crack that ends inside the window (crack front, needle-shaped void)
leaves the window connected, and the subset keeps voxels from both sides of the crack. That is the
same limitation as in 2-D and is out of scope here (it needs enriched shape functions or adaptive
subset sizes).

## 2. What exists today

The kernels already gate every voxel on the reference mask; the gate is the insertion point.

| where | line | today |
|---|---|---|
| `solver/numba_kernels.py::_precompute_one` | 242 | `if mask[zz, yy, xx] == 0: continue` (no linear index counter) |
| `solver/numba_kernels.py::_warp_and_sample` | 417 | same gate, writes `gbuf[idx] = nan`; the gradient and ZNCC loops skip NaN |
| `solver/numba_kernels.py::_zncc_from_buffer` | 454 | same gate on the reference side |
| `solver/cuda_kernels.py` | 1160, 549, ~936 | the same three gates, `v` is the linear sampled index |
| `solver/reference_kernels.py` | 200, 257, 377 | `m = mask[sl].ravel() > 0` |
| `mesh/grid_mesh.py::apply_mask_to_mesh` | 174 | node valid iff centre in mask and `frac >= min_valid_ratio` (summed-area table) |
| `solver/numba_kernels.py` | 271 | node rejected if `n_valid < 27` or `n_valid < min_valid_ratio * total` |

All loops run `dz` (outer), `dy`, `dx` (inner) with `range(-h, h + 1, stride)`; the NumPy
reference uses `subset_offsets()` with the same `ij` order. A per-node keep array in that order
lines up with `gbuf[idx]` on the CPU and `gbuf[blk, v]` on the GPU without any re-derivation.

pyALDIC differences worth knowing: it stores the keep set as float64 `(N, Sy, Sx)` for every node
(8x the minimum, and dense over all nodes); its `min_valid_ratio` is a dead parameter with 0.5
hard-coded in four places; its compiled path skips the Hessian conditioning check; its initial
guess is not split-aware; it has no on/off switch. This port fixes the first three and keeps the
last two as explicit, documented choices.

## 3. Design decisions

D1. **Per-subset labelling, never a global labelling of the mask.** A crack with a front is one
component globally; only the window-local labelling separates the two sides where the crack does
cross the window. Measured on a 256^3 volume with a planar crack that stops at a front (subset 32,
step 8): the global labelling gives 1 component; per-subset labelling changes 5.2 % of the solved
nodes and drops a median 24 % of their voxels.

D2. **Label at full resolution, store at the sampling stride.** The flood fill must run on the full
`(2h+1)^3` window even when `subset_stride > 1`: a strided sample grid can step over a one-voxel
crack. The result is then subsampled to the stride grid for storage.

D3. **Store keep rows only for subsets that touch the mask, bit-packed.** Dense `(N, S)` booleans
cost 3 GB at the default subset (33^3) with 82 800 nodes. Candidates are the nodes whose window
contains at least one masked voxel (the summed-area table already gives this: `frac < 1`). Packed
rows cost `S / 8` bytes each: 4.5 kB at 33^3. Measured: 15 MB for 3 360 straddling nodes.

D4. **One gate, three backends, one test.** The bit test is inserted next to the existing mask test
in the numba, CUDA and NumPy paths; parity tests compare all three on the same split case.

D5. **Reuse `min_valid_ratio`.** With splitting on, the fraction is the kept component over the
sampled subset; the existing rejection at `numba_kernels.py:271` applies unchanged because
`n_valid` becomes the kept count. No new status code in phase 1; a per-node `split_fraction`
diagnostic says what happened.

D6. **Default off.** `DVCPara.subset_split = False`; existing results, sessions and the MATLAB
agreement do not change unless the user turns it on.

D7. **The global step is a separate phase, cut per element the way pyALDIC does it.** Splitting
changes which voxels a node correlates, not `node_valid`; the FEM and FD operators would still
couple the two sides of a boundary. pyALDIC never had this problem because its mesh is trimmed by
`mesh/mark_bridging.py` (applied in the default pipeline, `core/pipeline.py:1178`): the material
pixels inside an element's bounding box are labelled (4-connected) and the element is removed when
its corner nodes fall in different components. The 3-D port is the same test on the hex8 box with
6-connectivity: corners in different components -> the element is dropped. An edge is "ok" when at
least one surviving element contains it, and that `edge_ok (N, 3)` table drives the FD stencils,
bad-node inpainting, the median test and the strain plane fit, so mesh topology and solver
validity never disagree. A plain segment test was rejected: a grid edge through a small pore would
cut the regulariser all over a porous material although the material is connected around the pore;
the box labelling keeps such elements, and it stops by itself at a crack front.

D8. **The initial guess stays split-blind in phase 1** (as in pyALDIC), with the same-side
neighbour fallback of phase 2 as the cheap fix.

## 4. Data layout

`solver/local_icgn.py::LocalContext` gains three fields, all `None` when the feature is off:

```python
split_index: NDArray[np.int64]     # (N,)  row into split_keep, -1 when the subset is not split
split_keep: NDArray[np.uint8]      # (n_split, ceil(S / 8))  bit i of the row = keep sampled voxel i
split_fraction: NDArray[np.float32]  # (N,) kept / in-mask sampled voxels, 1.0 when not split, nan when rejected
```

`S = _subset_count(hx, hy, hz, stride)`. Bit `i` is `(row[i >> 3] >> (i & 7)) & 1`. The context
is cached per reference frame (`pipeline.py:169`, two resident), so the arrays are computed once
per (reference, mesh, parameters) and reused by every local pass and every ADMM iteration.

`FrameResult` / `PipelineResult` gain `split_fraction: NDArray[np.float32] | None` next to `zncc`,
`n_iter`, `status`, so the GUI and the exports can show it like ZNCC.

## 5. Algorithms

### 5.1 Candidates

```
frac = subset_valid_fraction(mask, coords, winsize)      # existing summed-area table
candidates = node_valid & (frac < 1.0)                   # window touches the mask
```

### 5.2 Flood fill (numba, one kernel, parallel over candidates)

```
_flood_fill_centre(sub, out, stack)      # sub, out: (Sz, Sy, Sx) uint8; stack: int32[Sz*Sy*Sx]
    seed = (Sz//2, Sy//2, Sx//2)
    if sub[seed] == 0: return 0          # centre masked: nothing kept
    BFS over 6-neighbours inside the window, marking out = 1; return count
```

Per candidate: copy the full-resolution window of the mask into `sub` (out-of-volume voxels are 0
when border clipping is on, see opportunity O1), run the fill, then walk the `dz, dy, dx` stride
loops in kernel order and pack `out[...]` into the row. Cost measured with SciPy at 0.2 ms per
33^3 subset; the numba version is faster and runs in `prange`.

The NumPy reference is `scipy.ndimage.label(sub, structure=generate_binary_structure(3, 1))` and
`lab == lab[centre]`; a test asserts array equality with the numba fill on random masks, a
one-voxel diagonal crack (must separate under 6-connectivity) and a masked centre.

### 5.3 The gate

CPU (`_precompute_one`, `_warp_and_sample`, `_zncc_from_buffer`), with `row` the node's packed
row and `use_keep = split_index[n] >= 0`:

```
if mask[zz, yy, xx] == 0 or (use_keep and ((row[idx >> 3] >> (idx & 7)) & 1) == 0):
    idx += 1            # every sampled position advances idx, kept or not
    continue
```

`_precompute_one` gets an `idx` counter (it has none today). GPU: identical with `v` in place of
`idx` and `split_keep[row, v >> 3]`. NumPy: `m &= np.unpackbits(row, count=S).astype(bool)`.

Everything downstream is unchanged: `n_valid`, `meanf`, `bottomf` and the Hessian are already
accumulated over the gated voxels; the gradient and ZNCC loops already skip the NaN in `gbuf`;
the noise correction already scales with `n_valid / n_full` (`numba_kernels.py:597`). The one
approximation that remains is that the noise pattern's geometric moments are those of the full
cube; acceptable, and cheap to recompute per split node in the precompute if a test shows it matters.

### 5.4 Element cut for the global step (phase 3)

```
bridging_elements(mask, coords, elements) -> bool (E,)   # per hex8: label the in-mask voxels of the
                                                          # element's bounding box (6-connected, numba flood
                                                          # fill from one corner); True when the 8 corners
                                                          # are not all in one component
edge_ok: (N, 3) bool                                      # node n to its +x, +y, +z grid neighbour: some
                                                          # surviving element contains the edge
```

FEM: `active_elements` also drops bridging elements. FD: `has_prev` / `has_next` in
`global_operators._difference_operator` also require `edge_ok`. Inpainting (`fill_bad_nodes`,
`fill_nan_grid`), `universal_median_test` and the strain plane fit take neighbours only through ok
edges (path connectivity inside their window, decision 3). Nodes next to a dropped element join
`boundary_nodes` for the beta sweep. Cost: one `(step + 1)^3` labelling per element, a few seconds
for 80 000 elements.

## 6. Changes by file

| file | change | phase |
|---|---|---|
| `core/config.py` | `subset_split: bool = False` in section 4; no validation needed | 1 |
| `solver/numba_kernels.py` | `_flood_fill_centre`, `build_split_rows` (parallel); `idx` and the gate in `_precompute_one`; gate in `_warp_and_sample`, `_zncc_from_buffer`; `split_index`/`split_keep` threaded through `precompute_nodes`, `_icgn_12dof_parallel_jit`, `_icgn_3dof_parallel_jit`, `evaluate_zncc_parallel`; optional-arg normalisation in `icgn_12dof_parallel`, `icgn_3dof_parallel` | 1 |
| `solver/reference_kernels.py` | `centre_component_np` (SciPy); `keep` in `precompute_node_np`, `icgn_12dof_np`, `icgn_3dof_np`; unpack in the batch wrappers | 1 |
| `solver/local_icgn.py` | `LocalContext` fields; candidates + rows in `precompute_local_context`; pass-through in `local_icgn`; `split_fraction` into the result | 1 |
| `solver/subpb1_solver.py` | pass-through of the two arrays | 1 |
| `core/data_structures.py`, `core/pipeline.py` | `split_fraction` in `FrameResult`/`PipelineResult`, summary count "nodes split / rejected by the split" in the log | 1 |
| `solver/cuda_kernels.py` | gates in `precompute_kernel`, `icgn12_kernel`, `icgn3_kernel`; device upload of the rows once per context (`DeviceCache`) | 2 |
| `solver/init_disp.py` | same-side neighbour fallback for split nodes (O2) | 2 |
| `mesh/grid_mesh.py`, `solver/global_operators.py`, `solver/subpb2_solver.py` | `segment_crosses_mask`, `edge_ok`, FD/FEM cuts | 3 |
| `solver/local_icgn.py::fill_bad_nodes`, `utils/inpaint.py`, `utils/outlier_detection.py` | neighbourhoods across ok edges only | 3 |
| `strain/*` plane fit | neighbour exclusion by segment test | 3 |
| `gui/panels/param_panel.py`, translations (6) | check box "Split subsets at boundaries" + tooltip | 4 |
| `gui/names.py`, results panel, export | `split_fraction` as a displayable field | 4 |
| `synthetic.py` | `piecewise_displacement` (two bodies separated by a masked plane) | 1 |
| `tests/test_subset_split.py`, `tests/test_cuda_backend.py` | see section 8 | 1-2 |
| `scripts/make_subset_split_report.py` -> `reports/subset_split.pdf` | see section 8 | 1-3 |
| `docs/design.md`, `docs/user_guide.md`, `CHANGELOG.md` | contracts, parameter description | 4 |

## 7. Phases, acceptance, effort

| phase | content | accept when | effort |
|---|---|---|---|
| 0 probe | flood fill + NumPy path + two-body case | number: near-boundary error with/without split, local-only and full ADMM | 0.5 day |
| 1 CPU | numba kernels, context, diagnostics, parameter, tests | numba == NumPy to 1e-10 on split cases; all existing tests unchanged with the flag off | 1.5 days |
| 2 GPU + guess | CUDA gates, device rows, same-side initial guess | CUDA == numba within the existing tolerance; guess test on the two-body case | 1 day |
| 3 global step | segment test, FD/FEM/inpainting/median/strain cuts | two-body case: jump preserved through the full ADMM; MATLAB agreement unchanged with masks off | 1.5 days |
| 4 GUI, docs, report | check box, field, translations, PDF, docs | i18n audit 100 %, report regenerated, CHANGELOG | 0.5 day |

Effort assumes the existing parity-test pattern (`tests/test_kernels.py`) and the report skeleton
(`scripts/make_mask_report.py`).

## 8. Tests and report

`tests/test_subset_split.py`

- `test_flood_fill_matches_scipy`: random masks, several seeds, strides 1-3; diagonal one-voxel
  crack separates under 6-connectivity; masked centre gives an empty row.
- `test_packing_order_matches_subset_offsets`.
- `test_precompute_and_icgn_parity_with_split`: numba vs NumPy on the conftest affine pair with a
  masked wall through the volume.
- `test_two_bodies_keep_their_own_translation`: `piecewise_displacement`, split on: nodes next
  to the wall within 0.02 voxel of their side; split off: error of order half the jump.
- `test_min_valid_ratio_applies_to_the_kept_component`.
- `test_flag_off_is_bit_identical` to the current results.
- GPU: one `_case(..., split=True)` in `tests/test_cuda_backend.py`.

`reports/subset_split.pdf`

1. Slices of a split subset: kept, dropped and masked voxels.
2. Two-body case: displacement error against distance from the wall, split off/on, local-only vs
   full ADMM (this is the number that says how much the global step gives back, phase 3 closes it).
3. Crack with a front: where splitting changes nothing and why.
4. Porous phantom (bead volume): share of straddling nodes, error with/without split.
5. Cost: precompute time and packed memory against the number of straddling nodes.

## 9. Risks

| risk | effect | mitigation |
|---|---|---|
| global step smooths across the boundary | phase 1 gain partly undone in the ADMM result | phase 3; page 2 of the report quantifies it before and after |
| split-blind initial guess | NCC template contaminated by the far side, IC-GN converges to the wrong minimum | O2 same-side fallback; ZNCC and status expose failures |
| three backends drift | CPU/GPU results differ at boundaries | one gate, one parity test per backend on split cases |
| thin kept components | ill-conditioned Hessian | keep the conditioning check in every backend (pyALDIC's compiled path lacks it) |
| fewer voxels per subset | higher displacement noise near boundaries | unavoidable; report page 2 and 4 show it; `min_valid_ratio` is the user's knob |
| memory on the GPU | rows uploaded next to the 295 MB `gbuf` | bit-packed rows: tens of MB in practice; a budget check falls back to off with a log line |
| stride jumps over a crack | component wrongly connected | D2: label at full resolution |

## 10. Opportunities

O1. **Recover the half-subset ring at the volume faces.** Today a subset that leaves the volume is
rejected outright (`numba_kernels.py:227`), so every face of the volume loses a ring of half a
subset. Treating out-of-volume voxels as masked in the flood fill (as pyALDIC does at the image
border) makes those subsets ordinary split subsets: the ring comes back with the same mechanism,
subject to `min_valid_ratio`. Requires relaxing the bounds check when a keep row exists and the
sampling guard for out-of-volume positions.

O2. **Same-side initial guess for free.** A grid neighbour lies inside the window whenever
`step <= half`, so the keep row says directly which neighbours are on the node's side. Split nodes
take their starting displacement from the median of the same-side neighbours instead of a
contaminated NCC template.

O3. **One boundary criterion for the whole pipeline.** The segment test of phase 3 fixes coupling
across mask holes that exists today, splitting or not: the global step, inpainting, the median
test and the strain fit all reach across holes and through thin walls at present.

O4. **A per-node diagnostic.** `split_fraction` shown in the results panel and exported with the
fields tells the user where the subsets were cut and by how much, and makes the near-front nodes
(fraction 1.0 but low ZNCC) visible.

O5. **Porous materials.** Foams, trabecular bone and granular packings put a large share of
subsets on pore boundaries; the benefit there is broader than at a single crack. The bead phantom
page of the report measures it.

O6. **Free on the GPU.** Bit-packed rows and a one-bit test per sampled voxel cost nothing next to
the tricubic sample; the feature does not change the GPU chunking.

O7. **Groundwork for the crack front.** The keep rows and the edge cuts are exactly the inputs an
enriched shape function or an adaptive subset size at the front would need.

O8. **Back-port.** The live `min_valid_ratio`, the conditioning check in the compiled path and the
same-side guess are three fixes pyALDIC can take back.

## 11. Decisions (2026-09-07)

1. Border clipping (O1) is part of `subset_split`: out-of-volume voxels are masked in the flood
   fill, and a subset that leaves the volume is an ordinary split subset. Moved into phase 1.
2. `split_fraction` is stored with the results (session results archive) and exported with the
   other per-node fields, exactly like `zncc`.
3. The strain plane fit keeps only neighbours reachable from the centre node through ok edges
   inside the fit window (path connectivity, the stricter criterion), not the plain segment test.
