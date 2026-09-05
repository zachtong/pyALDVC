# Texture analysis: integration plan

Status: proposal, 2026-09-05. Source material: the autocorrelation (`acf_analysis.py`) and
RVE (`rve_analysis.py`) scripts written for the DVC Challenge feature analysis (kept locally
under `reference/`, not in the repository). Both reviews of those scripts agree on the
findings summarised in section 1; this document turns them into a design for pyALDVC.

## 1. What is kept, what is rewritten

Kept as ideas: the FFT autocorrelation of a mean-subtracted volume, threshold lengths
(1/e, 0.1, 0.01) read off a correlation profile, a size sweep to judge whether a region is
large enough for the statistic, and the synthetic sphere volumes for validation.

Rewritten entirely (none of the script code is imported):

| Defect in the scripts | Design decision here |
|---|---|
| FFT result cropped before `fftshift`; negative lags shifted by one whenever `next_fast_len(2N-1) > 2N-1` (6 of the 8 default sweep sizes) | Shift the full inverse FFT, then cut `[-(N-1), N-1]` around the true zero lag; test against `scipy.signal.correlate(mode="full")` on sizes 12, 13, 64 and non-cubic shapes, and assert `C(h) = C(-h)` |
| Zero-lag normalisation only: the finite window multiplies the ACF by `prod(1 - |h_j| / N_j)`, a size-dependent bias that leaks into the size sweep | Two estimators, both explicit: `window` (as the scripts) and `overlap` (divide by the overlap count); `overlap` is the default, lags are capped where the overlap drops below a configurable fraction (default 50 % of the volume), and the report compares the two |
| Radial average only; isotropic voxels assumed; shells run into the corners of the padded cube | Directional profiles along x, y, z and a radial profile; physical distances from the voxel spacing; shell statistics carry the actual mean radius of each shell and its coverage; profiles stop at the reliable lag |
| Negative correlations clamped to 1e-10 before the threshold search; the search mutates the array that is later exported | Profiles keep their sign; one crossing function for every use, returning a value and a status (`crossed`, `not_crossed`, `plateau`, `invalid`); log plots mask non-positive values instead of clamping |
| Two different threshold implementations (strict vs non-strict comparison, plateau and tail-rise behaviour differ) | One implementation: first strict downward crossing, linear interpolation between the two bracketing samples, no extrapolation |
| Zero-variance input divides by zero and produces NaN | Input validation: 3-D, finite, non-constant; otherwise a `TextureResult` with `status="no_texture"` |
| Nested sub-volumes at one centre; sliding-window CV with an absolute tolerance that lets a monotone drift pass; clipped duplicate sub-volumes give CV = 0 | Several sub-volumes per size at different positions (tiled, then random when tiling gives fewer than the requested count); a plateau criterion against the largest sizes with a minimum size span; deduplication by actual bounds; the sweep stops at the volume bounds |
| Report prints the CV threshold but not the absolute tolerance that actually passed | Every decision records the statistics it was made from and which criterion passed |
| Per-radius boolean masks (`O(M R)`), full float64 coordinate grids (24 GiB at 512^3) | Integer radii computed per z-slab, `bincount` for count, sum and sum of squares; the ACF is cropped to `[-max_lag, max_lag]` before any shell statistics, so memory scales with `(2 max_lag + 1)^3`, not with the volume |
| tkinter dialogs, `quit()`, `matplotlib.use("TkAgg")` at import, OpenCV trackbars | Pure functions in `al_dvc.texture`; the GUI is a separate window; no OpenCV dependency; denoising is not part of the analysis (the ACF of a filtered image describes the filtered image) |

The sphere generators are replaced by a Boolean model (Poisson sphere centres, overlaps
allowed), whose two-point correlation is known in closed form, so the 1/e length has an
analytic ground truth. Hard-core packings have no closed form and are used only as a
qualitative check.

## 2. Package layout

```
src/al_dvc/texture/
    __init__.py         public API: analyse_texture, sweep_sizes, recommend_parameters
    acf.py              autocorrelation(vol, spacing, max_lag, estimator) -> Autocorrelation
    profiles.py         directional_profiles, radial_profile (shell mean radius, coverage, count)
    crossing.py         correlation_length(profile, threshold) -> Crossing(value, status)
    rve.py              sweep_sizes(vol, mask, sizes, samples_per_size, ...) -> SizeSweep
    recommend.py        subset / step suggestions from directional lengths
    boolean_model.py    synthetic Boolean sphere volume + analytic correlation function
```

Contracts follow `docs/design.md`: volumes are `(nz, ny, nx)`, spacing is `(dx, dy, dz)`
or a scalar, every length is reported in voxels and in physical units, all indices are
0-based. Results are frozen dataclasses:

- `Autocorrelation`: the cropped, shifted ACF `(2Lz+1, 2Ly+1, 2Lx+1)`, `estimator`,
  `max_lag`, `spacing`, `variance`, `n_voxels`, `status`.
- `Profile`: `lag` (voxels), `distance` (physical), `mean`, `std`, `count`, `coverage`, `axis`
  (`"x" | "y" | "z" | "radial"`).
- `Crossing`: `threshold`, `value_voxels`, `value_physical`, `status`, `bracket`.
- `TextureResult`: the ACF, the four profiles, crossings for the three thresholds per
  profile, `noise_floor` (standard deviation of the ACF beyond three 1/e lengths),
  `periodicity` (lag and height of the largest secondary peak, `None` when absent),
  `recommendation`.
- `SizeSweep`: per size the sampled sub-volumes (actual bounds), the crossing values, their
  mean and spread, the plateau decision per threshold with the statistics behind it.

`recommend_parameters` is a heuristic and is labelled as such in the GUI: subset edge per
axis = `factor` times the 1/e length along that axis (default factor 2.5, rounded up to
an even value and clamped to `[8, 128]`), step = half the subset edge. The report validates
the heuristic on synthetic pairs against the actual displacement error; the user always
sees the lengths the suggestion was derived from and can apply, edit or ignore it.

## 3. GUI

A "Texture analysis" window (`gui/texture_window.py`) modelled on the strain window,
opened from the Analysis menu (Ctrl+X) and from a button under the Volumes section
("Analyse texture..."). It works on the reference volume restricted to the region of
interest's bounding box (or the whole volume) and runs in a worker thread with progress and
cancel.

Left: parameters (estimator, maximum lag, thresholds, samples per size, size schedule for
the sweep). Centre, two tabs: "Correlation" with the directional and radial profiles
(linear and log, threshold markers, noise floor band) and a slice of the ACF; "Size sweep"
with length versus sub-volume size per threshold, the spread per size and the plateau
decision. Right: the lengths table (three thresholds by four profiles, voxels and physical
units), the recommendation with "Apply to parameters" (sets the three subset edges, unlocks
the cube, sets the step) and export buttons (CSV of the profiles and lengths, JSON summary,
PNG of the figures) through the existing export helpers.

All labels go through `names.py` and the translation tables; the window is included in
the retranslate loop and in the GUI report.

CLI: `al-dvc texture VOLUME [--roi MASK] [--spacing dx dy dz] [--sweep] [--out DIR]` for
batch use, e.g. recomputing the 28 DVC Challenge datasets with the corrected estimator.

## 4. Tests and report

`tests/test_texture.py`:

- ACF equals `scipy.signal.correlate(mode="full")` (after normalisation) on `12^3`, `13^3`,
  `64^3`, `(7, 12, 10)`; symmetry `C(h) = C(-h)` to 1e-6; the `window` estimator
  reproduces the scripts' curve on a size that does not trigger the crop bug.
- Constant volume gives `status == "no_texture"`; NaN input is rejected.
- Boolean model: the measured 1/e length matches the analytic value within 3 % for two
  radii; the `overlap` estimator's error is independent of the sub-volume size while the
  `window` estimator's grows as the size shrinks.
- Anisotropic spacing: a volume stretched 2x along z reports the same physical length
  along z and twice the voxel length; an elongated Gaussian texture reports the longer
  length along its axis.
- Crossings: the review's curves (`[1, .8, .6]`, `[1, .3, .2]`, `[1, .1, .1, .05]`,
  `[1, .2, -.2]`, `[1, .3, .05, .005, .03]`) return the expected values and statuses.
- Size sweep: a plateau sequence converges, a monotone drift does not, duplicate bounds are
  removed, the sweep stops at the volume bounds, and every decision carries its statistics.
- GUI: the window opens offscreen, runs on a synthetic pair, "Apply to parameters" sets a
  non-cubic subset and unlocks the cube lock; the CLI writes CSV and JSON.

`scripts/make_texture_report.py` writes `reports/texture.pdf`: the analytic validation, the
crop-bug reproduction (old versus corrected curve on `64^3`), `window` versus `overlap`
across sub-volume sizes, directional profiles on an anisotropic texture, the size sweep on
the Boolean model with the decision, the subset-size heuristic against the measured DVC
error on synthetic pairs, timings for `128^3` to `512^3`, and the limitations (the
heuristic is a starting point; the sweep measures the stability of a texture statistic,
not a material RVE; the ACF describes the image, not the structure).

## 5. Order of work

1. `texture/acf.py`, `profiles.py`, `crossing.py`, `boolean_model.py` with their tests
   (the corrected estimator and its validation come first because every later number
   depends on them).
2. `rve.py` and `recommend.py` with tests; the report script.
3. The GUI window, the CLI, translations, the GUI report, README and CHANGELOG.
4. Not now: the DVC Challenge 2.0 analysis is in revision with its data and scripts already
   packaged, so those numbers are not recomputed before publication. The CLI makes a later
   recomputation a batch job, and an erratum can state that the original scripts carried the
   defects listed in section 1.

Estimated effort: about a day for each of steps 1 to 3.
