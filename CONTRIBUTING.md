# Contributing to pyALDVC

## Development setup

```bash
git clone https://github.com/zachtong/pyALDVC.git
cd pyALDVC
pip install -e .
pip install pytest pytest-xdist psutil ruff
pytest
```

## Ground rules

- Read `docs/design.md` first. Axis order, node ordering, DOF layout and
  warp parameterisation are contracts; changing them touches everything.
- Every Numba kernel has a NumPy reference implementation in
  `solver/reference_kernels.py`; keep them in sync and covered by
  `tests/test_kernels.py`.
- New features need pytest coverage against analytic ground truth (see
  `al_dvc.synthetic`) and, when they affect accuracy or speed, an entry in
  the validation or benchmark script under `scripts/` so the PDF reports
  stay current.
- Do not loosen a test tolerance to make a test pass; fix the code or
  document the limitation.
- Code, comments and commit messages are in English. Commit messages use
  the conventional format `type: description` with types
  `feat, fix, refactor, docs, test, chore, perf, ci`.

## Reporting bugs

Open an issue with the volume shapes and dtypes, the `DVCPara` you used
(`para_to_dict`), the traceback, and if possible a small synthetic
reproduction built with `al_dvc.synthetic`.
