# Examples

| file | what it shows |
|---|---|
| `scripting/run_synthetic.py` | generate a synthetic speckle pair with a known deformation, run AL-DVC, compare with ground truth, export everything |
| `scripting/batch_process.py` | process several samples listed in a YAML/JSON config without the CLI |
| `scripting/batch_config.example.yaml` | every config key understood by `al-dvc run` and `batch_process.py` |
| `scripting/plot_results.py` | re-plot fields from an exported `.npz` (no recomputation) |

Real volumetric datasets are not shipped (they are hundreds of MB). The
MATLAB ALDVC repository links example micro-CT and confocal stacks; any
`.mat` file from that pipeline loads directly with
`al_dvc.load_volume("vol_...mat")`.
