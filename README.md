<p align="center">
  <img src="assets/banner.png" alt="pyALDVC banner" width="100%"/>
</p>

<p align="center">
  Full-field 3-D displacement and strain from volumetric images (micro-CT, confocal, MRI, OCT).<br/>
  Hybrid local-global solver, GPU acceleration, a complete desktop application.
</p>

<p align="center">
  <a href="https://github.com/zachtong/pyALDVC/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/zachtong/pyALDVC/ci.yml?style=flat-square&label=CI" alt="CI"/></a>
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/GUI-PySide6-41cd52?style=flat-square" alt="PySide6"/>
  <img src="https://img.shields.io/badge/GPU-CUDA%20optional-76b900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA"/>
  <img src="https://img.shields.io/badge/License-BSD--3--Clause-22c55e?style=flat-square" alt="License"/>
  <a href="https://pypi.org/project/al-dvc/"><img src="https://img.shields.io/pypi/v/al-dvc?style=flat-square&label=PyPI" alt="PyPI"/></a>
</p>

<p align="center">
  <strong>Available in 7 languages</strong><br/>
  <img src="https://img.shields.io/badge/English-✓-22c55e?style=flat-square" alt="English"/>
  <img src="https://img.shields.io/badge/简体中文-✓-22c55e?style=flat-square" alt="Simplified Chinese"/>
  <img src="https://img.shields.io/badge/繁體中文-✓-22c55e?style=flat-square" alt="Traditional Chinese"/>
  <img src="https://img.shields.io/badge/日本語-✓-22c55e?style=flat-square" alt="Japanese"/>
  <img src="https://img.shields.io/badge/Deutsch-✓-22c55e?style=flat-square" alt="German"/>
  <img src="https://img.shields.io/badge/Français-✓-22c55e?style=flat-square" alt="French"/>
  <img src="https://img.shields.io/badge/Español-✓-22c55e?style=flat-square" alt="Spanish"/>
</p>

---

pyALDVC is the Python version of the MATLAB [ALDVC](https://github.com/FranckLab/ALDVC)
code (Yang, Hazlett, Landauer, Franck, *Exp. Mech.* 2020) and the volumetric sibling of
[pyALDIC](https://github.com/zachtong/pyALDIC). Local 12-DOF subsets are coupled to a
global finite-element step through the augmented Lagrangian (ADMM), so the field is
smooth and accurate where plain subset DVC breaks down, at sub-voxel precision.

<p align="center">
  <img src="assets/pyALDVC_demo.gif" alt="pyALDVC workflow: load volumes, draw a region of interest, run, strain post-processing, 3-D view" width="90%"/>
</p>

<p align="center"><sub>Load, draw the region of interest, run, strain, 3-D view. Synthetic foam, 200 x 224 x 256 voxels, 14 112 nodes, 11 s on an RTX 5090.</sub></p>

## What you get

- **A desktop application.** Volumes, region of interest and parameters on the left, slices and 3-D view in the middle, results and exports on the right. No code needed.
- **Local DVC or AL-DVC** with one switch; the ADMM penalty is tuned automatically.
- **Masks and regions of interest** drawn on the slices (rectangle, ellipse, polygon, brush, auto-threshold), so time and memory scale with the region, not the scan.
- **Texture analysis** that measures the correlation length of your scan and suggests the subset size and step.
- **Strain post-processing** with plane fitting, finite elements, finite differences or the solver's gradient; infinitesimal, Green-Lagrange, Euler-Almansi or Hencky measures.
- **A 3-D view** with field slices, node points, iso-surfaces, the deformed lattice and displacement arrows; orbit, frame and slice-sweep animations recorded as GIF or MP4.
- **GPU acceleration** on NVIDIA cards: a 1024 x 1024 x 306 micro-CT scan with 79 200 nodes in 23 s (3.6 min on a 24-core CPU), within 0.01 voxel of the MATLAB code.
- **Every format you have**: TIFF stacks and slice folders, MATLAB, NumPy, HDF5, NIfTI, NRRD, DICOM; exports to NumPy, MATLAB, CSV, ParaView and a PDF report.
- **Uncertainty** per node, checkpoints and resume for long sequences, batch runs, sessions, a command line and a Python API.

## Case studies

**Synthetic rotation.** The frames animation plays the deformation from the reference state as one continuous motion of the lattice.

<p align="center">
  <img src="assets/videos/rotation_frame_animation.gif" alt="Synthetic rotation: frames animation with smooth deformation on the deformed lattice" width="90%"/>
</p>

**Hydrogel indentation, micro-CT, 306 x 1024 x 1024 voxels.** The deformed lattice with displacement arrows:

<p align="center">
  <img src="assets/videos/indentation_deformed_lattice_orbit_with_arrow.gif" alt="Hydrogel indentation: deformed lattice with displacement arrows, orbit" width="90%"/>
</p>

The deformation played frame by frame:

<p align="center">
  <img src="assets/videos/indentation_frame_smooth_animation.gif" alt="Hydrogel indentation: frames animation with smooth deformation" width="90%"/>
</p>

The displacement field swept slice by slice along x, y and z:

<p align="center">
  <img src="assets/videos/indentation_sweep_x.gif" alt="Slice sweep along x" width="90%"/>
</p>
<p align="center">
  <img src="assets/videos/indentation_sweep_y.gif" alt="Slice sweep along y" width="90%"/>
</p>
<p align="center">
  <img src="assets/videos/indentation_sweep_z.gif" alt="Slice sweep along z" width="90%"/>
</p>

## Install

```bash
pip install al-dvc           # CPU: application, 3-D view, command line
pip install "al-dvc[gpu]"    # the same plus the CUDA backend for NVIDIA GPUs
```

No Python? Every [release](https://github.com/zachtong/pyALDVC/releases) ships a portable Windows bundle: unzip, double-click `pyALDVC.exe`.

## Quick start

```bash
al-dvc-gui                                                              # the application
al-dvc run --volumes scan/ -o results --winsize 32 --step 16 --export npz vtk report
```

```python
from al_dvc import dvcpara_default, run_aldvc, load_volume

para = dvcpara_default(winsize=32, winstepsize=16, voxel_size=(5.0, 5.0, 5.0), units="um")
result = run_aldvc(para, [load_volume("ref.tif"), load_volume("deformed.tif")])
U = result.result_disp[0].U            # (N, 3) displacement per node
E = result.result_strain[0]            # exx, eyy, ezz, exy, exz, eyz, principal, von Mises
```

## Learn more

- [User guide](docs/user_guide.md): the application step by step.
- [Technical notes](docs/technical.md): algorithm, Python API, every parameter, accuracy and throughput, agreement with the MATLAB code.
- [Design document](docs/design.md): conventions and design decisions.
- [Tutorial notebook](examples/tutorial_real_data.ipynb): a complete run on real or synthetic data.

## Citation

> J. Yang, L. Hazlett, A. K. Landauer, C. Franck. Augmented Lagrangian
> Digital Volume Correlation (ALDVC). *Experimental Mechanics* 60, 1205-1223
> (2020). https://doi.org/10.1007/s11340-020-00607-3

and this software (see `CITATION.cff`).

## License

BSD 3-Clause. Developed in Dr. Jin Yang's group at The University of Texas
at Austin. Based on the MATLAB ALDVC code by Jin Yang and the pyALDIC
architecture by Zixiang (Zach) Tong.
