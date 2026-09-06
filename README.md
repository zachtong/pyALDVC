<p align="center">
  <img src="assets/banner.png" alt="pyALDVC banner" width="100%"/>
</p>

<p align="center">
  Full-field 3-D displacement and strain from volumetric images (micro-CT, confocal, MRI, OCT).
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
[pyALDIC](https://github.com/zachtong/pyALDIC): a desktop application that turns a
sequence of 3-D scans into displacement and strain fields.

<p align="center">
  <img src="assets/pyALDVC_demo.gif" alt="pyALDVC workflow: load volumes, draw a region of interest, run, strain post-processing, 3-D view" width="90%"/>
</p>

## Why pyALDVC

- **Accurate where subset DVC breaks down.** Local subsets are coupled to a global smoothness step, so steep gradients, boundaries and noisy scans stay sub-voxel accurate.
- **Fast.** A 1024 x 1024 x 306 micro-CT scan with 79 200 nodes takes 23 s on an NVIDIA GPU, 3.6 min on a 24-core CPU.
- **Point and click.** Load the scans, draw the region of interest on the slices, run, look, export. No code.
- **Knows your data.** The texture analysis measures your scan and suggests the subset size and step.
- **See it in 3-D.** Field slices, the deformed lattice, displacement arrows; animations recorded as GIF or MP4.
- **Strain included.** Four gradient methods and four strain measures, computed after the run in their own window.
- **Every format.** TIFF, MATLAB, NumPy, HDF5, NIfTI, NRRD, DICOM in; NumPy, MATLAB, CSV, ParaView and a PDF report out.

## Case studies

**Synthetic rotation**

<p align="center">
  <img src="assets/videos/rotation_frame_animation.gif" alt="Synthetic rotation: frames animation with smooth deformation on the deformed lattice" width="90%"/>
</p>

**Hydrogel indentation, micro-CT, 306 x 1024 x 1024 voxels**

<p align="center">
  <img src="assets/videos/indentation_deformed_lattice_orbit_with_arrow.gif" alt="Hydrogel indentation: deformed lattice with displacement arrows, orbit" width="90%"/>
</p>
<p align="center">
  <img src="assets/videos/indentation_frame_smooth_animation.gif" alt="Hydrogel indentation: frames animation with smooth deformation" width="90%"/>
</p>
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
pip install al-dvc            # add "[gpu]" for the NVIDIA backend
al-dvc-gui
```

No Python? Every [release](https://github.com/zachtong/pyALDVC/releases) ships a portable Windows bundle: unzip, double-click `pyALDVC.exe`.

Read the [user guide](docs/user_guide.md) to get started.

## Citation

> J. Yang, L. Hazlett, A. K. Landauer, C. Franck. Augmented Lagrangian
> Digital Volume Correlation (ALDVC). *Experimental Mechanics* 60, 1205-1223
> (2020). https://doi.org/10.1007/s11340-020-00607-3

## License

BSD 3-Clause. Developed in Dr. Jin Yang's group at The University of Texas
at Austin.
