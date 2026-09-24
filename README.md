<p align="center">
  <a href="https://zachtong.github.io/pyALDVC/"><img src="https://raw.githubusercontent.com/zachtong/pyALDVC/main/assets/readme/banner.png" alt="pyALDVC: augmented Lagrangian digital volume correlation in Python" width="800"></a>
</p>

<h1 align="center">pyALDVC: A Python Implementation of Augmented Lagrangian Digital Volume Correlation</h1>

<h3 align="center">3-D displacement and strain, <i>inside the material</i></h3>

<p align="center">
  Open-source digital volume correlation for micro-CT, confocal and other 3-D scans:<br>
  a desktop application, a command line and a Python library, with optional GPU acceleration.
</p>

<p align="center">
  <a href="https://zachtong.github.io/pyALDVC/"><img src="https://raw.githubusercontent.com/zachtong/pyALDVC/main/assets/readme/button-website.png" alt="Visit the website" width="256"></a>
</p>
<p align="center">
  <b>Website: <a href="https://zachtong.github.io/pyALDVC/">zachtong.github.io/pyALDVC</a></b><br>
  <sub>cases, how it works, accuracy and speed, all in one page</sub>
</p>

<p align="center">
  <a href="https://zachtong.github.io/pyALDVC/#start"><img src="https://raw.githubusercontent.com/zachtong/pyALDVC/main/assets/readme/button-start.png" alt="Get started" width="138"></a>
  <a href="https://github.com/zachtong/pyALDVC/releases/latest"><img src="https://raw.githubusercontent.com/zachtong/pyALDVC/main/assets/readme/button-download.png" alt="Windows download" width="203"></a>
  <a href="https://github.com/zachtong/pyALDVC/blob/main/docs/user_guide.md"><img src="https://raw.githubusercontent.com/zachtong/pyALDVC/main/assets/readme/button-guide.png" alt="User guide" width="133"></a>
</p>

<p align="center">
  <a href="https://pypi.org/project/al-dvc/"><img src="https://img.shields.io/pypi/v/al-dvc?style=flat-square&label=PyPI&color=4f46e5" alt="PyPI version"></a>
  <a href="https://github.com/zachtong/pyALDVC/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/zachtong/pyALDVC/ci.yml?branch=main&style=flat-square&label=tested%20on%20Python%203.10%E2%80%933.12&logo=python&logoColor=white" alt="CI status: tested on Python 3.10 to 3.12"></a>
  <img src="https://img.shields.io/badge/interface-7%20languages-7c3aed?style=flat-square" alt="Interface in 7 languages">
  <a href="https://doi.org/10.5281/zenodo.22883767"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22883767-0b7285?style=flat-square" alt="Zenodo DOI 10.5281/zenodo.22883767"></a>
  <a href="https://github.com/zachtong/pyALDVC/blob/main/LICENSE"><img src="https://img.shields.io/badge/licence-BSD--3--Clause-16a34a?style=flat-square" alt="BSD 3-Clause licence"></a>
</p>

<p align="center">
  <a href="https://zachtong.github.io/pyALDVC/#hydrogel"><img src="https://raw.githubusercontent.com/zachtong/pyALDVC/main/assets/readme/hydrogel-orbit.gif" alt="Orbiting 3-D view of a hydrogel block with a circular dimple of downward displacement and arrows around it" width="720"></a>
  <br>
  <sub>Sphere indentation of a hydrogel, confocal scan of 1024&nbsp;×&nbsp;1024&nbsp;×&nbsp;306 voxels: vertical displacement
  on the deformed node grid, down to −10.5&nbsp;voxels (−4.5&nbsp;µm) under the sphere; 144,342 nodes in 71&nbsp;s on
  one GPU. Measured with pyALDVC (local subset solver).</sub>
</p>

<p align="center">
  <a href="https://zachtong.github.io/pyALDVC/#accuracy"><img src="https://raw.githubusercontent.com/zachtong/pyALDVC/main/assets/readme/stats.png" alt="Key numbers: 34 s on one NVIDIA RTX 5090 for a 1024 × 1024 × 306 confocal pair with 79,200 nodes (5.0 min on a 24-core CPU); 0.001 to 0.006 voxel displacement error on synthetic translation, rotation and 2 % strain; 0.005 to 0.020 voxel median difference from the MATLAB ALDVC code on the same scan; 14 GB peak volume memory of a masked 1024³ run (was 53 GB); texture analysis of a 256³ volume in 1.2 s; 7 interface languages" width="800"></a>
</p>

**pyALDVC** measures full-field displacement and strain inside a material from a sequence of 3-D scans. It is the
Python version of the MATLAB [ALDVC](https://github.com/FranckLab/ALDVC) code (Yang, Hazlett, Landauer, Franck,
*Exp. Mech.* 2020) and the volumetric sibling of [pyALDIC](https://github.com/zachtong/pyALDIC): free, open source,
and usable without writing a line of code.

## Cases

<table align="center">
  <tr>
    <td align="center" valign="top" width="50%">
      <a href="https://zachtong.github.io/pyALDVC/#foam"><img src="https://raw.githubusercontent.com/zachtong/pyALDVC/main/assets/readme/foam-slice-sweep.gif" alt="Foam cylinder with orthogonal micro-CT slices; a colour map of displacement magnitude sweeps through it" width="320"></a>
      <br><b>Foam under compression</b>
      <br><sub>Micro-CT slices, 987 × 1009 × 1856 voxels</sub>
    </td>
    <td align="center" valign="top" width="50%">
      <a href="https://zachtong.github.io/pyALDVC/#foam"><img src="https://raw.githubusercontent.com/zachtong/pyALDVC/main/assets/readme/foam-lattice.gif" alt="Foam cylinder drawn as a node grid coloured by displacement, with arrows along the compression axis" width="320"></a>
      <br><b>Foam: displacement on the node grid</b>
      <br><sub>48,720 nodes in about 3&nbsp;min; deformation exaggerated 2×</sub>
    </td>
  </tr>
  <tr>
    <td align="center" valign="top" width="50%">
      <a href="https://zachtong.github.io/pyALDVC/#rotation"><img src="https://raw.githubusercontent.com/zachtong/pyALDVC/main/assets/readme/rigid-rotation.gif" alt="A cube of synthetic beads, coloured by horizontal displacement, rotating step by step" width="360"></a>
      <br><b>Rigid-body rotation</b>
      <br><sub>Synthetic, six 5° steps to 30° (looped forward and back); tracked in 34&nbsp;s</sub>
    </td>
    <td align="center" valign="top" width="50%">
      <a href="https://zachtong.github.io/pyALDVC/#cavitation"><img src="https://raw.githubusercontent.com/zachtong/pyALDVC/main/assets/readme/lic-strain.gif" alt="Residual von Mises strain left by laser-induced cavitation" width="360"></a>
      <br><b>Laser-induced cavitation</b>
      <br><sub>Residual von Mises strain (unpublished data)</sub>
    </td>
  </tr>
</table>

<p align="center"><a href="https://zachtong.github.io/pyALDVC/#cases"><b>The full cases, with the experiments behind them, on the website →</b></a></p>

<sub>The foam, hydrogel and rotation fields were measured with pyALDVC (local subset solver) on one GPU. Their data are
part of the DVC Challenge 2.0 dataset, [doi:10.18434/mds2-4129](https://doi.org/10.18434/mds2-4129), described in
Tong, Z. et al. Digital Volume Correlation Challenge 2.0: A Comprehensive Dataset for Digital Volume Correlation
Benchmarking. Research Square preprint (2026). [https://doi.org/10.21203/rs.3.rs-9683321/v1](https://doi.org/10.21203/rs.3.rs-9683321/v1).
Foam data courtesy of NIST (Landauer et al., *Sci. Data* 10, 356, 2023).</sub>

## What it does

<table>
  <tr>
    <td width="50%" valign="top"><b>Point and click</b><br>Load the scans, draw the region of interest on the slices, run and export, in seven languages. No code.</td>
    <td width="50%" valign="top"><b>AL-DVC solver</b><br>Local subset fits coupled to one smooth, compatible field: cleaner gradients, and masked cracks and holes stay sharp.</td>
  </tr>
  <tr>
    <td valign="top"><b>NVIDIA GPU</b><br><code>pip install "al-dvc[gpu]"</code> runs the local solvers as CUDA kernels, typically within 10<sup>−5</sup> voxel of the CPU.</td>
    <td valign="top"><b>Large volumes</b><br>Sub-box local steps, streamed frames and on-the-fly gradients: a masked 1024³ run peaks at 14&nbsp;GB.</td>
  </tr>
  <tr>
    <td valign="top"><b>Texture analysis</b><br>Measures the correlation length of your scan and suggests the subset size and step.</td>
    <td valign="top"><b>Strain</b><br>Four gradient methods × four measures (infinitesimal, Green–Lagrange, Euler–Almansi, Hencky).</td>
  </tr>
  <tr>
    <td valign="top"><b>Statistics and rigid-body motion</b><br>Means with 95&nbsp;% confidence intervals, the noise floor, regions, profiles, a virtual extensometer.</td>
    <td valign="top"><b>3-D view and animations</b><br>Field slices, the deformed node grid and arrows; orbits and sweeps recorded as GIF or MP4.</td>
  </tr>
  <tr>
    <td valign="top"><b>Formats</b><br>TIFF, MATLAB, NumPy, HDF5, NIfTI, NRRD, DICOM in; NumPy, MATLAB, CSV, ParaView, PDF out.</td>
    <td valign="top"><b>Sessions, batch, command line</b><br>Save sessions, queue batches, resume from checkpoints, or script it with <code>al-dvc</code> and <code>al_dvc.run_aldvc</code>.</td>
  </tr>
</table>

## How it works

The method, and the tools that tell you how far to trust a result. Every figure is computed from synthetic volumes
with a known answer, by a script in this repository. The [website](https://zachtong.github.io/pyALDVC/#how) explains
each one in full, with the AL-DVC method itself.

**Tracking a sequence.** Accumulative tracking (the default) correlates every scan with the first, so errors do not
add up; incremental tracking correlates each scan with the one before and chains the steps, and follows motion too
large for one step.

<p align="center"><a href="https://zachtong.github.io/pyALDVC/#tracking"><img src="https://raw.githubusercontent.com/zachtong/pyALDVC/main/site/figures/tracking_modes.png" alt="Median error and converged nodes of accumulative and incremental tracking of a cylinder turning 5 degrees between scans, up to 45 degrees" width="800"></a></p>

**Texture analysis.** How far the grey values stay correlated sets the subset size: four correlation lengths per axis,
past which a larger subset barely lowers the error.

<p align="center"><a href="https://zachtong.github.io/pyALDVC/#texture"><img src="https://raw.githubusercontent.com/zachtong/pyALDVC/main/site/figures/texture_subset.png" alt="A synthetic sphere texture with subsets of 1, 2 and 4 correlation lengths, and the displacement error against subset size" width="800"></a></p>

**How precise is the result?** A noise floor from static scans, the predicted error of every node, and confidence
intervals that account for correlated neighbours.

<p align="center"><a href="https://zachtong.github.io/pyALDVC/#errors"><img src="https://raw.githubusercontent.com/zachtong/pyALDVC/main/site/figures/uncertainty_map.png" alt="Predicted displacement uncertainty, actual error and their calibration where the texture contrast fades" width="800"></a></p>

**Removing rigid-body motion.** A specimen that shifts or turns adds displacement that is not deformation, and a
rotation reads as false strain. A closed-form rigid fit removes it and reveals the deformation underneath.

<p align="center"><a href="https://zachtong.github.io/pyALDVC/#rigid"><img src="https://raw.githubusercontent.com/zachtong/pyALDVC/main/site/figures/rigid_removal.png" alt="Displacement arrows as measured and with the rigid motion removed, and the mean normal strains" width="800"></a></p>

## Accuracy and speed

| Test | Result |
|---|---|
| Synthetic rigid translation or 2 % strain | 0.003–0.006 voxel error |
| Synthetic 5° rotation, up to 8 voxels of motion | 0.001 voxel error |
| Synthetic translation (12.3, −9.6, 7.4) + 1 % strain | 0.004–0.005 voxel error |
| Synthetic 2 % strain with noise (SNR 6) | 0.011–0.012 voxel error |
| Confocal pair, 1024 × 1024 × 306, 79,200 nodes | 34 s on one RTX 5090 (5.0 min on a 24-core CPU) |
| Same pair, against MATLAB ALDVC (u, v, w) | median difference 0.005 / 0.006 / 0.020 voxel |
| Masked 1024³ run | 14 GB peak volume memory (was 53 GB) |

<sub>Synthetic rows: RMS error of each displacement component at the interior nodes, subset 16, step 8 voxels, default
settings. Confocal rows: subset 32, step 8 voxels, with the settings of the MATLAB example run. Tested on Python 3.10,
3.11 and 3.12 on every push to main.</sub>

<details>
<summary><b>Coming from the MATLAB ALDVC code?</b></summary>
<br>

| | MATLAB ALDVC | pyALDVC |
|---|---|---|
| Interface | scripts | desktop application in 7 languages, command line, Python library |
| GPU | – | NVIDIA CUDA, one install flag |
| Region of interest | box | masks drawn on the slices, automatic masks, per-frame masks |
| Subset size | by hand | suggested by texture analysis of the scan |
| Cracks and holes | subsets and smoothing reach across them | masked cracks and holes split the subsets and the node grid |
| Strain and statistics | in the run; mean and std of uniform strain | own window: 4 methods × 4 measures, regions, confidence intervals, series over frames, profiles, extensometer, noise floor, rigid-body motion removed |
| Large scans, long sequences | whole volume in memory | local steps over sub-boxes, streamed frames, gradients on the fly; checkpoints, resume, batch runs, sessions |
| Install | MATLAB licence | `pip install al-dvc`, or a portable Windows bundle |

</details>

## Install

```bash
conda create -n pyaldvc python=3.12 -y
conda activate pyaldvc
pip install al-dvc
al-dvc
```

`pip install "al-dvc[gpu]"` in place of the third line adds NVIDIA GPU support; it needs the NVIDIA driver, not the
CUDA Toolkit. The last command opens the application, and `al-dvc --self-test` checks the install. Optional:
`nibabel`, `pynrrd` and `pydicom` read NIfTI, NRRD and DICOM; `imageio` with `imageio-ffmpeg` records MP4.

**No Python?** Every [release](https://github.com/zachtong/pyALDVC/releases/latest) ships a portable Windows bundle
(CPU only): unzip it and double-click `pyALDVC.exe`. Then read the
[user guide](https://github.com/zachtong/pyALDVC/blob/main/docs/user_guide.md).

## Citation

If pyALDVC helps your research, please cite the software (the concept DOI always resolves to the latest release) and
the method:

> Tong, Z., Yang, J. pyALDVC: Augmented Lagrangian Digital Volume Correlation in Python. Zenodo (2026).
> https://doi.org/10.5281/zenodo.22883767
>
> Yang, J., Hazlett, L., Landauer, A. K., Franck, C. Augmented Lagrangian Digital Volume Correlation (ALDVC).
> *Experimental Mechanics* 60, 1205–1223 (2020). https://doi.org/10.1007/s11340-020-00607-3

<details>
<summary>BibTeX</summary>

```bibtex
@software{tong_pyaldvc,
  author    = {Tong, Zixiang and Yang, Jin},
  title     = {{pyALDVC}: Augmented Lagrangian Digital Volume Correlation in Python},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.22883767},
  url       = {https://github.com/zachtong/pyALDVC}
}

@article{yang_aldvc_2020,
  author  = {Yang, Jin and Hazlett, Lauren and Landauer, Alexander K. and Franck, Christian},
  title   = {Augmented {Lagrangian} Digital Volume Correlation ({ALDVC})},
  journal = {Experimental Mechanics},
  volume  = {60},
  number  = {9},
  pages   = {1205--1223},
  year    = {2020},
  doi     = {10.1007/s11340-020-00607-3}
}
```

</details>

## Licence

BSD 3-Clause. Developed by Zixiang Tong and Jin Yang in Dr. Jin Yang's group at The University of Texas at Austin.

<p align="center">
  <a href="https://zachtong.github.io/pyALDVC/"><img src="https://raw.githubusercontent.com/zachtong/pyALDVC/main/assets/readme/button-website.png" alt="Visit the website" width="256"></a>
</p>
