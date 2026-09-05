# Synthetic Test Image Generation

Scripts for generating 3D synthetic sphere-packing volumes used for validation and evaluation of the autocorrelation analysis pipeline.

## Scripts

- **`Balls_generation_random.py`** - Generates a 3D volume with randomly packed non-overlapping spheres. Configurable parameters: volume size, sphere radius, surface gap, and target density.
- **`Balls_generation_uniform.py`** - Generates a 3D volume with uniformly spaced spheres on a regular grid. Configurable parameters: volume size, sphere radius, and grid spacing.

## Usage

```bash
python Balls_generation_random.py
python Balls_generation_uniform.py
```

Each script outputs a 3D TIFF file (uint16) in the current directory. Edit the parameter section at the top of each script to adjust sphere radius, spacing, volume size, etc.

## Dependencies

- numpy
- tifffile
- scipy (for `Balls_generation_random.py` only)
