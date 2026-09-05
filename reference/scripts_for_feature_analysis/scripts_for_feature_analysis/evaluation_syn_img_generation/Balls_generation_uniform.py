#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uniform Sphere Grid Generator

DESCRIPTION:
Generates a 3D volume with uniformly spaced spheres arranged on a regular grid.

OUTPUTS:
- A 3D TIFF file (uint16) named '3d_spheres_{S}x{S}x{S}.tif'

DEPENDENCIES:
- numpy, tifffile

AUTHORS: Zixiang (Zach) Tong, @UT-Austin; Yujie Zhang, @UT-Austin
DATE: 2025.01.23
"""

import numpy as np
import tifffile

# Configuration parameters
size = 250     # Cube dimension (voxels)
radius = 10    # Sphere radius (voxels)
spacing = 40   # Center-to-center spacing (voxels)

# Create 3D volume array initialized to zero (black)
volume = np.zeros((size, size, size), dtype=np.uint16)

# Generate sphere center coordinate grid
centers = np.arange(0, size, spacing)
x_centers, y_centers, z_centers = np.meshgrid(centers, centers, centers, indexing='ij')

# Create sphere template
x, y, z = np.ogrid[-radius:radius + 1, -radius:radius + 1, -radius:radius + 1]
mask = (x ** 2 + y ** 2 + z ** 2 <= radius ** 2).astype(np.uint16) * 65535

# Place spheres at each grid point using slice operations
for cx in centers:
    x_slice = slice(max(0, cx - radius), min(size, cx + radius + 1))
    mx_slice = slice(max(0, radius - cx), min(2 * radius + 1, radius + (size - cx)))

    for cy in centers:
        y_slice = slice(max(0, cy - radius), min(size, cy + radius + 1))
        my_slice = slice(max(0, radius - cy), min(2 * radius + 1, radius + (size - cy)))

        for cz in centers:
            z_slice = slice(max(0, cz - radius), min(size, cz + radius + 1))
            mz_slice = slice(max(0, radius - cz), min(2 * radius + 1, radius + (size - cz)))

            # Apply sphere template via bitwise OR
            volume[x_slice, y_slice, z_slice] |= mask[mx_slice, my_slice, mz_slice]

# Save as 16-bit TIFF file
tifffile.imwrite(f'3d_spheres_{size}x{size}x{size}.tif', volume)
