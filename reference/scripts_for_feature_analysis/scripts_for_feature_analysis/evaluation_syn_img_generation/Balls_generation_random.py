#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Sphere Packing Generator

DESCRIPTION:
Generates a 3D volume with randomly packed non-overlapping spheres using a
rejection sampling algorithm with KD-tree-based nearest neighbor checks.

OUTPUTS:
- A 3D TIFF file (uint16) named '3d_random_spheres_rad_{R}px_gap_{G}px_size_{S}px.tif'

DEPENDENCIES:
- numpy, tifffile, scipy

AUTHORS: Zixiang (Zach) Tong, @UT-Austin; Yujie Zhang, @UT-Austin
DATE: 2025.02.16
"""

import numpy as np
import tifffile
from scipy.spatial import cKDTree

# Configuration parameters
size = 250            # Cube dimension (voxels)
radius = 12           # Sphere radius (voxels)
surface_gap = 18      # Minimum surface-to-surface gap between spheres (voxels)
target_density = 0.5  # Target packing density coefficient (0.0 to 1.0)

# Create 3D volume array
volume = np.zeros((size, size, size), dtype=np.uint16)

# Generate sphere template
x, y, z = np.ogrid[-radius:radius + 1, -radius:radius + 1, -radius:radius + 1]
mask = (x ** 2 + y ** 2 + z ** 2 <= radius ** 2).astype(np.uint16) * 65535

# Initialize coordinate storage (ensure 2D structure)
coords = np.empty((0, 3), dtype=np.int32)
min_center_dist = 2 * radius + surface_gap
max_spheres = int((size ** 3 * target_density) / ((4 / 3) * np.pi * radius ** 3))
attempts = 0

# Generate randomly placed spheres via rejection sampling
while coords.shape[0] < max_spheres and attempts < max_spheres * 100:
    candidate = np.random.randint(radius, size - radius, (1, 3))
    if coords.size == 0:
        coords = np.vstack([coords, candidate])
        kd_tree = cKDTree(coords)
    else:
        dist, _ = kd_tree.query(candidate, k=1, distance_upper_bound=min_center_dist)
        if dist[0] > min_center_dist:
            coords = np.vstack([coords, candidate])
            kd_tree = cKDTree(coords)  # Rebuild KD-tree with new point
    attempts += 1

# Write sphere data into volume
for i in range(coords.shape[0]):
    cx, cy, cz = coords[i, :]
    # Compute volume slice ranges (with boundary clamping)
    x_start = max(0, cx - radius)
    x_end = min(size, cx + radius + 1)
    y_start = max(0, cy - radius)
    y_end = min(size, cy + radius + 1)
    z_start = max(0, cz - radius)
    z_end = min(size, cz + radius + 1)

    # Compute template offset ranges
    dx_start = max(radius - cx, 0)
    dx_end = min(radius + (size - cx), 2 * radius + 1)
    dy_start = max(radius - cy, 0)
    dy_end = min(radius + (size - cy), 2 * radius + 1)
    dz_start = max(radius - cz, 0)
    dz_end = min(radius + (size - cz), 2 * radius + 1)

    # Apply sphere template to volume
    volume[x_start:x_end, y_start:y_end, z_start:z_end] |= \
        mask[dx_start:dx_end, dy_start:dy_end, dz_start:dz_end]

# Save as TIFF
tifffile.imwrite(f'3d_random_spheres_rad_{radius}px_gap_{surface_gap}px_size_{size}px.tif', volume)
