"""6-connected flood fill inside a small window (numba).

Shared by the subset splitting of the local kernels (component around the subset centre) and the
mesh cut (component around an element corner).
"""

from __future__ import annotations

from .._numba_compat import JIT_CACHE, njit


@njit(cache=JIT_CACHE)
def flood_fill_from(sub, out, queue, sz, sy, sx):
    """Mark in ``out`` the 6-connected component of ``sub`` (nonzero = material) that holds ``(sz, sy, sx)``.

    ``sub`` and ``out`` are ``(Sz, Sy, Sx)`` uint8 windows, ``queue`` an int64 scratch array of
    ``Sz * Sy * Sx`` entries; ``out`` is cleared first. Paths stay inside the window. Returns the
    number of voxels marked, 0 when the seed voxel itself is not material.
    """
    Sz = sub.shape[0]
    Sy = sub.shape[1]
    Sx = sub.shape[2]
    for iz in range(Sz):
        for iy in range(Sy):
            for ix in range(Sx):
                out[iz, iy, ix] = 0
    if sz < 0 or sz >= Sz or sy < 0 or sy >= Sy or sx < 0 or sx >= Sx or sub[sz, sy, sx] == 0:
        return 0
    head = 0
    tail = 0
    queue[tail] = (sz * Sy + sy) * Sx + sx
    tail += 1
    out[sz, sy, sx] = 1
    count = 1
    while head < tail:
        lin = queue[head]
        head += 1
        iz = lin // (Sy * Sx)
        rem = lin - iz * (Sy * Sx)
        iy = rem // Sx
        ix = rem - iy * Sx
        for k in range(6):
            jz = iz
            jy = iy
            jx = ix
            if k == 0:
                jz = iz - 1
            elif k == 1:
                jz = iz + 1
            elif k == 2:
                jy = iy - 1
            elif k == 3:
                jy = iy + 1
            elif k == 4:
                jx = ix - 1
            else:
                jx = ix + 1
            if jz < 0 or jz >= Sz or jy < 0 or jy >= Sy or jx < 0 or jx >= Sx:
                continue
            if sub[jz, jy, jx] == 0 or out[jz, jy, jx] != 0:
                continue
            out[jz, jy, jx] = 1
            count += 1
            queue[tail] = (jz * Sy + jy) * Sx + jx
            tail += 1
    return count


@njit(cache=JIT_CACHE)
def flood_fill_centre(sub, out, queue):
    """:func:`flood_fill_from` seeded at the centre voxel of the window."""
    return flood_fill_from(sub, out, queue, sub.shape[0] // 2, sub.shape[1] // 2, sub.shape[2] // 2)
