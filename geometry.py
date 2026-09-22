from __future__ import annotations

import math
from itertools import product

import numpy as np

from solver import Matrix, Solution, fmt


def row_kind(row) -> str:
    """'plane' | 'redundant' (0 0 0 | 0) | 'contradiction' (0 0 0 | k)."""
    if any(v != 0 for v in row[:-1]):
        return "plane"
    return "redundant" if row[-1] == 0 else "contradiction"


def row_caption(i: int, row) -> str | None:
    kind = row_kind(row)
    if kind == "redundant":
        return f"R{i + 1}: 0 = 0, redundant row"
    if kind == "contradiction":
        return f"R{i + 1}: 0 = {fmt(row[-1])}, contradiction"
    return None


def _cube_edges(R: float):
    verts = [np.array(v, dtype=float) for v in product((-R, R), repeat=3)]
    for a in verts:
        for b in verts:
            if (a < b).sum() == 1 and (a != b).sum() == 1:
                yield a, b


def plane_polygon(row, R: float) -> list[np.ndarray] | None:
    """Vertices (ordered around the boundary) of the plane clipped to the cube,
    or None if the row is not a plane or the plane misses the cube."""
    if row_kind(row) != "plane":
        return None
    n = np.array([float(v) for v in row[:3]])
    d = float(row[3])
    pts: list[np.ndarray] = []
    for a, b in _cube_edges(R):
        fa, fb = n @ a - d, n @ b - d
        if abs(fa) < 1e-12:
            pts.append(a)
        elif fa * fb < 0:
            pts.append(a + (b - a) * fa / (fa - fb))
    uniq: list[np.ndarray] = []
    for p in pts:
        if all(np.linalg.norm(p - q) > 1e-9 for q in uniq):
            uniq.append(p)
    if len(uniq) < 3:
        return None
    centre = np.mean(uniq, axis=0)
    u = uniq[0] - centre
    u /= np.linalg.norm(u)
    v = np.cross(n / np.linalg.norm(n), u)
    uniq.sort(key=lambda p: math.atan2((p - centre) @ v, (p - centre) @ u))
    return uniq


def clip_line(point, direction, R: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Segment of the line point + t*direction inside the cube (slab method)."""
    p = np.array([float(v) for v in point])
    d = np.array([float(v) for v in direction])
    lo, hi = -np.inf, np.inf
    for k in range(3):
        if abs(d[k]) < 1e-12:
            if abs(p[k]) > R:
                return None
            continue
        t1, t2 = (-R - p[k]) / d[k], (R - p[k]) / d[k]
        lo, hi = max(lo, min(t1, t2)), min(hi, max(t1, t2))
    if lo >= hi:
        return None
    return p + lo * d, p + hi * d


def view_range(matrices: list[Matrix], sol: Solution) -> float:
    """Half-width R of the view cube, computed ONCE from every frame so the axes
    never rescale mid-simulation. Keeps the solution point and each plane's
    closest point to the origin in view."""
    extent = 0.0
    A = np.array([[float(v) for v in r] for r in matrices[0]])
    ls = np.linalg.lstsq(A[:, :3], A[:, 3], rcond=None)[0]
    extent = max(extent, np.abs(ls).max())
    if sol.point is not None:
        extent = max(extent, max(abs(float(v)) for v in sol.point))
    for M in matrices:
        for row in M:
            if row_kind(row) == "plane":
                n = np.array([float(v) for v in row[:3]])
                closest = float(row[3]) * n / (n @ n)
                extent = max(extent, np.abs(closest).max())
    return float(min(max(math.ceil(extent * 1.4), 3), 20))


def tick_step(R: float) -> float:
    for s in (1, 2, 5, 10):
        if R / s <= 5:
            return s
    return 10
