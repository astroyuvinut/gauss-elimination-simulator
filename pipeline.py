from __future__ import annotations

import hashlib
import json
from pathlib import Path

from geometry import view_range
from scenes import render_clip
from solver import Matrix, Result, fmt, solve

CACHE_VERSION = "v6"  # bump when scenes.py changes so stale clips are not reused
RESULT_TITLE = {"unique": "The three planes meet in one point", "line": "The planes share a whole line",
                "plane": "The planes coincide", "space": "Every point solves the system",
                "none": "No point lies on all three planes"}
METHOD_LABEL = {"elimination": "Gauss Elimination", "jordan": "Gauss-Jordan"}


def result_info(res: Result) -> dict:
    """What the final clip should draw: a dot, a line, or just a caption."""
    sol = res.solution
    if sol.kind == "none":
        return {"kind": "none", "caption": "No common point: no solution"}
    assert sol.point is not None
    if sol.kind == "unique":
        return {"kind": "unique", "point": [float(v) for v in sol.point],
                "caption": "Unique solution  (" + ", ".join(fmt(v) for v in sol.point) + ")"}
    if len(sol.directions) == 1:
        return {"kind": "line", "point": [float(v) for v in sol.point],
                "direction": [float(v) for v in sol.directions[0]],
                "caption": "Planes meet in a line: infinitely many"}
    if len(sol.directions) == 2:
        return {"kind": "plane", "caption": "All planes coincide: infinitely many"}
    return {"kind": "space", "caption": "Every point is a solution"}


def build_specs(res: Result, duration: float = 2.4) -> list[dict]:
    """One spec per clip: intro, one per row operation, then the result."""
    frames = res.frames
    R = view_range([res.initial] + [f.after for f in frames], res.solution)
    to_list = lambda M: [[v for v in r] for r in M]  # noqa: E731  (Fractions kept exact)
    total = len(frames)
    specs = [{"kind": "intro", "index": 0, "before": to_list(res.initial), "after": to_list(res.initial),
              "rows": [], "op_text": f"{METHOD_LABEL[res.method]}: the starting system",
              "step_label": f"Start ({total} steps)"}]
    for n, f in enumerate(frames, 1):
        specs.append({"kind": f.op_type, "index": n, "before": to_list(f.before), "after": to_list(f.after),
                      "rows": list(f.changed_rows), "op_text": f.op_text, "step_label": f"Step {n} of {total}"})
    specs.append({"kind": "result", "index": total + 1, "before": to_list(res.final),
                  "after": to_list(res.final), "rows": [], "op_text": RESULT_TITLE[result_info(res)["kind"]],
                  "step_label": "Done", "result": result_info(res)})
    # widest string per column over the whole run -> matrix layout never shifts
    mats = [res.initial] + [f.after for f in frames]
    col_text = [max((fmt(M[i][j]) for M in mats for i in range(len(M))), key=len)
                for j in range(len(res.initial[0]))]
    for s in specs:
        s["R"] = R
        s["col_text"] = col_text
        s["duration"] = duration
    return specs


def cache_key(A: Matrix, method: str, quality: str, duration: float) -> str:
    raw = json.dumps([[fmt(v) for v in r] for r in A]) + method + quality + str(duration) + CACHE_VERSION
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def simulate(A: Matrix, method: str, out_root: Path, quality: str = "low_quality", duration: float = 2.4):
    """Yield (clip_path, spec, index, n_clips, result) for each clip, in order."""
    res = solve(A, method)
    specs = build_specs(res, duration)
    folder = Path(out_root) / cache_key(A, method, quality, duration)
    for i, spec in enumerate(specs):
        path = folder / f"clip_{i:02d}.mp4"
        if not path.exists():
            render_clip(spec, path, quality)
        yield path, spec, i, len(specs), res
