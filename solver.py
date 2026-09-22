"""Phase 1: exact Gauss elimination / Gauss-Jordan solver (no graphics).

Every row operation is recorded as a Frame so the renderer can animate it.
All arithmetic uses Fraction, so there is no floating-point drift.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction

Matrix = tuple[tuple[Fraction, ...], ...]


@dataclass(frozen=True)
class Frame:
    before: Matrix
    after: Matrix
    op_text: str
    op_type: str                  # "swap" | "scale" | "replace"
    changed_rows: tuple[int, ...]  # 0-based; two rows for a swap


@dataclass
class Solution:
    kind: str                     # "unique" | "infinite" | "none"
    rank_A: int
    rank_Ab: int
    nvars: int
    point: tuple[Fraction, ...] | None = None        # unique or particular solution
    directions: list[tuple[Fraction, ...]] = field(default_factory=list)  # null-space basis
    back_sub: list[str] = field(default_factory=list)  # back-substitution steps (elimination)


@dataclass
class Result:
    method: str                   # "elimination" | "jordan"
    initial: Matrix
    frames: list[Frame]
    final: Matrix
    solution: Solution


# ---------------------------------------------------------------- formatting

def fmt(q: Fraction) -> str:
    q = Fraction(q)
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def var_names(n: int) -> list[str]:
    return ["x", "y", "z"][:n] if n <= 3 else [f"x{i + 1}" for i in range(n)]


# ---------------------------------------------------------------- parsing

def parse_matrix(text: str) -> Matrix:
    """Parse "1 1 -1 -2; 2 -1 1 5; -1 2 2 1" (rows by ';' or newline, entries by
    spaces or commas; integers, decimals and a/b fractions allowed)."""
    rows = [r for r in re.split(r"[;\n]+", text.strip()) if r.strip()]
    if not rows:
        raise ValueError("Matrix is empty.")
    out = []
    for i, r in enumerate(rows, 1):
        try:
            out.append(tuple(Fraction(tok) for tok in re.split(r"[,\s]+", r.strip()) if tok))
        except (ValueError, ZeroDivisionError):
            raise ValueError(f"Row {i} has an entry that is not a number: {r.strip()!r}") from None
    width = len(out[0])
    if width < 2 or any(len(r) != width for r in out):
        raise ValueError("Every row needs the same number of entries (coefficients + right-hand side).")
    return tuple(out)


# ---------------------------------------------------------------- row operations

def _snap(M: list[list[Fraction]]) -> Matrix:
    return tuple(tuple(r) for r in M)


def _swap(M, frames, i, j):
    before = _snap(M)
    M[i], M[j] = M[j], M[i]
    frames.append(Frame(before, _snap(M), f"R{i + 1} ↔ R{j + 1}", "swap", (i, j)))


def _scale(M, frames, i, k: Fraction):
    before = _snap(M)
    M[i] = [k * v for v in M[i]]
    frames.append(Frame(before, _snap(M), f"R{i + 1} ← ({fmt(k)})·R{i + 1}", "scale", (i,)))


def _replace(M, frames, i, j, k: Fraction):
    """R_i <- R_i - k * R_j"""
    before = _snap(M)
    M[i] = [a - k * b for a, b in zip(M[i], M[j])]
    sign, mag = ("−", k) if k > 0 else ("+", -k)
    coef = "" if mag == 1 else f"{fmt(mag)}·"
    frames.append(Frame(before, _snap(M), f"R{i + 1} ← R{i + 1} {sign} {coef}R{j + 1}", "replace", (i,)))


def _reduce(A: Matrix, jordan: bool) -> tuple[list[Frame], Matrix, list[tuple[int, int]]]:
    """Row-reduce the augmented matrix A. Swaps only when the pivot is zero.
    Elimination clears below each pivot; Jordan scales pivots to 1 and clears
    above and below."""
    M = [list(r) for r in A]
    m, nvars = len(M), len(M[0]) - 1
    frames: list[Frame] = []
    pivots: list[tuple[int, int]] = []
    r = 0
    for c in range(nvars):
        if r >= m:
            break
        p = next((i for i in range(r, m) if M[i][c] != 0), None)
        if p is None:
            continue  # no pivot in this column: free variable
        if p != r:
            _swap(M, frames, r, p)
        if jordan and M[r][c] != 1:
            _scale(M, frames, r, 1 / M[r][c])
        for i in (range(m) if jordan else range(r + 1, m)):
            if i != r and M[i][c] != 0:
                _replace(M, frames, i, r, M[i][c] / M[r][c])
        pivots.append((r, c))
        r += 1
    return frames, _snap(M), pivots


def gauss_elimination(A: Matrix) -> list[Frame]:
    return _reduce(A, jordan=False)[0]


def gauss_jordan(A: Matrix) -> list[Frame]:
    return _reduce(A, jordan=True)[0]


# ---------------------------------------------------------------- classification

def rank(rows) -> int:
    return len(_reduce(tuple(tuple(r) + (Fraction(0),) for r in rows), jordan=False)[2])


def classify(A: Matrix) -> Solution:
    """unique / infinite / none from rank(A) vs rank([A|b]), plus the solution set
    read off the RREF."""
    nvars = len(A[0]) - 1
    rA = rank(tuple(r[:-1] for r in A))
    rAb = rank(A)
    if rA < rAb:
        return Solution("none", rA, rAb, nvars)
    _, R, pivots = _reduce(A, jordan=True)
    pivot_cols = {c for _, c in pivots}
    point = [Fraction(0)] * nvars
    for r, c in pivots:
        point[c] = R[r][-1]
    directions = []
    for f in (c for c in range(nvars) if c not in pivot_cols):
        d = [Fraction(0)] * nvars
        d[f] = Fraction(1)
        for r, c in pivots:
            d[c] = -R[r][f]
        directions.append(tuple(d))
    kind = "unique" if rA == nvars else "infinite"
    return Solution(kind, rA, rAb, nvars, tuple(point), directions)


def back_substitution(U: Matrix) -> list[str]:
    """Human-readable back-substitution steps for an upper-triangular system."""
    n = len(U[0]) - 1
    names = var_names(n)
    x: dict[int, Fraction] = {}
    steps = []
    for row in reversed(U[:n]):
        c = next((j for j in range(n) if row[j] != 0), None)
        if c is None:
            continue
        known = sum((row[j] * x[j] for j in range(c + 1, n)), Fraction(0))
        x[c] = (row[-1] - known) / row[c]
        lhs = f"{fmt(row[c])}{names[c]}" if row[c] != 1 else names[c]
        rhs = fmt(row[-1]) if known == 0 else f"{fmt(row[-1])} − ({fmt(known)})"
        steps.append(f"{lhs} = {rhs}  ⇒  {names[c]} = {fmt(x[c])}")
    return steps


def solve(A: Matrix, method: str) -> Result:
    if method not in ("elimination", "jordan"):
        raise ValueError(f"Unknown method {method!r}")
    frames, final, _ = _reduce(A, jordan=(method == "jordan"))
    sol = classify(A)
    if method == "elimination" and sol.kind == "unique":
        sol.back_sub = back_substitution(final)
    return Result(method, tuple(tuple(r) for r in A), frames, final, sol)


def describe_solution(sol: Solution) -> str:
    names = var_names(sol.nvars)
    if sol.kind == "none":
        return f"No solution: rank(A) = {sol.rank_A} < rank([A|b]) = {sol.rank_Ab}"
    if sol.kind == "unique":
        return "Unique solution: " + ", ".join(f"{n} = {fmt(v)}" for n, v in zip(names, sol.point))
    params = ["t", "s", "u", "v"][: len(sol.directions)]
    parts = []
    for i, name in enumerate(names):
        terms = [fmt(sol.point[i])] if sol.point[i] != 0 or not any(d[i] for d in sol.directions) else []
        for p, d in zip(params, sol.directions):
            if d[i] != 0:
                coef = "" if abs(d[i]) == 1 else f"{fmt(abs(d[i]))}"
                sign = "−" if d[i] < 0 else "+"
                terms.append(f"{sign} {coef}{p}" if terms else f"{'-' if d[i] < 0 else ''}{coef}{p}")
        parts.append(f"{name} = {' '.join(terms)}")
    return (f"Infinitely many solutions (rank {sol.rank_A} < {sol.nvars} unknowns): "
            + ", ".join(parts))
