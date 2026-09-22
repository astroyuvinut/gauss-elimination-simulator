"""Phase 2: check the solver against numpy on the four canonical cases."""
from fractions import Fraction

import numpy as np
import pytest

from solver import classify, gauss_elimination, gauss_jordan, parse_matrix, solve

CASES = {
    "unique": "1 1 -1 -2; 2 -1 1 5; -1 2 2 1",
    "zero_pivot": "0 2 1 7; 1 1 1 6; 2 1 -1 1",      # a11 = 0 forces a swap
    "infinite": "1 1 1 6; 2 2 2 12; 1 -1 2 5",
    "none": "1 1 1 6; 2 2 2 13; 1 -1 2 5",
}
EXPECTED_KIND = {"unique": "unique", "zero_pivot": "unique", "infinite": "infinite", "none": "none"}


def as_np(M):
    return np.array([[float(v) for v in r] for r in M])


def is_multiple(u, v):
    """True if u == k*v for some scalar k (v nonzero)."""
    j = next(i for i, x in enumerate(v) if x != 0)
    k = u[j] / v[j]
    return all(a == k * b for a, b in zip(u, v)), k


def check_op(frame):
    """Verify the changed row really is the result of the named elementary op."""
    B, A = frame.before, frame.after
    if frame.op_type == "swap":
        i, j = frame.changed_rows
        assert A[i] == B[j] and A[j] == B[i]
    elif frame.op_type == "scale":
        (i,) = frame.changed_rows
        ok, k = is_multiple(A[i], B[i])
        assert ok and k != 0
    else:  # replace: A[i] - B[i] must be a multiple of some other row
        (i,) = frame.changed_rows
        diff = tuple(a - b for a, b in zip(A[i], B[i]))
        assert any(diff) and any(is_multiple(diff, B[j])[0]
                                 for j in range(len(B)) if j != i and any(B[j]))


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("method", ["elimination", "jordan"])
def test_against_numpy(name, method):
    A = parse_matrix(CASES[name])
    Ab = as_np(A)
    res = solve(A, method)
    sol = res.solution

    assert sol.kind == EXPECTED_KIND[name]
    assert sol.rank_A == np.linalg.matrix_rank(Ab[:, :-1])
    assert sol.rank_Ab == np.linalg.matrix_rank(Ab)

    if sol.kind == "unique":
        expected = np.linalg.solve(Ab[:, :-1], Ab[:, -1])
        assert np.allclose([float(v) for v in sol.point], expected)
    if sol.kind == "infinite":
        A_, b_ = Ab[:, :-1], Ab[:, -1]
        p = np.array([float(v) for v in sol.point])
        assert np.allclose(A_ @ p, b_)
        for d in sol.directions:
            assert np.allclose(A_ @ np.array([float(v) for v in d]), 0)
        assert len(sol.directions) == A_.shape[1] - sol.rank_A


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("reduce", [gauss_elimination, gauss_jordan])
def test_frames_are_continuous_and_valid(name, reduce):
    A = parse_matrix(CASES[name])
    frames = reduce(A)
    assert frames, "every test system needs at least one row operation"
    assert frames[0].before == A
    for f, g in zip(frames, frames[1:]):
        assert f.after == g.before
    for f in frames:
        assert f.op_type in ("swap", "scale", "replace")
        # only the listed rows change
        for r in range(len(A)):
            if r not in f.changed_rows:
                assert f.before[r] == f.after[r]
        check_op(f)
        # row ops never change the rank of [A|b]
        assert np.linalg.matrix_rank(as_np(f.after)) == np.linalg.matrix_rank(as_np(f.before))


def test_zero_pivot_swaps_first():
    frames = gauss_elimination(parse_matrix(CASES["zero_pivot"]))
    assert frames[0].op_type == "swap" and frames[0].changed_rows == (0, 1)


def test_elimination_gives_upper_triangular():
    res = solve(parse_matrix(CASES["unique"]), "elimination")
    U = res.final
    assert all(U[i][j] == 0 for i in range(3) for j in range(i))
    assert res.solution.back_sub  # back-substitution steps were produced


def test_jordan_ends_at_identity():
    res = solve(parse_matrix(CASES["unique"]), "jordan")
    for i in range(3):
        assert res.final[i][:3] == tuple(Fraction(int(i == j)) for j in range(3))
        assert res.final[i][3] == res.solution.point[i]


def test_parse_accepts_fractions_decimals_newlines():
    A = parse_matrix("1/2, 0.5 1 2\n1 1 1 3")
    assert A[0][:2] == (Fraction(1, 2), Fraction(1, 2))


@pytest.mark.parametrize("bad", ["", "1 2 3; 4 5", "1 a 3 4"])
def test_parse_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        parse_matrix(bad)
