"""
Tests for the Material class elastic constant calculations.

Verifies that the Material class correctly computes the third elastic
constant (E, G, or ν) from any given pair, using the isotropic relation:

    G = E / (2 * (1 + ν))

and that it raises a ValueError when all three constants are supplied
simultaneously.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fem.material import Material


# Reference values: E=210e3 MPa, ν=0.3 → G=80769.23... MPa
E_REF = 210e3
NU_REF = 0.3
G_REF = E_REF / (2.0 * (1.0 + NU_REF))


def _close(a, b, rel_tol=1e-9):
    return math.isclose(a, b, rel_tol=rel_tol)


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1: E and ν provided → G computed
# ─────────────────────────────────────────────────────────────────────────────

def test_e_nu_given_computes_g():
    """G is correctly derived from E and ν."""
    mat = Material(1, E=E_REF, nu=NU_REF)
    assert _close(mat.E, E_REF), f"E mismatch: {mat.E}"
    assert _close(mat.nu, NU_REF), f"ν mismatch: {mat.nu}"
    assert _close(mat.G, G_REF), f"G mismatch: {mat.G}"


def test_positional_args_backward_compat():
    """Material(id, E, nu) positional call still works."""
    mat = Material(1, E_REF, NU_REF)
    assert _close(mat.E, E_REF)
    assert _close(mat.nu, NU_REF)
    assert _close(mat.G, G_REF)


def test_default_nu_backward_compat():
    """Material(id, E) with only E still works (ν defaults to 0.3)."""
    mat = Material(1, E_REF)
    assert _close(mat.E, E_REF)
    assert _close(mat.nu, 0.3)
    assert _close(mat.G, E_REF / (2.0 * (1.0 + 0.3)))


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2: E and G provided → ν computed
# ─────────────────────────────────────────────────────────────────────────────

def test_e_g_given_computes_nu():
    """ν is correctly derived from E and G."""
    mat = Material(2, E=E_REF, G=G_REF)
    assert _close(mat.E, E_REF), f"E mismatch: {mat.E}"
    assert _close(mat.G, G_REF), f"G mismatch: {mat.G}"
    assert _close(mat.nu, NU_REF), f"ν mismatch: {mat.nu}"


# ─────────────────────────────────────────────────────────────────────────────
# Mode 3: G and ν provided → E computed
# ─────────────────────────────────────────────────────────────────────────────

def test_g_nu_given_computes_e():
    """E is correctly derived from G and ν."""
    mat = Material(3, G=G_REF, nu=NU_REF)
    assert _close(mat.G, G_REF), f"G mismatch: {mat.G}"
    assert _close(mat.nu, NU_REF), f"ν mismatch: {mat.nu}"
    assert _close(mat.E, E_REF), f"E mismatch: {mat.E}"


# ─────────────────────────────────────────────────────────────────────────────
# Consistency: all three modes produce the same E, G, ν triple
# ─────────────────────────────────────────────────────────────────────────────

def test_all_modes_consistent():
    """All three input modes produce identical E, G, ν values."""
    m1 = Material(1, E=E_REF, nu=NU_REF)
    m2 = Material(2, E=E_REF, G=G_REF)
    m3 = Material(3, G=G_REF, nu=NU_REF)

    for attr in ("E", "G", "nu"):
        v1 = getattr(m1, attr)
        v2 = getattr(m2, attr)
        v3 = getattr(m3, attr)
        assert _close(v1, v2), f"{attr}: m1={v1}, m2={v2}"
        assert _close(v1, v3), f"{attr}: m1={v1}, m3={v3}"


# ─────────────────────────────────────────────────────────────────────────────
# Error: all three values provided → ValueError
# ─────────────────────────────────────────────────────────────────────────────

def test_all_three_raises():
    """Supplying E, G, and ν simultaneously raises ValueError."""
    raised = False
    try:
        Material(4, E=E_REF, G=G_REF, nu=NU_REF)
    except ValueError:
        raised = True
    assert raised, "Expected ValueError when all three of E, G, ν are provided"


def test_error_message_mentions_three():
    """ValueError message references the three-parameter restriction."""
    try:
        Material(4, E=E_REF, G=G_REF, nu=NU_REF)
    except ValueError as exc:
        assert "two" in str(exc).lower() or "three" in str(exc).lower(), (
            f"Unexpected error message: {exc}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Error: fewer than two values provided
# ─────────────────────────────────────────────────────────────────────────────

def test_too_few_raises():
    """Supplying only G (without E or ν) raises ValueError."""
    raised = False
    try:
        Material(5, G=G_REF)
    except ValueError:
        raised = True
    assert raised, "Expected ValueError when only one of E, G, ν is provided"


# ─────────────────────────────────────────────────────────────────────────────
# id is stored correctly
# ─────────────────────────────────────────────────────────────────────────────

def test_id_stored():
    """Material id is preserved."""
    mat = Material(99, E=E_REF, nu=NU_REF)
    assert mat.id == 99


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_e_nu_given_computes_g,
        test_positional_args_backward_compat,
        test_default_nu_backward_compat,
        test_e_g_given_computes_nu,
        test_g_nu_given_computes_e,
        test_all_modes_consistent,
        test_all_three_raises,
        test_error_message_mentions_three,
        test_too_few_raises,
        test_id_stored,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  ✗ {t.__name__}: {exc}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    if failed:
        sys.exit(1)
