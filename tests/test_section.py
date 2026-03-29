"""
Tests for every formula in section.py, including shear coefficients.

Covers:
  - RectangularBar
  - RectangularTube
  - CircularBar
  - CircularTube
  - TrapezoidalBar
  - TrapezoidalTube
  - HexagonalBar
  - HexagonalTube
  - IBeam
  - CSection
  - LSection
  - TSection
  - ZSection
  - HatSection
  - GeneralSection
  - Section.normal_stress
  - create_section factory
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fem.section import (
    RectangularBar,
    RectangularTube,
    CircularBar,
    CircularTube,
    TrapezoidalBar,
    TrapezoidalTube,
    HexagonalBar,
    HexagonalTube,
    IBeam,
    CSection,
    LSection,
    TSection,
    ZSection,
    HatSection,
    GeneralSection,
    create_section,
)


def _close(a, b, rel_tol=1e-9):
    return math.isclose(a, b, rel_tol=rel_tol)


# =============================================================================
# RectangularBar
# =============================================================================

def test_rectangular_bar_area():
    """area = width * height"""
    s = RectangularBar(1, width=0.1, height=0.2)
    assert _close(s.area, 0.1 * 0.2), f"area={s.area}"


def test_rectangular_bar_inertia():
    """inertia = width * height^3 / 12"""
    s = RectangularBar(1, width=0.1, height=0.2)
    expected = 0.1 * 0.2**3 / 12
    assert _close(s.inertia, expected), f"inertia={s.inertia}"


def test_rectangular_bar_shear_coefficient():
    """RectangularBar inherits default shear coefficient 5/6"""
    s = RectangularBar(1, width=0.1, height=0.2)
    assert _close(s.shear_coefficient, 5 / 6), f"kappa={s.shear_coefficient}"


# =============================================================================
# RectangularTube
# =============================================================================

def test_rectangular_tube_area():
    """area = outer_area - inner_area"""
    w, h, t = 0.1, 0.2, 0.01
    s = RectangularTube(1, width=w, height=h, thickness=t)
    expected = w * h - (w - 2 * t) * (h - 2 * t)
    assert _close(s.area, expected), f"area={s.area}"


def test_rectangular_tube_inertia():
    """inertia = (w*h^3 - (w-2t)*(h-2t)^3) / 12"""
    w, h, t = 0.1, 0.2, 0.01
    s = RectangularTube(1, width=w, height=h, thickness=t)
    expected = (w * h**3 - (w - 2 * t) * (h - 2 * t)**3) / 12
    assert _close(s.inertia, expected), f"inertia={s.inertia}"


def test_rectangular_tube_shear_coefficient():
    """RectangularTube inherits default shear coefficient 5/6"""
    s = RectangularTube(1, width=0.1, height=0.2, thickness=0.01)
    assert _close(s.shear_coefficient, 5 / 6), f"kappa={s.shear_coefficient}"


# =============================================================================
# CircularBar
# =============================================================================

def test_circular_bar_area():
    """area = pi * (d/2)^2"""
    d = 0.05
    s = CircularBar(1, diameter=d)
    expected = math.pi * (d / 2)**2
    assert _close(s.area, expected), f"area={s.area}"


def test_circular_bar_inertia():
    """inertia = (pi/64) * d^4"""
    d = 0.05
    s = CircularBar(1, diameter=d)
    expected = (math.pi / 64) * d**4
    assert _close(s.inertia, expected), f"inertia={s.inertia}"


def test_circular_bar_shear_coefficient():
    """CircularBar shear coefficient = 9/10"""
    s = CircularBar(1, diameter=0.05)
    assert _close(s.shear_coefficient, 9 / 10), f"kappa={s.shear_coefficient}"


# =============================================================================
# CircularTube
# =============================================================================

def test_circular_tube_area():
    """area = pi/4 * (D^2 - d^2), d = D - 2*t"""
    D, t = 0.1, 0.01
    d = D - 2 * t
    s = CircularTube(1, outer_diameter=D, thickness=t)
    expected = math.pi / 4 * (D**2 - d**2)
    assert _close(s.area, expected), f"area={s.area}"


def test_circular_tube_inertia():
    """inertia = (pi/64) * (D^4 - d^4), d = D - 2*t"""
    D, t = 0.1, 0.01
    d = D - 2 * t
    s = CircularTube(1, outer_diameter=D, thickness=t)
    expected = (math.pi / 64) * (D**4 - d**4)
    assert _close(s.inertia, expected), f"inertia={s.inertia}"


def test_circular_tube_shear_coefficient():
    """CircularTube shear coefficient = 9/10"""
    s = CircularTube(1, outer_diameter=0.1, thickness=0.01)
    assert _close(s.shear_coefficient, 9 / 10), f"kappa={s.shear_coefficient}"


# =============================================================================
# TrapezoidalBar
# =============================================================================

def test_trapezoidal_bar_area():
    """area = 0.5 * (b1 + b2) * h"""
    b1, b2, h = 0.1, 0.05, 0.15
    s = TrapezoidalBar(1, base1=b1, base2=b2, height=h)
    expected = 0.5 * (b1 + b2) * h
    assert _close(s.area, expected), f"area={s.area}"


def test_trapezoidal_bar_inertia():
    """inertia = (h^3/36) * (b1^2 + 4*b1*b2 + b2^2) / (b1 + b2)"""
    b1, b2, h = 0.1, 0.05, 0.15
    s = TrapezoidalBar(1, base1=b1, base2=b2, height=h)
    expected = (h**3 / 36) * (b1**2 + 4 * b1 * b2 + b2**2) / (b1 + b2)
    assert _close(s.inertia, expected), f"inertia={s.inertia}"


def test_trapezoidal_bar_shear_coefficient():
    """TrapezoidalBar inherits default shear coefficient 5/6"""
    s = TrapezoidalBar(1, base1=0.1, base2=0.05, height=0.15)
    assert _close(s.shear_coefficient, 5 / 6), f"kappa={s.shear_coefficient}"


def test_trapezoidal_bar_rectangle_special_case():
    """When base1 == base2, TrapezoidalBar should match RectangularBar"""
    b, h = 0.1, 0.2
    trap = TrapezoidalBar(1, base1=b, base2=b, height=h)
    rect = RectangularBar(2, width=b, height=h)
    assert _close(trap.area, rect.area), f"areas differ: {trap.area} vs {rect.area}"
    assert _close(trap.inertia, rect.inertia), f"inertias differ: {trap.inertia} vs {rect.inertia}"


# =============================================================================
# TrapezoidalTube
# =============================================================================

def test_trapezoidal_tube_area():
    """area = outer_area - inner_area (trapezoid formula)"""
    b1, b2, h, t = 0.12, 0.08, 0.15, 0.01
    ib1 = b1 - 2 * t
    ib2 = b2 - 2 * t
    ih = h - 2 * t
    s = TrapezoidalTube(1, base1=b1, base2=b2, height=h, thickness=t)
    expected = 0.5 * (b1 + b2) * h - 0.5 * (ib1 + ib2) * ih
    assert _close(s.area, expected), f"area={s.area}"


def test_trapezoidal_tube_inertia():
    """inertia = outer_inertia - inner_inertia (trapezoid formula)"""
    b1, b2, h, t = 0.12, 0.08, 0.15, 0.01
    ib1 = b1 - 2 * t
    ib2 = b2 - 2 * t
    ih = h - 2 * t
    s = TrapezoidalTube(1, base1=b1, base2=b2, height=h, thickness=t)
    outer_I = (h**3 / 36) * (b1**2 + 4 * b1 * b2 + b2**2) / (b1 + b2)
    inner_I = (ih**3 / 36) * (ib1**2 + 4 * ib1 * ib2 + ib2**2) / (ib1 + ib2)
    expected = outer_I - inner_I
    assert _close(s.inertia, expected), f"inertia={s.inertia}"


def test_trapezoidal_tube_shear_coefficient():
    """TrapezoidalTube inherits default shear coefficient 5/6"""
    s = TrapezoidalTube(1, base1=0.12, base2=0.08, height=0.15, thickness=0.01)
    assert _close(s.shear_coefficient, 5 / 6), f"kappa={s.shear_coefficient}"


# =============================================================================
# HexagonalBar
# =============================================================================

def test_hexagonal_bar_area():
    """area = 3*sqrt(3)/2 * s^2"""
    s_val = 0.05
    s = HexagonalBar(1, side=s_val)
    expected = (3 * math.sqrt(3) / 2) * s_val**2
    assert _close(s.area, expected), f"area={s.area}"


def test_hexagonal_bar_inertia():
    """inertia = 5*sqrt(3)/16 * s^4"""
    s_val = 0.05
    s = HexagonalBar(1, side=s_val)
    expected = (5 * math.sqrt(3) / 16) * s_val**4
    assert _close(s.inertia, expected), f"inertia={s.inertia}"


def test_hexagonal_bar_shear_coefficient():
    """HexagonalBar inherits default shear coefficient 5/6"""
    s = HexagonalBar(1, side=0.05)
    assert _close(s.shear_coefficient, 5 / 6), f"kappa={s.shear_coefficient}"


# =============================================================================
# HexagonalTube
# =============================================================================

def test_hexagonal_tube_area():
    """area = outer_area - inner_area (hexagon formula)"""
    a_outer, t = 0.06, 0.005
    a_inner = a_outer - 2 * t
    s = HexagonalTube(1, outer_side=a_outer, thickness=t)
    expected = (3 * math.sqrt(3) / 2) * a_outer**2 - (3 * math.sqrt(3) / 2) * a_inner**2
    assert _close(s.area, expected), f"area={s.area}"


def test_hexagonal_tube_inertia():
    """inertia = outer_inertia - inner_inertia (hexagon formula)"""
    a_outer, t = 0.06, 0.005
    a_inner = a_outer - 2 * t
    s = HexagonalTube(1, outer_side=a_outer, thickness=t)
    expected = (5 * math.sqrt(3) / 16) * a_outer**4 - (5 * math.sqrt(3) / 16) * a_inner**4
    assert _close(s.inertia, expected), f"inertia={s.inertia}"


def test_hexagonal_tube_shear_coefficient():
    """HexagonalTube inherits default shear coefficient 5/6"""
    s = HexagonalTube(1, outer_side=0.06, thickness=0.005)
    assert _close(s.shear_coefficient, 5 / 6), f"kappa={s.shear_coefficient}"


# =============================================================================
# IBeam
# =============================================================================

def test_ibeam_area():
    """area = tw*(h - 2*tf) + 2*b*tf"""
    h, b, tw, tf = 0.2, 0.1, 0.01, 0.015
    s = IBeam(1, h=h, b=b, tw=tw, tf=tf)
    expected = tw * (h - 2 * tf) + 2 * b * tf
    assert _close(s.area, expected), f"area={s.area}"


def test_ibeam_inertia():
    """inertia = tw*(h-2tf)^3/12 + 2*(b*tf^3/12 + b*tf*(h/2 - tf/2)^2)"""
    h, b, tw, tf = 0.2, 0.1, 0.01, 0.015
    s = IBeam(1, h=h, b=b, tw=tw, tf=tf)
    inertia_web = tw * (h - 2 * tf)**3 / 12
    inertia_flange = 2 * (b * tf**3 / 12 + b * tf * (h / 2 - tf / 2)**2)
    expected = inertia_web + inertia_flange
    assert _close(s.inertia, expected), f"inertia={s.inertia}"


def test_ibeam_shear_coefficient():
    """IBeam inherits default shear coefficient 5/6"""
    s = IBeam(1, h=0.2, b=0.1, tw=0.01, tf=0.015)
    assert _close(s.shear_coefficient, 5 / 6), f"kappa={s.shear_coefficient}"


# =============================================================================
# CSection
# =============================================================================

def test_csection_area():
    """area = tw*h + 2*b*tf"""
    h, b, tw, tf = 0.15, 0.07, 0.008, 0.012
    s = CSection(1, h=h, b=b, tw=tw, tf=tf)
    expected = tw * h + 2 * b * tf
    assert _close(s.area, expected), f"area={s.area}"


def test_csection_inertia():
    """inertia = tw*h^3/12 + 2*(b*tf^3/12 + b*tf*(h/2 - tf/2)^2)"""
    h, b, tw, tf = 0.15, 0.07, 0.008, 0.012
    s = CSection(1, h=h, b=b, tw=tw, tf=tf)
    inertia_web = tw * h**3 / 12
    inertia_flange = 2 * (b * tf**3 / 12 + b * tf * (h / 2 - tf / 2)**2)
    expected = inertia_web + inertia_flange
    assert _close(s.inertia, expected), f"inertia={s.inertia}"


def test_csection_shear_coefficient():
    """CSection inherits default shear coefficient 5/6"""
    s = CSection(1, h=0.15, b=0.07, tw=0.008, tf=0.012)
    assert _close(s.shear_coefficient, 5 / 6), f"kappa={s.shear_coefficient}"


# =============================================================================
# LSection
# =============================================================================

def test_lsection_area():
    """area = b*t + (h - t)*t"""
    b, h, t = 0.08, 0.12, 0.008
    s = LSection(1, b=b, h=h, t=t)
    expected = b * t + (h - t) * t
    assert _close(s.area, expected), f"area={s.area}"


def test_lsection_inertia():
    """inertia = (b*t^3)/12 + ((h - t)*t^3)/12"""
    b, h, t = 0.08, 0.12, 0.008
    s = LSection(1, b=b, h=h, t=t)
    expected = (b * t**3) / 12 + ((h - t) * t**3) / 12
    assert _close(s.inertia, expected), f"inertia={s.inertia}"


def test_lsection_shear_coefficient():
    """LSection inherits default shear coefficient 5/6"""
    s = LSection(1, b=0.08, h=0.12, t=0.008)
    assert _close(s.shear_coefficient, 5 / 6), f"kappa={s.shear_coefficient}"


# =============================================================================
# TSection
# =============================================================================

def test_tsection_area():
    """area = tw*(h - tf) + b*tf"""
    b, h, tw, tf = 0.1, 0.18, 0.01, 0.015
    s = TSection(1, b=b, h=h, tw=tw, tf=tf)
    expected = tw * (h - tf) + b * tf
    assert _close(s.area, expected), f"area={s.area}"


def test_tsection_inertia():
    """inertia = tw*(h-tf)^3/12 + (b*tf^3/12 + b*tf*(h - tf/2)^2)"""
    b, h, tw, tf = 0.1, 0.18, 0.01, 0.015
    s = TSection(1, b=b, h=h, tw=tw, tf=tf)
    inertia_web = tw * (h - tf)**3 / 12
    inertia_flange = b * tf**3 / 12 + b * tf * (h - tf / 2)**2
    expected = inertia_web + inertia_flange
    assert _close(s.inertia, expected), f"inertia={s.inertia}"


def test_tsection_shear_coefficient():
    """TSection inherits default shear coefficient 5/6"""
    s = TSection(1, b=0.1, h=0.18, tw=0.01, tf=0.015)
    assert _close(s.shear_coefficient, 5 / 6), f"kappa={s.shear_coefficient}"


# =============================================================================
# ZSection
# =============================================================================

def test_zsection_area():
    """area = tw*h + 2*b*tf"""
    h, b, tw, tf = 0.15, 0.07, 0.008, 0.012
    s = ZSection(1, h=h, b=b, tw=tw, tf=tf)
    expected = tw * h + 2 * b * tf
    assert _close(s.area, expected), f"area={s.area}"


def test_zsection_inertia():
    """inertia = tw*h^3/12 + 2*(b*tf^3/12 + b*tf*(h/2 - tf/2)^2)"""
    h, b, tw, tf = 0.15, 0.07, 0.008, 0.012
    s = ZSection(1, h=h, b=b, tw=tw, tf=tf)
    inertia_web = tw * h**3 / 12
    inertia_flange = 2 * (b * tf**3 / 12 + b * tf * (h / 2 - tf / 2)**2)
    expected = inertia_web + inertia_flange
    assert _close(s.inertia, expected), f"inertia={s.inertia}"


def test_zsection_shear_coefficient():
    """ZSection inherits default shear coefficient 5/6"""
    s = ZSection(1, h=0.15, b=0.07, tw=0.008, tf=0.012)
    assert _close(s.shear_coefficient, 5 / 6), f"kappa={s.shear_coefficient}"


# =============================================================================
# HatSection
# =============================================================================

def test_hatsection_area():
    """area = tw*h + 2*b*tf"""
    h, b, tw, tf = 0.12, 0.06, 0.007, 0.010
    s = HatSection(1, h=h, b=b, tw=tw, tf=tf)
    expected = tw * h + 2 * b * tf
    assert _close(s.area, expected), f"area={s.area}"


def test_hatsection_inertia():
    """inertia = tw*h^3/12 + 2*(b*tf^3/12 + b*tf*(h/2 - tf/2)^2)"""
    h, b, tw, tf = 0.12, 0.06, 0.007, 0.010
    s = HatSection(1, h=h, b=b, tw=tw, tf=tf)
    inertia_web = tw * h**3 / 12
    inertia_flange = 2 * (b * tf**3 / 12 + b * tf * (h / 2 - tf / 2)**2)
    expected = inertia_web + inertia_flange
    assert _close(s.inertia, expected), f"inertia={s.inertia}"


def test_hatsection_shear_coefficient():
    """HatSection inherits default shear coefficient 5/6"""
    s = HatSection(1, h=0.12, b=0.06, tw=0.007, tf=0.010)
    assert _close(s.shear_coefficient, 5 / 6), f"kappa={s.shear_coefficient}"


# =============================================================================
# GeneralSection
# =============================================================================

def test_general_section_stores_area_inertia():
    """GeneralSection stores user-supplied area and inertia unchanged"""
    A, I = 0.0025, 3.5e-6
    s = GeneralSection(1, area=A, inertia=I)
    assert _close(s.area, A), f"area={s.area}"
    assert _close(s.inertia, I), f"inertia={s.inertia}"


def test_general_section_shear_coefficient():
    """GeneralSection inherits default shear coefficient 5/6"""
    s = GeneralSection(1, area=0.001, inertia=1e-7)
    assert _close(s.shear_coefficient, 5 / 6), f"kappa={s.shear_coefficient}"


# =============================================================================
# normal_stress (Section base class method)
# =============================================================================

def test_normal_stress_pure_axial():
    """normal_stress = N/A when M=0"""
    s = RectangularBar(1, width=0.1, height=0.2)
    N = 1000.0
    sigma = s.normal_stress(N=N, M=0, y=0.05)
    expected = N / s.area
    assert _close(sigma, expected), f"sigma={sigma}"


def test_normal_stress_pure_bending():
    """normal_stress = -M*y/I when N=0"""
    s = RectangularBar(1, width=0.1, height=0.2)
    M = 500.0
    y = 0.1
    sigma = s.normal_stress(N=0, M=M, y=y)
    expected = -M * y / s.inertia
    assert _close(sigma, expected), f"sigma={sigma}"


def test_normal_stress_combined():
    """normal_stress = N/A - M*y/I (combined loading)"""
    s = CircularBar(1, diameter=0.05)
    N, M, y = 2000.0, 300.0, 0.02
    sigma = s.normal_stress(N=N, M=M, y=y)
    expected = N / s.area - M * y / s.inertia
    assert _close(sigma, expected), f"sigma={sigma}"


def test_normal_stress_raises_without_area_inertia():
    """normal_stress raises ValueError when area/inertia not set"""
    s = GeneralSection.__new__(GeneralSection)
    s.id = 99
    s.area = None
    s.inertia = None
    s.shear_coefficient = 5 / 6
    raised = False
    try:
        s.normal_stress(N=100, M=0, y=0)
    except ValueError:
        raised = True
    assert raised, "Expected ValueError when area/inertia is None"


# =============================================================================
# create_section factory
# =============================================================================

def test_create_section_rectangular_bar():
    s = create_section("rectangular_bar", 1, width=0.1, height=0.2)
    assert isinstance(s, RectangularBar)
    assert _close(s.area, 0.1 * 0.2)


def test_create_section_rectangular_tube():
    s = create_section("rectangular_tube", 1, width=0.1, height=0.2, thickness=0.01)
    assert isinstance(s, RectangularTube)


def test_create_section_trapezoidal_bar():
    s = create_section("trapezoidal_bar", 1, base1=0.1, base2=0.06, height=0.15)
    assert isinstance(s, TrapezoidalBar)


def test_create_section_trapezoidal_tube():
    s = create_section("trapezoidal_tube", 1, base1=0.1, base2=0.06, height=0.15, thickness=0.01)
    assert isinstance(s, TrapezoidalTube)


def test_create_section_circular_bar():
    s = create_section("circular_bar", 1, diameter=0.05)
    assert isinstance(s, CircularBar)


def test_create_section_circular_tube():
    s = create_section("circular_tube", 1, outer_diameter=0.1, thickness=0.01)
    assert isinstance(s, CircularTube)


def test_create_section_hexagonal_bar():
    s = create_section("hexagonal_bar", 1, side=0.05)
    assert isinstance(s, HexagonalBar)


def test_create_section_hexagonal_tube():
    s = create_section("hexagonal_tube", 1, outer_side=0.06, thickness=0.005)
    assert isinstance(s, HexagonalTube)


def test_create_section_ibeam():
    s = create_section("ibeam", 1, h=0.2, b=0.1, tw=0.01, tf=0.015)
    assert isinstance(s, IBeam)


def test_create_section_c_section():
    s = create_section("c_section", 1, h=0.15, b=0.07, tw=0.008, tf=0.012)
    assert isinstance(s, CSection)


def test_create_section_l_section():
    s = create_section("l_section", 1, b=0.08, h=0.12, t=0.008)
    assert isinstance(s, LSection)


def test_create_section_t_section():
    s = create_section("t_section", 1, b=0.1, h=0.18, tw=0.01, tf=0.015)
    assert isinstance(s, TSection)


def test_create_section_z_section():
    s = create_section("z_section", 1, h=0.15, b=0.07, tw=0.008, tf=0.012)
    assert isinstance(s, ZSection)


def test_create_section_hat_section():
    s = create_section("hat_section", 1, h=0.12, b=0.06, tw=0.007, tf=0.010)
    assert isinstance(s, HatSection)


def test_create_section_general():
    s = create_section("general", 1, area=0.001, inertia=1e-7)
    assert isinstance(s, GeneralSection)


def test_create_section_unknown_raises():
    """create_section raises ValueError for unknown section type"""
    raised = False
    try:
        create_section("unknown_type", 1)
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for unknown section type"


def test_create_section_case_insensitive():
    """create_section is case-insensitive for section type names"""
    s = create_section("Rectangular_Bar", 1, width=0.1, height=0.2)
    assert isinstance(s, RectangularBar)


# =============================================================================
# Runner
# =============================================================================

if __name__ == "__main__":
    tests = [
        # RectangularBar
        test_rectangular_bar_area,
        test_rectangular_bar_inertia,
        test_rectangular_bar_shear_coefficient,
        # RectangularTube
        test_rectangular_tube_area,
        test_rectangular_tube_inertia,
        test_rectangular_tube_shear_coefficient,
        # CircularBar
        test_circular_bar_area,
        test_circular_bar_inertia,
        test_circular_bar_shear_coefficient,
        # CircularTube
        test_circular_tube_area,
        test_circular_tube_inertia,
        test_circular_tube_shear_coefficient,
        # TrapezoidalBar
        test_trapezoidal_bar_area,
        test_trapezoidal_bar_inertia,
        test_trapezoidal_bar_shear_coefficient,
        test_trapezoidal_bar_rectangle_special_case,
        # TrapezoidalTube
        test_trapezoidal_tube_area,
        test_trapezoidal_tube_inertia,
        test_trapezoidal_tube_shear_coefficient,
        # HexagonalBar
        test_hexagonal_bar_area,
        test_hexagonal_bar_inertia,
        test_hexagonal_bar_shear_coefficient,
        # HexagonalTube
        test_hexagonal_tube_area,
        test_hexagonal_tube_inertia,
        test_hexagonal_tube_shear_coefficient,
        # IBeam
        test_ibeam_area,
        test_ibeam_inertia,
        test_ibeam_shear_coefficient,
        # CSection
        test_csection_area,
        test_csection_inertia,
        test_csection_shear_coefficient,
        # LSection
        test_lsection_area,
        test_lsection_inertia,
        test_lsection_shear_coefficient,
        # TSection
        test_tsection_area,
        test_tsection_inertia,
        test_tsection_shear_coefficient,
        # ZSection
        test_zsection_area,
        test_zsection_inertia,
        test_zsection_shear_coefficient,
        # HatSection
        test_hatsection_area,
        test_hatsection_inertia,
        test_hatsection_shear_coefficient,
        # GeneralSection
        test_general_section_stores_area_inertia,
        test_general_section_shear_coefficient,
        # normal_stress
        test_normal_stress_pure_axial,
        test_normal_stress_pure_bending,
        test_normal_stress_combined,
        test_normal_stress_raises_without_area_inertia,
        # create_section factory
        test_create_section_rectangular_bar,
        test_create_section_rectangular_tube,
        test_create_section_trapezoidal_bar,
        test_create_section_trapezoidal_tube,
        test_create_section_circular_bar,
        test_create_section_circular_tube,
        test_create_section_hexagonal_bar,
        test_create_section_hexagonal_tube,
        test_create_section_ibeam,
        test_create_section_c_section,
        test_create_section_l_section,
        test_create_section_t_section,
        test_create_section_z_section,
        test_create_section_hat_section,
        test_create_section_general,
        test_create_section_unknown_raises,
        test_create_section_case_insensitive,
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
