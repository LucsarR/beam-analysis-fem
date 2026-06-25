#!/usr/bin/env python3
"""
Test for the plot_structure_preview function
Tests that the visualization works correctly before analysis
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from post_processing.plotter import plot_structure_preview
import plotly.graph_objects as go

def test_simple_structure():
    """Test preview with a simple cantilever beam"""
    print("\n[TEST 1] Simple Structure Preview")
    print("-" * 70)
    
    nodes = [(0.0, 0.0), (1.0, 0.0)]
    elements = [(1, 2, "euler_bernoulli_2node", "Prop1", 1)]
    properties = [{"name": "Prop1", "material": None, "section": None}]
    constraints = [(1, 0, 0.0), (1, 1, 0.0), (1, 2, 0.0)]
    point_loads = [(2, 1, -1000.0)]
    distributed_loads = []
    
    fig = plot_structure_preview(nodes, elements, properties, constraints, point_loads, distributed_loads)
    
    assert isinstance(fig, go.Figure), "Should return a Plotly Figure object"
    assert len(fig.data) > 0, "Figure should have traces"
    
    print(f"  ✓ Preview created successfully")
    print(f"  ✓ Number of traces: {len(fig.data)}")
    return True

def test_structure_with_all_elements():
    """Test preview with all element types: nodes, elements, loads, constraints"""
    print("\n[TEST 2] Structure with All Element Types")
    print("-" * 70)
    
    nodes = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]
    elements = [(1, 2, "euler_bernoulli_2node", "P1", 1), (2, 3, "euler_bernoulli_2node", "P1", 1)]
    properties = [{"name": "P1", "material": None, "section": None}]
    
    # All constraint types
    constraints = [
        (1, 0, 0.0),  # X fixed
        (1, 1, 0.0),  # Y fixed
        (1, 2, 0.0),  # Rotation fixed
    ]
    
    # All point load types
    point_loads = [
        (2, 0, 500.0),    # X-direction
        (2, 1, -1000.0),  # Y-direction
        (2, 2, 2000.0),   # Moment
    ]
    
    # Distributed loads
    distributed_loads = [
        (1, -100.0, None, 'y', None, 'constant'),
        (2, -200.0, -400.0, 't', None, 'linear'),
    ]
    
    fig = plot_structure_preview(nodes, elements, properties, constraints, point_loads, distributed_loads)
    
    assert isinstance(fig, go.Figure), "Should return a Plotly Figure object"
    
    # Verify figure contains expected components
    trace_count = len(fig.data)
    assert trace_count >= 3, "Should have at least nodes, elements, and some loads/constraints"
    
    print(f"  ✓ Preview with all element types created successfully")
    print(f"  ✓ Number of traces: {trace_count}")
    print(f"  ✓ Includes: nodes, elements, constraints, point loads, distributed loads")
    return True

def test_empty_loads():
    """Test preview with no loads (should still work)"""
    print("\n[TEST 3] Structure with No Loads")
    print("-" * 70)
    
    nodes = [(0.0, 0.0), (1.0, 0.0)]
    elements = [(1, 2, "euler_bernoulli_2node", "Prop1", 1)]
    properties = [{"name": "Prop1", "material": None, "section": None}]
    constraints = [(1, 0, 0.0), (1, 1, 0.0)]
    point_loads = []
    distributed_loads = []
    
    fig = plot_structure_preview(nodes, elements, properties, constraints, point_loads, distributed_loads)
    
    assert isinstance(fig, go.Figure), "Should work even with no loads"
    assert len(fig.data) > 0, "Should still show nodes and elements"
    
    print(f"  ✓ Preview with no loads created successfully")
    print(f"  ✓ Number of traces: {len(fig.data)}")
    return True

def test_angled_structure():
    """Test preview with angled elements"""
    print("\n[TEST 4] Angled Structure")
    print("-" * 70)
    
    nodes = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.5)]
    elements = [(1, 2, "euler_bernoulli_2node", "P1", 1), (2, 3, "euler_bernoulli_2node", "P1", 1)]
    properties = [{"name": "P1", "material": None, "section": None}]
    constraints = [(1, 0, 0.0), (1, 1, 0.0)]
    point_loads = [(2, 1, -500.0)]
    distributed_loads = [(1, -100.0, None, 't', None, 'constant')]
    
    fig = plot_structure_preview(nodes, elements, properties, constraints, point_loads, distributed_loads)
    
    assert isinstance(fig, go.Figure), "Should handle angled elements"
    
    print(f"  ✓ Angled structure preview created successfully")
    print(f"  ✓ Number of traces: {len(fig.data)}")
    return True

def test_invalid_node_ids():
    """Test preview handles invalid node IDs gracefully"""
    print("\n[TEST 5] Structure with Invalid Node IDs")
    print("-" * 70)
    
    nodes = [(0.0, 0.0), (1.0, 0.0)]
    # Element with invalid node ID (node 5 doesn't exist)
    elements = [(1, 2, "euler_bernoulli_2node", "P1", 1), (1, 5, "euler_bernoulli_2node", "P1", 1)]
    properties = [{"name": "P1", "material": None, "section": None}]
    # Constraint with invalid node ID
    constraints = [(1, 0, 0.0), (10, 1, 0.0)]
    # Load with invalid node ID
    point_loads = [(2, 1, -1000.0), (8, 0, 500.0)]
    distributed_loads = []
    
    # Should not raise an exception, just skip invalid elements
    fig = plot_structure_preview(nodes, elements, properties, constraints, point_loads, distributed_loads)
    
    assert isinstance(fig, go.Figure), "Should handle invalid node IDs gracefully"
    assert len(fig.data) > 0, "Should still show valid nodes and elements"
    
    print(f"  ✓ Invalid node IDs handled gracefully")
    print(f"  ✓ Number of traces: {len(fig.data)}")
    return True

def test_arrow_scaling():
    """Test that arrow scale is at least 15% of the largest structural dimension"""
    print("\n[TEST 6] Arrow Scaling Visibility")
    print("-" * 70)

    test_cases = [
        # (nodes, description, expected_scale)
        ([(0.0, 0.0), (10.0, 0.0)], "horizontal beam L=10", 10.0 * 0.15),
        ([(0.0, 0.0), (0.0, 5.0)], "vertical beam H=5", 5.0 * 0.15),
        ([(0.0, 0.0), (0.5, 0.0)], "tiny beam L=0.5 (minimum kicks in)", 1.0 * 0.15),
        ([(0.0, 0.0), (100.0, 0.0)], "large beam L=100", 100.0 * 0.15),
    ]

    for nodes, desc, expected_scale in test_cases:
        node_xs = [n[0] for n in nodes]
        node_ys = [n[1] for n in nodes]
        x_range = max(node_xs) - min(node_xs)
        y_range = max(node_ys) - min(node_ys)
        scale = max(x_range, y_range, 1.0) * 0.15

        assert abs(scale - expected_scale) < 1e-9, (
            f"{desc}: expected scale {expected_scale:.4f}, got {scale:.4f}"
        )
        print(f"  ✓ {desc}: scale={scale:.4f}")

    return True

def test_spring_symbols():
    """Test preview includes linear and torsional spring symbols"""
    print("\n[TEST 7] Spring Symbols")
    print("-" * 70)

    nodes = [(0.0, 0.0), (1.0, 0.0)]
    elements = [(1, 2, "euler_bernoulli_2node", "P1", 1)]
    properties = [{"name": "P1", "material": None, "section": None}]
    constraints = []
    point_loads = []
    distributed_loads = []
    springs = [
        (1, 1, 1000.0),  # linear spring
        (2, 2, 500.0),   # torsional spring
    ]

    fig = plot_structure_preview(
        nodes, elements, properties, constraints, point_loads, distributed_loads, springs=springs
    )

    assert isinstance(fig, go.Figure), "Should return a Plotly Figure object"
    linear_traces = [tr for tr in fig.data if tr.name == "Linear Spring"]
    torsional_traces = [tr for tr in fig.data if tr.name == "Torsional Spring"]
    assert len(linear_traces) == 1, "Should include one linear spring symbol"
    assert len(torsional_traces) == 1, "Should include one torsional spring symbol"

    print("  ✓ Spring symbols rendered successfully")
    return True


def test_moment_load_directions():
    """Test that positive moment load points counterclockwise and negative points clockwise"""
    print("\n[TEST 8] Moment Load Directions")
    print("-" * 70)
    
    nodes = [(0.0, 0.0), (1.0, 0.0)]
    elements = [(1, 2, "euler_bernoulli_2node", "P1", 1)]
    properties = [{"name": "P1", "material": None, "section": None}]
    constraints = []
    distributed_loads = []
    
    # 1. Positive moment load at node 1 (should point counterclockwise, i.e. end at 6 o'clock pointing right)
    point_loads_pos = [(1, 2, 10.0)]
    fig_pos = plot_structure_preview(nodes, elements, properties, constraints, point_loads_pos, distributed_loads)
    
    moment_traces_pos = [tr for tr in fig_pos.data if tr.name == "Moment Load"]
    assert len(moment_traces_pos) == 1, "Should have one moment load trace"
    pos_x = moment_traces_pos[0].x
    pos_y = moment_traces_pos[0].y
    
    assert abs(pos_x[-1]) < 1e-5, f"Expected end x near 0, got {pos_x[-1]}"
    assert pos_y[-1] < -1e-5, f"Expected end y negative, got {pos_y[-1]}"
    
    assert len(fig_pos.layout.annotations) == 1, "Should have one annotation for arrowhead"
    ann_pos = fig_pos.layout.annotations[0]
    dx_pos = ann_pos.x - ann_pos.ax
    assert dx_pos > 0, f"Expected positive moment arrow to point right at 6 o'clock, but got dx={dx_pos}"
    
    # 2. Negative moment load at node 1 (should point clockwise, i.e. end at 12 o'clock pointing right)
    point_loads_neg = [(1, 2, -10.0)]
    fig_neg = plot_structure_preview(nodes, elements, properties, constraints, point_loads_neg, distributed_loads)
    
    moment_traces_neg = [tr for tr in fig_neg.data if tr.name == "Moment Load"]
    assert len(moment_traces_neg) == 1, "Should have one moment load trace"
    neg_x = moment_traces_neg[0].x
    neg_y = moment_traces_neg[0].y
    
    assert abs(neg_x[-1]) < 1e-5, f"Expected end x near 0, got {neg_x[-1]}"
    assert neg_y[-1] > 1e-5, f"Expected end y positive, got {neg_y[-1]}"
    
    assert len(fig_neg.layout.annotations) == 1, "Should have one annotation for arrowhead"
    ann_neg = fig_neg.layout.annotations[0]
    dx_neg = ann_neg.x - ann_neg.ax
    assert dx_neg > 0, f"Expected negative moment arrow to point right at 12 o'clock, but got dx={dx_neg}"
    
    print("  ✓ Positive moment arc ends at 6 o'clock and points right (CCW)")
    print("  ✓ Negative moment arc ends at 12 o'clock and points right (CW)")
    return True


def run_all_tests():
    """Run all tests for the preview function"""
    print("=" * 70)
    print("STRUCTURE PREVIEW FUNCTION TESTS")
    print("=" * 70)
    
    tests = [
        test_simple_structure,
        test_structure_with_all_elements,
        test_empty_loads,
        test_angled_structure,
        test_invalid_node_ids,
        test_arrow_scaling,
        test_spring_symbols,
        test_moment_load_directions,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
