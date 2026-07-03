# Tests

This directory contains tests that verify the FEM implementation is working correctly, covering element formulations, section properties, material definitions, mesh integration, and convergence with mesh refinement.

## Test Files

### Element Tests

#### 1. `test_euler_bernoulli.py`
Tests for 2-node Euler-Bernoulli beam elements with analytical solutions:
- Simply supported beams under sinusoidal and uniform loads
- Cantilever beams under point and distributed loads
- Custom load functions (sinusoidal, exponential)
- Comparison of bending moment and shear force against analytical results
- Single-element and multi-element (subdivided) meshes

#### 2. `test_euler_bernoulli_3node.py`
Tests the 3-node Euler-Bernoulli beam element:
- Element creation and DOF structure (9 DOFs with central node rotation)
- Stiffness matrix properties (symmetry, positive semi-definiteness)
- Cantilever beam solutions compared to analytical results
- Simply supported beam with distributed loads

#### 3. `test_euler_bernoulli_3node_updated.py`
Extended tests for the updated 3-node Euler-Bernoulli element:
- Verifies 9 DOFs with active central node rotation
- Mesh convergence verification
- Cantilever and simply supported beam comparisons

#### 4. `test_timoshenko.py`
Tests for 2-node Timoshenko beam elements:
- Slender beams (shear deformation negligible → matches Euler-Bernoulli)
- Thick beams (shear deformation significant → larger deflection)
- Custom load functions (sinusoidal, exponential)
- Single-element and multi-element meshes

#### 5. `test_timoshenko_3node.py`
Tests for 3-node Timoshenko beam elements:
- Element creation and DOF structure (9 DOFs with central node rotation)
- Stiffness matrix properties
- Cantilever beam with point load vs. analytical Timoshenko solution
- Simply supported beam with distributed load
- Verification that Timoshenko deflections exceed Euler-Bernoulli (shear effects)

#### 6. `test_reddy_bickford.py`
**Tests for standard Reddy-Bickford and Modified Reddy-Bickford (MRBT) 2-node beam elements.**

Comprehensive test suite validating:
- Parameter computation (D1, E1, F1, G1) for rectangular sections ✅
- Stiffness matrix properties (symmetry, positive semi-definiteness) ✅
- Cantilever and simply supported beam behavior (standard RBT and MRBT formulations)
- Comparison with Euler-Bernoulli and Timoshenko theories
- Mesh convergence with refinement ✅
- Force recovery methods (moment, shear, normal force)
- Inter-element boundary rotation/slope compatibility and shear locking resistance (MRBT)

**Status**: Test suite complete. **Formulations are fully verified and working.** All tests pass and match reference literature solutions (Heyliger & Reddy, 1988; Rodrigues et al., 2024).

#### 7. `test_forces_polynomial_degrees.py`
Tests verifying the polynomial degrees of internal forces for Euler-Bernoulli and Timoshenko 3-node elements. Validates the consistency of force recovery against the respective polynomial degrees of shape functions.

#### 8. `test_gauss_point_selection.py`
Tests the configuration and correct propagation of numerical integration parameters (number of Gauss integration points) across all element types.

### Component Tests

#### 9. `test_section.py`
Tests for every formula in `section.py`:
- All 14 cross-section types: rectangular bar/tube, circular bar/tube, trapezoidal bar/tube, hexagonal bar/tube, I-beam, C-section, L-section, T-section, Z-section, hat section, general
- Area and second moment of area calculations
- Shear coefficients (default and section-specific)
- `normal_stress` method
- `create_section` factory function

#### 10. `test_material.py`
Tests for the `Material` class elastic constant calculations:
- Computing G from E and ν
- Computing ν from E and G
- Computing E from G and ν
- Error handling when all three constants are supplied simultaneously

#### 11. `test_reactions.py`
Tests reaction force calculations at constrained DOFs:
- Cantilever beam reactions (vertical, horizontal, moment)
- Simply supported beam reactions
- Verification that equilibrium is satisfied (sum of reactions equals applied loads)

### Mesh & Integration Tests

#### 12. `test_mesh_integration.py`
Tests the integration of the `Mesh` class with other FEM components:
- Basic mesh creation and operations
- Node and element management
- Integration with Material and Section classes
- Integration with Constraint and Load classes
- Integration with Analysis classes
- Integration with post-processing components
- Support for different element types (Euler-Bernoulli, Timoshenko)
- Mesh query and export methods

#### 13. `test_mesh_convergence.py`
Tests convergence of solutions with mesh refinement for canonical problems:
- **Euler-Bernoulli Cantilever**: Verifies displacement, moment, and shear convergence
- **Timoshenko Cantilever**: Verifies convergence including shear deformation effects
- **Simply Supported Beam**: Tests with distributed loads
- **Element Comparison**: Compares Euler-Bernoulli vs Timoshenko behavior

These tests use analytical solutions to validate numerical convergence.

#### 14. `test_complex_structures.py`
Tests mesh behaviour with complex, realistic scenarios:
- **Angled Cantilever (45°)**: Verifies mesh handles inclined elements correctly
- **L-Shaped Frame**: Tests structures with elements at 90° to each other
- **Multiple Loads**: Beam with multiple point loads and distributed loads simultaneously
- **Timoshenko Angled Beam**: Verifies Timoshenko elements work correctly at angles
- **Forces Convergence**: Verifies that moment and shear force calculations converge

### Visualization & Stress Tests

#### 15. `test_structure_preview.py`
Tests the `plot_structure_preview` function used in the app before analysis:
- Verifies the function returns a Plotly `Figure` object with traces
- Checks that constraints, loads, and elements are represented correctly

#### 16. `test_improvements.py`
Verifies the QOL improvements made to the post-processing functions, including:
- Correctly retrieving central node displacements for deformed shape plots on 3-node elements
- Verification of the generation of various normal and shear stress distribution plots

#### 17. `test_normal_stress_distribution_plot.py`
Tests the plotting logic of normal stress distributions on cross-sections, ensuring correct coordinates and titles.

#### 18. `test_normal_stress_side_view_plot.py`
Tests the plotting logic of normal stress side views across elements, ensuring correctness across element orientations and sections.

#### 19. `test_normal_stress_verification.py`
Verifies normal stresses for standard elements, checking polynomial fits against analytical solutions.

#### 20. `test_shear_stress_distribution_plot.py`
Tests transverse shear stress distribution plotting, including comparison plots, side views, and Reddy formulations.

#### 21. `verify_convergence_reddy.py`
A verification script that checks Reddy-Bickford (RBT) and Modified Reddy-Bickford (MRBT) convergence for slope and rotation, comparing results directly to reference values.

#### 22. `create_mockup.py`
Script that generates a visual mockup showing the reaction forces display in the Streamlit app.

## Running the Tests

### Recommended: Run the entire test suite with `pytest`
```bash
pytest
```

### Alternatively, run individual test files:
```bash
export PYTHONPATH=$(pwd)   # run from the repository root
python tests/test_euler_bernoulli.py
python tests/test_euler_bernoulli_3node.py
python tests/test_euler_bernoulli_3node_updated.py
python tests/test_timoshenko.py
python tests/test_timoshenko_3node.py
python tests/test_reddy_bickford.py
python tests/test_reactions.py
python tests/test_section.py
python tests/test_material.py
python tests/test_mesh_integration.py
python tests/test_mesh_convergence.py
python tests/test_complex_structures.py
python tests/test_structure_preview.py
python tests/test_forces_polynomial_degrees.py
python tests/test_gauss_point_selection.py
python tests/test_improvements.py
python tests/test_normal_stress_distribution_plot.py
python tests/test_normal_stress_side_view_plot.py
python tests/test_normal_stress_verification.py
python tests/test_shear_stress_distribution_plot.py
python tests/verify_convergence_reddy.py
```

## Test Results Summary

All analytical tests verify that:
✅ Mesh correctly generates nodes and elements
✅ Mesh integrates properly with all FEM components
✅ Displacement solutions converge with mesh refinement
✅ Force solutions (moment, shear) converge with mesh refinement
✅ Euler-Bernoulli 2-node and 3-node elements produce accurate results
✅ Timoshenko 2-node and 3-node elements include shear deformation effects
✅ Reddy-Bickford (RBT) and Modified Reddy-Bickford (MRBT) elements are fully verified and match analytical reference solutions
✅ Timoshenko deflections are larger than Euler-Bernoulli (includes shear)
✅ Mesh handles elements at various angles (not just horizontal)
✅ Mesh handles multiple loads (point and distributed) simultaneously
✅ Mesh handles complex boundary conditions with multiple constraints
✅ Reaction forces satisfy global equilibrium
✅ All 14 cross-section types compute correct area and inertia
✅ Material class correctly derives the third elastic constant from any two

## Convergence Criteria

Tests verify convergence by checking that:
1. Solution differences decrease as the mesh is refined
2. For problems with analytical solutions, numerical error approaches zero
3. Euler-Bernoulli and Timoshenko elements show consistent behaviour
4. Timoshenko deflections are larger than Euler-Bernoulli (includes shear)
5. Reddy-Bickford and MRBT elements converge correctly, with MRBT showing significantly faster convergence on coarse meshes

## Test Coverage

### Element Types Tested:
- ✅ Euler-Bernoulli 2-node elements
- ✅ Euler-Bernoulli 3-node elements
- ✅ Timoshenko 2-node elements
- ✅ Timoshenko 3-node elements
- ✅ Reddy-Bickford 2-node elements (standard RBT)
- ✅ Reddy-Bickford MRBT 2-node elements

### Load Types Tested:
- ✅ Point loads (single and multiple)
- ✅ Distributed loads (uniform, linear, sinusoidal, exponential)
- ✅ Combined point and distributed loads
- ✅ Custom load functions

### Boundary Conditions Tested:
- ✅ Cantilever (fixed-free)
- ✅ Simply supported (pin-roller)
- ✅ Multiple constraints at different nodes

### Geometry Tested:
- ✅ Horizontal beams
- ✅ Inclined beams (30°, 45°)
- ✅ L-shaped frames (90° corners)
- ✅ Beams of various lengths and cross-sections

## References

The analytical solutions used for validation are based on classical beam theory:
- Logan, D.L. "A First Course in the Finite Element Method"
- Reddy, J.N. "An Introduction to the Finite Element Method"
- Timoshenko, S.P. "Strength of Materials" (1955)
- Cowper, G.R. "The Shear Coefficient in Timoshenko's Beam Theory" (1966)
