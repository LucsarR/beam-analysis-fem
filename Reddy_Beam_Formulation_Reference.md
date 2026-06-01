# Finite Element Formulation for Improved Reddy Beam (RBT & MRBT)

This document serves as a comprehensive reference guide for implementing the Finite Element Method (FEM) using the Reddy Beam Theory (RBT) and the Modified Reddy Beam Theory (MRBT) for framed structures, as described in the provided paper.

---

## 1. Kinematics and Theories (RBT)

The Reddy beam model is a higher-order shear deformation theory that allows cross-sections to warp, ensuring that the profile of the warped section intersects the upper and lower surfaces orthogonally (satisfying zero shear stress boundary conditions at the top and bottom fibers).

### Displacement Field
The displacement field in the $x$ (axial) and $y$ (transversal) directions is given by:
$$u(x,y) = u_0(x) - y\theta(x) + \alpha y^3 \left( \theta(x) - \frac{dv_0(x)}{dx} \right)$$
$$v(x,y) = v_0(x)$$

* $u_0(x)$ is the axial displacement of the centroidal axis.
* $v_0(x)$ is the transverse displacement of the centroidal axis.
* $\theta(x)$ is the rotation of the cross-section.
* $\alpha$ is the Reddy constant. For a rectangular section of height $h$, $\alpha = \frac{4}{3h^2}$.

### Strain Field
The non-zero components of the linear Green-Lagrange strain tensor are:
$$\varepsilon_x = \frac{du_0}{dx} - y\frac{d\theta}{dx} + \alpha y^3 \left( \frac{d\theta}{dx} - \frac{d^2v_0}{dx^2} \right)$$
$$\gamma_{xy} = \left( \theta(x) - \frac{dv_0}{dx} \right) (3\alpha y^2 - 1)$$

---

## 2. Interpolating (Shape) Functions

### 2.1 Standard RBT Shape Functions
In standard Finite Element applications of the Reddy Beam Theory, independent interpolations are typically used:
* **Transverse Displacement ($v_0$)**: Hermitian cubic polynomials.
* **Axial Displacement ($u_0$)**: Linear polynomials.
* **Rotation ($\theta$)**: Linear polynomials.

**Limitation:** Because the rotational components are independent of transverse displacements, standard RBT exhibits numerical instabilities (shear locking phenomena) and requires a very dense discretization (many elements per member) to converge to the exact analytical solution.

### 2.2 Modified RBT (MRBT) Shape Functions
To overcome the limitations of the standard RBT, the MRBT derives shape functions directly from the analytical homogenous solution of the differential equations governing the Reddy beam.

The exact analytical solution contains exponential terms $e^{\mu x}$ and $e^{-\mu x}$ where $\mu = \frac{2\sqrt{105}}{h\sqrt{1+\nu}}$. These exponentials cause numerical instabilities. Thus, the MRBT replaces the exponential terms with their Taylor series expansion truncated to the second order:
$$e^{\pm \mu x} \approx 1 \pm \mu x + \frac{\mu^2 x^2}{2}$$

The modified polynomial kinematics become:
$$v_0(x) = -\frac{c_1 h^2 (1+\nu)}{420 EI}x + c_2 + c_3 \left(1 - \mu x + \frac{\mu^2 x^2}{2}\right) + c_4 \left(1 + \mu x + \frac{\mu^2 x^2}{2}\right) - \frac{c_5 x^3}{6 EI} - \frac{c_6 x^2}{2 EI}$$

$$\theta(x) = \frac{c_1 h^2 (1+\nu)}{168 EI} + \frac{\mu}{4}c_3 \left(1 - \mu x + \frac{\mu^2 x^2}{2}\right) - \frac{\mu}{4}c_4 \left(1 + \mu x + \frac{\mu^2 x^2}{2}\right) - \frac{c_5 x^2}{2 EI} - \frac{c_6 x}{EI} + c_7$$

The shape functions $[N]$ are formulated by evaluating the nodal variables at $x = 0$ and $x = L$, yielding a transformation matrix $[H]$ such that the constants $\{C\}$ map to nodal displacements $\{d'\}$. Thus, $[N] = [X][H]^{-1}$.

---

## 3. Stiffness Matrices

For a 2D frame element, the nodal displacement vector is $\{d\} = [u_1, v_1, \theta_1, u_2, v_2, \theta_2]^T$.

### 3.1 Standard RBT Matrix
Derived from standard Hermitian and Linear interpolations, the RBT stiffness matrix $K^{RBT}$ possesses artificially high rigidity regarding transverse displacements due to the uncoupled nature of $\theta$ and $v_0$. Key cross-coupled rigidity terms take the form $\frac{16GA}{25L}$ and $\frac{4GA}{75}$. (Refer to Eq. 33 of the text for the fully expanded explicit matrix).

### 3.2 Modified MRBT Matrix
Using the truncated polynomial approximations $[N]$, the stiffness matrix $K^{MRBT}$ incorporates shear deformation mechanics directly into coupled terms. 
The equivalent modified $6 \times 6$ local stiffness matrix closely resembles the classical Euler-Bernoulli (EBBT) layout but integrates Timoshenko-like shear flexibility without requiring arbitrary shear correction factors. 

**Structure of the MRBT Stiffness Matrix:**
$$K^{MRBT} = \int_0^L [B]^T [D] [B] \, dx$$
*(During implementation, this matrix should be assembled using numerical integration like Gauss-Legendre quadrature over the domain $0$ to $L$ utilizing the explicit polynomial derivatives of $[N]$ derived above, or explicitly matching the constants in Eq 34).*

---

## 4. Equivalent Nodal Loads

For linearly distributed loads (e.g., $q(x)$ acting transversely), the equivalent nodal loads $f_i$ for the element are obtained by integrating the shape functions over the element length:

$$f_i = \int_0^L N_i^v(x) \cdot q(x) \, dx$$

Because the proposed interpolation functions for transverse displacement $N^v(x)$ in the MRBT are cubic polynomials derived from the series expansion, the equivalent nodal loads for MRBT correspond exactly to the standard Bernoulli-Euler beam theory fixed-end forces.

For a uniformly distributed load $q_0$:
* $f_{y1} = f_{y2} = \frac{q_0 L}{2}$
* $M_1 = \frac{q_0 L^2}{12}$
* $M_2 = -\frac{q_0 L^2}{12}$

For a triangular load $q_1(x) = q_1 \frac{x}{L}$:
* $f_{y1} = \frac{3 q_1 L}{20}$
* $M_1 = \frac{q_1 L^2}{30}$
* $f_{y2} = \frac{7 q_1 L}{20}$
* $M_2 = -\frac{q_1 L^2}{20}$

---

## 5. Shear Stress Post-Processing

One of the primary advantages of the MRBT is its accurate prediction of the local shear stress fields without requiring highly refined meshes. 

**Shear Strain Computation:**
$$\gamma_{xy} = \left( \theta(x) - \frac{dv_0(x)}{dx} \right) (3\alpha y^2 - 1)$$
Where $\theta(x)$ and $\frac{dv_0(x)}{dx}$ are calculated using the element shape functions $[N]$ and the solved nodal displacements.

**Shear Stress Computation:**
By applying the material's shear modulus $G$:
$$\tau_{xy}(x,y) = G \cdot \gamma_{xy} = G \left( \theta(x) - \frac{dv_0}{dx} \right) (3\alpha y^2 - 1)$$

**Implementation Note:** 1. EBBT provides an undetermined shear stress.
2. TBT provides a constant shear stress across the thickness.
3. RBT provides a parabolic shear stress, but requires massive discretization (e.g., 40-160 elements) to converge.
4. **MRBT implementation** accurately yields the correct parabolic shear stress variation, satisfying $\tau_{xy} = 0$ at $y = \pm h/2$, and requires only a minimal number of elements (e.g., 4 elements) to converge within 1% error.
