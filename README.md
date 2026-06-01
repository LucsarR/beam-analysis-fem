# FEM Beam Analysis Tool

A project by **Lucas Sarmento** under the supervision of **Prof. Rafael Marques Lins**.

This tool performs finite element analysis on beams using three different formulations: **Euler-Bernoulli**, **Timoshenko**, and **Reddy-Bickford**. The goal is to provide a simple-to-use numerical tool for educational purposes, built with a friendly Streamlit interface.

## ✨ Features

* Analysis using three distinct beam theories
* Multiple element types:
  * **Euler-Bernoulli 2-node**: Standard linear beam element
  * **Euler-Bernoulli 3-node**: Enhanced element with central node for improved accuracy with distributed loads
  * **Timoshenko 2-node**: Includes shear deformation effects
  * **Timoshenko 3-node**: Enhanced element with central node for improved accuracy with distributed loads and shear deformation
  * **Reddy-Bickford RBT 2-node**: Third-order shear deformation theory element
  * **Reddy-Bickford MRBT 2-node**: Modified Reddy formulation with coupled shear-flexibility terms
* Interactive web interface for setting up simulations
* Visualization of displacement, shear force, bending moment, and normal force diagrams
* Support for point loads and distributed loads (constant, linear, and custom functions)
* Spring supports (translational and rotational)
* Reaction force calculation at all constrained DOFs
* Multiple cross-section types:
  * Solid sections: rectangular bar, circular bar, trapezoidal bar, hexagonal bar
  * Hollow sections: rectangular tube, circular tube, trapezoidal tube, hexagonal tube
  * Structural sections: I-beam, C-section, L-section, T-section, Z-section, hat section, general
* Material definition from any two of E, G, and ν (third is computed automatically)
* Project save/load functionality

## 📚 Element Types

### Euler-Bernoulli 2-Node
Standard 2-node beam element based on classical beam theory. Suitable for slender beams where shear deformation is negligible. Each end node has 3 DOFs: axial displacement (u), transverse displacement (v), and rotation (θ).

### Euler-Bernoulli 3-Node
Enhanced 3-node element with a central node for improved accuracy:
- Uses quadratic shape functions for axial displacement
- Uses Hermite cubic polynomials + bubble function for bending
- Provides better representation of distributed loads
- Central node has full 3 DOFs (u, v, θ) — same as end nodes
- 9 DOFs total per element

### Timoshenko 2-Node
Includes shear deformation effects. More accurate for thick beams and high frequencies. Uses the field-consistent (phi-corrected) stiffness matrix to avoid shear locking.

### Timoshenko 3-Node
Enhanced 3-node element with shear deformation effects:
- Uses quadratic shape functions for axial displacement, transverse displacement, and rotation
- Provides better representation of distributed loads
- Includes shear deformation for thick beams
- Uses selective reduced integration to avoid shear locking
- All three nodes have full rotation DOFs
- 9 DOFs total per element

### Reddy-Bickford 2-Node
Third-order shear deformation theory element:
- Based on Reddy's third-order beam theory
- Parabolically distributed shear strain through the thickness
- Does not require a shear correction factor
- Each node has 4 DOFs: u, v, θ, dv/dx
- 8 DOFs total per element

## 🚀 How to Run

### Prerequisites
* **Python 3.9** or higher

### Instructions
1.  **Clone the repository and navigate into the folder:**
    ```sh
    git clone https://github.com/LucsarR/beam-analysis-fem.git
    cd beam-analysis-fem
    ```

2.  **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3.  **Launch the app:**
    ```sh
    streamlit run app.py
    ```

4.  **Run tests:**
    ```sh
    export PYTHONPATH=$(pwd)
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
    ```

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **NumPy** & **SciPy** & **SymPy**
* **Matplotlib** & **Seaborn** & **Plotly**

## 📖 References

The element implementations are based on:
- Reddy, J.N. "An Introduction to the Finite Element Method" (2006)
- Bathe, K.J. "Finite Element Procedures" (1996)
- Logan, D.L. "A First Course in the Finite Element Method" (2017)
- Timoshenko, S.P. "Strength of Materials" (1955)
- Cowper, G.R. "The Shear Coefficient in Timoshenko's Beam Theory" (1966)
- EN 1993-1-1 (Eurocode 3), Annexes for steel member shear area approximations (A_v ≈ A_web for thin-walled open sections)
