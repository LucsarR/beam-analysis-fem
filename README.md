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
* Interactive web interface for setting up simulations
* Visualization of displacement, shear force, and bending moment diagrams
* Support for point loads and distributed loads (constant, linear, and custom functions)
* Multiple cross-section types (rectangular, circular, I-beam, etc.)
* Project save/load functionality

## 📚 Element Types

### Euler-Bernoulli 2-Node
Standard 2-node beam element based on classical beam theory. Suitable for slender beams where shear deformation is negligible.

### Euler-Bernoulli 3-Node
Enhanced 3-node element with a central node for improved accuracy:
- Uses quadratic shape functions for axial displacement
- Uses Hermite cubic polynomials + bubble function for bending
- Provides better representation of distributed loads
- Exact for point loads at nodes
- Central node has displacement DOFs only (no rotation)

### Timoshenko 2-Node
Includes shear deformation effects. More accurate for thick beams and high frequencies.

### Timoshenko 3-Node
Enhanced 3-node element with shear deformation effects:
- Uses quadratic shape functions for axial and transverse displacement
- Uses quadratic shape functions for rotation
- Provides better representation of distributed loads
- Includes shear deformation for thick beams
- Uses selective reduced integration to avoid shear locking
- All three nodes have rotation DOFs (unlike Euler-Bernoulli 3-node)

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
    python tests/test_euler_bernoulli_3node.py
    python tests/test_mesh_integration.py
    ```

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **NumPy** & **SciPy**
* **Matplotlib** & **Seaborn** & **Plotly**

## 📖 References

The 3-node Euler-Bernoulli element implementation is based on:
- Reddy, J.N. "An Introduction to the Finite Element Method" (2006)
- Bathe, K.J. "Finite Element Procedures" (1996)
- Logan, D.L. "A First Course in the Finite Element Method" (2017)