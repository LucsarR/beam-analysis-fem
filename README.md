# FEM Beam Analysis Tool

A project by **Lucas Sarmento** under the supervision of **Prof. Rafael Marques Lins**.

This tool performs finite element analysis on beams using three different formulations: **Euler-Bernoulli**, **Timoshenko**, and **Reddy-Bickford**. The goal is to provide a simple-to-use numerical tool for educational purposes, built with a friendly Streamlit interface.

## ✨ Features

* Analysis using three distinct beam theories.
* Interactive web interface for setting up the simulation.
* Visualization of displacement, shear force, and bending moment diagrams.

## 🚀 How to Run

### Prerequisites
* **Python 3.9** or higher

### Instructions
1.  **Clone the repository and navigate into the folder:**
    ```sh
    git clone [https://github.com/LucsarR/beam-analysis-fem.git](https://github.com/LucsarR/beam-analysis-fem.git)
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

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **NumPy** & **SciPy**
* **Matplotlib** & **Seaborn**