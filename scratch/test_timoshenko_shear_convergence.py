import numpy as np
from fem.mesh import Mesh
from fem.material import Material
from fem.section import RectangularBar
from fem.constraint import Constraint
from fem.load import DistributedLoad
from fem.analysis import BeamAnalysis
from fem.analysis import get_element_dof_indices
from post_processing.forces import ElementResults

# Beam properties
L = 5.0
E = 1e9
nu = 0.3
b = 0.05
h = 0.1

# We will create a mesh with 10 elements of timoshenko_3node
n_elements = 10
mesh = Mesh()
mat = Material(1, E, nu)
sec = RectangularBar(1, b, h)

# Create nodes
nodes = []
for i in range(n_elements + 1):
    x = L * i / n_elements
    node = mesh.add_node(x, 0)
    nodes.append(node)

# Create elements
elements = []
for i in range(n_elements):
    el = mesh.add_element(nodes[i], nodes[i+1], mat, sec, 'timoshenko_3node')
    elements.append(el)

# Cantilever fixed at left
mesh.constraints.add(Constraint(nodes[0], 0, 0.0))
mesh.constraints.add(Constraint(nodes[0], 1, 0.0))
mesh.constraints.add(Constraint(nodes[0], 2, 0.0))

# Uniform distributed load on all elements
for el in elements:
    dist_load = DistributedLoad(magnitude_start=-10.0, magnitude_end=-10.0, direction='y')
    dist_load.element = el
    mesh.distributed_loads.append(dist_load)

# Solve
analysis = BeamAnalysis(mesh)
analysis.assemble()
displacements = analysis.solve()

# Now sample recovered shear force at nodes
print("x\tRecovered V\tExact V")
for i in range(n_elements + 1):
    x = L * i / n_elements
    # To get shear force at x, we use the element that contains x
    if i < n_elements:
        el = elements[i]
        dof_indices = get_element_dof_indices(el, analysis.dpn)
        el_disps = displacements[dof_indices]
        V_rec = el.shear_force(0.0, el_disps)  # Start of element i
    else:
        el = elements[-1]
        dof_indices = get_element_dof_indices(el, analysis.dpn)
        el_disps = displacements[dof_indices]
        V_rec = el.shear_force(el.length, el_disps)  # End of last element
        
    V_exact = 50.0 - 10.0 * x
    print(f"{x:.2f}\t{V_rec:.6f}\t{V_exact:.6f}")
