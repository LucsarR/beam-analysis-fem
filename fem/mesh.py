import numpy as np

class Mesh:
    """
    Class representing a finite element mesh.
    
    The Mesh class is the central container for all FEM components and manages:
    - Nodes: spatial points in the structure
    - Elements: beam elements connecting nodes (EulerBernoulli, Timoshenko)
    - Constraints: boundary conditions applied to nodes
    - Loads: point loads and distributed loads on the structure
    
    Integration with FEM framework:
    - Works with Material and Section classes to define element properties
    - Integrates with Analysis classes for system assembly and solving
    - Supports post-processing through StructureResults and ElementResults
    - All element types (EulerBernoulli, Timoshenko) provide consistent interfaces
    
    Attributes:
        nodes (list): List of Node objects in the mesh
        elements (list): List of Element objects (various types) in the mesh
        point_loads (list): List of PointLoad objects applied to nodes
        distributed_loads (list): List of DistributedLoad objects applied to elements
        constraints (ConstraintSet): Boundary conditions applied to the mesh
        node_id_counter (int): Counter for assigning unique node IDs
        element_id_counter (int): Counter for assigning unique element IDs
    """
    def __init__(self):
        self.nodes = []
        self.elements = []
        self.node_id_counter = 1
        self.element_id_counter = 1
        self.point_loads = []
        self.distributed_loads = []
        from fem.constraint import ConstraintSet
        self.constraints = ConstraintSet()

    def add_node(self, x, y):
        """
        Add a node to the mesh at the specified coordinates.
        
        Args:
            x (float): X-coordinate of the node
            y (float): Y-coordinate of the node
            
        Returns:
            Node: The created node object with assigned ID
        """
        from fem.node import Node
        node = Node(self.node_id_counter, x, y)
        self.nodes.append(node)
        self.node_id_counter += 1
        return node

    def add_element(self, node_start, node_end, material, section, element_type="euler_bernoulli_2node", stiffness_integration="analytical"):
        """
        Add an element to the mesh connecting two nodes.
        
        Args:
            node_start (Node): Starting node of the element
            node_end (Node): Ending node of the element
            material (Material): Material properties for the element
            section (Section): Cross-section properties for the element
            element_type (str): Type of element to create. Options:
                - "euler_bernoulli_2node": 2-node Euler-Bernoulli beam (default)
                - "euler_bernoulli_3node": 3-node Euler-Bernoulli beam
                - "timoshenko_2node": 2-node Timoshenko beam
                - "timoshenko_3node": 3-node Timoshenko beam
            stiffness_integration (str): Stiffness matrix formulation ("analytical" or "numerical")
                
        Returns:
            Element: The created element object with assigned ID
            
        Raises:
            NotImplementedError: If element_type is not supported
        """
        from fem.element import (
            EulerBernoulliElement2Node, EulerBernoulliElement3Node,
            TimoshenkoElement2Node, TimoshenkoElement3Node,
            ReddyBickfordElement2Node, MRBTElement2Node
        )
        if element_type == "euler_bernoulli_2node":
            element = EulerBernoulliElement2Node(
                self.element_id_counter, node_start, node_end, material, section,
                stiffness_integration=stiffness_integration
            )
        elif element_type == "euler_bernoulli_3node":
            # For 3-node element, create or find the central node
            x_center = (node_start.x + node_end.x) / 2
            y_center = (node_start.y + node_end.y) / 2
            # Check if central node already exists
            node_center = None
            for node in self.nodes:
                if np.isclose(node.x, x_center) and np.isclose(node.y, y_center):
                    node_center = node
                    break
            if node_center is None:
                node_center = self.add_node(x_center, y_center)
            element = EulerBernoulliElement3Node(self.element_id_counter, node_start, node_end, material, section, node_center)
        elif element_type == "timoshenko_2node":
            element = TimoshenkoElement2Node(
                self.element_id_counter, node_start, node_end, material, section,
                stiffness_integration=stiffness_integration
            )
        elif element_type == "timoshenko_3node":
            # For 3-node element, create or find the central node
            x_center = (node_start.x + node_end.x) / 2
            y_center = (node_start.y + node_end.y) / 2
            # Check if central node already exists
            node_center = None
            for node in self.nodes:
                if np.isclose(node.x, x_center) and np.isclose(node.y, y_center):
                    node_center = node
                    break
            if node_center is None:
                node_center = self.add_node(x_center, y_center)
            element = TimoshenkoElement3Node(self.element_id_counter, node_start, node_end, material, section, node_center)
        elif element_type == "reddy_bickford_2node":
            element = ReddyBickfordElement2Node(self.element_id_counter, node_start, node_end, material, section)
        elif element_type == "mrbt_2node":
            element = MRBTElement2Node(self.element_id_counter, node_start, node_end, material, section)
        else:
            raise NotImplementedError(f"Element type '{element_type}' not implemented.")
        self.elements.append(element)
        self.element_id_counter += 1
        return element

    def generate_1d_mesh(self, x_start, y_start, x_end, y_end, n_elements, material, section, element_type="euler_bernoulli_2node", stiffness_integration="analytical"):
        """
        Generate a structured 1D mesh between two points.
        
        Creates evenly spaced nodes along a line and connects them with elements.
        Useful for quickly creating beam meshes.
        
        Args:
            x_start (float): X-coordinate of start point
            y_start (float): Y-coordinate of start point
            x_end (float): X-coordinate of end point
            y_end (float): Y-coordinate of end point
            n_elements (int): Number of elements to create
            material (Material): Material properties for all elements
            section (Section): Cross-section properties for all elements
            element_type (str): Type of elements to create (default: "euler_bernoulli_2node")
            stiffness_integration (str): Stiffness matrix formulation ("analytical" or "numerical")
            
        Returns:
            list: List of created Node objects (length: n_elements + 1)
        """
        # Structured mesh between two points
        nodes = []
        for i in range(n_elements + 1):
            x = x_start + (x_end - x_start) * i / n_elements
            y = y_start + (y_end - y_start) * i / n_elements
            node = self.add_node(x, y)
            nodes.append(node)
        for i in range(n_elements):
            self.add_element(
                nodes[i], nodes[i+1], material, section, element_type,
                stiffness_integration=stiffness_integration
            )
        return nodes

    def get_node_by_id(self, node_id):
        """
        Retrieve a node by its ID.
        
        Args:
            node_id (int): ID of the node to find
            
        Returns:
            Node: The node with the specified ID, or None if not found
        """
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_element_by_id(self, element_id):
        """
        Retrieve an element by its ID.
        
        Args:
            element_id (int): ID of the element to find
            
        Returns:
            Element: The element with the specified ID, or None if not found
        """
        for element in self.elements:
            if element.id == element_id:
                return element
        return None

    def export_mesh(self):
        """
        Export mesh data for visualization and post-processing.
        
        Returns:
            dict: Dictionary with keys:
                - 'nodes': List of tuples (id, x, y) for each node
                - 'elements': List of tuples (id, node_start_id, node_end_id) for each element
        """
        # Export mesh data for visualization/post-processing
        node_data = [(node.id, node.x, node.y) for node in self.nodes]
        element_data = [(el.id, el.node_start.id, el.node_end.id) for el in self.elements]
        return {"nodes": node_data, "elements": element_data}
