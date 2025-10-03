class Mesh:
    """
    Class representing a finite element mesh.
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
        from fem.node import Node
        node = Node(self.node_id_counter, x, y)
        self.nodes.append(node)
        self.node_id_counter += 1
        return node

    def add_element(self, node_start, node_end, material, section, element_type="euler_bernoulli_2node"):
        from fem.element import EulerBernoulliElement2Node, EulerBernoulliElement3Node,TimoshenkoElement2Node
        if element_type == "euler_bernoulli_2node":
            element = EulerBernoulliElement2Node(self.element_id_counter, node_start, node_end, material, section)
        elif element_type == "euler_bernoulli_3node":
            element = EulerBernoulliElement3Node(self.element_id_counter, node_start, node_end, material, section)
        elif element_type == "timoshenko_2node":
            element = TimoshenkoElement2Node(self.element_id_counter, node_start, node_end, material, section)
        else:
            raise NotImplementedError(f"Element type '{element_type}' not implemented.")
        self.elements.append(element)
        self.element_id_counter += 1
        return element

    def generate_1d_mesh(self, x_start, y_start, x_end, y_end, n_elements, material, section, element_type="euler_bernoulli_2node"):
        # Structured mesh between two points
        nodes = []
        for i in range(n_elements + 1):
            x = x_start + (x_end - x_start) * i / n_elements
            y = y_start + (y_end - y_start) * i / n_elements
            node = self.add_node(x, y)
            nodes.append(node)
        for i in range(n_elements):
            self.add_element(nodes[i], nodes[i+1], material, section, element_type)
        return nodes

    def get_node_by_id(self, node_id):
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_element_by_id(self, element_id):
        for element in self.elements:
            if element.id == element_id:
                return element
        return None

    def export_mesh(self):
        # Export mesh data for visualization/post-processing
        node_data = [(node.id, node.x, node.y) for node in self.nodes]
        element_data = [(el.id, el.node_start.id, el.node_end.id) for el in self.elements]
        return {"nodes": node_data, "elements": element_data}