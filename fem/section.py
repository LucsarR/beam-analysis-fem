from abc import ABC, abstractmethod
import math

class Section(ABC):
    """
    Abstract base class for cross-sectional properties.
    """
    def __init__(self, id):
        self.id = id
        self.area = None
        self.inertia = None

    @abstractmethod
    def compute_properties(self):
        pass

class RectangularBar(Section):
    def __init__(self, id, width, height):
        super().__init__(id)
        self.width = width
        self.height = height
        self.compute_properties()

    def compute_properties(self):
        self.area = self.width * self.height
        self.inertia = (self.width * self.height**3) / 12

class RectangularTube(Section):
    def __init__(self, id, width, height, thickness):
        super().__init__(id)
        self.width = width
        self.height = height
        self.thickness = thickness
        self.compute_properties()

    def compute_properties(self):
        outer_area = self.width * self.height
        inner_area = (self.width - 2*self.thickness) * (self.height - 2*self.thickness)
        self.area = outer_area - inner_area
        self.inertia = ((self.width * self.height**3) - ((self.width - 2*self.thickness) * (self.height - 2*self.thickness)**3)) / 12

class CircularBar(Section):
    def __init__(self, id, diameter):
        super().__init__(id)
        self.diameter = diameter
        self.compute_properties()

    def compute_properties(self):
        self.area = math.pi * (self.diameter/2)**2
        self.inertia = (math.pi/64) * self.diameter**4

class CircularTube(Section):
    def __init__(self, id, outer_diameter, thickness):
        super().__init__(id)
        self.outer_diameter = outer_diameter
        self.thickness = thickness
        self.compute_properties()

    def compute_properties(self):
        inner_diameter = self.outer_diameter - 2*self.thickness
        self.area = math.pi/4 * (self.outer_diameter**2 - inner_diameter**2)
        self.inertia = (math.pi/64) * (self.outer_diameter**4 - inner_diameter**4)

class TrapezoidalBar(Section):
    def __init__(self, id, base1, base2, height):
        super().__init__(id)
        self.base1 = base1
        self.base2 = base2
        self.height = height
        self.compute_properties()

    def compute_properties(self):
        self.area = 0.5 * (self.base1 + self.base2) * self.height
        # Inertia about centroidal axis (parallel to bases)
        self.inertia = (self.height**3 / 36) * (self.base1**2 + 4*self.base1*self.base2 + self.base2**2) / (self.base1 + self.base2)

class TrapezoidalTube(Section):
    def __init__(self, id, base1, base2, height, thickness):
        super().__init__(id)
        self.base1 = base1
        self.base2 = base2
        self.height = height
        self.thickness = thickness
        self.compute_properties()

    def compute_properties(self):
        outer_area = 0.5 * (self.base1 + self.base2) * self.height
        inner_base1 = self.base1 - 2*self.thickness
        inner_base2 = self.base2 - 2*self.thickness
        inner_height = self.height - 2*self.thickness
        inner_area = 0.5 * (inner_base1 + inner_base2) * inner_height
        self.area = outer_area - inner_area
        # Approximate inertia (neglecting corner effects)
        outer_inertia = (self.height**3 / 36) * (self.base1**2 + 4*self.base1*self.base2 + self.base2**2) / (self.base1 + self.base2)
        inner_inertia = (inner_height**3 / 36) * (inner_base1**2 + 4*inner_base1*inner_base2 + inner_base2**2) / (inner_base1 + inner_base2)
        self.inertia = outer_inertia - inner_inertia

class HexagonalBar(Section):
    def __init__(self, id, side):
        super().__init__(id)
        self.side = side
        self.compute_properties()

    def compute_properties(self):
        self.area = (3 * math.sqrt(3) / 2) * self.side**2
        self.inertia = (5 * math.sqrt(3) / 16) * self.side**4

class HexagonalTube(Section):
    def __init__(self, id, outer_side, thickness):
        super().__init__(id)
        self.outer_side = outer_side
        self.thickness = thickness
        self.compute_properties()

    def compute_properties(self):
        inner_side = self.outer_side - 2*self.thickness
        outer_area = (3 * math.sqrt(3) / 2) * self.outer_side**2
        inner_area = (3 * math.sqrt(3) / 2) * inner_side**2
        self.area = outer_area - inner_area
        outer_inertia = (5 * math.sqrt(3) / 16) * self.outer_side**4
        inner_inertia = (5 * math.sqrt(3) / 16) * inner_side**4
        self.inertia = outer_inertia - inner_inertia

class IBeam(Section):
    def __init__(self, id, h, b, tw, tf):
        super().__init__(id)
        self.h = h  # total height
        self.b = b  # flange width
        self.tw = tw  # web thickness
        self.tf = tf  # flange thickness
        self.compute_properties()

    def compute_properties(self):
        area_web = self.tw * (self.h - 2*self.tf)
        area_flange = self.b * self.tf * 2
        self.area = area_web + area_flange
        inertia_web = (self.tw * (self.h - 2*self.tf)**3) / 12
        inertia_flange = 2 * ((self.b * self.tf**3) / 12 + self.b * self.tf * ((self.h/2 - self.tf/2)**2))
        self.inertia = inertia_web + inertia_flange

class CSection(Section):
    def __init__(self, id, h, b, tw, tf):
        super().__init__(id)
        self.h = h
        self.b = b
        self.tw = tw
        self.tf = tf
        self.compute_properties()

    def compute_properties(self):
        area_web = self.tw * self.h
        area_flange = self.b * self.tf * 2
        self.area = area_web + area_flange
        inertia_web = (self.tw * self.h**3) / 12
        inertia_flange = 2 * ((self.b * self.tf**3) / 12 + self.b * self.tf * ((self.h/2 - self.tf/2)**2))
        self.inertia = inertia_web + inertia_flange

class LSection(Section):
    def __init__(self, id, b, h, t):
        super().__init__(id)
        self.b = b
        self.h = h
        self.t = t
        self.compute_properties()

    def compute_properties(self):
        self.area = self.b * self.t + (self.h - self.t) * self.t
        # Approximate inertia about axis through corner
        self.inertia = (self.b * self.t**3) / 12 + ((self.h - self.t) * self.t**3) / 12

class TSection(Section):
    def __init__(self, id, b, h, tw, tf):
        super().__init__(id)
        self.b = b
        self.h = h
        self.tw = tw
        self.tf = tf
        self.compute_properties()

    def compute_properties(self):
        area_web = self.tw * (self.h - self.tf)
        area_flange = self.b * self.tf
        self.area = area_web + area_flange
        inertia_web = (self.tw * (self.h - self.tf)**3) / 12
        inertia_flange = (self.b * self.tf**3) / 12 + self.b * self.tf * ((self.h - self.tf/2)**2)
        self.inertia = inertia_web + inertia_flange

class ZSection(Section):
    def __init__(self, id, h, b, tw, tf):
        super().__init__(id)
        self.h = h
        self.b = b
        self.tw = tw
        self.tf = tf
        self.compute_properties()

    def compute_properties(self):
        area_web = self.tw * self.h
        area_flange = self.b * self.tf * 2
        self.area = area_web + area_flange
        inertia_web = (self.tw * self.h**3) / 12
        inertia_flange = 2 * ((self.b * self.tf**3) / 12 + self.b * self.tf * ((self.h/2 - self.tf/2)**2))
        self.inertia = inertia_web + inertia_flange

class HatSection(Section):
    def __init__(self, id, h, b, tw, tf):
        super().__init__(id)
        self.h = h
        self.b = b
        self.tw = tw
        self.tf = tf
        self.compute_properties()

    def compute_properties(self):
        area_web = self.tw * self.h
        area_flange = self.b * self.tf * 2
        self.area = area_web + area_flange
        inertia_web = (self.tw * self.h**3) / 12
        inertia_flange = 2 * ((self.b * self.tf**3) / 12 + self.b * self.tf * ((self.h/2 - self.tf/2)**2))
        self.inertia = inertia_web + inertia_flange

class GeneralSection(Section):
    def __init__(self, id, area, inertia):
        super().__init__(id)
        self.area = area
        self.inertia = inertia

    def compute_properties(self):
        pass  # Already set by user

# Factory function
def create_section(section_type, id, **kwargs):
    section_classes = {
        "rectangular_bar": RectangularBar,
        "rectangular_tube": RectangularTube,
        "trapezoidal_bar": TrapezoidalBar,
        "trapezoidal_tube": TrapezoidalTube,
        "circular_bar": CircularBar,
        "circular_tube": CircularTube,
        "hexagonal_bar": HexagonalBar,
        "hexagonal_tube": HexagonalTube,
        "ibeam": IBeam,
        "c_section": CSection,
        "l_section": LSection,
        "t_section": TSection,
        "z_section": ZSection,
        "hat_section": HatSection,
        "general": GeneralSection,
    }
    cls = section_classes.get(section_type.lower())
    if cls:
        return cls(id, **kwargs)
    else:
        raise ValueError(f"Unknown section type: {section_type}")

# Example usage:
# section = create_section("rectangular_bar", 1, width=0.2, height=0.4)