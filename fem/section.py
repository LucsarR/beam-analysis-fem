from abc import ABC, abstractmethod
import math
import numpy as np

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

    def normal_stress(self, N, M, y):
        """
        Returns normal stress sigma(y) at position y given axial force N and moment M.
        """
        if self.area is None or self.inertia is None:
            raise ValueError("Section area/inertia not set.")
        return N / self.area - M * y / self.inertia

    def xy_grid(self, n_points=100):
        """
        Returns X, Y, mask arrays for the section shape in local coordinates.
        Default: None (not implemented).
        """
        return None, None, None

class RectangularBar(Section):
    def __init__(self, id, width, height):
        super().__init__(id)
        self.width = width
        self.height = height
        self.compute_properties()

    def compute_properties(self):
        self.area = self.width * self.height
        self.inertia = (self.width * self.height**3) / 12

    def xy_grid(self, n_points=100):
        x_min, x_max = -self.width/2, self.width/2
        y_min, y_max = -self.height/2, self.height/2
        x = np.linspace(x_min, x_max, n_points)
        y = np.linspace(y_min, y_max, n_points)
        X, Y = np.meshgrid(x, y)
        mask = np.ones_like(X, dtype=bool)
        return X, Y, mask

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

    def xy_grid(self, n_points=100):
        x_min, x_max = -self.width/2, self.width/2
        y_min, y_max = -self.height/2, self.height/2
        x = np.linspace(x_min, x_max, n_points)
        y = np.linspace(y_min, y_max, n_points)
        X, Y = np.meshgrid(x, y)
        # Mask: outside inner rectangle or inside outer rectangle
        inner_x_min, inner_x_max = x_min + self.thickness, x_max - self.thickness
        inner_y_min, inner_y_max = y_min + self.thickness, y_max - self.thickness
        mask = ~((inner_x_min < X) & (X < inner_x_max) & (inner_y_min < Y) & (Y < inner_y_max))
        return X, Y, mask

class CircularBar(Section):
    def __init__(self, id, diameter):
        super().__init__(id)
        self.diameter = diameter
        self.compute_properties()

    def compute_properties(self):
        self.area = math.pi * (self.diameter/2)**2
        self.inertia = (math.pi/64) * self.diameter**4

    def xy_grid(self, n_points=100):
        r = self.diameter/2
        x = np.linspace(-r, r, n_points)
        y = np.linspace(-r, r, n_points)
        X, Y = np.meshgrid(x, y)
        mask = X**2 + Y**2 <= r**2
        return X, Y, mask

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

    def xy_grid(self, n_points=100):
        r_outer = self.outer_diameter/2
        r_inner = r_outer - self.thickness
        x = np.linspace(-r_outer, r_outer, n_points)
        y = np.linspace(-r_outer, r_outer, n_points)
        X, Y = np.meshgrid(x, y)
        R2 = X**2 + Y**2
        mask = (R2 <= r_outer**2) & (R2 >= r_inner**2)
        return X, Y, mask

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

    def xy_grid(self, n_points=100):
        y_min, y_max = -self.height/2, self.height/2
        y = np.linspace(y_min, y_max, n_points)
        # Linear interpolation for width at each y
        width_y = self.base1 + (self.base2 - self.base1) * ((y - y_min) / (y_max - y_min))
        X = np.zeros((n_points, n_points))
        Y = np.zeros((n_points, n_points))
        mask = np.zeros((n_points, n_points), dtype=bool)
        for i, yi in enumerate(y):
            w = width_y[i]
            x_min, x_max = -w/2, w/2
            x = np.linspace(x_min, x_max, n_points)
            X[i, :] = x
            Y[i, :] = yi
            mask[i, :] = True
        return X, Y, mask

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

    def xy_grid(self, n_points=100):
        y_min, y_max = -self.height/2, self.height/2
        y = np.linspace(y_min, y_max, n_points)
        width_y_outer = self.base1 + (self.base2 - self.base1) * ((y - y_min) / (y_max - y_min))
        width_y_inner = (self.base1 - 2*self.thickness) + (self.base2 - 2*self.thickness - (self.base1 - 2*self.thickness)) * ((y - (y_min + self.thickness)) / (y_max - y_min - 2*self.thickness))
        X = np.zeros((n_points, n_points))
        Y = np.zeros((n_points, n_points))
        mask = np.zeros((n_points, n_points), dtype=bool)
        for i, yi in enumerate(y):
            w_outer = width_y_outer[i]
            x_min, x_max = -w_outer/2, w_outer/2
            x = np.linspace(x_min, x_max, n_points)
            X[i, :] = x
            Y[i, :] = yi
            # Mask out inner region
            if (y_min + self.thickness) < yi < (y_max - self.thickness):
                w_inner = width_y_inner[i]
                x_min_in, x_max_in = -w_inner/2, w_inner/2
                mask[i, :] = (x < x_min_in) | (x > x_max_in)
            else:
                mask[i, :] = True
        return X, Y, mask

class HexagonalBar(Section):
    def __init__(self, id, side):
        super().__init__(id)
        self.side = side
        self.compute_properties()

    def compute_properties(self):
        self.area = (3 * math.sqrt(3) / 2) * self.side**2
        self.inertia = (5 * math.sqrt(3) / 16) * self.side**4

    def xy_grid(self, n_points=100):
        # Regular hexagon centered at (0,0)
        a = self.side
        r = a  # Approximate bounding box
        x = np.linspace(-r, r, n_points)
        y = np.linspace(-r, r, n_points)
        X, Y = np.meshgrid(x, y)
        # Hexagon mask: |x| <= a, |y| <= sqrt(3)*a/2, |y| <= sqrt(3)*(a-|x|)/2
        mask = (np.abs(X) <= a) & (np.abs(Y) <= np.sqrt(3)*a/2) & (np.abs(Y) <= np.sqrt(3)*(a-np.abs(X))/2)
        return X, Y, mask

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

    def xy_grid(self, n_points=100):
        a = self.outer_side
        ai = a - 2*self.thickness
        r = a
        x = np.linspace(-r, r, n_points)
        y = np.linspace(-r, r, n_points)
        X, Y = np.meshgrid(x, y)
        mask_outer = (np.abs(X) <= a) & (np.abs(Y) <= np.sqrt(3)*a/2) & (np.abs(Y) <= np.sqrt(3)*(a-np.abs(X))/2)
        mask_inner = (np.abs(X) <= ai) & (np.abs(Y) <= np.sqrt(3)*ai/2) & (np.abs(Y) <= np.sqrt(3)*(ai-np.abs(X))/2)
        mask = mask_outer & ~mask_inner
        return X, Y, mask

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

    def xy_grid(self, n_points=100):
        x_min, x_max = -self.b/2, self.b/2
        y_min, y_max = -self.h/2, self.h/2
        x = np.linspace(x_min, x_max, n_points)
        y = np.linspace(y_min, y_max, n_points)
        X, Y = np.meshgrid(x, y)
        # Flanges: |y| > (h/2 - tf)
        mask_flange = (np.abs(Y) > (self.h/2 - self.tf)) & (np.abs(X) <= self.b/2)
        # Web: |x| <= tw/2 and |y| <= (h/2 - tf)
        mask_web = (np.abs(X) <= self.tw/2) & (np.abs(Y) <= (self.h/2 - self.tf))
        mask = mask_flange | mask_web
        return X, Y, mask

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

    def xy_grid(self, n_points=100):
        x_min, x_max = -self.b/2, self.b/2
        y_min, y_max = -self.h/2, self.h/2
        x = np.linspace(x_min, x_max, n_points)
        y = np.linspace(y_min, y_max, n_points)
        X, Y = np.meshgrid(x, y)
        # Flanges: y > (h/2 - tf) or y < -(h/2 - tf)
        mask_flange = ((Y > (self.h/2 - self.tf)) | (Y < -(self.h/2 - self.tf))) & (X >= -self.b/2) & (X <= self.b/2)
        # Web: X between -b/2 and -b/2+tw, Y between -(h/2-tf) and (h/2-tf)
        mask_web = (X >= -self.b/2) & (X <= -self.b/2 + self.tw) & (np.abs(Y) <= (self.h/2 - self.tf))
        mask = mask_flange | mask_web
        return X, Y, mask

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

    def xy_grid(self, n_points=100):
        x_min, x_max = 0, self.b
        y_min, y_max = 0, self.h
        x = np.linspace(x_min, x_max, n_points)
        y = np.linspace(y_min, y_max, n_points)
        X, Y = np.meshgrid(x, y)
        # Horizontal leg: Y <= t
        mask_h = (Y <= self.t)
        # Vertical leg: X <= t
        mask_v = (X <= self.t)
        mask = mask_h | mask_v
        # Center at (b/2, h/2)
        X = X - self.b/2
        Y = Y - self.h/2
        return X, Y, mask

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

    def xy_grid(self, n_points=100):
        x_min, x_max = -self.b/2, self.b/2
        y_min, y_max = -self.h/2, self.h/2
        x = np.linspace(x_min, x_max, n_points)
        y = np.linspace(y_min, y_max, n_points)
        X, Y = np.meshgrid(x, y)
        # Flange: Y > (h/2 - tf)
        mask_flange = (Y > (self.h/2 - self.tf)) & (np.abs(X) <= self.b/2)
        # Web: |X| <= tw/2 and Y <= (h/2 - tf)
        mask_web = (np.abs(X) <= self.tw/2) & (Y <= (self.h/2 - self.tf)) & (Y >= -self.h/2)
        mask = mask_flange | mask_web
        return X, Y, mask

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

    def xy_grid(self, n_points=100):
        # Approximate as two flanges and a web, offset in x
        x_min, x_max = -self.b, self.b
        y_min, y_max = -self.h/2, self.h/2
        x = np.linspace(x_min, x_max, n_points)
        y = np.linspace(y_min, y_max, n_points)
        X, Y = np.meshgrid(x, y)
        # Lower flange: Y < -(h/2 - tf), X in [0, b]
        mask_lower = (Y < -(self.h/2 - self.tf)) & (X >= 0) & (X <= self.b)
        # Upper flange: Y > (h/2 - tf), X in [-b, 0]
        mask_upper = (Y > (self.h/2 - self.tf)) & (X >= -self.b) & (X <= 0)
        # Web: X in [-tw/2, tw/2], Y in [-(h/2-tf), (h/2-tf)]
        mask_web = (np.abs(X) <= self.tw/2) & (np.abs(Y) <= (self.h/2 - self.tf))
        mask = mask_lower | mask_upper | mask_web
        return X, Y, mask

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

    def xy_grid(self, n_points=100):
        x_min, x_max = -self.b/2, self.b/2
        y_min, y_max = -self.h/2, self.h/2
        x = np.linspace(x_min, x_max, n_points)
        y = np.linspace(y_min, y_max, n_points)
        X, Y = np.meshgrid(x, y)
        # Flanges: Y > (h/2 - tf) or Y < -(h/2 - tf)
        mask_flange = ((Y > (self.h/2 - self.tf)) | (Y < -(self.h/2 - self.tf))) & (np.abs(X) <= self.b/2)
        # Web: |X| <= tw/2 and |Y| <= (h/2 - tf)
        mask_web = (np.abs(X) <= self.tw/2) & (np.abs(Y) <= (self.h/2 - self.tf))
        mask = mask_flange | mask_web
        return X, Y, mask

class GeneralSection(Section):
    def __init__(self, id, area, inertia):
        super().__init__(id)
        self.area = area
        self.inertia = inertia

    def compute_properties(self):
        pass  # Already set by user

    def xy_grid(self, n_points=100):
        # Not defined for general section
        return None, None, None

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