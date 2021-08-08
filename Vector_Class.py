"""
A vector class for vector algebra operations

Author: Oscar A. Nieves
Updated: August 8, 2021
"""
import math

class Vector:
    """ This is a vector class containing vector operations
    in 3 dimensions for physics or engineering applications,
    including dot product, cross product, magnitude, unit 
    vector, finding the angle between two vectors.
    
    Y and Z coordinates are set to 0 by default in case we
    are dealing with a 1-dimensional vector;.
    """
    def __init__(self,x,y=0,z=0):
        self.x, self.y, self.z = x, y, z
    
    # Add
    def add(self, other):
        nx = self.x + other.x
        ny = self.y + other.y
        nz = self.z + other.z
        return Vector(nx,ny,nz)
    
    # Subtract
    def sub(self, other):
        nx = self.x - other.x
        ny = self.y - other.y
        nz = self.z - other.z
        return Vector(nx,ny,nz)
    
    # Magnitude
    def norm(self):
        return math.sqrt(self.x**2 + self.y**2 +\
                         self.z**2)
    
    # Unit vector
    def unit(self):
        Mag = self.norm()
        nx = self.x/Mag
        ny = self.y/Mag
        nz = self.z/Mag
        return Vector(nx,ny,nz)
    
    # Dot product
    def dot(self, other):
        return self.x*other.x + self.y*other.y +\
            self.z*other.z
    
    # Cross product
    def cross(self, other):
        nx = self.y*other.z - self.z*other.y
        ny = self.z*other.x - self.x*other.z
        nz = self.x*other.y - self.y*other.x
        return Vector(nx,ny,nz)
    
    def angle(self, other):
        dot0 = self.dot(other)
        den0 = self.norm() * other.norm()
        return math.acos( dot0/den0 )

# ------ Examples ------  #
a = Vector(1,3,4)
b = Vector(-2,7,-5)

mag_a = a.norm()
print('||a|| = ' + str(mag_a))

unit_b = b.unit()
print('b unit = ' + str([unit_b.x, unit_b.y, unit_b.z]))

c = a.dot(b)
print('a dot b = ' + str(c))   

d = a.cross(b)
print('a cross b = ' + str([d.x, d.y, d.z]))

e = a.add(b)
print('a + b = ' + str([e.x, e.y, e.z]))
     
ang = 180*(a.angle(b))/math.pi # degrees
print('theta = ' + str(ang) + ' deg')