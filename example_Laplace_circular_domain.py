# -*- coding: utf-8 -*-
"""
Uses bvp_solver.py to solve Laplace's equation on a circular domain.

Author: Oscar A. Nieves
"""
import numpy as np
import bvp_solver as bvp

# Input parameters
Nmin = 101 # minimum points in grid along either direction
xlims = (-5,5)
ylims = (-5,5)

[X, Y, dx, dy, Nx, Ny] = \
    bvp.optimized_mesh(xlims=xlims, ylims=ylims, Nmin=Nmin, Nmax=None, 
                       tolerance=1e-6)

diameter = Nx*dx
radius = diameter/2
x0, y0 = 0, 0

# Define solution domain and compute boundary
Omega = bvp.Circle(X,Y,diameter,dx,dy,x0,y0)
domain = Omega.area
inner_domain = Omega.hole
boundary_list = [Omega.boundary]

# Define boundary value functions for each boundary
def BC_func(x,y,theta):
    return radius**2 * np.sin( np.angle(x+1j*y) + theta ) * \
                       np.cos( np.angle(x+1j*y) + theta ) 

boundary_values_list = [BC_func(X,Y,0)]
                       
# Left-hand side of the bvp in the form L*u = b where L is a linear operator
# containing derivatives and coefficient functions, u(x,y) is the solution vector
# to be determined and b is the source function (also callable and optional)
LHS = bvp.Laplacian(Nx, Ny, dx, dy)

# Compute solution and plot
solution = bvp.solve_bvp(X = X, 
                         Y = Y, 
                         dx = dx, 
                         dy = dy, 
                         domain = domain, 
                         inner_domain = inner_domain,
                         boundary = boundary_list, 
                         boundary_values = boundary_values_list, 
                         LHS = LHS, 
                         source = None, 
                         plot_solution = True)