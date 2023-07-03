# -*- coding: utf-8 -*-
"""
Uses bvp_solver.py to solve Poissons's equation on a rectangular domain
with Dirichlet non-zero boundary conditions and a non-zero source.

Author: Oscar A. Nieves
"""

import numpy as np
import bvp_solver as bvp

# Input parameters
Nmin = 101 # minimum points in grid along either direction
xlims = (0,1)
ylims = (0,1)

[X, Y, dx, dy, Nx, Ny] = \
    bvp.optimized_mesh(xlims=xlims, ylims=ylims, Nmin=Nmin, Nmax=None, 
                       tolerance=1e-6)

x0 = np.mean(xlims)
y0 = np.mean(ylims)
width = xlims[1]-xlims[0]
height = ylims[1]-ylims[0]

# Define solution domain and compute boundary
Omega = bvp.Rectangle(X,Y,width,height,dx,dy,x0,y0)
domain = Omega.area
inner_domain = Omega.hole
boundary_list = [Omega.lower_boundary, Omega.upper_boundary,
                 Omega.left_boundary, Omega.right_boundary]

# Define boundary value functions for each boundary
BCs_on = 1
boundary_values_list = [BCs_on*np.sin(2*np.pi*X), BCs_on*np.sin(2*np.pi*X),
                        BCs_on*2*np.sin(2*np.pi*Y), BCs_on*2*np.sin(2*np.pi*Y)]

# Source function
f_source = 100*(X**2 + Y**2)
                       
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
                         source = f_source, 
                         plot_solution = True)

# Compute gradient field
stream = bvp.grad(solution, dx, dy)

# Plot vector field
bvp.plot_stream(X=X, Y=Y, A=solution, stream=stream, density=2.0, color="black")