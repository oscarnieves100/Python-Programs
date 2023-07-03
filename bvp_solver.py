# -*- coding: utf-8 -*-
"""
%-----------------------------------------------------------------------------%
Description of the module:

A self-contained module (dependencies: NumPy, Numba, Scipy and Matplotlib) for 
solving 2-dimensional boundary-value problems associated with linear partial 
differential equations (PDEs) on simply OR multiply connected domains 
(e.g. containing holes).

It is originally intended for solving steady-state problems (in rectangular
coordinates only) which are expressible in the following general form:

    L*u(x,y) = b(x,y)

where L is some linear operator containing partial derivative and coefficient
functions, u(x,y) is the vector of solutions to be determined, and b(x,y) is 
an optional source term. In this way, we can solve equations such as:
    
    ∇^2 u(x,y) = 0 --> Laplace's equation
    ∇^2 u(x,y) = f(x,y) --> Poisson equation
    ∇^2 u(x,y) + k^2 u(x,y) = f(x,y) --> Helmholtz equation
    
It uses the finite difference method (FDM) with centered finite difference
schemes. Currently, the module contains functions to create 1st and 2nd order
partial derivative matrices, namely Dx, Dxx, Dy and Dyy, and the Laplacian
operator. However, the main function 'solve_pde()' takes in any left-hand
side operator LHS as an input, so as the user you can define this L matrix
using anything you like, even your own derivative matrices of arbitrary order.
%-----------------------------------------------------------------------------%    
Author: Oscar A. Nieves
%-----------------------------------------------------------------------------%
"""
import numpy as np
import numba as nb
import scipy as sp
import time
from scipy.sparse import csc_matrix 
import matplotlib.pyplot as plt

###############################################################################
# Geometry objects and Boundary points extraction
###############################################################################
class Shape:
    """A generic shape object consisting of 2 attributes (binary matrices):
        a filled area (of the shape) and the shape's boundary
    
    Args:
        area: a binary matrix with 1's in every point (x,y) that lies within
                the shape's domain
        boundary: a binary matrix with 1's in every point along the boundary 
                    enclosing the shape
    
    Functions:
        union: creates the union of two shapes: self and other, and then computes
                the binary matrix corresponding to the combined (overlapping)
                areas, the new boundary enclosing it and the hole enclosed
                within the boundary.
        insert_hole: inserts a hole of given shape into the current shape. It also
                     adds the inserted hole object to the self.subdomains list
                     which can be used for keeping track of all the holes and their
                     attributes (e.g. area, boundary). This function verifies if the
                     given hole shape lies entirely within the current Shape's area,
                     otherwise it will throw an error.
    """
    def __init__(self, 
                 area: np.ndarray = np.array([1]), 
                 boundary: np.ndarray = np.array([1]), 
        ): 
        if type(area) is not sp.sparse._csc.csc_matrix: area = csc_matrix(area)
        if type(boundary) is not sp.sparse._csc.csc_matrix: boundary = csc_matrix(boundary)
        self.area = area
        self.boundary = boundary
        self.hole = csc_matrix( np.logical_xor(area.todense(), boundary.todense()) )
        self.subdomains = []
        self.outer_boundary = self.extract_outer_boundary()
    
    def union(self, other):
        self_area = self.area.todense()
        other_area = other.area.todense()
        new_area = np.logical_or(self_area, other_area)
        new_boundary = extract_boundary(new_area)
        self.area = csc_matrix(new_area)
        self.boundary = csc_matrix(new_boundary)
        self.outer_boundary = self.extract_outer_boundary()
        self.hole = csc_matrix( np.logical_xor(self.area.todense(), 
                                               self.boundary.todense()) )
        
    def difference(self, other):
        self_area = self.area.todense()
        other_area = other.area.todense()
        intersection = np.logical_and(self_area, other_area)
        new_area = np.logical_and(self_area, np.logical_not(intersection))
        new_boundary = extract_boundary(new_area)
        self.area = csc_matrix(new_area)
        self.boundary = csc_matrix(new_boundary)
        self.outer_boundary = csc_matrix(new_boundary)
        self.hole = csc_matrix( np.logical_xor(self.area.todense(), 
                                               self.boundary.todense()) )
    
    def insert_hole(self, other):
        self_area = self.area.todense()
        other_area = other.area.todense()
        assert other_area in self_area
        new_area = np.logical_xor(self_area, other.hole.todense())
        new_boundary = extract_boundary(new_area)
        self.area = csc_matrix(new_area)
        self.boundary = csc_matrix(new_boundary)
        self.hole = csc_matrix( np.logical_xor(self.area.todense(), 
                                               self.boundary.todense()) )
        self.subdomains.append( other )
        self.outer_boundary = self.extract_outer_boundary()
        
    def extract_outer_boundary(self):
        outer_boundary = self.boundary
        if len(self.subdomains) > 0:
            for n in range(len(self.subdomains)):
                outer_boundary -= self.subdomains[n].boundary
        return csc_matrix(outer_boundary)
            
class Rectangle(Shape):
    """A rectangle shape
    
    Args:
        X, Y: 2D arrays of coordinates produced from np.meshgrid(x,y) where x
              and y are 1D arrays of coordinates enclosing the full solution domain
        width, height: width and height of rectangle
        dx, dy: the step-sizes in given directions
        x0, y0: the center coordinates of the rectangle.
    """
    def __init__(self, 
                 X: np.ndarray, 
                 Y: np.ndarray, 
                 width: float, 
                 height: float, 
                 dx: float,
                 dy: float,
                 x0: float, 
                 y0: float,
         ):
        super().__init__()
        self.center = (x0, y0)
        self.width = width
        self.height = height
        self.area = csc_matrix( (abs(X-x0) <= width/2) * (abs(Y-y0) <= height/2) )
        self.boundary = extract_boundary( self.area.todense() )
        self.hole = csc_matrix( np.logical_xor(self.area.todense(), 
                                               self.boundary.todense()) )
        self.left_boundary = csc_matrix( np.array(self.boundary.todense()) * \
                                        (X < x0 - (width-dx)/2) )
        self.right_boundary = csc_matrix( np.array(self.boundary.todense()) * \
                                         (X > x0 + (width-dx)/2) )
        self.upper_boundary = csc_matrix( np.array(self.boundary.todense()) * \
                                       (Y > y0 + (height-dy)/2) )
        self.lower_boundary = csc_matrix( np.array(self.boundary.todense()) * \
                                          (Y < y0 - (height-dy)/2) )
         
class Superellipse(Shape):
    """A superellipse shape defined by the following parametric equation:
        | (x-x0)/(w/2) |**p + | (y-y0)/(h/2) |**p <= 1. The power p defines the
        "roundness" of the corners, so if p = 2 you get an ellipse, p = 4 you get
        a traditional superellipse, if p --> infinity you get a rectangle.
        w and h denote the "total" width and height of the shape.
    
    Args:
        X, Y: 2D arrays of coordinates produced from np.meshgrid(x,y) where x
              and y are 1D arrays of coordinates enclosing the full solution domain
        width, height: width and height of shape
        dx, dy: the step-sizes in given directions
        x0, y0: the center coordinates of the shape.
    """
    def __init__(self, 
                 X: np.ndarray, 
                 Y: np.ndarray, 
                 width: float, 
                 height: float,
                 power: float,
                 dx: float,
                 dy: float,
                 x0: float, 
                 y0: float,
         ):
        super().__init__()
        self.center = (x0, y0)
        self.width = width
        self.height = height
        self.area = csc_matrix( abs((X-x0)/(width/2))**power + \
                    abs((Y-y0)/(height/2))**power <= 1 )
        self.boundary = extract_boundary( self.area.todense() )
        self.hole = csc_matrix( np.logical_xor(self.area.todense(), 
                                               self.boundary.todense()) )

class Ellipse(Superellipse):
    """A regular ellipse shape. A subclass of Superellipse
    
    Args:
        X, Y: 2D arrays of coordinates produced from np.meshgrid(x,y) where x
              and y are 1D arrays of coordinates enclosing the full solution domain
        width, height: width and height of shape
        dx, dy: the step-sizes in given directions
        x0, y0: the center coordinates of the shape.
    """
    def __init__(self, 
                 X: np.ndarray, 
                 Y: np.ndarray, 
                 width: float, 
                 height: float,
                 dx: float,
                 dy: float,
                 x0: float, 
                 y0: float,
         ):
        super().__init__(X=X, Y=Y, width=width, height=height, power=2, dx=dx,
                         dy=dx, x0=x0, y0=x0)
        
class Circle(Ellipse):
    """A circle shape. A subclass of Ellipse
    
    Args:
        X, Y: 2D arrays of coordinates produced from np.meshgrid(x,y) where x
              and y are 1D arrays of coordinates enclosing the full solution domain
        width, height: width and height of shape
        dx, dy: the step-sizes in given directions
        x0, y0: the center coordinates of the shape.
    """
    def __init__(self, 
                 X: np.ndarray, 
                 Y: np.ndarray, 
                 diameter: float,
                 dx: float,
                 dy: float,
                 x0: float, 
                 y0: float,
         ):
        super().__init__(X=X, Y=Y, width=diameter, height=diameter, dx=dx, 
                         dy=dx, x0=x0, y0=x0)

###############################################################################
# Boundary points extraction functions
###############################################################################
def extract_boundary(A: np.ndarray) -> np.ndarray:
    """Given a binary matrix A containing some filled shape(s) (e.g. circle), this
    function determines which points (pixels) make up the boundary enclosing said
    shape(s), and outputs a binary matrix containing those boundary points only.
    
    Note: it is important that A is a completely filled shape, otherwise the
    detected boundary will include inner points (e.g. the boundary of subdomains)
    and this may not be desirable for the user.
    
    Args:
        A: the binary matrix describing the shape/geometry of the domain
    """
    A_trial = np.ones((3,3)); find_edges(A_trial) # compilation step

    # Extract points from outer frame 
    if type(A) is np.matrix: A = np.array(A)
    edge = np.zeros(np.shape(A))
    edge[0,:] = 1
    edge[-1,:] = 1
    edge[:,0] = 1
    edge[:,-1] = 1
    outer_frame = edge.astype(bool) * A
        
    # Define boundary
    boundary = find_edges(A).astype(bool) | outer_frame 
    
    return csc_matrix(boundary)

@nb.njit(parallel=True)
def find_edges(A: np.ndarray) -> np.ndarray:
    (Nx, Ny) = np.shape(A)
    B = np.zeros(np.shape(A))
    for ii in nb.prange(1,Nx-1):
        for jj in nb.prange(1,Ny-1):
            condition = A[ii-1,jj] and A[ii+1,jj] and A[ii,jj+1] and \
                        A[ii,jj-1] and A[ii+1,jj+1] and A[ii-1,jj+1] and\
                        A[ii+1,jj-1] and A[ii-1,jj-1]
                        
            if A[ii,jj] and not condition: B[ii,jj] = 1
    return B
    
###############################################################################
# Derivative matrices
###############################################################################
def Dx(Nx: int, Ny: int, dx: float):
    B = sp.sparse.diags( [np.ones((Nx-1,)), -np.ones((Nx-1,))],
                        [+1, -1], format="csc")
    D = 1/(2*dx) * sp.sparse.block_diag( [B]*Ny, format="csc" )
    return D

def Dy(Nx: int, Ny: int, dy: float):
    N = Nx*(Ny - 1)
    B = sp.sparse.diags([np.ones((N,)), -np.ones((N,))], [+Nx,-Nx], 
                        format="csc")
    D = 1/(2*dy) * B
    return D

def Dxx(Nx: int, Ny: int, dx: float):
    B = sp.sparse.diags([np.ones((Nx-1,)), -2*np.ones((Nx,)), np.ones((Nx-1,))],
                         [1,0,-1], format="csc")
    D = 1/dx**2 * sp.sparse.block_diag( [B]*Ny, format="csc" )
    return D

def Dyy(Nx: int, Ny: int, dy: float):
    N = Nx*(Ny-1)
    B = sp.sparse.diags([np.ones((N,)), -2*np.ones((Nx*Ny,)), np.ones((N,))],
                         [+Nx,0,-Nx], format="csc")
    D = 1/dy**2 * B
    return D

def Laplacian(Nx: int, Ny: int, dx: float, dy: float):
    return Dxx(Nx,Ny,dx) + Dyy(Nx,Ny,dy) 

def grad(A: np.ndarray, dx: float, dy: float) -> list:
    (Ny,Nx) = np.shape(A)
    A = csc_matrix( np.array(A.todense()).flatten()[:,None] )
    grad_x = np.array((Dx(Nx, Ny, dx) @ A).todense()).reshape((Ny,Nx))
    grad_y = np.array((Dy(Nx, Ny, dy) @ A).todense()).reshape((Ny,Nx))
    return [csc_matrix( grad_x ), csc_matrix( grad_y )]

###############################################################################
# Linear PDE solver
###############################################################################
def solve_bvp(X: np.ndarray,
              Y: np.ndarray,
              dx: float,
              dy: float,
              domain: np.ndarray,
              inner_domain: np.ndarray,
              boundary: list,
              boundary_values: list,
              LHS: np.ndarray,
              source: np.ndarray = None,
              plot_solution: bool=True) -> np.ndarray:
    """Solves a boundary-value problem of the form LHS*u = source
    where LHS is the left-hand side operator, source is the source function 
    (optional) and return u(x,y) as the solution over the prescribed domain,
    and given the boundary values.
    
    Args:
        X, Y: meshgrid matrices of coordinates for a rectangle of size Nx by Ny,
              where Nx is the number of points in x-direction and Ny is the number
              of points in the y-direction
        dx, dy: step-size in each direction
        domain: a (Ny,Nx) binary matrix in csc_matrix sparse format (scipy sparse)
        inner_domain: a (Ny,Nx) binary matrix similar to domain but excluding boundary 
                      points
        boundary: (Ny,Nx) binary matrix containing boundary point locations in sparse
                  csc_matrix format
        boundary_values: the boundary condition values in sparse csc_matrix format of
                         dimension (Ny,Nx)
        LHS: left-hand side operator matrix in sparse csc_matrix format of dimension
             (Ny,Nx)
        source: source function in sparse csc_matrix format of dimension (Ny,Nx)
        plot_solution: True if you want plots
    """
    # Start timer
    start_time = time.time()
    
    # Handle source
    if source is None: source = np.zeros(np.shape(X))
    
    # Apply boundary conditions
    N = len(boundary)
    u_boundary = 0.0
    boundary_points = 0
    for n in range(N):
        u_boundary += csc_matrix( boundary[n].multiply(csc_matrix(boundary_values[n])) )
        boundary_points += boundary[n]
    
    # Solve system
    if source is not sp.sparse._csc.csc_matrix: source = csc_matrix(source)
    solution = linear_solver(LHS = LHS, 
                             source = source, 
                             inner_domain = inner_domain,
                             boundary = boundary_points, 
                             boundary_values = u_boundary)
    
    # Close timer
    final_time = time.time() - start_time
    print("#----- computation time = %s seconds -----#" %(np.round(final_time,2)))
    
    # plot solutions (optional)
    if plot_solution:
        CM = 'RdBu'
        
        fig, ax = plt.subplots( figsize=(6,6), dpi=500 )
        c = ax.pcolor(X, Y, np.array(u_boundary.todense()), cmap=CM)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_title(r"$u|_{\partial\Omega}(x,y)$")
        ax.grid(visible=None)
        fig.colorbar(c, ax=ax)
        fig.tight_layout()
        plt.show()
        
        fig, ax = plt.subplots( figsize=(6,6), dpi=500 )
        c = ax.pcolor(X, Y, np.array(solution.todense()), cmap=CM)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_title(r"$u(x,y)$")
        ax.grid(visible=None)
        fig.colorbar(c, ax=ax)
        fig.tight_layout()
        plt.show()
    
    return solution

def linear_solver(LHS: np.ndarray, 
                  source: np.ndarray, 
                  inner_domain: np.ndarray,
                  boundary: np.ndarray,
                  boundary_values: np.ndarray):
    """Given a system of equations of the form L*u = b where L is the LHS operator,
    u is the solution vector and b is the source vector, we insert the boundary values
    for u by re-arranging the linear system and removing redundancies. 
    
    Suppose that we start with N equations and LHS has size [N x N], and then
    we specify m boundary values at boundary where m < N, it follows that
    there are now N-m unknowns u, and so we want to create a reduced system of
    N-M equations. 
    
    The outputs of this function are a reduced LHS matrix of size [(N-m) x (N-m)],
    and a RHS column vector of size [(N-m) x 1], where RHS contains both the
    source values and the boundary conditions applied at the points OTHER than
    at the boundary points.
    
    If we let the reduced system be defined as L'*u' = b', then 
    u' = inverse(L')*b' will give us the values of u at all the points OTHER
    than at the boundary points. This means that to recover the full solution
    for u we must then parse u' with all the boundary values we started with,
    which can be done with a separate function called reconstruct_system().
    """
    assert type(LHS) is sp.sparse._csc.csc_matrix
    assert type(boundary) is sp.sparse._csc.csc_matrix
    assert type(boundary_values) is sp.sparse._csc.csc_matrix
    assert type(source) is sp.sparse._csc.csc_matrix
    N = np.prod( boundary.shape )
    
    boundary_vector = np.array( boundary.todense() ).flatten()[None,:]
    BC_vector = np.array( boundary_values.todense() ).flatten()[:,None]
    inner_domain_vector = np.array( inner_domain.todense() ).flatten()[None,:]
    source_in = np.array( source.todense() ).flatten()[:,None]
    
    # Move boundary values to RHS of equation system  
    LHS_mask = csc_matrix( LHS.multiply(boundary_vector) )
    RHS = csc_matrix( source_in - LHS_mask @ BC_vector )
    
    # remove columns
    LHS_reduced_cols = \
        csc_matrix( LHS.transpose()[inner_domain_vector[0,:]].transpose() )
    
    # remove rows
    LHS_system = LHS_reduced_cols[inner_domain_vector[0,:]]
    RHS_system = RHS[inner_domain_vector[0,:]]
    print("# of equations to solve: %s out of %s grid points" %(LHS_system.shape[0],
                                                                N))

    # Solve u(x,y) inside domain
    solution_inner = sp.sparse.linalg.spsolve(LHS_system, RHS_system)
    print("Solution on inner points complete. Reconstructing full system...")

    # Full solution
    full_solution = np.zeros((N,))
    full_solution[inner_domain_vector[0,:]] = solution_inner
    full_solution[boundary_vector[0,:]] = BC_vector[:,0][boundary_vector[0,:]]
    
    full_solution = np.reshape( full_solution, np.shape(boundary) )
    print("Done.")
    
    return csc_matrix( full_solution )

###############################################################################
# Meshing functions for improving solution accuracy
###############################################################################
def optimized_mesh(xlims: tuple, ylims: tuple, Nmin: int, Nmax: int=None, 
                   tolerance: float=1e-12):
    """This function takes a minimum number of points Nmin, as well as the
    x-axis limits xlims=(x_min, x_max) and y-axis limits ylims=(y_min, y_max)
    to compute the best step-sizes dx and dy which ensure a uniform grid.
    
    It outputs meshgrid matrices X and Y, as well as the step-sizes dx and dy
    """
    if Nmax is None: Nmax = 20*Nmin
    
    # compile function
    mesh_search(xlims=(0,1), ylims=(0,1), Nmin=3, Nmax=6, tolerance=1e-6)
    
    # call function
    Nx, Ny = \
        mesh_search(xlims=xlims, ylims=ylims, Nmin=Nmin, Nmax=Nmax, 
                    tolerance=tolerance)
    
    x = np.linspace(xlims[0], xlims[1], Nx)
    y = np.linspace(ylims[0], ylims[1], Ny)
    dx = np.abs(x[1]-x[0])
    dy = np.abs(y[1]-y[0])
    [X,Y] = np.meshgrid(x,y)
    
    return [X, Y, dx, dy, Nx, Ny]
    
@nb.njit
def mesh_search(xlims: tuple, ylims: tuple, Nmin:int, Nmax: int, 
                tolerance: float=1e-12):    
    dx = np.abs(xlims[1] - xlims[0])/(Nmin-1)
    dy = np.abs(ylims[1] - ylims[0])/(Nmin-1)
    step_diff = np.abs(dx - dy)
    Nx_prev = Nmin
    Ny_prev = Nmin
    Nx = Nmin
    Ny = Nmin
    while step_diff > tolerance and Nx <= Nmax and Ny <= Nmax:
        Nx = Nx_prev
        Ny = Ny_prev
        if dy < dx:
            Nx = Nx + 1
        elif dy > dx:
            Ny = Ny + 1
        dx = np.abs(xlims[1] - xlims[0])/(Nx-1)
        dy = np.abs(ylims[1] - ylims[0])/(Ny-1)
        step_diff = np.abs(dx - dy)
        Nx_prev = Nx
        Ny_prev = Ny
    
    return Nx, Ny

###############################################################################
# Other useful functions
###############################################################################    
def plot_matrix(A):
    if type(A) is sp.sparse._csc.csc_matrix: 
        A_input = A.todense()
    else:
        A_input = A
    plt.matshow(A_input); plt.colorbar()
    return 0

def plot_stream(X:np.ndarray, Y: np.ndarray, A: np.ndarray, 
                stream: np.ndarray, density: float, color: str="white"):
    CM = 'RdBu'
    
    fig, ax = plt.subplots( figsize=(6,6), dpi=500 )
    c = ax.pcolor(X, Y, np.array(A.todense()), cmap=CM)
    ax.streamplot(X, Y, np.array(stream[0].todense()), np.array(stream[1].todense() ), 
                   density=density, linewidth=None, color=color)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(r"$\nabla u(x,y)$")
    ax.grid(visible=None)
    fig.colorbar(c, ax=ax)
    fig.tight_layout()
    plt.show()