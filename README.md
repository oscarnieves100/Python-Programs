# Python-Programs
Python scripts for numerical and statistical applications.

Tested using Python 3.8 in Spyder 4.1.5

# PDE solver module 
- bvp_solver.py <-- this is the main standalone module
- example_Laplace_circular_domain.py <-- example of using bvp_solver.py
- example_Laplace_doughnut_domain.py <-- another example of using bvp_solver.py

"bvp_solver.py" is a standalone module that solves 2D linear boundary value problems on arbitrarily shaped domains with Dirichlet boundary conditions. It uses a finite difference scheme with centred differences in Cartesian coordinates. It contains the main solver function "solve_bvp()" as well as classes for geometric shapes and their boundary points as sparse binary matrices. The entire module is based on sparse matrices in CSC format. You can create your own domains by using the Shape class methods such as union() or insert_hole(), all of which update the boundary points. Alternatively, if you have a binary matrix containing the domain shape (e.g. from a greyscale image), you can transform it into a Shape object and then use it with the module.

For examples on how to use this module, refer to "example_Laplace_circular_domain.py" or "example_Laplace_doughnut_domain.py"
