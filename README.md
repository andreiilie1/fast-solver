Fast solver for a class of differential equations

This is my 3rd year student project for the CS Course at University of Oxford. It's implementing, testing and comparing
various methods for solving a class of differential equations. The main results in the thesis shall be related to
the Multigrid method and the Preconditioned Conjugate Gradients using Multigrid as a preconditiner. 

The code so far implements Jacobi, Gauss Seidel, SSOR, Conjugate Gradients, Steepest Descent, Multigrid and Preconditioned
Conjugate Gradients with Multigrid as a preconditioner. It also contains the code for discretizing the differential equations and creating
the associated linear system. In order to generate solutions at different time coordinates, it can also be used with Backward Euler method,
whis is implemented as well.
