"""
FEniCS HW1 program.

  -Laplace(u) = f  in the unit circle
            u = u0  on the boundary

f = -.4 / (.1 + x^2 + y^2)^2
u0 = log(.1 + x^2 + y^2)
"""

from __future__ import print_function
from time import time

from fenics import *
from mshr import *

def run(grid_size, element_order, do_plots=False):

    # Create mesh and define function space.
    domain = Circle(Point(0, 0), 1)
    mesh = generate_mesh(domain, grid_size)
    V = FunctionSpace(mesh, 'P', element_order)

    # Define boundary condition.
    u0 = Expression('std::log(.1 + x[0] * x[0] + x[1] * x[1])', degree=2)

    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u0, boundary)

    # Define f.
    f = Expression('-.4 / pow(.1 + x[0] * x[0] + x[1] * x[1], 2)', degree=2)

    # Define variational problem.
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v)) * dx
    L = f * v * dx

    start = time()

    # Compute solution.
    u = Function(V)
    solve(a == L, u, bc)

    elapsed_time = time() - start

    # Compute error in L2 norm.
    error = errornorm(u0, u, 'L2')

    u0 = interpolate(u0, V)
    u = interpolate(u, V)
    W = VectorFunctionSpace(mesh, 'P', element_order)

    grad_u0 = project(grad(u0), W)
    grad_u = project(grad(u), W)

    error_grad = errornorm(grad_u0, grad_u, 'L2')
    print(f'Solution error: {error}\nGradient error: {error_grad}')

    if do_plots:
        import numpy as np
        import matplotlib.pyplot as plt

        # Exact solution.
        plot(u0, title='Exact Solution')
        plt.savefig('exact_solution.png')
        plt.close()

        # Numerical solution.
        plot(u, title='Solution')
        plt.savefig('solution.png')
        plt.close()

        # Solution error.
        plot(u - u0, title='Solution Error')
        plt.savefig('solution_error.png')
        plt.close()

        # Grad error.
        plot(dot(grad_u - grad_u0, grad_u - grad_u0), title='Gradient Error')
        plt.savefig('grad_error.png')
        plt.close()

        # # Save solution to file in VTK format
        # vtkfile_w = File('poisson_membrane/deflection.pvd')
        # vtkfile_w << w
        # vtkfile_p = File('poisson_membrane/load.pvd')
        # vtkfile_p << p

        # Curve plot along x = 0 comparing f and u.
        # tol = 0.001  # avoid hitting points outside the domain
        # y = np.linspace(-1 + tol, 1 - tol, 101)
        # points = [(0, y_) for y_ in y]  # 2D points
        # u_line = np.array([u(point) for point in points])
        # f_line = np.array([f(point) for point in points])
        # plt.plot(y, u_line, 'k', linewidth=2)
        # plt.plot(y, f_line, 'b--', linewidth=2)
        # plt.grid(True)
        # plt.xlabel('$y$')
        # plt.legend(['u', 'f'], loc='upper left')
        # plt.savefig('plot.png')

    return error, error_grad, elapsed_time


if __name__ == '__main__':
    run(100, 1, do_plots=True)
