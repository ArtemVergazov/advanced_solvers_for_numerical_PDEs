from __future__ import print_function
from time import time

from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

channel = Rectangle(Point(0, 0), Point(2.2, .41))
cylinder1 = Circle(Point(.2, .2), .05)
cylinder2 = Circle(Point(1.2, .2), .05)
domain = channel - cylinder1 - cylinder2
mesh_sizes = [4, 8, 16, 32, 64, 128, 256]

times = []
iter_counts = []

for mesh_size in mesh_sizes:
    mesh = generate_mesh(domain, mesh_size)

    inflow = 'near(x[0], 0)'
    outflow = 'near(x[0], 2.2)'
    walls = 'near(x[1], 0) || near(x[1], .41)'
    cylinder1 = 'on_boundary && x[0] > .1 && x[0] < .3 && x[1] > .1 && x[1] < .3'
    cylinder2 = 'on_boundary && x[0] > 1.1 && x[0] < 1.3 && x[1] > .1 && x[1] < .3'
    inflow_profile = '-4. * 1.5 * x[1] * (.41 - x[1]) / .41 / .41', '0'

    V1 = VectorElement('Lagrange', mesh.ufl_cell(), 2)
    Q1 = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, V1 * Q1)

    bcu_inflow = DirichletBC(W.sub(0), Expression(inflow_profile, degree=2), inflow)
    bcu_walls = DirichletBC(W.sub(0), Constant((0, 0)), walls)
    bcp_outflow = DirichletBC(W.sub(1), Constant(0), outflow)
    bcu_cylinder1 = DirichletBC(W.sub(0), Constant((0, 0)), cylinder1)
    bcu_cylinder2 = DirichletBC(W.sub(0), Constant((0, 0)), cylinder2)
    bcs = [bcu_inflow, bcu_walls, bcu_cylinder1, bcu_cylinder2, bcp_outflow]

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    f = Constant((0., 0.))

    a = inner(grad(u), grad(v)) * dx + div(v) * p * dx + q * div(u) * dx
    L = inner(f, v) * dx    
    b = inner(grad(u), grad(v)) * dx + p * q * dx

    A, bb = assemble_system(a, L, bcs)
    P, btmp = assemble_system(b, L, bcs)

    solver = KrylovSolver('minres', 'amg')
    solver.set_operators(A, P)

    U = Function(W)

    start = time()
    num_iter = solver.solve(U.vector(), bb)
    times.append(time() - start)
    iter_counts.append(num_iter)

    print(f'grid size: {mesh_size}, time: {times[-1]}, iterations: {iter_counts[-1]}\n')

    u, p = U.split()

    p1 = plot(u)
    plt.colorbar(p1)
    plt.savefig('plot_u.png')
    plt.close()

    p2 = plot(p)
    plt.colorbar(p2)
    plt.savefig('plot_p.png')
    plt.close()
