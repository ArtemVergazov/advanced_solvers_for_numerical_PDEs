from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import numpy as np

mesh_size = 10


# Create mesh and define function space.
mesh = IntervalMesh(mesh_size, 0, 1)
V = FunctionSpace(mesh, 'P', 1)

afun = lambda x: np.ones_like(x)
bfun = lambda x: np.zeros_like(x)
ffun = lambda x: np.ones_like(x)
# a = Expression('sin(x[0]) + 3', degree=2)
# b = Expression('cos(x[0]) + 3', degree=2)
a = Expression('1.', degree=2)
b = Expression('0.', degree=2)

a = interpolate(a, V)
b = interpolate(b, V)

# Define boundary condition.
u_D = Constant(0.)
boundary = lambda x, on_boundary: on_boundary
bc = DirichletBC(V, u_D, boundary)

# Define variational problem.
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(1.)
lhs = a * dot(grad(u), grad(v)) * dx + b * u * v * dx
rhs = f * v * dx

# Compute solution.
u = Function(V)
solve(lhs == rhs, u, bc)

# Plot solution and error.
plot(u)

x = np.linspace(0, 1)
u_ana = x / 2 - x**2 / 2
plt.plot(x, u_ana, '--')

steps = np.ones(mesh_size) / (mesh_size - 1)
grid = steps.cumsum().tolist()
grid.insert(0, 0.)
grid = np.array(grid)
u_ana2 = grid / 2 - grid**2 / 2
uvec = u.vector().get_local()
plt.savefig(f'solution_for_{mesh_size}_steps.png')
plt.close()

from tools import num_grad
residual = -num_grad(afun(grid) * num_grad(uvec, steps), steps) + bfun(grid) * uvec - ffun(grid)

mu_sqr = residual[:-1]**2 * steps
error_indicators_sqr = (steps**2 * mu_sqr / np.pi**2 / afun(grid[:-1] + steps / 2))[:-1]

plt.plot(grid[:-2], error_indicators_sqr, 'o')
plt.title(f'A-posteriori error = {np.sqrt(np.sum(error_indicators_sqr))}')
plt.savefig(f'error_for_{mesh_size}_steps.png')

# # Compute error in L2 norm.
# error_L2 = errornorm(u_D, u, 'L2')

# # Compute maximum error at vertices
# vertex_values_u_D = u_D.compute_vertex_values(mesh)
# vertex_values_u = u.compute_vertex_values(mesh)
# import numpy as np
# error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# # Print errors
# print('error_L2  =', error_L2)
# print('error_max =', error_max)
