from __future__ import print_function
from time import time

from fenics import *
from dolfin import *
from mshr import *
import numpy as np


def boundary(x, on_boundary):
    return on_boundary


mesh = UnitSquareMesh(100, 100)
nu = 1e-2
tau = .05

V = VectorFunctionSpace(mesh, 'P', 1)

u_init = Expression(
    ('exp(-3 * x[0] * x[0] - 3 * x[1] * x[1])', 'exp(-3 * x[0] * x[0] - 3 * x[1] * x[1])'),
    degree=2,
)

u_prev = interpolate(u_init, V)

u = Function(V)
v = TestFunction(V)

F = inner(u, v) * dx + tau * (nu * inner(grad(u), grad(v)) * dx + \
    inner(grad(u) * u, v) * dx) - \
    inner(u_prev, v) * dx

u1 = Function(V)
t = 0
data = np.zeros((u_prev.vector().get_local().shape[0], 21))
data[:, 0] = np.copy(u_prev.vector().get_local().ravel())
i = 0
t1 = time()
while t < 1:
    print(t + tau)
    t += tau
    solve(F == 0, u)
    u_prev.assign(u)
    print(i)
    i += 1
    data[:, i] = u.vector().get_local()
print(f'time: {time() - t1}')
with open('data.npz', 'wb') as f:
    np.save(f, data)
with open('data.npy', 'wb') as f:
    np.save(f, data)
