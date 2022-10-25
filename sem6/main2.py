# from dolfin.cpp.generation import BoxMesh
# from dolfin.cpp.generation import Point
from dolfin import *

N = 60
lx, ly, lz = 0.2, 0.1, 1.0
mesh = BoxMesh(Point(0, 0, 0), Point(lx, ly, lz), int(lx*N), int(ly*N), int(lz*N))

# from dolfin.function.functionspace import FunctionSpace
# from dolfin.function.argument import TrialFunction
# from dolfin.function.argument import TestFunction

from ufl import inner
from ufl import grad
from ufl import dx

V = FunctionSpace(mesh, 'Lagrange', 1)

u_h = TrialFunction(V)
v_h = TestFunction(V)

a = inner(grad(u_h), grad(v_h)) * dx
b = inner(u_h, v_h) * dx

# from dolfin.cpp.la import PETScMatrix
# from dolfin.fem.assembling import assemble

A = PETScMatrix(V.mesh().mpi_comm())
assemble(a, tensor=A)

B = PETScMatrix(V.mesh().mpi_comm())
assemble(b, tensor=B)

bc_dofs = []
for bc in bcs:
    bc_dofs.extend(list(bc.get_boundary_values().keys()))

A.mat().zeroRowsColumnsLocal(bc_dofs, diag=diag_value)
B.mat().zeroRowsColumnsLocal(bc_dofs, diag=1.)

import scipy

A_array = scipy.sparse.csr_matrix(A.mat().getValuesCSR()[::-1])
B_array = scipy.sparse.csr_matrix(B.mat().getValuesCSR()[::-1])

k = 20  # number of eigenvalues
which = 'SM'  # smallest magnityde
w, v = scipy.linalg.eigh(A_array, B_array, k=k, which=which)

import numpy as np
from dolfin.cpp.la import SLEPcEigenSolver

# instantiate solver
solver = SLEPcEigenSolver(A, B)

# update parameters
solver.parameters['solver'] = 'krylov-schur'
solver.parameters['spectrum'] = 'smallest magnitude'
solver.parameters['problem_type'] = 'gen_hermitian'
solver.parameters['tolerance'] = 1e-4

# solve
n_eig = 20
solver.solve(n_eig)

# collect eigenpairs
w, v = [], []
for i in range(solver.get_number_converged()):
    r, _, rv, _ = solver.get_eigenpair(i)
    w.append(r)
    v.append(rv)

w = np.array(w)
v = np.array(v).T


