"""
Created on Mon Sep 26 12:42:17 2022

@author: user
"""

from __future__ import print_function
import time
import os

from fenics import *
from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

LOGFILE = 'log.txt'
if os.path.exists(LOGFILE):
    os.remove(LOGFILE) 


class Right(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] >= 0 + DOLFIN_EPS


class Left(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] <= 0 - DOLFIN_EPS

    
class Sigma(UserExpression):
    def __init__(self, subdomains, D):
        super().__init__()
        self.subdomains = subdomains
        self.D = D

    def eval_cell(self, values, x, cell):
        values[0] = 1. if self.subdomains[cell.index] == 0 else self.D
    
    def value_shape(self):
        return ()


def boundary(x, on_boundary):
    return on_boundary


circle = Circle(Point(0, 0), 1)
right = Right()
left = Left()

num_vertices = np.array([100, 1000, 10000])
size = np.sqrt(4 * num_vertices / np.pi).round()

f = Expression('exp(-100 * (pow(x[0] + .1, 2) + x[1] * x[1])) - exp(-100 * (pow(x[0] - .1, 2) + x[1] * x[1]))', degree=2)

params = LinearVariationalSolver.default_parameters()

for i, s in enumerate(size):

    print(f'Solving for size = {s}')

    mesh = generate_mesh(circle, s)
    domains = MeshFunction('size_t', mesh, 2)
    domains.set_all(0)

    left.mark(domains, 0)
    right.mark(domains, 1)

    W = FunctionSpace(mesh, 'P', 1)
    u = TrialFunction(W)
    v = TestFunction(W)
    dx = Measure('dx', domain=mesh, subdomain_data=domains)

    solvers = {
        'SOR': PETScKrylovSolver('cg', 'sor'),
        'ICC': PETScKrylovSolver('cg', 'icc'),
        'AMG': PETScKrylovSolver('cg', 'amg'),
        'default': ...,
    }

    for solver in solvers.keys():

        if solver != 'default':

            params = solvers[solver].parameters
            params['error_on_nonconvergence'] = False
            params['maximum_iterations'] = 1000

    for D in {1, 100}:
        print(f'D = {D}')
        
        sigma = Sigma(domains, D)
        a = (sigma * inner(grad(u), grad(v))) * dx
        L = f * v * dx
        A = assemble(a)
        b = assemble(L)

        for solver in solvers.keys():

            u1 = Function(W)
            u1vec = u1.vector()
            start = time.time()
            num_iter = solve(a == L, u1) if solver == 'default' else solvers[solver].solve(A, u1vec, b)

            p = plot(-grad(u1) * sigma)
            plt.colorbar(p)
            plt.savefig('sigmagrad_u.pdf')
            plt.close()

            p = plot(u1)
            plt.colorbar(p)
            plt.savefig('u.pdf')
            plt.close()

            elapsed_time = time.time() - start
            msg = f'solver = {solver}, time = {elapsed_time}, num_iter = {num_iter}\n'
            print(msg)
            with open(LOGFILE, 'a') as logfile:
                logfile.write(msg)

p = plot(-grad(u1) * sigma)
plt.colorbar(p)
plt.savefig('sigmagrad_u.png')
plt.close()

p = plot(u1)
plt.colorbar(p)
plt.savefig('u.png')
plt.close()
