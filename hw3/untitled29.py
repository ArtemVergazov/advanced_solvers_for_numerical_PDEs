#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:47:45 2022

@author: user
"""

from __future__ import print_function
from fenics import *
from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import time
from sympy import exp, sin, pi
import sympy as sym
from memory_profiler import profile
import petsc4py
from scipy.sparse import csr_matrix
#import pydmd

def boundary(x, on_boundary):
    return on_boundary

mesh = UnitSquareMesh(100, 100)
nu = 1e-2
tau = 0.05

V = VectorFunctionSpace(mesh, 'P', 1)
u_init = Expression (('exp(-3*x[0]*x[0]-3*x[1]*x[1])', 'exp(-3*x[0]*x[0]-3*x[1]*x[1])'), degree = 2 )
u_prev= interpolate(u_init, V) #ensure exact interpolation

u = Function(V)
v = TestFunction(V)

F = inner(u,v)*dx+tau*(nu*inner(grad(u), grad(v))*dx+inner(grad(u)*u,v)*dx) - inner(u_prev,v)*dx
u1 = Function(V)
t = 0
data = np.zeros((u_prev.vector().get_local().shape[0], 21))
print(data.shape)
data[:,0] = np.copy(u_prev.vector().get_local().ravel())
i = 0
t1 = time.time()
while t<1:
    print(t+tau)
    t+=tau
    solve(F == 0,u)
    u_prev.assign(u)
    print(i)
    i+=1
    data[:,i] = u.vector().get_local()
print('time')
print(time.time()-t1)
print(data.shape)
with open('/media/user/Elements/hw3solvers'+'.npz', 'wb') as f:
    np.save(f, data)
with open('/media/user/Elements/hw3solvers'+'.npy', 'wb') as f:
    np.save(f, data)




