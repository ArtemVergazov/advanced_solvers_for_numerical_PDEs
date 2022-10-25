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

with open('//home/user/solver3dat/hw3solvers.npy', 'rb') as f:
    a = np.load(f)
mesh = UnitSquareMesh(100, 100)
V = VectorFunctionSpace(mesh, 'P', 1)
u = Function(V)
v = Function(V)
w = Function(V)

for r in [1, 2, 5, 10]:
    print(f'Considering rank {r}')
    with open(f'//home/user/solver3dat/hw3solvers_pydmd_{r}.npy', 'rb') as f:
        b = np.load(f)
 
    with open(f'//home/user/solver3dat/hw3solvers_pydmd_r_{r}.npy', 'rb') as f:
        c = np.load(f)
    
    for i in [12, 16, 20]:
        print(f'Timestep number {i}')
        u.vector().set_local(a[:,i])
        v.vector().set_local(b[:,i])
        w.vector().set_local(c[:,i])
        p=plot(u, title = f'Plot of FEM-BE solution at timestep number {i}')
        plt.colorbar(p)
        plt.savefig(f'//home/user/solver3dat/plot_fembe_step_{i}.png')
        plt.close()
        p=plot(v, title = f'Plot of DMD solution without explicit coercion of reals at timestep number {i} rank {r}')
        plt.colorbar(p)
        plt.savefig(f'//home/user/solver3dat/plot_dmd_step_{i}_rank_{r}.png')
        plt.close()
        p=plot(w, title = f'Plot of DMD solution with explicit coercion of reals at timestep number {i} rank {r}')
        plt.colorbar(p)
        plt.savefig(f'//home/user/solver3dat/plot_dmd_r_step_{i}_rank_{r}.png')
        plt.close()
        print(f'L2 error of DMD: {np.linalg.norm(a[:,i] - b[:,i], ord = 2)}')
        print(f'L2 error of DMD-coerced: {np.linalg.norm(a[:,i] - c[:,i], ord = 2)}')
        print(f'Relative L2 error of DMD: {np.linalg.norm(a[:,i] - b[:,i], ord = 2)/np.linalg.norm(a[:,i],ord = 2)}')
        print(f'Relative L2 error of DMD-coerced: {np.linalg.norm(a[:,i] - c[:,i], ord = 2)/np.linalg.norm(a[:,i],ord = 2)}')




