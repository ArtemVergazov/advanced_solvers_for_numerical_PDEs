#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 05:43:35 2022

@author: user
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 03:00:12 2022

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import time

def DMD_approx_a(data,r):
    X1,X2 = data[:, :-1], data[:, 1:]
    u, s, v = np.linalg.svd(X1, full_matrices = False)
    S = np.matmul(u[:, : r].conj().T, np.matmul(X2, v[: r, :].conj().T))* np.reciprocal(s[: r])
    u1, s1 = np.linalg.eig(S)
    adjusted_modes = np.matmul(
        X2, np.matmul(
            v[: r, :].conj().T, np.matmul(
                np.diag(np.reciprocal(s[: r])), s1
                )
            ))
    return np.matmul(
        adjusted_modes, np.matmul(
            np.diag(u1), np.linalg.pinv(adjusted_modes)
            )
        )


with open('/home/s_polyanskiy/solver3/hw3solvers.npy', 'rb') as f:
    a = np.load(f)
    print(a.shape)


b = np.copy(a)
b[:, 10:] = 0
rs = [10, 5, 2, 1]
for r in rs: 
    print(r)
    t0 = time.time()
    for i in range(10, 21):
        print(i)
        t1 = time.time()
        A = DMD_approx_a(b[:,i-10:i],r)
        b[:, i] = np.matmul(A, b[:, i-1]).real
        print('Time:')
        print(time.time()-t1)
    print('Full cycle time:')
    print(time.time()-t0)
