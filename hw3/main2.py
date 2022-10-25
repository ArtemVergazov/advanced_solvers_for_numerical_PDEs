from time import time
import numpy as np


def DMD_approx_a(data,r):
    X1, X2 = data[:, :-1], data[:, 1:]
    u, s, v = np.linalg.svd(X1, full_matrices=False)
    ur, sr, vr = u[:, :r].conj().T, np.reciprocal(s[:r]), v[:r, :].conj().T
    S = (ur @ X2 @ vr) * sr
    u1, s1 = np.linalg.eig(S)
    adj_modes = X2 @ vr @ np.diag(sr) @ s1
    return adj_modes @ np.diag(u1) @ np.linalg.pinv(adj_modes)


with open('data.npy', 'rb') as f:
    a = np.load(f)

b = np.copy(a)
b[:, 10:] = 0
rs = [10, 5, 2, 1]
for r in rs: 
    print(r)
    t0 = time()
    for i in range(10, 21):
        print(i)
        t1 = time()
        A = DMD_approx_a(b[:, i - 10 : i], r)
        b[:, i] = (A @ b[:, i - 1]).real
        print(f'time: {time() - t1}')
    print(f'Full cycle time: {time() - t0}')
