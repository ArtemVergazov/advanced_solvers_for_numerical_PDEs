import numpy as np


def num_grad(f, steps):
    res = ((f[1:] - f[:-1]) / steps).tolist()
    res.append((f[-1] - f[-2]) / steps[-1])
    
    return np.array(res)
