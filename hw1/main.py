import json
import numpy as np
from run import run

# Due to memory limits, we do not consider 100_000 and 1 000 000 vertices grid.
mesh_sizes = np.logspace(2, 4, num=3)  # total number of vertices
resolutions = np.sqrt(4 * mesh_sizes / np.pi)  # vertices at diameter

d = {
    'Grid resolution': resolutions.tolist(),
    'P1 Element error': [],
    'P1 Element grad error': [],
    'P1 Element CPU time': [],
    'P2 Element error': [],
    'P2 Element grad error': [],
    'P2 Element CPU time': [],
}
for element_order in 1, 2:
    for res in resolutions:
        res = round(res)

        print(f'Solving for resolution = {res}, element order = {element_order}')

        error, error_grad, t = run(res, element_order)

        d[f'P{element_order} Element error'].append(error)
        d[f'P{element_order} Element grad error'].append(error_grad)
        d[f'P{element_order} Element CPU time'].append(t)

        with open('results.json', 'w') as f:
            json.dump(d, f, indent=4)
