# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# test 2D/3D array manipulation

# %%
# %load_ext autoreload
# %autoreload 2

# %%
# import copy as cp

# import porespy as ps
import numpy as np
# import scipy as sp

# from matplotlib import cm
# import matplotlib.pyplot as plt
# import seaborn as sns

# import helper

# %%
x = np.ma.array(
    [
        [0, 1, 2, 3], 
        [0, 1, 0, 1], 
        [0, 0, 1, 2], 
        [0, 1, 2, 3],
    ],
    mask=[
        [0, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
    ]
)
y = np.ma.array(
    [
        [0, 1, 0, 1], 
        [0, 1, 2, 3], 
        [0, 1, 2, 3],
        [0, 0, 1, 2], 
    ],
    mask=[
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
    ]
)

x, np.rot90(y, k=-1)

# %%
xy = np.ma.array(
    [
        [[0, 0], [1, 0], [2, 0], [3, 0]], 
        [[0, 0], [1, 1], [0, 1], [1, 1]], 
        [[0, 1], [0, 2], [1, 2], [2, 0]], 
        [[0, 2], [1, 3], [2, 3], [3, 1]],
    ],
    mask=[
        [[0, 1], [0, 0], [0, 0], [0, 1]],
        [[1, 0], [1, 0], [0, 0], [0, 1]],
        [[1, 0], [0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0], [0, 0]],
    ]
)

for ij in np.ndindex(xy.shape[:2]):
    val = xy.mask[ij]
    xy.mask[ij] = [1, 1] if np.any(xy.mask[ij]) else val

# xy

# %%
y_rot = np.rot90(y, k=-1)

x_data = x.data.ravel()
x_mask = x.mask.ravel()

y_data = y_rot.data.ravel()
y_mask = y_rot.mask.ravel()

x_y = np.column_stack((x_data, y_data)).reshape(4, 4, 2)
x_y_mask = np.column_stack((x_mask, y_mask)).reshape(4, 4, 2)

for ij in np.ndindex(x_y_mask.shape[:2]):
    val = x_y_mask[ij]
    x_y_mask[ij] = [1, 1] if np.any(x_y_mask[ij]) else val

x_y = np.ma.array(x_y, mask=x_y_mask)

print(f'data: {(x_y.data == xy.data).all()}')
print(f'mask: {(x_y.mask == xy.mask).all()}')

# %%
x3d = np.array([
    [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
    ], [
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ], [
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ]
])

# %%
print(x3d[0, :, :])
print(np.rot90(x3d, axes=(2, 0))[0, :, :])

# %%
