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
# calc distance from edges separatly for solid and void

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import porespy as ps
import numpy as np
import pandas as pd
import scipy as sp

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import colors

import helper

# %% tags=[]
im_size = 32
dim = 2
porosity = 0.5
blobiness = 0.5
im_shape = np.ones(dim, dtype=np.int32) * im_size

np.random.seed(0)
img = ~ps.generators.blobs(im_shape, porosity=porosity, blobiness=blobiness)
img = img.astype(np.int32)

fig, axes = plt.subplots(1, 1, figsize=(5, 5))
img_show = img if dim == 2 else img[0, :, :] if dim == 3 else []
axes.imshow(img_show)
print(f'porosity: { helper.image_porosity(img) }')

# %% tags=[]
x_borders = None
x_borders_solid = None
x_borders_void = None
y_borders = None

for y in np.arange(img.shape[-2]):
    row = img[y, :]
    borders = row[1:] != row[:-1]
    borders = np.append(True, borders)
    indices = np.where(borders)[0]
    points = [(y, idx) for idx in indices]
    x_borders = points if x_borders is None else np.append(x_borders, points, axis=0)
    x_borders_solid = np.array([(y, x) for y, x in x_borders if img[y, x] == 1])
    x_borders_void = np.array([(y, x) for y, x in x_borders if img[y, x] == 0])

for x in np.arange(img.shape[-1]):
    row = img[:, x]
    borders = row[1:] != row[:-1]
    borders = np.append(True, borders)
    indices = np.where(borders)[0]
    points = [(idy, x) for idy in indices]
    y_borders = points if y_borders is None else np.append(y_borders, points, axis=0)
    y_borders_solid = np.array([(y, x) for y, x in y_borders if img[y, x] == 1])
    y_borders_void = np.array([(y, x) for y, x in y_borders if img[y, x] == 0])

fig, axis = plt.subplots(1, 4, figsize=(20, 5))
axis[0].imshow(img)
axis[0].scatter([x for y, x in x_borders_solid], [y for y, x in x_borders_solid], c='red', marker='.')
axis[1].imshow(img)
axis[1].scatter([x for y, x in x_borders_void], [y for y, x in x_borders_void], c='white', marker='.')
axis[2].imshow(img)
axis[2].scatter([x for y, x in y_borders_solid], [y for y, x in y_borders_solid], c='red', marker='.')
axis[3].imshow(img)
axis[3].scatter([x for y, x in y_borders_void], [y for y, x in y_borders_void], c='white', marker='.')

# %% tags=[]

# %%
