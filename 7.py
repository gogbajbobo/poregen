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
# use border distances for generator in 2D case

# %% tags=[]
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

# %%
im_size = 64
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

# %%
x_distances_solid, x_distances_void, y_distances_solid, y_distances_void = helper.border_distances_for_image(img)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(x_distances_solid)
axes[1].imshow(x_distances_void)
axes[2].imshow(y_distances_solid)
axes[3].imshow(y_distances_void)

# %%
