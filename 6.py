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
# use pixel's left and top segment lengths for generator in 2D case

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import porespy as ps
import numpy as np
import pandas as pd

from matplotlib import cm
import matplotlib.pyplot as plt

import helper

# %%
im_size = 8
dim = 2
porosity = 0.5
im_shape = np.ones(dim, dtype=np.int32) * im_size
np.random.seed(0)
img = ~ps.generators.blobs(im_shape, porosity=porosity, blobiness=1)

# %%
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
img_show = img if dim == 2 else img[0, :, :] if dim == 3 else []
axes.imshow(img_show)
print(f'porosity: { helper.image_porosity(img) }')

# %%
image_statistics = helper.image_statistics(img)
edge_distances = image_statistics['edge_distances']
eds_y = (np.int32(edge_distances[1]) + 1).T
eds_x = np.int32(edge_distances[0]) + 1
print(eds_y)
print(eds_x)

# %%
y_size = img.shape[-2]
x_size = img.shape[-1]
y_grid = np.arange(y_size)
x_grid = np.arange(x_size)
indices = pd.MultiIndex.from_tuples(list(np.ndindex(y_size, x_size)))

df = pd.DataFrame(columns=['isSolid', 'leftLength', 'leftIsSolid', 'topLength', 'topIsSolid'], index=indices)

for y in y_grid:
    for x in x_grid:
        
        is_solid = bool(img[y, x])
        left_length = np.NaN
        left_is_solid = np.NaN
        rigth_length = np.NaN
        rigth_is_solid = np.NaN
        
        if x > 0:
            prev_x = x - 1
            is_masked = eds_x.mask[y, prev_x]
            if not is_masked:
                left_length = eds_x.data[y, prev_x]
                left_is_solid = bool(img[y, prev_x])

        if y > 0:
            prev_y = y - 1
            is_masked = eds_y.mask[prev_y, x]
            if not is_masked:
                rigth_length = eds_y.data[prev_y, x]
                rigth_is_solid = bool(img[prev_y, x])
                
        df.loc[(y, x)] = pd.Series({
            'isSolid': is_solid, 
            'leftLength': left_length, 
            'leftIsSolid': left_is_solid, 
            'topLength': rigth_length, 
            'topIsSolid': rigth_is_solid,
        })

df[df.notna().all(axis='columns')].shape

# %%
