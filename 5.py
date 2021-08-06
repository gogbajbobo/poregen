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
# use switch and continue matricies for generator in 2D case

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import porespy as ps
import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt

import helper

# %%
im_size = 8
dim = 2
porosity = 0.5
im_shape = np.ones(dim, dtype=np.int32) * im_size
np.random.seed(0)
img = ~ps.generators.blobs(im_shape, porosity=porosity, blobiness=.5)

# %%
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
img_show = img if dim == 2 else img[0, :, :] if dim == 3 else []
axes.imshow(img_show)
print(f'porosity: { helper.image_porosity(img) }')

# %%
image_statistics = helper.image_statistics(img)
edge_distances = image_statistics['edge_distances']
segments_lengths = image_statistics['segments_lengths']
# print(np.int32(img_show))
# print(np.int32(edge_distances[0].data + 1))
# print(np.int32(edge_distances[1].data + 1))
# print(segments_lengths[0])
# print(segments_lengths[1])

# %%
x = edge_distances[0]
y = edge_distances[1]

x_data = x.data.ravel()
y_data = y.T.data.ravel()

eds_shape = (*img.shape, dim)
eds = np.column_stack((y_data, x_data)).reshape(eds_shape)

print(np.int32(img))
# print(eds[:, :, 0])
# print(eds[:, :, 1])

# %%
els_x = np.pad(edge_distances[0], 1)
els_x[0, :] = -1
els_x[:, 0] = -1
mask_x = els_x != 0
els_x[-1, :] = -1
# els_x = np.ma.array(np.roll(els_x, 1) + 1, mask=mask_x)
els_x = np.roll(els_x, 1) + 1
els_x[mask_x] = 0
els_x = els_x[1:, 1:]
els_x

# %%
els_y = np.pad(edge_distances[1], 1)
els_y[0, :] = -1
els_y[:, 0] = -1
mask_y = els_y != 0
els_y[-1, :] = -1
# els_y = np.ma.array(np.roll(els_y, 1) + 1, mask=mask_y)
els_y = np.roll(els_y, 1) + 1
els_y[mask_y] = 0
els_y = els_y.T[1:, 1:]
els_y

# %%
els_shape = (*tuple(i + 1 for i in img.shape), dim)
els = np.column_stack((els_y.ravel(), els_x.ravel())).reshape(els_shape)
# els

# %%
fig, axes = plt.subplots(1, 1, figsize=(15, 15))
im = np.pad(np.float32(img_show), 1, constant_values=.5)[1:, 1:]
axes.imshow(im)

for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        color = 'black' if im[i, j] else 'w'
        if i < eds.shape[0] and j < eds.shape[1]:
            axes.text(j, i, eds[i, j], ha='right', va='bottom', c=color, size='x-large')
        axes.text(j, i, els[i, j], ha='left', va='top', c=color, size='xx-large')

# %%
