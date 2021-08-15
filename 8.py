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
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
import porespy as ps
import numpy as np
import pandas as pd
import scipy as sp

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

import helper

# %% tags=[]
im_size = 64
dim = 2
porosity = 0.5
blobiness = 1
im_shape = np.ones(dim, dtype=np.int32) * im_size

np.random.seed(0)
img = ~ps.generators.blobs(im_shape, porosity=porosity, blobiness=blobiness)
img = img.astype(np.int32)

fig, axes = plt.subplots(1, 1, figsize=(5, 5))
img_show = img if dim == 2 else img[0, :, :] if dim == 3 else []
axes.imshow(img_show)
print(f'porosity: { helper.image_porosity(img) }')

# %% tags=[]
df, dff = helper.dataframes_from_image_with_nan_at_edges(img)
dff.describe()

# %% tags=[]
lengths_map = np.empty(img.shape, dtype=object)
for y, x in np.ndindex(img.shape[-2], img.shape[-1]):
    str_value = np.array2string(df.loc[(y, x)][['leftLength', 'topLength']].values)
    str_value = np.char.replace(str_value, 'nan', 'NA')
    lengths_map[y, x] = str_value
lengths_map.shape

# %% tags=[]
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(img[:20, :20], square=True, ax=ax, annot=lengths_map[:20, :20], fmt='')

# %% tags=[]
df.loc[(19, 7)]

# %% tags=[]
ssdf = dff[(dff.leftIsSolid == True) & (dff.topIsSolid == True)][['isSolid', 'leftLength', 'topLength']]
vvdf = dff[(dff.leftIsSolid == False) & (dff.topIsSolid == False)][['isSolid', 'leftLength', 'topLength']]
svdf = dff[(dff.leftIsSolid == True) & (dff.topIsSolid == False)][['isSolid', 'leftLength', 'topLength']]
vsdf = dff[(dff.leftIsSolid == False) & (dff.topIsSolid == True)][['isSolid', 'leftLength', 'topLength']]

# %% tags=[]
ssdf_hist, _, _ = np.histogram2d(ssdf.leftLength, ssdf.topLength, bins=[ssdf.leftLength.max() - 1, ssdf.topLength.max() - 1])
vvdf_hist, _, _ = np.histogram2d(vvdf.leftLength, vvdf.topLength, bins=[vvdf.leftLength.max() - 1, vvdf.topLength.max() - 1])
svdf_hist, _, _ = np.histogram2d(svdf.leftLength, svdf.topLength, bins=[svdf.leftLength.max() - 1, svdf.topLength.max() - 1])
vsdf_hist, _, _ = np.histogram2d(vsdf.leftLength, vsdf.topLength, bins=[vsdf.leftLength.max() - 1, vsdf.topLength.max() - 1])
fig, axes = plt.subplots(2, 2, figsize=(20, 20))
sns.heatmap(ssdf_hist, annot=True, linewidths=.5, ax=axes[0, 0])
sns.heatmap(vvdf_hist, annot=True, linewidths=.5, ax=axes[0, 1])
sns.heatmap(svdf_hist, annot=True, linewidths=.5, ax=axes[1, 0])
sns.heatmap(vsdf_hist, annot=True, linewidths=.5, ax=axes[1, 1])

# %%
