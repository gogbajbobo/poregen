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

sss = ssdf[ssdf.isSolid == True]
ssv = ssdf[ssdf.isSolid == False]
vvs = vvdf[vvdf.isSolid == True]
vvv = vvdf[vvdf.isSolid == False]
svs = svdf[svdf.isSolid == True]
svv = svdf[svdf.isSolid == False]
vss = vsdf[vsdf.isSolid == True]
vsv = vsdf[vsdf.isSolid == False]


# %% tags=[]
def df_hist(df):
    df_hist, _, _ = np.histogram2d(df.leftLength, df.topLength, bins=[df.leftLength.max() - 1, df.topLength.max() - 1])
    return df_hist

sss_hist = df_hist(sss)
ssv_hist = df_hist(ssv)
vvs_hist = df_hist(vvs)
vvv_hist = df_hist(vvv)
svs_hist = df_hist(svs)
svv_hist = df_hist(svv)
vss_hist = df_hist(vss)
vsv_hist = df_hist(vsv)

fig, axes = plt.subplots(4, 2, figsize=(20, 40))
sns.heatmap(sss_hist, annot=True, linewidths=.5, ax=axes[0, 0])
sns.heatmap(ssv_hist, annot=True, linewidths=.5, ax=axes[0, 1])
sns.heatmap(vvs_hist, annot=True, linewidths=.5, ax=axes[1, 0])
sns.heatmap(vvv_hist, annot=True, linewidths=.5, ax=axes[1, 1])
sns.heatmap(svs_hist, annot=True, linewidths=.5, ax=axes[2, 0])
sns.heatmap(svv_hist, annot=True, linewidths=.5, ax=axes[2, 1])
sns.heatmap(vss_hist, annot=True, linewidths=.5, ax=axes[3, 0])
sns.heatmap(vsv_hist, annot=True, linewidths=.5, ax=axes[3, 1])

# %% tags=[]
sss_hist_kde = sp.stats.gaussian_kde(sss_hist[0, :])
# plt.imshow(sss_hist_kde)
# sss_hist.shape
sss_hist_kde

# %% tags=[]
sss_hist[0, :]

# %% tags=[]
ds = np.array([[y, x] for y, x in np.ndindex(sss_hist.shape[-2], sss_hist.shape[-1]) if sss_hist[y, x] > 0]).T
weights = np.array([sss_hist[y, x] for y, x in np.ndindex(sss_hist.shape[-2], sss_hist.shape[-1]) if sss_hist[y, x] > 0])
sss_hist_kde = sp.stats.gaussian_kde(ds, weights=weights, bw_method=.01)
sss_hist_kde_img = np.array([sss_hist_kde.pdf([y, x])[0] for y, x in np.ndindex(sss_hist.shape[-2], sss_hist.shape[-1])]).reshape(sss_hist.shape)

fig, axes = plt.subplots(1, 3, figsize=(30, 10))
axes[0].imshow(sss_hist)
axes[1].imshow(sss_hist_kde_img)
axes[2].imshow(sss_hist_kde_img - sss_hist)
# axes[0].imshow(sss_hist, norm=colors.LogNorm())
# axes[1].imshow(sss_hist_kde_img, norm=colors.LogNorm())

# %%
