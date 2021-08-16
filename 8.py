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
def calc_hist_kde(hist, factor=.1):
    ds = np.array([[y, x] for y, x in np.ndindex(hist.shape[-2], hist.shape[-1]) if hist[y, x] > 0]).T
    weights = np.array([hist[y, x] for y, x in np.ndindex(hist.shape[-2], hist.shape[-1]) if hist[y, x] > 0])
    hist_kde = sp.stats.gaussian_kde(ds, weights=weights, bw_method=factor)
    hist_kde_img = np.array([hist_kde.pdf([y, x])[0] for y, x in np.ndindex(hist.shape[-2], hist.shape[-1])]).reshape(hist.shape)
    hist_kde_img /= np.sum(hist_kde_img)
    hist_kde_img *= np.sum(hist)

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    axes[0].imshow(hist)
    axes[1].imshow(hist_kde_img)
    axes[2].imshow(hist_kde_img - hist)
#     axes[0].imshow(hist, norm=colors.LogNorm())
#     axes[1].imshow(hist_kde_img, norm=colors.LogNorm())
#     axes[2].imshow(hist_kde_img - hist, norm=colors.LogNorm())
    
    return hist_kde, hist_kde_img


# %% tags=[]
sss_hist_kde, sss_hist_kde_img = calc_hist_kde(sss_hist)
ssv_hist_kde, ssv_hist_kde_img = calc_hist_kde(ssv_hist, .3)
vvs_hist_kde, vvs_hist_kde_img = calc_hist_kde(vvs_hist, .3)
vvv_hist_kde, vvv_hist_kde_img = calc_hist_kde(vvv_hist)
svs_hist_kde, svs_hist_kde_img = calc_hist_kde(svs_hist)
svv_hist_kde, svv_hist_kde_img = calc_hist_kde(svv_hist)
vss_hist_kde, vss_hist_kde_img = calc_hist_kde(vss_hist)
vsv_hist_kde, vsv_hist_kde_img = calc_hist_kde(vsv_hist)


# %% tags=[]
def get_solid_probability(count_solid, count_void):
    y_range = np.min([count_solid.shape[-2], count_void.shape[-2]])
    x_range = np.min([count_solid.shape[-1], count_void.shape[-1]])
    return count_solid[:y_range, :x_range] / (count_solid[:y_range, :x_range] + count_void[:y_range, :x_range])

psss = get_solid_probability(sss_hist_kde_img, ssv_hist_kde_img)
pvvs = get_solid_probability(vvs_hist_kde_img, vvv_hist_kde_img)
psvs = get_solid_probability(svs_hist_kde_img, svv_hist_kde_img)
pvss = get_solid_probability(vss_hist_kde_img, vsv_hist_kde_img)

fig, axes = plt.subplots(2, 2, figsize=(20, 20))
axes[0, 0].imshow(psss)
axes[0, 1].imshow(pvvs)
axes[1, 0].imshow(psvs)
axes[1, 1].imshow(pvss)

# %% tags=[]
f = open('/Users/grimax/Desktop/log.txt', 'w')
f.close()

new_img = np.empty(img.shape, dtype=np.int32)
new_lengths = np.empty((*img.shape, 2), dtype=np.int32)  # use [y, x] order

new_img[0, :] = img[0, :]
new_img[:, 0] = img[:, 0]
new_lengths[0, :] = np.array([
    [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 1], [0, 1], [0, 2], 
    [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 1], [0, 2], [0, 3],
    [0, 4], [0, 5], [0, 6], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
    [0, 6], [0, 1], [0, 2], [0, 1], [0, 2], [0, 3], [0, 1], [0, 2],
    [0, 3], [0, 4], [0, 5], [0, 6], [0, 1], [0, 2], [0, 3], [0, 4],
    [0, 5], [0, 6], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6],
    [0, 7], [0, 8], [0, 9], [0, 1], [0, 2], [0, 3], [0, 4], [0, 1],
    [0, 2], [0, 1], [0, 2], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
])
new_lengths[:, 0] = np.array([
    [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0],
    [8, 0], [9, 0], [10, 0], [11, 0], [12, 0], [13, 0], [14, 0], [15, 0],
    [16, 0], [17, 0], [1, 0], [2, 0], [3, 0], [1, 0], [2, 0], [3, 0],
    [4, 0], [5, 0], [6, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0],
    [6, 0], [7, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0],
    [7, 0], [8, 0], [9, 0], [10, 0], [1, 0], [1, 0], [2, 0], [3, 0],
    [4, 0], [5, 0], [6, 0], [7, 0], [1, 0], [2, 0], [3, 0], [4, 0],
    [5, 0], [1, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0],
])

def calc_probability(p_map, tl, ll):
    if tl < p_map.shape[-2] and ll < p_map.shape[-1]:
        p = p_map[tl, ll]
        return sp.stats.bernoulli.rvs(p), p
#         return p > 0.5, p
    return 0, 0

for y in np.arange(img.shape[-2])[1:]:
    for x in np.arange(img.shape[-1])[1:]:
        
        left_neibor = new_img[y, x - 1]
        left_length = new_lengths[y, x -1][-1]
        ll = left_length + 1
        top_neibor = new_img[y - 1, x]
        top_length = new_lengths[y - 1, x][-2]
        tl = top_length + 1
        result = None
        p = None
        
        if left_neibor == top_neibor:
            if left_neibor == True:
                result, p = calc_probability(psss, tl, ll)
            else:
                result, p = calc_probability(pvvs, tl, ll)
        else:
            if left_neibor == True:
                result, p = calc_probability(psvs, tl, ll)
            else:
                result, p = calc_probability(pvss, tl, ll)

        new_img[y, x] = result
        new_top_length = top_length + 1 if result == top_neibor else 1
        new_left_length = left_length + 1 if result == left_neibor else 1
        new_lengths[y, x] = np.array([new_top_length, new_left_length])
        
        f = open('/Users/grimax/Desktop/log.txt', 'a')
        f.write(f'x: {x}, y: {y}\n')
        f.write(f'left_neibor: {left_neibor}\n')
        f.write(f'top_neibor: {top_neibor}\n')
        f.write(f'left_length: {left_length}\n')
        f.write(f'top_length: {top_length}\n')
        f.write(f'p: {p}\n')
        f.write(f'result: {result}\n')
        f.write(f'new_top_length: {new_top_length}\n')
        f.write(f'new_left_length: {new_left_length}\n\n')
        f.close()

plt.imshow(new_img)
print(f'porosity: { helper.image_porosity(new_img) }')

# %% tags=[]
plt.imshow(img)

# %%
