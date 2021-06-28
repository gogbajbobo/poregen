# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import porespy as ps
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

import helper

# %%
im_size = 128
dim = 2
im_shape = np.ones(dim, dtype=np.int32) * im_size
np.random.seed(0)
img = ~ps.generators.blobs(im_shape, porosity=0.3, blobiness=2)
# img = np.array(img, dtype=np.int32)

# %%
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
axes.imshow(img)

# %%
segments_lengths = helper.get_segments_lengths_from_image(img)
ndim = img.ndim

fig, axes = plt.subplots(ndim * 2, 1, figsize=(10, ndim * 10))
for d in np.arange(ndim):
    pores_lengths = segments_lengths[d]['pores']
    p_max_value = np.max(pores_lengths)
    print(f'{ d } p_max_value { p_max_value }')
    p_hist, p_edges = np.histogram(pores_lengths, bins=p_max_value)
    p_hist_kde = sp.stats.gaussian_kde(p_hist)
    p_hist = p_hist / np.sum(p_hist)
    p_linspace = np.linspace(0, p_max_value, num=p_max_value + 1)
    p_pdf = p_hist_kde.pdf(p_linspace)
    
    axes[2 * d].bar(p_edges[:-1], p_hist, width=np.diff(p_edges), edgecolor="black", align="edge")
    axes[2 * d].plot(p_linspace, p_pdf, c='red')

    solid_lengths = segments_lengths[d]['solid']
    s_max_value = np.max(solid_lengths)
    print(f'{ d } s_max_value { s_max_value }')
    s_hist, s_edges = np.histogram(solid_lengths, bins=s_max_value)
    s_hist_kde = sp.stats.gaussian_kde(s_hist)
    s_hist = s_hist / np.sum(s_hist)
    s_linspace = np.linspace(0, s_max_value, num=s_max_value + 1)
    s_pdf = s_hist_kde.pdf(s_linspace)

    axes[2 * d + 1].bar(s_edges[:-1], s_hist, width=np.diff(s_edges), edgecolor="black", align="edge")
    axes[2 * d + 1].plot(s_pdf, c='red')

# %%
fig, axes = plt.subplots(ndim * 2, 1, figsize=(10, ndim * 10))
for d in np.arange(ndim):
    pores_lengths = segments_lengths[d]['pores']
    solid_lengths = segments_lengths[d]['solid']
    p_max_value = np.max(pores_lengths)
    s_max_value = np.max(solid_lengths)
    sns.histplot(pores_lengths, ax=axes[2 * d], bins=p_max_value, stat='probability', kde=True)
    sns.histplot(solid_lengths, ax=axes[2 * d + 1], bins=s_max_value, stat='probability', kde=True)

# %%

# %%

# %%
