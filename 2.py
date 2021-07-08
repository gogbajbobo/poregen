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

from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns

import helper

# %%
im_size = 128
dim = 2
im_shape = np.ones(dim, dtype=np.int32) * im_size
np.random.seed(0)
img = ~ps.generators.blobs(im_shape, porosity=0.3, blobiness=.5)
# img = np.array(img, dtype=np.int32)

# %%
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
img_show = img if dim == 2 else img[0, :, :] if dim == 3 else []
axes.imshow(img_show)
print(f'porosity: { helper.image_porosity(img) }')

# %%
segments_lengths = helper.segments_lengths_from_image(img)
ndim = img.ndim
segments_kdes = {}
segments_cdfs = {}

fig, axes = plt.subplots(ndim * 2, 2, figsize=(20, ndim * 10))

for d in np.arange(ndim):
    segments_kdes[d] = {}
    segments_cdfs[d] = {}

    pores_lengths = segments_lengths[d]['pores']
    p_hist, p_edges = helper.hist_of_lengths(pores_lengths)
    p_kde, p_pdf, p_cdf, p_linspace = helper.kde_of_lengths(pores_lengths)
    segments_kdes[d]['pores'] = p_kde
    segments_cdfs[d]['pores'] = p_cdf
    
    axes[2 * d, 0].bar(p_edges[:-1], p_hist, width=np.diff(p_edges), edgecolor="black", align="edge")
    axes[2 * d, 0].plot(p_linspace, p_pdf, c='red')
    axes[2 * d, 1].plot([p_cdf(x) for x in p_linspace])
    axes[2 * d, 1].set_ylim([0, 1])

    solid_lengths = segments_lengths[d]['solid']
    s_hist, s_edges = helper.hist_of_lengths(solid_lengths)
    s_kde, s_pdf, s_cdf, s_linspace = helper.kde_of_lengths(solid_lengths)
    segments_kdes[d]['solid'] = s_kde
    segments_cdfs[d]['solid'] = s_cdf
    
    axes[2 * d + 1, 0].bar(s_edges[:-1], s_hist, width=np.diff(s_edges), edgecolor="black", align="edge")
    axes[2 * d + 1, 0].plot(s_linspace, s_pdf, c='red')
    axes[2 * d + 1, 1].plot([s_cdf(x) for x in s_linspace])
    axes[2 * d + 1, 1].set_ylim([0, 1])

# %%
