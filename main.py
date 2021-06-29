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
img_show = img if dim == 2 else img[0, :, :] if dim == 3 else []
axes.imshow(img_show)

# %%
segments_lengths = helper.segments_lengths_from_image(img)
ndim = img.ndim
segments_pdfs = {}

fig, axes = plt.subplots(ndim * 2, 1, figsize=(10, ndim * 10))

for d in np.arange(ndim):
    segments_pdfs[d] = {}

    pores_lengths = segments_lengths[d]['pores']
    p_hist, p_edges = helper.hist_of_lengths(pores_lengths)
    _, p_pdf, p_linspace = helper.kde_of_lengths(pores_lengths)
    segments_pdfs[d]['pores'] = p_pdf
    
    axes[2 * d].bar(p_edges[:-1], p_hist, width=np.diff(p_edges), edgecolor="black", align="edge")
    axes[2 * d].plot(p_linspace, p_pdf, c='red')

    solid_lengths = segments_lengths[d]['solid']
    s_hist, s_edges = helper.hist_of_lengths(solid_lengths)
    _, s_pdf, s_linspace = helper.kde_of_lengths(solid_lengths)
    segments_pdfs[d]['solid'] = s_pdf
    
    axes[2 * d + 1].bar(s_edges[:-1], s_hist, width=np.diff(s_edges), edgecolor="black", align="edge")
    axes[2 * d + 1].plot(s_linspace, s_pdf, c='red')
    
print(segments_pdfs)

# %%
