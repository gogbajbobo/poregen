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
im_size = 16
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
synt_img_h = np.array([])
synt_img_v = np.array([])

kdes_h = segments_kdes[0]
kdes_v = segments_kdes[0]

while len(synt_img_h) < im_size:
    segment = helper.get_sample(kdes_h['pores'])
    synt_img_h = np.append(synt_img_h, np.zeros(segment))
    segment = helper.get_sample(kdes_h['solid'])
    synt_img_h = np.append(synt_img_h, np.ones(segment))

while len(synt_img_v) < im_size:
    segment = helper.get_sample(kdes_v['pores'])
    synt_img_v = np.append(synt_img_v, np.zeros(segment))
    segment = helper.get_sample(kdes_v['solid'])
    synt_img_v = np.append(synt_img_v, np.ones(segment))

synt_img_h = synt_img_h[:im_size]
synt_img_v = synt_img_v[:im_size]

fig, axes = plt.subplots(1, 2, figsize=(20, 5))
axes[0].plot(synt_img_h)
axes[1].plot(synt_img_v)
# print(f'porosity: { 1 - np.sum(synt_img)/ synt_img.size}')

# %%
synt_img_2d = np.zeros((im_size, im_size))
synt_img_2d[0, :] = synt_img_h
synt_img_2d[:, 0] = synt_img_v
plt.imshow(synt_img_2d)

# %%
for y in np.linspace(1, im_size - 1, num=im_size - 1, dtype=np.int32):
    for x in np.linspace(1, im_size - 1, num=im_size - 1, dtype=np.int32):
        line_h = synt_img_2d[y, 0:x]
        line_v = synt_img_2d[0:y, x]
        last_seg_h = helper.segments_from_row(line_h)[-1]
        last_seg_v = helper.segments_from_row(line_v)[-1]
        kind_h = 'pores' if 0 in last_seg_h else 'solid'
        kind_v = 'pores' if 0 in last_seg_v else 'solid'
        probability_h = 1 - segments_cdfs[0][kind_h](x)
        probability_v = 1 - segments_cdfs[1][kind_v](y)
        print(last_seg_h, last_seg_v)
        print(probability_h, probability_v)

# %%
