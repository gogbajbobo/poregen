# -*- coding: utf-8 -*-
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
synt_img_h = np.array([])
synt_img_v = np.array([])

kdes_h = segments_kdes[0]
kdes_v = segments_kdes[0]

# line_length = im_size
line_length = 10_000

while len(synt_img_h) < line_length:
    segment = helper.get_sample(kdes_h['pores'])
    synt_img_h = np.append(synt_img_h, np.zeros(segment))
    segment = helper.get_sample(kdes_h['solid'])
    synt_img_h = np.append(synt_img_h, np.ones(segment))

while len(synt_img_v) < line_length:
    segment = helper.get_sample(kdes_v['pores'])
    synt_img_v = np.append(synt_img_v, np.zeros(segment))
    segment = helper.get_sample(kdes_v['solid'])
    synt_img_v = np.append(synt_img_v, np.ones(segment))

synt_img_h = synt_img_h[:line_length]
synt_img_v = synt_img_v[:line_length]

fig, axes = plt.subplots(1, 2, figsize=(20, 5))
axes[0].plot(synt_img_h)
axes[1].plot(synt_img_v)

print(f'porosity h: { 1 - np.sum(synt_img_h)/ synt_img_h.size }') # porosity tends to 0.38 @ large lengths — this is not the 0.3!
print(f'porosity v: { 1 - np.sum(synt_img_v)/ synt_img_v.size }') # this method is incorrect

# %%
synt_img_2d = np.zeros((im_size, im_size))
synt_img_2d[0, :] = synt_img_h
synt_img_2d[:, 0] = synt_img_v
plt.imshow(synt_img_2d)

# %%
f = open('/Users/grimax/Desktop/log.txt', 'w')
f.close()

for y in np.linspace(1, im_size - 1, num=im_size - 1, dtype=np.int32):
    for x in np.linspace(1, im_size - 1, num=im_size - 1, dtype=np.int32):

        line_h = synt_img_2d[y, 0:x]
        last_seg_h = helper.segments_from_row(line_h)[-1]
        kind_h = 'pores' if 0 in last_seg_h else 'solid'
        p_h = 1 - segments_cdfs[0][kind_h](len(last_seg_h))

        line_v = synt_img_2d[0:y, x]
        last_seg_v = helper.segments_from_row(line_v)[-1]
        kind_v = 'pores' if 0 in last_seg_v else 'solid'
        p_v = 1 - segments_cdfs[1][kind_v](len(last_seg_v))

        p_solid = 0
        p_pores = 0

        if kind_h == kind_v:
            if kind_h == 'pores':
                if np.min([p_h, p_v]) == 0:
                    p_pores = 0
                    p_solid = 1
                else:
                    p_pores = 1 if np.sqrt(p_h * p_v) > 0.5 else 0
                    p_solid = 1 - p_pores
            else:
                if np.min([p_h, p_v]) == 0:
                    p_solid = 0
                    p_pores = 1
                else:
                    p_solid = 1 if np.sqrt(p_h * p_v) > 0.5 else 0
                    p_pores = 1 - p_solid
        else:
            if kind_h == 'pores':
                p_pores = 1 if np.sqrt(p_h * (1 - p_v)) > 0.5 else 0
                p_solid = 1 - p_pores
#                 p_pores = p_h / (p_h + p_v)
#                 p_solid = p_v / (p_h + p_v)
            else:
                p_solid = 1 if np.sqrt(p_h * (1 - p_v)) > 0.5 else 0
                p_pores = 1 - p_solid
#                 p_solid = p_h / (p_h + p_v)
#                 p_pores = p_v / (p_h + p_v)
        result = sp.stats.bernoulli.rvs(p_solid)
        synt_img_2d[y, x] = result
        
#         f = open('/Users/grimax/Desktop/log.txt', 'a')
#         f.write(f'x: {x}, y: {y}\n')
#         f.write(f'last_seg_h: {last_seg_h}\n')
#         f.write(f'p_h: {p_h}\n')
#         f.write(f'last_seg_v: {last_seg_v}\n')
#         f.write(f'p_v: {p_v}\n')
#         f.write(f'p_solid: {p_solid}, p_pores: {p_pores}, result: {result}\n\n')

#         f.close()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(synt_img_2d)
axes[1].imshow(img_show)
print(f'porosity: {1 - np.sum(synt_img_2d)/synt_img_2d.size}')

# %%
synt_img_segments_lengths = helper.segments_lengths_from_image(synt_img_2d)
ndim = synt_img_2d.ndim
synt_img_segments_kdes = {}
synt_img_segments_cdfs = {}

fig, axes = plt.subplots(ndim * 2, 2, figsize=(20, ndim * 10))

for d in np.arange(ndim):
    synt_img_segments_kdes[d] = {}
    synt_img_segments_cdfs[d] = {}

    pores_lengths = synt_img_segments_lengths[d]['pores']
    p_hist, p_edges = helper.hist_of_lengths(pores_lengths)
    p_kde, p_pdf, p_cdf, p_linspace = helper.kde_of_lengths(pores_lengths)
    synt_img_segments_kdes[d]['pores'] = p_kde
    synt_img_segments_cdfs[d]['pores'] = p_cdf
    
    axes[2 * d, 0].bar(p_edges[:-1], p_hist, width=np.diff(p_edges), edgecolor="black", align="edge")
    axes[2 * d, 0].plot(p_linspace, p_pdf, c='red')
    axes[2 * d, 1].plot([p_cdf(x) for x in p_linspace])
    axes[2 * d, 1].set_ylim([0, 1])

    solid_lengths = synt_img_segments_lengths[d]['solid']
    s_hist, s_edges = helper.hist_of_lengths(solid_lengths)
    s_kde, s_pdf, s_cdf, s_linspace = helper.kde_of_lengths(solid_lengths)
    synt_img_segments_kdes[d]['solid'] = s_kde
    synt_img_segments_cdfs[d]['solid'] = s_cdf
    
    axes[2 * d + 1, 0].bar(s_edges[:-1], s_hist, width=np.diff(s_edges), edgecolor="black", align="edge")
    axes[2 * d + 1, 0].plot(s_linspace, s_pdf, c='red')
    axes[2 * d + 1, 1].plot([s_cdf(x) for x in s_linspace])
    axes[2 * d + 1, 1].set_ylim([0, 1])

# %%
fig, ax = plt.subplots(2, 2, subplot_kw={"projection": "3d"}, figsize=(20, 20))

def m_func(i, j):
    _i = i / 100
    _j = j / 100
    return _i * _j

data = np.fromfunction(m_func, (101, 101))
sqrt_data = np.sqrt(data)

def diff_data_func(i, j):
    _i = i / 100
    _j = j / 100
    return np.max([_i, _j]) -  _i * _j

def diff_sqrt_data_func(i, j):
    _i = i / 100
    _j = j / 100
    return np.max([_i, _j]) -  np.sqrt(_i * _j)

diff_data = np.fromfunction(np.vectorize(diff_data_func), (101, 101))
diff_sqrt_data = np.fromfunction(np.vectorize(diff_sqrt_data_func), (101, 101))

x = y = np.linspace(0, 1, 101)
x, y = np.meshgrid(x, y)
ax[0, 0].plot_surface(x, y, data, cmap=cm.coolwarm)
ax[0, 1].plot_surface(x, y, sqrt_data, cmap=cm.coolwarm)
ax[1, 0].plot_surface(x, y, diff_data, cmap=cm.coolwarm)
ax[1, 1].plot_surface(x, y, diff_sqrt_data, cmap=cm.coolwarm)

# %%
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].imshow(data, cmap=cm.coolwarm)
ax[0, 1].imshow(sqrt_data, cmap=cm.coolwarm)
ax[1, 0].imshow(diff_data, cmap=cm.coolwarm)
ax[1, 1].imshow(diff_sqrt_data, cmap=cm.coolwarm)

# %%
