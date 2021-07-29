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
# use p_remain / (p_remain + p_change) probability for generator in 2D case

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
im_size = 32
dim = 2
porosity = 0.3
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
print(np.int32(img_show[0]))
print(np.int32(edge_distances[0][0]))
print(segment_lengths[0])

# %%
im = np.copy(img_show[0])
ed = np.ma.copy(edge_distances[0][0])
sl = cp.deepcopy(segment_lengths[0])
pores_mask = np.ma.mask_or(ed.mask, im)
solid_mask = np.ma.mask_or(ed.mask, ~im)
ed_pores = ed.copy()
ed_pores.mask = pores_mask
ed_solid = ed.copy()
ed_solid.mask = solid_mask
print(ed_pores)
print(ed_solid)

# %%
print(sl['pores'])
helper.hist_of_lengths(sl['pores'], density=False)

# %%
im = np.copy(img_show)
ed = np.ma.copy(edge_distances[0])

pores_mask = np.ma.mask_or(ed.mask, im)
solid_mask = np.ma.mask_or(ed.mask, ~im)
ed_pores = ed.copy()
ed_pores.mask = pores_mask
ed_solid = ed.copy()
ed_solid.mask = solid_mask

# for edp in ed_pores:
#     print(np.int32(edp))

max_value = np.max(ed_pores) + 1
hist, edges = np.histogram(ed_pores.compressed(), range=(0, max_value), bins=max_value)
hist


# %%
ed0 = edge_distances[0].ravel()
ed1 = edge_distances[1].ravel()
mask_c = np.ma.mask_or(edge_distances[0].mask, edge_distances[1].mask)
edc = np.column_stack((ed0.data, ed1.data)).reshape(32, 32, 2)
edc = np.ma.array(edc, mask=mask_c)
print(edc[0:1])

# %%
segments_lengths = helper.segments_lengths_from_image(img)
ndim = img.ndim

segments_hists = {}
segments_hist_cdfs = {}
segments_pdfs = {}
segments_cdfs = {}

fig, axes = plt.subplots(ndim * 2, 2, figsize=(20, ndim * 10))

for d in np.arange(ndim):
    segments_hists[d] = {}
    segments_hist_cdfs[d] = {}
    segments_pdfs[d] = {}
    segments_cdfs[d] = {}

    pores_lengths = segments_lengths[d]['pores']
    p_hist, p_edges, p_hist_cdf = helper.hist_of_lengths(pores_lengths)
    p_kde, p_pdf, p_cdf, p_linspace = helper.kde_of_lengths(pores_lengths)
    segments_hists[d]['pores'] = p_hist
    segments_hist_cdfs[d]['pores'] = p_hist_cdf
    segments_pdfs[d]['pores'] = p_pdf
    segments_cdfs[d]['pores'] = p_cdf
    
    axes[2 * d, 0].bar(p_edges[:-1], p_hist, width=np.diff(p_edges), edgecolor="black", align="edge")
    axes[2 * d, 0].plot(p_linspace, p_pdf, c='red')
    axes[2 * d, 1].plot([p_cdf(x) for x in p_linspace])
    axes[2 * d, 1].plot([p_hist_cdf(x) for x in p_linspace])
    axes[2 * d, 1].set_ylim([0, 1])

    solid_lengths = segments_lengths[d]['solid']
    s_hist, s_edges, s_hist_cdf = helper.hist_of_lengths(solid_lengths)
    s_kde, s_pdf, s_cdf, s_linspace = helper.kde_of_lengths(solid_lengths)
    segments_hists[d]['solid'] = s_hist
    segments_hist_cdfs[d]['solid'] = s_hist_cdf
    segments_pdfs[d]['solid'] = s_pdf
    segments_cdfs[d]['solid'] = s_cdf
    
    axes[2 * d + 1, 0].bar(s_edges[:-1], s_hist, width=np.diff(s_edges), edgecolor="black", align="edge")
    axes[2 * d + 1, 0].plot(s_linspace, s_pdf, c='red')
    axes[2 * d + 1, 1].plot([s_cdf(x) for x in s_linspace])
    axes[2 * d + 1, 1].plot([s_hist_cdf(x) for x in s_linspace])
    axes[2 * d + 1, 1].set_ylim([0, 1])

# %%
synt_img_2d = np.zeros((im_size, im_size))

# f = open('/Users/grimax/Desktop/log.txt', 'w')
# f.close()

_pdfs = segments_hists.copy()
_cdfs = segments_hist_cdfs.copy()

for y in np.linspace(0, im_size - 1, num=im_size - 1, dtype=np.int32):
    for x in np.linspace(0, im_size - 1, num=im_size - 1, dtype=np.int32):

        ph = None
        pv = None
        line_h = None
        line_v = None
        last_seg_h = None
        last_seg_v = None
        result = None
        
        if x == 0 and y == 0:
            result = sp.stats.bernoulli.rvs(1 - porosity)
            synt_img_2d[y, x] = result
            
        elif y == 0:
            line_h = synt_img_2d[y, 0:x]
            last_seg_h = helper.segments_from_row(line_h)[-1]
            kind_h = 'pores' if 0 in last_seg_h else 'solid'
            ph_change = _pdfs[0][kind_h][last_seg_h.size]
            ph_remain = 1 - _cdfs[0][kind_h](last_seg_h.size)
            ph = ph_remain / (ph_remain + ph_change)
            ph = 1 - ph if kind_h == 'pores' else ph
            result = sp.stats.bernoulli.rvs(ph)

        elif x == 0:
            line_v = synt_img_2d[0:y, x]
            last_seg_v = helper.segments_from_row(line_v)[-1]
            kind_v = 'pores' if 0 in last_seg_v else 'solid'
            pv_change = _pdfs[1][kind_v][last_seg_v.size]
            pv_remain = 1 - _cdfs[1][kind_v](last_seg_v.size)
            pv = pv_remain / (pv_remain + pv_change)
            pv = 1 - pv if kind_v == 'pores' else pv
            result = sp.stats.bernoulli.rvs(pv)

        else:
            line_h = synt_img_2d[y, 0:x]
            last_seg_h = helper.segments_from_row(line_h)[-1]
            kind_h = 'pores' if 0 in last_seg_h else 'solid'
            ph_change = _pdfs[0][kind_h][last_seg_h.size]
            ph_remain = 1 - _cdfs[0][kind_h](last_seg_h.size)
            ph = ph_remain / (ph_remain + ph_change)
            ph = 1 - ph if kind_h == 'pores' else ph

            line_v = synt_img_2d[0:y, x]
            last_seg_v = helper.segments_from_row(line_v)[-1]
            kind_v = 'pores' if 0 in last_seg_v else 'solid'
            pv_change = _pdfs[1][kind_v][last_seg_v.size]
            pv_remain = 1 - _cdfs[1][kind_v](last_seg_v.size)
            pv = pv_remain / (pv_remain + pv_change)
            pv = 1 - pv if kind_v == 'pores' else pv

            result = sp.stats.bernoulli.rvs((ph + pv) / 2)
        
        synt_img_2d[y, x] = result

#         f = open('/Users/grimax/Desktop/log.txt', 'a')
#         f.write(f'x: {x}, y: {y}\n')
#         f.write(f'line_h: {line_h}\n')
#         f.write(f'last_seg_h: {last_seg_h}\n')
#         f.write(f'ph: {ph}\n')
#         f.write(f'line_v: {line_v}\n')
#         f.write(f'last_seg_v: {last_seg_v}\n')
#         f.write(f'pv: {pv}\n')
#         f.write(f'result: {result}\n\n')

#         f.close()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(synt_img_2d)
axes[1].imshow(img_show)
print(f'porosity: {1 - np.sum(synt_img_2d)/synt_img_2d.size}')

# %%
x = synt_img_2d[1,].copy()
segments = helper.segments_from_row(x)
edge_distance = [np.array([idx for idx, _ in enumerate(segment)]) for segment in segments]
edge_distance

# %%
