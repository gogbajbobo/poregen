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
# use p_remain / (p_remain + p_change) probability for generator in 1D case

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
porosity = 0.3
img = helper.one_D_generator(im_size, sigma=1, porosity=porosity, seed=1)
plt.figure(figsize=(20,5))
plt.plot(img)
print(f'porosity: { helper.image_porosity(img) }')

# %%
segments_lengths = helper.segments_lengths_from_image(img)[1]
# print(segments_lengths)

segments_hists = {}
segments_hist_cdfs = {}
segments_kdes = {}
segments_pdfs = {}
segments_cdfs = {}

fig, axes = plt.subplots(2, 2, figsize=(20, 10))

pores_lengths = segments_lengths['pores']
p_hist, p_edges, p_hist_cdf = helper.hist_of_lengths(pores_lengths)
p_kde, p_pdf, p_cdf, p_linspace = helper.kde_of_lengths(pores_lengths)
segments_hists['pores'] = p_hist
segments_hist_cdfs['pores'] = p_hist_cdf
segments_kdes['pores'] = p_kde
segments_pdfs['pores'] = p_pdf
segments_cdfs['pores'] = p_cdf

axes[0, 0].bar(p_edges[:-1], p_hist, width=np.diff(p_edges), edgecolor="black", align="edge")
axes[0, 0].plot(p_linspace, p_pdf, c='red')
axes[0, 1].plot([p_cdf(x) for x in p_linspace])
axes[0, 1].plot([p_hist_cdf(x) for x in p_linspace])
axes[0, 1].set_ylim([0, 1])

solid_lengths = segments_lengths['solid']
s_hist, s_edges, s_hist_cdf = helper.hist_of_lengths(solid_lengths)
s_kde, s_pdf, s_cdf, s_linspace = helper.kde_of_lengths(solid_lengths)
segments_hists['solid'] = s_hist
segments_hist_cdfs['solid'] = s_hist_cdf
segments_kdes['solid'] = s_kde
segments_pdfs['solid'] = s_pdf
segments_cdfs['solid'] = s_cdf

axes[1, 0].bar(s_edges[:-1], s_hist, width=np.diff(s_edges), edgecolor="black", align="edge")
axes[1, 0].plot(s_linspace, s_pdf, c='red')
axes[1, 1].plot([s_cdf(x) for x in s_linspace])
axes[1, 1].plot([s_hist_cdf(x) for x in s_linspace])
axes[1, 1].set_ylim([0, 1])

# print(segments_hists)

# %%
sample_length = 1024*1024
total_length = {}
total_length['pores'] = np.int32(np.round(sample_length * porosity))
total_length['solid'] = sample_length - total_length['pores']
print(f'pores total_length: {total_length["pores"]}, solid total_length: {total_length["solid"]}')
print(f'segments_hists: {segments_hists}')

# %%
_hists = segments_hists.copy()
_hist_cdfs = segments_hist_cdfs.copy()
synt_img = np.zeros(sample_length)
last_seg = np.array([], dtype=np.int32)

for i in np.arange(sample_length):
    if i == 0:
        synt_img[i] = sp.stats.bernoulli.rvs(1 - porosity)
        last_seg = np.array([synt_img[i]])
    else:
#         print(f'last_seg: {last_seg}')
        kind = 'pores' if 0 in last_seg else 'solid'
        p_change = _hists[kind][last_seg.size]
        p_remain = 1 - _hist_cdfs[kind](last_seg.size)
        p = p_remain / (p_remain + p_change)
#         print(f'p_change: {p_change}')
#         print(f'p_remain: {p_remain}')
#         print(f'p {p}')
        result = sp.stats.bernoulli.rvs(1 - p if kind == 'pores' else p)
        synt_img[i] = result
        boundary = result != last_seg[0]
        if boundary:
#             print('\n')
            last_seg = np.array([result])
        else:
            last_seg = np.append(last_seg, [result])

plt.figure(figsize=(20, 5))
plt.plot(synt_img[:128])
print(f'synt_img porosity: {helper.image_porosity(synt_img)}')

# %%
new_segments_lengths = helper.segments_lengths_from_image(np.array([synt_img]).reshape([sample_length, 1]))[1]
new_segments_lengths

# %%
pn_lengths = new_segments_lengths['pores']
pn_hist, pn_edges, pn_hist_cdf = helper.hist_of_lengths(pn_lengths)
print(f'p_hist: {p_hist}')
print(f'pn_hist: {pn_hist}')
print(f'p diff: {pn_hist - p_hist}')

sn_lengths = new_segments_lengths['solid']
sn_hist, sn_edges, sn_hist_cdf = helper.hist_of_lengths(sn_lengths)
print(f's_hist: {s_hist}')
print(f'sn_hist: {sn_hist}')
print(f's diff: {sn_hist - s_hist}')

# %%
