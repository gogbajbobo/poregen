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
im_size = 1024*1024
img = helper.one_D_generator(im_size, sigma=4, porosity=0.3, seed=0)
plt.figure(figsize=(20,5))
plt.plot(img)
print(f'porosity: { helper.image_porosity(img) }')

# %%
segments_lengths = helper.segments_lengths_from_image(img)[1]
print(segments_lengths)

segments_hists = {}
segments_kdes = {}
segments_cdfs = {}

fig, axes = plt.subplots(2, 2, figsize=(20, 10))

pores_lengths = segments_lengths['pores']
p_hist, p_edges = helper.hist_of_lengths(pores_lengths)
p_kde, p_pdf, p_cdf, p_linspace = helper.kde_of_lengths(pores_lengths)
segments_hists['pores'] = p_hist
segments_kdes['pores'] = p_kde
segments_cdfs['pores'] = p_cdf

axes[0, 0].bar(p_edges[:-1], p_hist, width=np.diff(p_edges), edgecolor="black", align="edge")
axes[0, 0].plot(p_linspace, p_pdf, c='red')
axes[0, 1].plot([p_cdf(x) for x in p_linspace])
axes[0, 1].set_ylim([0, 1])

solid_lengths = segments_lengths['solid']
s_hist, s_edges = helper.hist_of_lengths(solid_lengths)
s_kde, s_pdf, s_cdf, s_linspace = helper.kde_of_lengths(solid_lengths)
segments_hists['solid'] = s_hist
segments_kdes['solid'] = s_kde
segments_cdfs['solid'] = s_cdf

axes[1, 0].bar(s_edges[:-1], s_hist, width=np.diff(s_edges), edgecolor="black", align="edge")
axes[1, 0].plot(s_linspace, s_pdf, c='red')
axes[1, 1].plot([s_cdf(x) for x in s_linspace])
axes[1, 1].set_ylim([0, 1])

# print(segments_hists)

# %%
synt_img = np.array([])
kdes = segments_kdes

line_length = im_size
# line_length = 100_000

while len(synt_img) < line_length:
    segment = helper.get_sample(kdes['pores'])
    synt_img = np.append(synt_img, np.zeros(segment))
    segment = helper.get_sample(kdes['solid'])
    synt_img = np.append(synt_img, np.ones(segment))

synt_img = synt_img[:line_length]

plt.figure(figsize=(20, 5))
plt.plot(synt_img)

print(f'porosity: { helper.image_porosity(synt_img) }')

# %%
t_segments_lengths = helper.segments_lengths_from_image(np.array([synt_img]))[0]
print(t_segments_lengths)

t_segments_hists = {}

fig, axes = plt.subplots(2, 2, figsize=(20, 10))

pores_lengths = t_segments_lengths['pores']
p_hist, p_edges = helper.hist_of_lengths(pores_lengths)
p_kde, p_pdf, p_cdf, p_linspace = helper.kde_of_lengths(pores_lengths)
t_segments_hists['pores'] = p_hist

axes[0, 0].bar(p_edges[:-1], p_hist, width=np.diff(p_edges), edgecolor="black", align="edge")
axes[0, 0].plot(p_linspace, p_pdf, c='red')
axes[0, 1].plot([p_cdf(x) for x in p_linspace])
axes[0, 1].set_ylim([0, 1])

solid_lengths = t_segments_lengths['solid']
s_hist, s_edges = helper.hist_of_lengths(solid_lengths)
s_kde, s_pdf, s_cdf, s_linspace = helper.kde_of_lengths(solid_lengths)
t_segments_hists['solid'] = s_hist

axes[1, 0].bar(s_edges[:-1], s_hist, width=np.diff(s_edges), edgecolor="black", align="edge")
axes[1, 0].plot(s_linspace, s_pdf, c='red')
axes[1, 1].plot([s_cdf(x) for x in s_linspace])
axes[1, 1].set_ylim([0, 1])

# print(segments_hists)

# %%
print(sp.stats.mode(segments_lengths['pores']))
print(sp.stats.mode(segments_lengths['solid']))
print(sp.stats.mode(t_segments_lengths['pores']))
print(sp.stats.mode(t_segments_lengths['solid']))

# %%
print(np.mean(segments_lengths['pores']))
print(np.mean(segments_lengths['solid']))
print(np.mean(t_segments_lengths['pores']))
print(np.mean(t_segments_lengths['solid']))

# %%
print(np.std(segments_lengths['pores']))
print(np.std(segments_lengths['solid']))
print(np.std(t_segments_lengths['pores']))
print(np.std(t_segments_lengths['solid']))

# %%
print(np.median(segments_lengths['pores']))
print(np.median(segments_lengths['solid']))
print(np.median(t_segments_lengths['pores']))
print(np.median(t_segments_lengths['solid']))

# %%
p_hists_diff = np.zeros(np.max([segments_hists['pores'].size, t_segments_hists['pores'].size]))
p_hists_diff[:segments_hists['pores'].size] = segments_hists['pores']
p_hists_diff[:t_segments_hists['pores'].size] -= t_segments_hists['pores']

s_hists_diff = np.zeros(np.max([segments_hists['solid'].size, t_segments_hists['solid'].size]))
s_hists_diff[:segments_hists['solid'].size] = segments_hists['solid']
s_hists_diff[:t_segments_hists['solid'].size] -= t_segments_hists['solid']

print(np.sum(p_hists_diff))
print(np.sum(s_hists_diff))

# %%
