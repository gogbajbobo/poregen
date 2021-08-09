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
# use pixel's left and top segment lengths for generator in 2D case

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

import helper

# %% tags=[]
import seaborn as sns
sns.set_theme()
sns.set_style("white")

# %% tags=[]
im_size = 128
dim = 2
porosity = 0.3
blobiness = 1
im_shape = np.ones(dim, dtype=np.int32) * im_size
np.random.seed(0)
img = ~ps.generators.blobs(im_shape, porosity=porosity, blobiness=blobiness)

# %% tags=[]
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
img_show = img if dim == 2 else img[0, :, :] if dim == 3 else []
axes.imshow(img_show)
print(f'porosity: { helper.image_porosity(img) }')

# %% tags=[]
image_statistics = helper.image_statistics(img)
edge_distances = image_statistics['edge_distances']
eds_y = (np.int32(edge_distances[1]) + 1).T
eds_x = np.int32(edge_distances[0]) + 1
# print(eds_y)
# print(eds_x)

# %% tags=[]
y_size = img.shape[-2]
x_size = img.shape[-1]
y_grid = np.arange(y_size)
x_grid = np.arange(x_size)
indices = pd.MultiIndex.from_tuples(list(np.ndindex(y_size, x_size)))

df = pd.DataFrame(columns=['isSolid', 'leftLength', 'leftIsSolid', 'topLength', 'topIsSolid'], index=indices)

for y in y_grid:
    for x in x_grid:
        
        is_solid = bool(img[y, x])
        left_length = np.NaN
        left_is_solid = np.NaN
        rigth_length = np.NaN
        rigth_is_solid = np.NaN
        
        if x > 0:
            prev_x = x - 1
            is_masked = eds_x.mask[y, prev_x]
            if not is_masked:
                left_length = eds_x.data[y, prev_x]
                left_is_solid = bool(img[y, prev_x])

        if y > 0:
            prev_y = y - 1
            is_masked = eds_y.mask[prev_y, x]
            if not is_masked:
                rigth_length = eds_y.data[prev_y, x]
                rigth_is_solid = bool(img[prev_y, x])

        df.loc[(y, x)] = pd.Series({
            'isSolid': is_solid, 
            'leftLength': left_length, 
            'leftIsSolid': left_is_solid, 
            'topLength': rigth_length, 
            'topIsSolid': rigth_is_solid,
        })

print(df.info())
print(df.shape)
        
dff = df[df.notna().all(axis='columns')].astype(np.int32)

print(dff.info())
print(dff.shape)

# %% tags=[]
# dff.head()

# %% tags=[]
# fig, axes = plt.subplots(1, 1, figsize=(10, 10))
# sns.histplot(dff, x='leftLength', y='topLength', bins=(dff['leftLength'].max(), dff['topLength'].max()))

# %% tags=[]
# dff1 = dff[(dff['isSolid'] == dff['leftIsSolid']) & (dff['isSolid'] == dff['topIsSolid'])]
# print(dff1.shape)
# dff1.head()

# %% tags=[]
# dff2 = dff[dff['leftIsSolid'] != dff['topIsSolid']]
# print(dff2.shape)
# dff2.head()

# %% tags=[]
# fig, axes = plt.subplots(1, 2, figsize=(20, 10))
# sns.histplot(dff1, x='leftLength', y='topLength', bins=(dff1['leftLength'].max(), dff1['topLength'].max()), ax=axes[0])
# sns.histplot(dff2, x='leftLength', y='topLength', bins=(dff2['leftLength'].max(), dff2['topLength'].max()), ax=axes[1])

# %% tags=[]
# g = sns.PairGrid(dff, vars=['leftLength', 'topLength'])
# g.map_diag(sns.histplot)
# g.map_offdiag(sns.scatterplot)

# %% tags=[]
import statsmodels.api as sm

# %% tags=[]
X = dff[['leftLength', 'leftIsSolid', 'topLength', 'topIsSolid']]
Y = dff[['isSolid']]
x_train = X[:500]
y_train = Y[:500]
x_test = X[500:]
y_test = Y[500:]
log_reg = sm.Logit(y_train, x_train).fit()
print(log_reg.summary())
predicted_train = log_reg.predict(x_train) > .5
predicted_test = log_reg.predict(x_test) > .5
print(f'train score: {(predicted_train.to_numpy().ravel() == y_train.to_numpy().ravel()).mean()}')
print(f'test score: {(predicted_test.to_numpy().ravel() == y_test.to_numpy().ravel()).mean()}')

# %% tags=[]
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix

# %% tags=[]
skl_log_reg = LogisticRegression(fit_intercept = False, penalty='none')
X_train = x_train
Y_train = y_train.to_numpy().ravel()
X_test = x_test
Y_test = y_test.to_numpy().ravel()
skl_log_reg.fit(X_train, Y_train)
print(skl_log_reg.coef_)
print(f'train: {skl_log_reg.score(X_train, Y_train)}')
print(f'test: {skl_log_reg.score(X_test, Y_test)}')
plot_confusion_matrix(skl_log_reg, X_test, Y_test)

# %% tags=[]
parameters = {'C': [.0001, .001, .01, .1, 1, 10, 100], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
Logistic = LogisticRegression(max_iter=10000)
grid_search_CV = GridSearchCV(Logistic, parameters)
grid_search_CV.fit(X_train, Y_train)
print(f'best estimator: {grid_search_CV.best_estimator_}')
print(f'best estimator coefs: {grid_search_CV.best_estimator_.coef_}')
print(f'best score: {grid_search_CV.best_score_}')
print(f'train accuracy: {grid_search_CV.best_estimator_.score(X_train, Y_train)}')
print(f'test accuracy: {grid_search_CV.best_estimator_.score(X_test, Y_test)}')
plot_confusion_matrix(grid_search_CV.best_estimator_, X_test, Y_test)

# %% tags=[]
new_img = np.empty(im_shape, dtype=np.int32)
new_img_eds = np.empty((*im_shape, 2), dtype=np.int32)

segments_lengths = image_statistics['segments_lengths']

synt_line_h = np.array([])
synt_line_v = np.array([])

while len(synt_line_h) < im_shape[-1]:
    segment = np.random.choice(segments_lengths[0]['solid'])
    synt_line_h = np.append(synt_line_h, np.ones(segment))
    segment = np.random.choice(segments_lengths[0]['pores'])
    synt_line_h = np.append(synt_line_h, np.zeros(segment))

while len(synt_line_v) < im_shape[-2]:
    segment = np.random.choice(segments_lengths[1]['solid'])
    synt_line_v = np.append(synt_line_v, np.ones(segment))
    segment = np.random.choice(segments_lengths[1]['pores'])
    synt_line_v = np.append(synt_line_v, np.zeros(segment))

synt_line_h = synt_line_h[:im_shape[-1]]
synt_line_v = synt_line_v[:im_shape[-2]]

# print(synt_line_h)
# print(synt_line_v)

new_img[0, :] = synt_line_h
new_img[:, 0] = synt_line_v

# print(new_img[0, :])
# print(new_img[:, 0])

new_img_eds[0, :] = np.array([1, 0])
new_img_eds[:, 0] = np.array([0, 1])
new_img_eds[0, 0] = np.array([0, 0])

# print(new_img_eds[0, :])
# print(new_img_eds[:, 0])

def calc_result(prediction):
#     return np.int32(prediction > .5)
    return sp.stats.bernoulli.rvs(prediction)

for y in y_grid[1:]:
    for x in x_grid[1:]:
        leftLength = new_img_eds[y, x - 1][1]
        leftIsSolid = new_img[y, x - 1]
        topLength = new_img_eds[y - 1, x][0]
        topIsSolid = new_img[y - 1, x]
        prediction = grid_search_CV.best_estimator_.predict([[leftLength, leftIsSolid, topLength, topIsSolid]])
        result = calc_result(prediction)
        new_img_eds[y, x] = np.array([
            topLength + 1 if result == topIsSolid else 1, 
            leftLength + 1 if result == leftIsSolid else 1
        ])
        new_img[y, x] = result

new_img.shape

# %% tags=[]
fig, axes = plt.subplots(1, 1, figsize=(10, 10))
axes.imshow(new_img[:, :])
print(f'porosity: { helper.image_porosity(new_img) }')

# %%
