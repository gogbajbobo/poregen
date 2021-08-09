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

# %%
import porespy as ps
import numpy as np
import pandas as pd

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import colors

import helper

# %% tags=[]
import seaborn as sns
sns.set_theme()
sns.set_style("white")

# %% tags=[]
im_size = 32
dim = 2
porosity = 0.5
im_shape = np.ones(dim, dtype=np.int32) * im_size
np.random.seed(0)
img = ~ps.generators.blobs(im_shape, porosity=porosity, blobiness=1)

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
dff.head()

# %% tags=[]
fig, axes = plt.subplots(1, 1, figsize=(10, 10))
sns.histplot(dff, x='leftLength', y='topLength', bins=(dff['leftLength'].max(), dff['topLength'].max()))

# %% tags=[]
dff1 = dff[(dff['isSolid'] == dff['leftIsSolid']) & (dff['isSolid'] == dff['topIsSolid'])]
print(dff1.shape)
dff1.head()

# %% tags=[]
dff2 = dff[dff['leftIsSolid'] != dff['topIsSolid']]
print(dff2.shape)
dff2.head()

# %% tags=[]
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
sns.histplot(dff1, x='leftLength', y='topLength', bins=(dff1['leftLength'].max(), dff1['topLength'].max()), ax=axes[0])
sns.histplot(dff2, x='leftLength', y='topLength', bins=(dff2['leftLength'].max(), dff2['topLength'].max()), ax=axes[1])

# %% tags=[]
g = sns.PairGrid(dff, vars=['leftLength', 'topLength'])
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)

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
parameters = {'C': [.0001, .001, .01, .1, 1, 10], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
Logistic = LogisticRegression()
grid_search_CV = GridSearchCV(Logistic, parameters)
grid_search_CV.fit(X_train, Y_train)
print(f'best estimator: {grid_search_CV.best_estimator_}')
print(f'best score: {grid_search_CV.best_score_}')
print(f'train accuracy: {grid_search_CV.best_estimator_.score(X_train, Y_train)}')
print(f'test accuracy: {grid_search_CV.best_estimator_.score(X_test, Y_test)}')

# %%
