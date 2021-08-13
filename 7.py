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
# use border distances for generator in 2D case

# %% tags=[]
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

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split

import helper

# %% tags=[]
im_size = 64
dim = 2
porosity = 0.5
blobiness = 1
im_shape = np.ones(dim, dtype=np.int32) * im_size

np.random.seed(0)
img = ~ps.generators.blobs(im_shape, porosity=porosity, blobiness=blobiness)
img = img.astype(np.int32)

fig, axes = plt.subplots(1, 1, figsize=(5, 5))
img_show = img if dim == 2 else img[0, :, :] if dim == 3 else []
axes.imshow(img_show)
print(f'porosity: { helper.image_porosity(img) }')

# %% tags=[]
x_distances_solid, x_distances_void, y_distances_solid, y_distances_void = helper.border_distances_for_image(img)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(x_distances_solid)
axes[1].imshow(x_distances_void)
axes[2].imshow(y_distances_solid)
axes[3].imshow(y_distances_void)

# %% tags=[]
df = helper.dataframe_with_distances_from_image(img, (x_distances_solid, x_distances_void, y_distances_solid, y_distances_void))
df

# %% tags=[]
df.corr().style.background_gradient(axis=None)

# %% tags=[]
X = df[['xDistanceSolid', 'xDistanceVoid', 'yDistanceSolid', 'yDistanceVoid']]
Y = df[['isSolid']]
x_train, x_test, y_train, y_test = train_test_split(X, Y)
log_reg = sm.Logit(y_train, x_train).fit()
print(log_reg.summary())
predicted_train = log_reg.predict(x_train) > .5
predicted_test = log_reg.predict(x_test) > .5
print(f'train score: {(predicted_train.to_numpy().ravel() == y_train.to_numpy().ravel()).mean()}')
print(f'test score: {(predicted_test.to_numpy().ravel() == y_test.to_numpy().ravel()).mean()}')

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
predict_image = skl_log_reg.predict(X)
y = [y for y, x in Y[Y.isSolid != predict_image].index]
x = [x for y, x in Y[Y.isSolid != predict_image].index]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
img_show = img if dim == 2 else img[0, :, :] if dim == 3 else []
axes[0].imshow(img_show)
axes[1].imshow(predict_image.reshape(img.shape))
axes[2].imshow(img_show)
axes[2].scatter(x, y, color='red', marker='.')

# %%
xds = np.full(img.shape, -1, dtype=np.int32)
xdv = np.full(img.shape, -1, dtype=np.int32)
yds = np.full(img.shape, -1, dtype=np.int32)
ydv = np.full(img.shape, -1, dtype=np.int32)
new_img = np.full(img.shape, -1, dtype=np.int32)

for y in np.arange(img.shape[-2]):
    for x in np.arange(img.shape[-1]):
        
        if y == 0 and x == 0:
            result = sp.stats.bernoulli.rvs(porosity)
            new_img[y, x] = result
            if result:
                xds[y, x] = 0
                yds[y, x] = 0
            else:
                xdv[y, x] = 0
                ydv[y, x] = 0
        elif y = 0:
            
