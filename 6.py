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

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split

import helper

# %% tags=[]
import seaborn as sns
sns.set_theme()
sns.set_style("white")
sns.color_palette("viridis", as_cmap=True)

# %% tags=[]
im_size = 128
dim = 2
porosity = 0.3
blobiness = 1
im_shape = np.ones(dim, dtype=np.int32) * im_size

np.random.seed(0)
img = ~ps.generators.blobs(im_shape, porosity=porosity, blobiness=blobiness)

fig, axes = plt.subplots(1, 1, figsize=(5, 5))
img_show = img if dim == 2 else img[0, :, :] if dim == 3 else []
axes.imshow(img_show)
print(f'porosity: { helper.image_porosity(img) }')

# %% tags=[]
df = helper.dataframe_from_image(img)
df.head()

# %% tags=[]
X = df[['leftLength', 'leftIsSolid', 'topLength', 'topIsSolid']]
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
y = [y for y, x in Y[Y.isSolid != skl_log_reg.predict(X)].index]
x = [x for y, x in Y[Y.isSolid != skl_log_reg.predict(X)].index]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
img_show = img if dim == 2 else img[0, :, :] if dim == 3 else []
axes[0].imshow(img_show)
axes[1].imshow(img_show)
axes[1].scatter(x, y, color='red', marker='.')

# %% tags=[]
np.random.seed(1)
img_test = ~ps.generators.blobs(im_shape, porosity=porosity, blobiness=blobiness)
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
img_test_show = img_test if dim == 2 else img_test[0, :, :] if dim == 3 else []
axes.imshow(img_test_show)
print(f'porosity: { helper.image_porosity(img_test) }')

# %% tags=[]
df_test = helper.dataframe_from_image(img_test)
df_test.head()

# %% tags=[]
XX = df_test[['leftLength', 'leftIsSolid', 'topLength', 'topIsSolid']]
YY = df_test[['isSolid']]

y = [y for y, x in YY[YY.isSolid != skl_log_reg.predict(XX)].index]
x = [x for y, x in YY[YY.isSolid != skl_log_reg.predict(XX)].index]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
img_test_show = img_test if dim == 2 else img_test[0, :, :] if dim == 3 else []
axes[0].imshow(img_test_show)
axes[1].imshow(img_test_show)
axes[1].scatter(x, y, color='red', marker='.')

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
f = open('/Users/grimax/Desktop/log.txt', 'w')
f.close()

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

# LogReg = skl_log_reg
LogReg = grid_search_CV.best_estimator_

for y in y_grid[1:]:
    for x in x_grid[1:]:
        leftLength = new_img_eds[y, x - 1][1]
        leftIsSolid = new_img[y, x - 1]
        topLength = new_img_eds[y - 1, x][0]
        topIsSolid = new_img[y - 1, x]
        prediction = LogReg.predict_proba([[leftLength, leftIsSolid, topLength, topIsSolid]])[0][1]
#         prediction = grid_search_CV.best_estimator_.predict([[leftLength, leftIsSolid, topLength, topIsSolid]])
        result = calc_result(prediction)
        new_img_eds[y, x] = np.array([
            topLength + 1 if result == topIsSolid else 1, 
            leftLength + 1 if result == leftIsSolid else 1
        ])
        new_img[y, x] = result
        f = open('/Users/grimax/Desktop/log.txt', 'a')
        f.write(f'y: {y}, x: {x}\n')
        f.write(f'leftLength: {leftLength}\n')
        f.write(f'leftIsSolid: {leftIsSolid}\n')
        f.write(f'topLength: {topLength}\n')
        f.write(f'topIsSolid: {topIsSolid}\n')
        f.write(f'prediction: {prediction}\n')
        f.write(f'new_img_eds value: {new_img_eds[y, x]}\n')
        f.write(f'result: {result}\n')
        f.write(f'\n')

        f.close()

new_img.shape

# %% tags=[]
fig, axes = plt.subplots(1, 1, figsize=(15, 15))
axes.imshow(new_img[:, :])
print(f'porosity: { helper.image_porosity(new_img) }')

# %% tags=[]
pd.crosstab(dff.isSolid, dff.leftIsSolid).plot(kind='bar')

# %% tags=[]
corr = dff.corr()
corr.style.background_gradient(axis=None)
# plt.matshow(corr)

# %% tags=[]
dff[dff.isSolid == True].leftLength.hist()

# %% tags=[]
grid_search_CV.best_estimator_.predict_proba([[160, 0, 160, 0]])

# %% tags=[]
dff[(dff.isSolid == False) & (dff.leftIsSolid == True)].leftLength.hist()

# %% tags=[]
plot_range = np.arange(-300, 300)
plt.plot(plot_range, [grid_search_CV.best_estimator_.predict_proba([[i, 0, i, 0]])[0][1] for i in plot_range])

# %% tags=[]
dff1 = dff.copy()
dff1[['leftLengthSolid']] = (dff1.leftLength.values * dff1.leftIsSolid.values).reshape(-1, 1)
dff1[['leftLengthVoid']] = (dff1.leftLength.values * (1 - dff1.leftIsSolid.values)).reshape(-1, 1)
dff1[['topLengthSolid']] = (dff1.topLength.values * dff1.topIsSolid.values).reshape(-1, 1)
dff1[['topLengthVoid']] = (dff1.topLength.values * (1 - dff1.topIsSolid.values)).reshape(-1, 1)
dff1

# %% tags=[]
dff2 = dff1.copy()
del dff2['leftLength']
del dff2['leftIsSolid']
del dff2['topLength']
del dff2['topIsSolid']
# dff2['leftLength'] = dff2.leftLengthSolid - dff2.leftLengthVoid
# dff2['topLength'] = dff2.topLengthSolid - dff2.topLengthVoid
# del dff2['leftLengthSolid']
# del dff2['leftLengthVoid']
# del dff2['topLengthSolid']
# del dff2['topLengthVoid']
dff2

# %% tags=[]
corr = dff2.corr()
corr.style.background_gradient(axis=None)

# %% tags=[]
df_test = dff[['isSolid', 'leftLength', 'leftIsSolid']]
df_test[(df_test.leftIsSolid == False) & (df_test.isSolid == False)].size

# %% tags=[]
corr = df_test.corr()
corr.style.background_gradient(axis=None)

# %% tags=[]
# pd.crosstab(df_test[df_test.leftIsSolid == False].leftLength, df_test.isSolid).plot()
df_test[df_test.isSolid == False].leftLength.hist()

# %% tags=[]
X = df_test[['leftLength', 'leftIsSolid']]
Y = df_test[['isSolid']]
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
plot_range = np.arange(-200, 300)
plt.plot(plot_range, [skl_log_reg.predict_proba([[i, 0]])[0] for i in plot_range])

# %% tags=[]
parameters = {'C': [1e-07, 1e-06, 1e-05, 1e-04, .001, .01, .1, 1, 10, 100], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
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
plot_range = np.arange(-200, 300)
plt.plot(plot_range, [grid_search_CV.best_estimator_.predict_proba([[i, 1]])[0][1] for i in plot_range])

# %% tags=[]
# f = open('/Users/grimax/Desktop/log.txt', 'w')
# f.close()

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

# LogReg = skl_log_reg
LogReg = grid_search_CV.best_estimator_

for y in y_grid[1:]:
    for x in x_grid[1:]:
        leftIsSolid = new_img[y, x - 1]
        topIsSolid = new_img[y - 1, x]
        leftLength = new_img_eds[y, x - 1][1]
        topLength = new_img_eds[y - 1, x][0]
        lestLengthSolid = leftLength * leftIsSolid
        lestLengthVoid = leftLength * (not leftIsSolid)
        topLengthSolid = topLength * topIsSolid
        topLengthVoid = topLength * (not topIsSolid)
#         prediction = LogReg.predict_proba([[lestLengthSolid, lestLengthVoid, topLengthSolid, topLengthVoid]])[0][1]
        prediction = LogReg.predict([[lestLengthSolid, lestLengthVoid, topLengthSolid, topLengthVoid]])
        result = calc_result(prediction)
        new_img_eds[y, x] = np.array([
            topLength + 1 if result == topIsSolid else 1, 
            leftLength + 1 if result == leftIsSolid else 1
        ])
        new_img[y, x] = result
#         f = open('/Users/grimax/Desktop/log.txt', 'a')
#         f.write(f'y: {y}, x: {x}\n')
#         f.write(f'leftLength: {leftLength}\n')
#         f.write(f'leftIsSolid: {leftIsSolid}\n')
#         f.write(f'topLength: {topLength}\n')
#         f.write(f'topIsSolid: {topIsSolid}\n')
#         f.write(f'prediction: {prediction}\n')
#         f.write(f'new_img_eds value: {new_img_eds[y, x]}\n')
#         f.write(f'result: {result}\n')
#         f.write(f'\n')

#         f.close()

fig, axes = plt.subplots(1, 1, figsize=(15, 15))
axes.imshow(new_img[:, :])
print(f'porosity: { helper.image_porosity(new_img) }')

# %% tags=[]
dff2[(dff2.leftLengthSolid > 30) & (dff2.topLengthSolid > 30)]

# %%
