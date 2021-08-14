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
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
import pandas as pd

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import colors

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split

# %% tags=[]
df = pd.DataFrame(columns=['Y', 'X'])
for i in np.arange(0, 100):
    df.loc[i] = pd.Series({'Y': i > 20, 'X': i})
df = df.astype(np.int32)
df

# %% tags=[]
X = df[['X']]
Y = df[['Y']]
x_train, x_test, y_train, y_test = train_test_split(X, Y)
log_reg = sm.Logit(y_train, x_train).fit()
print(log_reg.summary())
predicted_train = log_reg.predict(x_train) > .5
predicted_test = log_reg.predict(x_test) > .5
print(f'train score: {(predicted_train.to_numpy().ravel() == y_train.to_numpy().ravel()).mean()}')
print(f'test score: {(predicted_test.to_numpy().ravel() == y_test.to_numpy().ravel()).mean()}')

# %% tags=[]
skl_log_reg = LogisticRegression()
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
plot_range = np.arange(-100, 100)
plt.plot(plot_range, [skl_log_reg.predict_proba([[i]])[0][1] for i in plot_range])

# %%
