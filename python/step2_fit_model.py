#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 22:38:11 2025

@author: jiancao
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read data
print('\n\nReading processed data......')
df = pd.read_pickle('../data/df_processed.pkl')
print(df.head())

# Prepare train/test sets
print('\n\nPreparing train/test sets......')
# Define outcome (Y), treatment (T), and covariates (X)
Y = df['re78'].values       # outcome variable: earnings in 1978
T = df['treat'].values      # treatment indicator
# Use remaining columns as covariates
X = df.drop(columns=['treat', 're78']).values

# Split into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test, T_train, T_test = train_test_split(
    X, Y, T, test_size=0.2, random_state=1234
)
import pickle
with open('../data/X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)
with open('../data/Y_test.pkl', 'wb') as f:
    pickle.dump(Y_test, f)


# Grid search
print('\n\nGrid searching the best model......')
param_grid = {
    'n_estimators': [300, 500],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2']
}

best_model = None
best_params = None
best_cate_std = np.inf

from sklearn.model_selection import ParameterGrid
from econml.dml import CausalForestDML
# For causal inference, one approach is to choose the hyperparameters that provide stable (low-variance) CATE estimates.
# Here, as a demo, we use the standard deviation of the estimated conditional average Treatment Effects (CATE)
# on the test set as a proxy (lower variance can indicate more stable estimates).

for params in ParameterGrid(param_grid):
    model = CausalForestDML(
        model_y='forest',
        model_t='forest',
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        max_features=params['max_features'],
        random_state=1234,
        verbose=0
    )
    model.fit(Y_train, T_train, X=X_train)
    # Estimate the CATE on the test set
    cate_test = model.effect(X_test)
    cate_std = np.std(cate_test)
    print(f"Params: {params}, CATE std: {cate_std:.2f}")

    if cate_std < best_cate_std:
        best_cate_std = cate_std
        best_params = params
        best_model = model

print("\nBest Hyperparameters:", best_params)

# Save the best model
with open('../data/best_causal_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print('\n\nBest model saved......')





