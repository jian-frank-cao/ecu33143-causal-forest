#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 22:25:06 2025

@author: jiancao
"""

import pandas as pd

# Read data
print('\n\nReading data......')
df = pd.read_csv('../data/lalonde_data.csv', index_col=0)
print(df.shape)
print(df.head())

# Describe data
print('\n\nDescribe data......')
print(df.describe())

# Check missing values
print('\n\nCheck missing values......')
print("Missing values per column:\n", df.isnull().sum())

# Check column types
print('\n\nCheck column types......')
print(df.dtypes)

# Save processed data
df.to_pickle('../data/df_processed.pkl')
print('\n\nProcessed data saved!')