#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 22:51:49 2025

@author: jiancao
"""

import matplotlib.pyplot as plt

# Read model
import pickle
print('\n\nReading the best model......')
with open('../data/best_causal_model.pkl', 'rb') as f:
    best_model = pickle.load(f)
with open('../data/X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open('../data/Y_test.pkl', 'rb') as f:
    Y_test = pickle.load(f)

# Validate the model
# Estimate the Average Treatment Effect (ATE) and its confidence interval on the test set.
ate_point = best_model.ate(X_test)
ate_lb, ate_ub = best_model.ate_interval(X_test, alpha=0.05)
print(f"\nEstimated ATE: {ate_point:.2f}")
print(f"95% Confidence Interval for ATE: ({ate_lb:.2f}, {ate_ub:.2f})")

# Also, compute CTE for each test observation
cate_pred = best_model.effect(X_test)


# Plot 1: Histogram of Estimated CTE
plt.figure(figsize=(8, 5))
plt.hist(cate_pred, bins=30, edgecolor='k', alpha=0.7)
plt.title("Distribution of Estimated Conditional Treatment Effects (CATE)")
plt.xlabel("Estimated Treatment Effect")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("../figures/cate_distribution.pdf", format="pdf", bbox_inches="tight")
print('\n\nFigure cate_distribution saved!')

# Plot 2: Scatter Plot of CTE vs. a Covariate (e.g., Age)
# Assuming the first column in X corresponds to age
age_test = X_test[:, 0]
plt.figure(figsize=(8, 5))
plt.scatter(age_test, cate_pred, alpha=0.6, edgecolor='k')
plt.title("Estimated Treatment Effect vs. Age")
plt.xlabel("Age")
plt.ylabel("Estimated Treatment Effect")
plt.grid(True)
plt.savefig("../figures/cate_scatter.pdf", format="pdf", bbox_inches="tight")
print('\n\nFigure cate_scatter saved!')

# Plot 3: Visualize the ATE with its Confidence Interval
plt.figure(figsize=(6, 4))
plt.errorbar(1, ate_point, yerr=[[ate_point - ate_lb], [ate_ub - ate_point]], fmt='o', color='red', capsize=5)
plt.xlim(0.5, 1.5)
plt.xticks([1], ['ATE'])
plt.title("Average Treatment Effect with 95% Confidence Interval")
plt.ylabel("Treatment Effect")
plt.grid(True)
plt.savefig("../figures/ate.pdf", format="pdf", bbox_inches="tight")
print('\n\nFigure ate saved!')

print('\n\nAll done!')
