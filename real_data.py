# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:06:47 2024

@author: scott
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn_quantile import RandomForestQuantileRegressor
from utils import *

# Set random seed for reproducibility
np.random.seed(30)

###### read data
nmdf=pd.read_csv('/real_data/weight_NM.csv')
#nmdf = pd.read_csv('real_data/weight_GA.csv')


### randomly generate s and X(s) (s is uniformly over the unit grid [0,1]*[0,1])
s = np.array(nmdf[["X","Y"]])
Xs = s
Ys = np.array(nmdf.average_score)

# Split into training and testing sets
Xs_train, Xs_test, Ys_train, Ys_test, s_train, s_test = train_test_split(
    Xs, Ys, s, test_size=0.2, random_state=60
)

# Split into training and calibration sets
Xs_train, Xs_cal, Ys_train, Ys_cal, s_train, s_cal = train_test_split(
    Xs_train, Ys_train, s_train, test_size=0.5, random_state=80
)

lx = np.max(nmdf.X)-np.min(nmdf.X)
ly = np.max(nmdf.Y)-np.min(nmdf.Y)

### train the prediction model
# Train the prediction method on the training setk = 10
k=8
nn_model = NearestNeighbors(n_neighbors=k)  # +2 to account for removing min and max
nn_model.fit(s_train)

def predict_with_neighbors_average(s_test, s_train, Ys_train, nn_model, k_neighbors):
    predictions = []
    distances, indices = nn_model.kneighbors(s_test)
    
    for neighbor_indices in indices:
        neighbor_scores = Ys_train[neighbor_indices[1:]]  # Exclude the test point itself
        trimmed_scores = np.sort(neighbor_scores) # Exclude the smallest and largest scores
        average_score = np.mean(trimmed_scores)
        predictions.append(average_score)
    
    return np.array(predictions)


### compute the residuals on calibration set
# Predict on the calibration set
Ys_cal_pred = predict_with_neighbors_average(s_cal, s_train, Ys_train, nn_model, k)
res_cal = Ys_cal - Ys_cal_pred
print(np.mean(np.abs(res_cal)))
# Predict on the test set
Ys_test_pred =predict_with_neighbors_average(s_test, s_train, Ys_train, nn_model, k)
res_test = Ys_test - Ys_test_pred



### compute the prediction interval on test set
nn = NearestNeighbors(n_neighbors=10)
nn.fit(s_cal)

# Find the k-nearest neighbors in the calibration set for each point in the test set
distances, indices_cal = nn.kneighbors(s_cal)
distances, indices_test = nn.kneighbors(s_test)

# For each test point, get the residuals of its k nearest neighbors from the calibration set
feature_cal = res_cal[indices_cal[:,1:]]
feature_test = res_cal[indices_test[:,:-1]]


### Proposed method: LSCP
# Train quantile regression on calibration data
qrf = RandomForestQuantileRegressor(n_estimators=30, max_depth=5, q=[0.05, 0.95],n_jobs=-1)
qrf.fit(feature_cal, res_cal)
# predict interval on test data
res_test_5,  res_test_95 = qrf.predict(feature_test)
# compute accuracy
in_interval_LSCP = (res_test >= res_test_5) & (res_test <= res_test_95)
accuracy_LSCP = np.mean(in_interval_LSCP)
width_LSCP = res_test_95 - res_test_5
print("LSCP average width:",np.mean(res_test_95-res_test_5),", accuracy:", accuracy_LSCP)

### EnbPI
# Compute the empirical quantile in neighborhood
nn1 = NearestNeighbors(n_neighbors=151)
nn1.fit(s_cal)
distances, indices_cal = nn1.kneighbors(s_cal)
distances, indices_test = nn1.kneighbors(s_test)
feature_cal = res_cal[indices_cal[:,1:]]
feature_test = res_cal[indices_test[:,:-1]]
qn5, qn95 = np.quantile(feature_test, q=[0.05,0.95],axis=1)
in_interval_EnbPI = (res_test >= qn5) & (res_test <= qn95)
accuracy_EnbPI = np.mean(in_interval_EnbPI)
width_EnbPI = qn95 - qn5
print("EnbPI average width:",np.mean(qn95-qn5),", accuracy:", accuracy_EnbPI)

### Global spatial CP
# Compute the global empirical quantile
q5, q95 = np.quantile(res_cal, [0.05, 0.95])
in_interval_GSCP = (res_test >= q5) & (res_test <= q95)
accuracy_GSCP = np.mean(in_interval_GSCP)
width_GSCP = q95 - q5
print("GSCP average width:",np.mean(q95-q5),", accuracy:", accuracy_GSCP)

### smoothed Local spatial CP (SLSCP)
# select bandwidth
rg = [0.01*i for i in range(1,51)]
#bd = select_bd(feature_cal, res_cal, s_cal, rg)
bd = 0.2
weights = np.exp(-distances[:,:-1]**2/(2*bd**2))
qs5, qs95 = weighted_quantile(feature_test, [0.05, 0.95], weights=weights)
in_interval_SLSCP = (res_test >= qs5) & (res_test <= qs95)
accuracy_SLSCP = np.mean(in_interval_SLSCP)
width_SLSCP = qs95 - qs5
print("SLSCP average width:",np.mean(qs95-qs5),", accuracy:", accuracy_SLSCP,", select bandwidth=", bd)

### localized CP (LCP)
# select bandwidth
rg = [0.01*i for i in range(1,51)]
#bd1 = select_bd(s_cal, res_cal, s_cal, rg, neighbors="all")
bd1 = 0.3
# compute interval
nn2 = NearestNeighbors(n_neighbors=s_cal.shape[0])
nn2.fit(s_cal)
distances, indices_test = nn2.kneighbors(s_test)
feature_test = res_cal[indices_test[:,:-1]]
weights = np.exp(-distances[:,:-1]**2/(2*bd1**2))
ql5, ql95 = weighted_quantile(feature_test, [0.05, 0.95], weights=weights)
in_interval_LCP = (res_test >= ql5) & (res_test <= ql95)
accuracy_LCP = np.mean(in_interval_LCP)
width_LCP = ql95 - ql5
print("LCP average width:",np.mean(ql95-ql5),", accuracy:", accuracy_LCP,", select bandwidth=", bd1)

def compute_metrics_on_grid(widths, coverages, grid_size=10):
    x_edges = np.linspace(np.min(s_test[:, 0]), np.max(s_test[:, 0]), grid_size + 1)
    y_edges = np.linspace(np.min(s_test[:, 1]), np.max(s_test[:, 1]), grid_size + 1)
    area_widths = []
    area_coverages = []

    for i in range(grid_size):
        for j in range(grid_size):
            area_mask = (
                (s_test[:, 0] >= x_edges[i]) & (s_test[:, 0] < x_edges[i + 1]) &
                (s_test[:, 1] >= y_edges[j]) & (s_test[:, 1] < y_edges[j + 1])
            )
            if np.sum(area_mask) > 0:
                area_widths.append(np.mean(widths[area_mask]))
                area_coverages.append(np.mean(coverages[area_mask]))
    return area_widths, area_coverages

# Compute metrics for each method
widths_and_coverages = {
    "LSCP": compute_metrics_on_grid(width_LSCP, in_interval_LSCP),
    "EnbPI": compute_metrics_on_grid(width_EnbPI, in_interval_EnbPI),
    "GSCP": compute_metrics_on_grid(np.full_like(res_test, width_GSCP), in_interval_GSCP),
    "SLSCP": compute_metrics_on_grid(width_SLSCP, in_interval_SLSCP),
    "LCP": compute_metrics_on_grid(width_LCP, in_interval_LCP)
}

# Prepare data for violin plots
coverage_data = [wc[1] for wc in widths_and_coverages.values()]
width_data = [wc[0] for wc in widths_and_coverages.values()]

draw_violin(coverage_data, "coverage")
draw_violin(width_data, "width")