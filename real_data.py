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
nmdf=pd.read_csv('C:/Users/scott/Desktop/Network Measurement Project/data/weight_NM.csv')
nmdf=pd.read_csv('C:/Users/scott/Desktop/Network Measurement Project/data/weight_GA10ds.csv')

### randomly generate s and X(s) (s is uniformly over the unit grid [0,1]*[0,1])
s = np.array(nmdf[["X","Y"]])
Xs = s
Ys = np.array(nmdf.Score)

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
# Train the prediction method on the training set
k = 320
knn = KNeighborsRegressor(n_neighbors=k)
knn.fit(s_train, Ys_train)


### compute the residuals on calibration set
# Predict on the calibration set
Ys_cal_pred = knn.predict(s_cal)
res_cal = Ys_cal - Ys_cal_pred
# Predict on the test set
Ys_test_pred = knn.predict(s_test)
res_test = Ys_test - Ys_test_pred

### compute the prediction interval on test set
nn = NearestNeighbors(n_neighbors=4)
nn.fit(s_cal)

# Find the k-nearest neighbors in the calibration set for each point in the test set
distances, indices_cal = nn.kneighbors(s_cal)
distances, indices_test = nn.kneighbors(s_test)

# For each test point, get the residuals of its k nearest neighbors from the calibration set
feature_cal = res_cal[indices_cal[:,1:]]
feature_test = res_cal[indices_test[:,:-1]]


### Proposed method: LSCP
# Train quantile regression on calibration data
qrf = RandomForestQuantileRegressor(n_estimators=30, max_depth=10, q=[0.05, 0.95],n_jobs=-1)
qrf.fit(feature_cal, res_cal)
# predict interval on test data
res_test_5,  res_test_95 = qrf.predict(feature_test)
# compute accuracy
in_interval_LSCP = (res_test >= res_test_5) & (res_test <= res_test_95)
accuracy_LSCP = np.mean(in_interval_LSCP)
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
print("EnbPI average width:",np.mean(qn95-qn5),", accuracy:", accuracy_EnbPI)

### Global spatial CP
# Compute the global empirical quantile
q5, q95 = np.quantile(res_cal, [0.05, 0.95])
in_interval_GSCP = (res_test >= q5) & (res_test <= q95)
accuracy_GSCP = np.mean(in_interval_GSCP)
print("GSCP average width:",np.mean(q95-q5),", accuracy:", accuracy_GSCP)

### smoothed Local spatial CP (SLSCP)
# select bandwidth
rg = [0.01*i for i in range(1,51)]
bd = select_bd(feature_cal, res_cal, s_cal, rg)
weights = np.exp(-distances[:,:-1]**2/(2*bd**2))
qs5, qs95 = weighted_quantile(feature_test, [0.05, 0.95], weights=weights)
in_interval_SLSCP = (res_test >= qs5) & (res_test <= qs95)
accuracy_SLSCP = np.mean(in_interval_SLSCP)
print("SLSCP average width:",np.mean(qs95-qs5),", accuracy:", accuracy_SLSCP,", select bandwidth=", bd)

### localized CP (LCP)
# select bandwidth
rg = [0.01*i for i in range(1,51)]
#bd1 = select_bd(s_cal, res_cal, s_cal, rg, neighbors="all")
bd1 = 0.5
# compute interval
nn2 = NearestNeighbors(n_neighbors=s_cal.shape[0])
nn2.fit(s_cal)
distances, indices_test = nn2.kneighbors(s_test)
feature_test = res_cal[indices_test[:,:-1]]
weights = np.exp(-distances[:,:-1]**2/(2*bd1**2))
ql5, ql95 = weighted_quantile(feature_test, [0.05, 0.95], weights=weights)
in_interval_LCP = (res_test >= ql5) & (res_test <= ql95)
accuracy_LCP = np.mean(in_interval_LCP)
print("LCP average width:",np.mean(ql95-ql5),", accuracy:", accuracy_LCP,", select bandwidth=", bd1)

### plot the intervals for each method
plt.figure(figsize=(10, 6))
# Plot LSCP interval
plt.fill_between(range(res_test_5.shape[0]), Ys_test_pred+res_test_5, Ys_test_pred+res_test_95, color='lightblue', alpha=0.5, label="LSCP Interval")
plt.plot(res_test,  label="Test Data", color="blue")
# Plot EnbPI interval
plt.fill_between(range(res_test_5.shape[0]), Ys_test_pred+qn5, Ys_test_pred+qn95, color='lightgreen', alpha=0.3, label="EnbPI Interval")
# Plot GSCP interval (global constant)
plt.fill_between(range(res_test_5.shape[0]), Ys_test_pred+q5, Ys_test_pred+q95, color='grey', alpha=0.3, label="GSCP Interval")
plt.title('Prediction Interval')
plt.xlabel('Test Data Index')
plt.ylabel('Prediction / Interval')
plt.legend()
plt.show()


### plot the interval width over the unit square for each method
grid_size = 100
x_edges = np.linspace(0, 1, grid_size + 1)*lx+np.min(nmdf.X)
y_edges = np.linspace(0, 1, grid_size + 1)*ly+np.min(nmdf.Y)
x_grid, y_grid = np.meshgrid(x_edges, y_edges)
s_grid = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
sg_cal = s_grid
distances, indices_grid = nn.kneighbors(s_grid)
distances1, indices_grid1 = nn1.kneighbors(s_grid)
distances2, indices_grid2 = nn2.kneighbors(s_grid)
feature_grid = res_cal[indices_grid[:,:-1]]
feature_grid1 = res_cal[indices_grid1[:,:-1]]
feature_grid2 = res_cal[indices_grid2[:,:-1]]
# LSCP
res_grid_5,  res_grid_95 = qrf.predict(feature_grid)
len_grid_lscp = (res_grid_95-res_grid_5).reshape((grid_size+1,-1))
fig, ax = plt.subplots()
im=ax.imshow(len_grid_lscp)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('interval length', rotation=270, labelpad=15)
plt.axis('off')
plt.show()
# EnbPI
qg5, qg95 = np.quantile(feature_grid1, q=[0.05,0.95],axis=1)
len_grid_lscp = (qg95-qg5).reshape((grid_size+1,-1))
fig, ax = plt.subplots()
im=ax.imshow(len_grid_lscp)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('interval length', rotation=270, labelpad=15)
plt.axis('off')
plt.show()
# GSCP
q5, q95 = np.quantile(res_cal, [0.05, 0.95])
fig, ax = plt.subplots()
im=ax.imshow(np.full((grid_size+1,grid_size+1),q95-q5))
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('interval length', rotation=270, labelpad=15)
plt.axis('off')
plt.show()
# SLSCP
weights = np.exp(-distances1[:,:-1]**2/(2*bd**2))
qs5, qs95 = weighted_quantile(feature_grid1, [0.05, 0.95], weights=weights)
fig, ax = plt.subplots()
im=ax.imshow((qs95-qs5).reshape((grid_size+1,-1)))
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('interval length', rotation=270, labelpad=15)
plt.axis('off')
plt.show()
# LCP
weights = np.exp(-distances2[:,:-1]**2/(2*bd1**2))
ql5, ql95 = weighted_quantile(feature_grid2, [0.05, 0.95], weights=weights)
fig, ax = plt.subplots()
im=ax.imshow((ql95-ql5).reshape((grid_size+1,-1)))
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('interval length', rotation=270, labelpad=15)
plt.axis('off')
plt.show()