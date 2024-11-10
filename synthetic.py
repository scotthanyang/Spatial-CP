# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:41:21 2024

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
np.random.seed(40)

###### generate data
### hyperparameters
n_data = 5000
grid_size = 100
x_edges = np.linspace(0, 1, grid_size + 1)
y_edges = np.linspace(0, 1, grid_size + 1)
x_grid, y_grid = np.meshgrid(x_edges, y_edges)
s_grid = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

### randomly generate s and X(s) (s is uniformly over the unit grid [0,1]*[0,1])
s, Xs = syn_gp(n_data = n_data, grid_size=grid_size)
eps = syn_gp(s=s, grid_size=grid_size)
X_grid = Xs[n_data:]
eps_grid = eps[n_data:] 
Xs = Xs[:n_data]
eps = eps[:n_data]

### select scenerio
sce=1
Ys = gen_Ys(s=s, Xs=Xs, eps=eps, scenerio=sce)
Y_grid = gen_Ys(s=s_grid, Xs=X_grid, eps=eps_grid, scenerio=sce)

# Split into training and testing sets
Xs_train, Xs_test, Ys_train, Ys_test, s_train, s_test, eps_train, eps_test = train_test_split(
    Xs, Ys, s, eps, test_size=0.2, random_state=60
)

# Split into training and calibration sets
Xs_train, Xs_cal, Ys_train, Ys_cal, s_train, s_cal, eps_train, eps_cal = train_test_split(
    Xs_train, Ys_train, s_train, eps_train, test_size=0.5, random_state=80
)


### train the prediction model
# Train the prediction method on the training set
k = 5
knn = KNeighborsRegressor(n_neighbors=k)
Xc_train = np.concatenate([s_train,Xs_train.reshape((-1,1))],axis=1)
knn.fit(Xc_train, Ys_train)


### compute the residuals on calibration set
# Predict on the calibration set
Xc_cal = np.concatenate([s_cal,Xs_cal.reshape((-1,1))],axis=1)
Ys_cal_pred = knn.predict(Xc_cal)
res_cal = Ys_cal - Ys_cal_pred
# Predict on the test set
Xc_test = np.concatenate([s_test,Xs_test.reshape((-1,1))],axis=1)
Ys_test_pred = knn.predict(Xc_test)
res_test = Ys_test - Ys_test_pred

### compute the prediction interval on test set
nn = NearestNeighbors(n_neighbors=51)
nn.fit(s_cal)

# Find the k-nearest neighbors in the calibration set for each point in the test set
Xc_test = np.concatenate([s_test,Xs_test.reshape((-1,1))],axis=1)
distances, indices_cal = nn.kneighbors(s_cal)
distances, indices_test = nn.kneighbors(s_test)

# For each test point, get the residuals of its k nearest neighbors from the calibration set
feature_cal = res_cal[indices_cal[:,1:]]
feature_test = res_cal[indices_test[:,:-1]]
res_cal_neb = res_cal[indices_cal[:,1:6]]
res_test_neb = res_cal[indices_test[:,:5]]

### Proposed method: LSCP
# Train quantile regression on calibration data
qrf = RandomForestQuantileRegressor(n_estimators=30, max_depth=5, q=[0.05, 0.95],n_jobs=-1)
qrf.fit(feature_cal, res_cal)
# predict interval on test data
res_test_5,  res_test_95 = qrf.predict(feature_test)
# compute accuracy
in_interval_LSCP = (res_test >= res_test_5) & (res_test <= res_test_95)
accuracy_LSCP = np.mean(in_interval_LSCP)
print("LSCP average width:",np.mean(res_test_95-res_test_5),", accuracy:", accuracy_LSCP)

### EnbPI
# Compute the empirical quantile in neighborhood
qn5, qn95 = np.quantile(feature_test, q=[0.05,0.95],axis=1)
in_interval_EnbPI = (res_test >= qn5) & (res_test <= qn95)
accuracy_EnbPI = np.mean(in_interval_EnbPI)
print("EnbPI average width:",np.mean(qn95-qn5),", accuracy:", accuracy_EnbPI)

### Global spatial CP
# Compute the global empirical quantile
#res_cal1 = res_cal/np.std(res_cal_neb,axis=1)
#feature_cal1 = res_cal1[indices_cal[:,1:]]
#feature_test1 = res_cal1[indices_test[:,:-1]]
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
sc_cal = Xc_cal / np.linalg.norm(Xc_cal, axis=0, keepdims=True)
bd1 = select_bd(feature_cal, res_cal, sc_cal, rg, neighbors="all")
# compute interval
nn1 = NearestNeighbors(n_neighbors=s_cal.shape[0])
nn1.fit(sc_cal)
Xc_test = np.concatenate([s_test,Xs_test.reshape((-1,1))],axis=1)
sc_test = Xc_test / np.linalg.norm(Xc_test, axis=0, keepdims=True)
distances, indices_test = nn1.kneighbors(sc_test)
feature_test = res_cal[indices_test[:,:-1]]
weights = np.exp(-distances[:,:-1]**2/(2*bd1**2))
ql5, ql95 = weighted_quantile(feature_test, [0.05, 0.95], weights=weights)
in_interval_LCP = (res_test >= ql5) & (res_test <= ql95)
accuracy_LCP = np.mean(in_interval_LCP)
print("LCP average width:",np.mean(ql95-ql5),", accuracy:", accuracy_LCP,", select bandwidth=", bd1)



### plot the interval width over the unit square for each method
sc_grid = np.concatenate([s_grid,X_grid.reshape((-1,1))],axis=1)
Yg_pred = knn.predict(sc_grid)
res_grid = Y_grid - Yg_pred
Ys_cal_pred = knn.predict(Xc_cal)
res_cal = Ys_cal - Ys_cal_pred

sg_grid = np.concatenate([s_grid,X_grid.reshape((-1,1))],axis=1)
sg_grid = sg_grid / np.linalg.norm(sg_grid, axis=0, keepdims=True)
distances, indices_grid = nn.kneighbors(s_grid)
distances1, indices_grid1 = nn1.kneighbors(sg_grid)
feature_grid = res_cal[indices_grid[:,:-1]]
feature_grid1 = res_cal[indices_grid1[:,:-1]]
# residual plot
fig, ax = plt.subplots()
im=ax.imshow(np.abs(res_grid).reshape((grid_size+1,-1)))
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('interval length', rotation=270, labelpad=15)
plt.axis('off')
plt.show()

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
qg5, qg95 = np.quantile(feature_grid, q=[0.05,0.95],axis=1)
len_grid_enbpi = (qg95-qg5).reshape((grid_size+1,-1))
fig, ax = plt.subplots()
im=ax.imshow(len_grid_enbpi)
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
weights = np.exp(-distances[:,:-1]**2/(2*bd**2))
qs5, qs95 = weighted_quantile(feature_grid, [0.05, 0.95], weights=weights)
fig, ax = plt.subplots()
im=ax.imshow((qs95-qs5).reshape((grid_size+1,-1)))
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('interval length', rotation=270, labelpad=15)
plt.axis('off')
plt.show()
# LCP
weights = np.exp(-distances1[:,:-1]**2/(2*bd1**2))
ql5, ql95 = weighted_quantile(feature_grid1, [0.05, 0.95], weights=weights)
fig, ax = plt.subplots()
im=ax.imshow((ql95-ql5).reshape((grid_size+1,-1)))
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('interval length', rotation=270, labelpad=15)
plt.axis('off')
plt.show()


# Divide the grid into 10x10 areas
n_areas = 10
area_size = grid_size // n_areas
points_per_area = area_size * area_size

# Initialize dictionaries to store counts of points within intervals for each method in each area
interval_counts = {
    'LSCP': np.zeros((n_areas, n_areas)),
    'EnbPI': np.zeros((n_areas, n_areas)),
    'GSCP': np.zeros((n_areas, n_areas)),
    'SLSCP': np.zeros((n_areas, n_areas)),
    'LCP': np.zeros((n_areas, n_areas))
}

# Reshape intervals to match grid structure
res_grid = res_grid.reshape((grid_size+1, grid_size+1))
res_grid_5, res_grid_95 = res_grid_5.reshape((grid_size+1, grid_size+1)), res_grid_95.reshape((grid_size+1, grid_size+1))
qg5, qg95 = qg5.reshape((grid_size+1, grid_size+1)), qg95.reshape((grid_size+1, grid_size+1))
qs5, qs95 = qs5.reshape((grid_size+1, grid_size+1)), qs95.reshape((grid_size+1, grid_size+1))
ql5, ql95 = ql5.reshape((grid_size+1, grid_size+1)), ql95.reshape((grid_size+1, grid_size+1))

# Loop over each area and count points within each interval
for i in range(n_areas):
    for j in range(n_areas):
        # Define area boundaries
        x_start, x_end = i * area_size, (i + 1) * area_size
        y_start, y_end = j * area_size, (j + 1) * area_size

        # Extract the subgrid for the current area
        area_res_grid = res_grid[x_start:x_end, y_start:y_end]

        # LSCP interval counts
        area_res_grid_5 = res_grid_5[x_start:x_end, y_start:y_end]
        area_res_grid_95 = res_grid_95[x_start:x_end, y_start:y_end]
        interval_counts['LSCP'][i, j] = np.sum((area_res_grid >= area_res_grid_5) & (area_res_grid <= area_res_grid_95))/points_per_area 

        # EnbPI interval counts
        area_qg5 = qg5[x_start:x_end, y_start:y_end]
        area_qg95 = qg95[x_start:x_end, y_start:y_end]
        interval_counts['EnbPI'][i, j] = np.sum((area_res_grid >= area_qg5) & (area_res_grid <= area_qg95))/points_per_area 
        # GSCP interval counts
        interval_counts['GSCP'][i, j] = np.sum((area_res_grid >= q5) & (area_res_grid <= q95))/points_per_area

        # SLSCP interval counts
        area_qs5 = qs5[x_start:x_end, y_start:y_end]
        area_qs95 = qs95[x_start:x_end, y_start:y_end]
        interval_counts['SLSCP'][i, j] = np.sum((area_res_grid >= area_qs5) & (area_res_grid <= area_qs95))/points_per_area

        # LCP interval counts
        area_ql5 = ql5[x_start:x_end, y_start:y_end]
        area_ql95 = ql95[x_start:x_end, y_start:y_end]
        interval_counts['LCP'][i, j] = np.sum((area_res_grid >= area_ql5) & (area_res_grid <= area_ql95))/points_per_area

interval_widths = {
    "LSCP": len_grid_lscp,
    "EnbPI": len_grid_enbpi,
    "GSCP": np.full((grid_size+1, grid_size+1), q95-q5),  # GSCP has constant interval width
    "SLSCP": (qs95-qs5).reshape((grid_size+1, -1)),
    "LCP": (ql95-ql5).reshape((grid_size+1, -1))
}

area_size = 10
area_widths = {method: [] for method in interval_widths.keys()}

# Compute average interval width in each 10x10 area
for method, width_grid in interval_widths.items():
    for i in range(0, grid_size+1, area_size):
        for j in range(0, grid_size+1, area_size):
            # Select the area (10x10 subarray)
            area = width_grid[i:i+area_size, j:j+area_size]
            # Compute average width and store it
            average_width = np.mean(area)
            area_widths[method].append(average_width)


# Display results
data_cov = [interval_counts['LSCP'].reshape((-1)), interval_counts['EnbPI'].reshape((-1)), interval_counts['GSCP'].reshape((-1)), interval_counts['SLSCP'].reshape((-1)),interval_counts['LCP'].reshape((-1))]
draw_violin(data_cov, 'coverage')
    
data_width = [area_widths["LSCP"], area_widths["EnbPI"], area_widths["GSCP"], area_widths["SLSCP"], area_widths["LCP"]]
draw_violin(data_width, 'width')
