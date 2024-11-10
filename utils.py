# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:54:13 2024

@author: scott
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import warnings 
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold

# synthetic Gaussian Process
def syn_gp(s = 0, n_data = 1000, n_points = 100, grid_size = 100):
    X = np.random.uniform(0, 1, (n_points, 2))  # Random points in 2D space (locations)
    # Define the Matérn kernel (covariance function)
    # Length scale controls the range of correlations, nu controls the smoothness
    kernel = Matern(length_scale=0.1, nu=0.7, length_scale_bounds="fixed") 
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10)
    y = np.zeros(n_points)  # Mean of the process is zero
    gp.fit(X, y)
    
    x_edges = np.linspace(0, 1, grid_size + 1)
    y_edges = np.linspace(0, 1, grid_size + 1)
    x_grid, y_grid = np.meshgrid(x_edges, y_edges)
    s_grid = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    
    if isinstance(s,int):
        X_new = np.random.uniform(0, 1, (n_data, 2)) 
        X_cont = np.concatenate([X_new, s_grid], axis=0)
        y_mean_new, y_cov_new = gp.predict(X_cont, return_cov=True)
        y_samples_new = np.random.multivariate_normal(y_mean_new, y_cov_new)
        return X_new, y_samples_new
    else:
        X_new = s
        X_cont = np.concatenate([s, s_grid], axis=0)
        y_mean_new, y_cov_new = gp.predict(X_cont, return_cov=True)
        y_samples_new = np.random.multivariate_normal(y_mean_new, y_cov_new)
        return y_samples_new
    
# Generate Y(s) according to different scenerios
def gen_Ys(s, Xs, eps, scenerio=1):
    # Y(s) = X(s) + eps(s)
    if scenerio == 1:
        return Xs + eps
    if scenerio == 2:
        return Xs*np.abs(eps)
    if scenerio == 3:
        return Xs + np.sin(np.linalg.norm(s,axis=1))*eps
    
# kernel regression method
def kern(x,y,sigma):
    x = x.reshape(2,)
    y = y.reshape(2,)
    return (1/(2*np.pi*sigma**2))*np.exp(-((x[0]-y[0])**2+(x[1]-y[1])**2)/(2*sigma**2))

def kernel_pred(X, neigh, c=0.01, case="fixed"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mbps = []
        kernel = []
        for ind in range(len(neigh.X)):
            nrow = neigh.iloc[ind]
            neighb = np.array([nrow.X,nrow.Y])
            mbps.append(nrow.Score)        
            if case == "fixed":
                kernel.append(kern(X,neighb,c))
        if sum(kernel) == 0:
            kernel = np.zeros((len(neigh.X),))
            kernel[0] = 1
        est = (np.array(mbps).reshape(1,-1)@np.array(kernel).reshape(-1,1))[0,0]/sum(kernel)
    return est

def kernel_pred_set(test, train, c=0.01, case="fixed", k=5):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(train[['X','Y']])
        est = []
        for i in range(len(test.X)):
            R_point = np.array([test.X.iloc[i], test.Y.iloc[i]]).reshape(1,-1)
            distances, indices = nbrs.kneighbors(R_point)
            distances = distances[0]
            indices = indices[0]
            if distances[0] == 0:
                distances = distances[1:]
                indices = indices[1:]
            est.append(kernel_pred(R_point, train.iloc[indices],c=c,case="fixed"))
    return est


# weighted quantile
def weighted_quantile(data, quantiles, weights=None):
    data = np.asarray(data)
    quantiles = np.asarray(quantiles)
    
    if weights is None:
        weights = np.ones(data.shape)
    else:
        weights = np.asarray(weights)
        if weights.ndim == 1:
            weights = np.tile(weights, (data.shape[0], 1))  # Repeat weights for each row
    
    # Initialize an array to store the quantiles for each row
    result = np.zeros((data.shape[0], len(quantiles)))
    
    # Compute weighted quantiles for each row
    for i in range(data.shape[0]):
        # Sort data and weights for this row
        sorter = np.argsort(data[i])
        data_sorted = data[i, sorter]
        weights_sorted = weights[i, sorter]
        
        # Compute cumulative sum of weights
        cumulative_weights = np.cumsum(weights_sorted)
        cumulative_weights /= cumulative_weights[-1]  # Normalize to 1
        
        # Compute quantiles for this row
        result[i, :] = np.interp(quantiles, cumulative_weights, data_sorted)
    
    return result[:,0], result[:,1]


def select_bd(X, Y, s, rg, fold=5, neighbors=101):
    kf = KFold(n_splits=fold)
    # Store the errors for each fold
    min_l = 10**6
    min_bd = 0

    # Perform 5-fold cross-validation
    for bd in rg:
        l = 0
        accuracy = 0
        best_acc = 0
        for train_index, test_index in kf.split(X):
            # Split the data
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            s_train, s_test = s[train_index], s[test_index]
            
            ### compute the prediction interval on test set
            if neighbors == "all":
                nn = NearestNeighbors(n_neighbors=X_train.shape[0])
                nn.fit(s_train)
            else:                
                nn = NearestNeighbors(n_neighbors=neighbors)
                nn.fit(s_train)

            # Find the k-nearest neighbors in the calibration set for each point in the test set
            distances, indices_train = nn.kneighbors(s_train)
            distances, indices_test = nn.kneighbors(s_test)

            # For each test point, get the residuals of its k nearest neighbors from the calibration set
            feature_train = Y[indices_train[:,1:]]
            feature_test = Y[indices_test[:,:-1]]
            
            # Train the model
            weights = np.exp(-distances**2/(2*bd**2))
            qs5, qs95 = weighted_quantile(feature_test[:,:-1], [0.05, 0.95], weights=weights)
            in_interval = (Y_test >= qs5) & (Y_test <= qs95)
            accuracy += np.mean(in_interval)/fold
            l += np.mean(qs95-qs5)/fold
        if accuracy >= 0.9 and best_acc < 0.9:
            min_l = l
            min_bd = bd
        if accuracy >= 0.9 and l < min_l:
            min_l = l
            min_bd = bd
        if accuracy < 0.9 and 0.9-accuracy < 0.05 and best_acc < 0.9 and l < min_l:
            min_l = l
            min_bd = bd
                
    return min_bd

def draw_violin(data, cat='coverage', tag = 0.9):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightgrey', 'palegoldenrod']
    violin_parts = ax.violinplot(data, showmeans=True, showmedians=False, showextrema=False)

    # Set color for each violin plot
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')  # Optional: Set edge color
        pc.set_alpha(0.7)  # Optional: Set transparency

    # Set x-tick labels for the five methods
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(['LSCP', 'EnbPI', 'GSCP', 'SLSCP', 'LCP'])
    ax.set_xlabel('Method')

    # Set y-axis label based on category
    if cat == 'coverage':
        ax.set_ylabel('Coverage')
        # Add horizontal dashed line at 0.9 for coverage
        ax.axhline(y=tag, color='red', linestyle='--', linewidth=1, label='Target Coverage (0.9)')
    elif cat == 'width':
        ax.set_ylabel('Width')

    # Show legend if there's a line for coverage
    if cat == 'coverage':
        ax.legend()

    plt.show()