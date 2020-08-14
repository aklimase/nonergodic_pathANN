#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 12:45:34 2020

@author: aklimase
"""
import gpflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from gpflow.utilities import print_summary
import sys
import os
sys.path.append(os.path.abspath('/Users/aklimase/Documents/USGS/nonergodic_ANN/'))
from preprocessing import transform_dip, readindata, transform_data, create_grid, grid_data
from model_plots import gridded_plots, obs_pre, plot_resid, plot_outputs, plot_rawinputs, gridded_plots
import random
import pandas as pd
import seaborn as sns
sns.reset_defaults()
sns.set_style('whitegrid')
sns.set_context('talk')
sns.set_context(context='talk',font_scale=0.7)


N = 10000 # number of points
D = 4  # number of input dimensions
M = 15  # number of inducing points
L = 2  # number of latent GPs
P = 2  # number of observations = output dimensions

folder_path = '/Users/aklimase/Documents/USGS/models/ANN13_VGP4/VGP4_5iter/'
# folder_path = '/Users/aklimase/Documents/USGS/models/baseANN/ANN13_VGP4/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

n=4
train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimase/Documents/USGS/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimase/Documents/USGS/data/cybertestyeti10_residfeb.csv', n = n)
#read in ANN residuals as model targets
ANNresid_train = pd.read_csv('/Users/aklimase/Documents/USGS/models/ANN13_VGP4/ANN13/'+ 'ANNresiduals_train.csv').iloc[:, 1:11]
ANNresid_test= pd.read_csv('/Users/aklimase/Documents/USGS/models/ANN13_VGP4/ANN13/'+ 'ANNresiduals_test.csv').iloc[:, 1:11]

randindex = random.sample(range(0, len(train_data1)), N)
randindextest= random.sample(range(0, len(test_data1)), N)

#randomly choose subset for quick training
x_train = train_data1[randindex]
#start with 2 periods
y_train = np.asarray(ANNresid_train)[randindex]
#randomely choose subset for quick training
x_test = test_data1[randindextest]
#start with 1 period
y_test = np.asarray(ANNresid_test)[randindextest]

y_train_target1 = train_targets1[randindex]
y_test_target1 = test_targets1[randindextest]

period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]

GMresid = np.zeros((N,10))
GMresid_test = np.zeros((N,10))
var_train = np.zeros((N,10))
var_test = np.zeros((N,10))
#loop through each period
for i in range(len(y_train[0])):
    print('GPR for period: ', period[i])
    y_traini = y_train[:,i].reshape(N,1)
    y_testi = y_test[:,i].reshape(N,1)

    #active dims is input dimensions
    k = gpflow.kernels.Matern12(active_dims=[0], lengthscales=0.1) + gpflow.kernels.Matern12(
        active_dims=[1], lengthscales=0.1)+ gpflow.kernels.Matern12(active_dims=[2], lengthscales=0.1)+ gpflow.kernels.Matern12(active_dims=[3], lengthscales=0.1)
    
    print_summary(k)
    
    # meanf = gpflow.mean_functions.Linear()

    m = gpflow.models.GPR(data=(x_train, y_traini), kernel=k, mean_function=None)
    
    print_summary(m)
    
    #optimize model parameters (variance and length scale)
    opt = gpflow.optimizers.Scipy()
    
    opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=5))#, method = 'CG')
    print_summary(m)
    # np.asarray([np.linspace(min(X[:,i]),max(X[:,i]), 100) for i in range(D)]).T  # test points must be of shape (N, D)
    
    # mean, var = m.predict_f(xx)
    mean_testi, var_testi = m.predict_f(x_test)
    mean_traini, var_traini = m.predict_f(x_train)

    #add predictions back to the targets
    GMresid[:,i] = np.asarray(mean_traini).flatten()
    GMresid_test[:,i] = np.asarray(mean_testi).flatten()
    var_train[:,i] = np.asarray(var_traini).flatten()
    var_test[:,i] = np.asarray(var_testi).flatten()
    
#test data
mean_x_test_allT = GMresid_test
predict_epistemic_allT = np.sqrt(var_test)

#training data
mean_x_train_allT = GMresid
predict_epistemic_train_allT = np.sqrt(var_train)

resid = y_train  -mean_x_train_allT
resid_test = y_test -mean_x_test_allT

# pre = predict_mean_train
# pre_test = predict_mean


n=13
x_trainANN, x_testANN, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimase/Documents/USGS/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimase/Documents/USGS/data/cybertestyeti10_residfeb.csv', n = n)
Rindex = np.where(feature_names == 'Rrup')[0][0]
x_trainANN = x_trainANN[randindex]
x_testANN = x_testANN[randindextest]

plot_resid(resid, resid_test, folder_path)
plot_outputs(folder_path, mean_x_test_allT, predict_epistemic_allT, mean_x_train_allT, predict_epistemic_train_allT, x_train, y_train, x_test, y_test, Rindex, period, feature_names)
plot_rawinputs(x_raw = x_trainANN, mean_x_allT = mean_x_train_allT, y=y_train, feature_names=feature_names, period = period, folder_path = folder_path + 'train/')
plot_rawinputs(x_raw = x_testANN, mean_x_allT = mean_x_test_allT, y=y_test, feature_names=feature_names, period = period, folder_path = folder_path + 'test/')

obs_pre(y_train, y_test, GMresid, GMresid_test, period, folder_path)

#%%
#grid the predictions
df, lon, lat = create_grid(dx = 0.1)

n=6
x_train6, x_test6, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimase/Documents/USGS/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimase/Documents/USGS/data/cybertestyeti10_residfeb.csv', n = n)
x_train6 = x_train6[randindex]
x_test6 = x_test6[randindextest]

hypoR, sitelat, sitelon, evlat, evlon, target, gridded_targetsnorm_list, gridded_counts = grid_data(x_train6, y_train, df=df, nsamples = N)     
gridded_targetsnorm_list = np.asarray(gridded_targetsnorm_list)

griddednorm_mean=np.zeros((len(gridded_targetsnorm_list),10))
for i in range(len(gridded_targetsnorm_list)):
    # for j in range(10):
    griddednorm_mean[i] = np.mean(gridded_targetsnorm_list[i],axis=0)

#find the cells with no paths (nans)
nan_ind=np.argwhere(np.isnan(griddednorm_mean)).flatten()
# set nan elements for empty array
for i in nan_ind:
    griddednorm_mean[i] = 0
    
gridded_plots(griddednorm_mean, gridded_counts, period, lat, lon, evlon, evlat, sitelon, sitelat, folder_path + 'traingrid/')

#%%
#test
hypoRtest, sitelat, sitelon, evlat, evlon, target, gridded_targetsnorm_list, gridded_counts = grid_data(x_test6, y_test, df=df, nsamples = N)     
gridded_targetsnorm_list = np.asarray(gridded_targetsnorm_list)

griddednorm_mean=np.zeros((len(gridded_targetsnorm_list),10))
for i in range(len(gridded_targetsnorm_list)):
    # for j in range(10):
    griddednorm_mean[i] = np.mean(gridded_targetsnorm_list[i],axis=0)

#find the cells with no paths (nans)
nan_ind=np.argwhere(np.isnan(griddednorm_mean)).flatten()
# set nan elements for empty array
for i in nan_ind:
    griddednorm_mean[i] = 0
    
gridded_plots(griddednorm_mean, gridded_counts, period, lat, lon, evlon, evlat, sitelon, sitelat, folder_path + 'testgrid/')

#%%
#test predictions

# period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]

# Rindex = np.where(feature_names == 'Rrup')[0][0]




# #
# ## plot
# #first period vs distance
# plt.figure(figsize=(12, 6))
# plt.plot(X[:,1], Y[:,0], "kx", mew=2, label = 'data')
# plt.plot(xx[:,1], mean[:,0], "C0", lw=2, label = 'mean prediction')
# plt.xlabel('Rhypo')
# plt.ylabel('target T=10s')
# plt.fill_between(
#     xx[:, 1],
#     mean[:,0] - 1.96 * np.sqrt(var[:, 0]),
#     mean[:,0] + 1.96 * np.sqrt(var[:, 0]),
#     color="C0",
#     alpha=0.2,
# )
# plt.legend()