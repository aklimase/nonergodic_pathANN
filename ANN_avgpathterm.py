#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 16:38:02 2020

@author: aklimasewski

grids data targets, takes average, and saves files
calculated average path term per period as the path length per cell x average target per cell 
uses average path term per period as new input features

"""
import sys
import os
sys.path.append(os.path.abspath('/Users/aklimase/Documents/nonergodic_ANN'))
from preprocessing import transform_dip, readindata, transform_data
from model_plots import plot_resid, obs_pre, plot_outputs
from grid_funcs import create_grid, grid_data, mean_grid_save, gridded_plots, avgpath_resid
from build_ANN import create_ANN
import shapely
import shapely.geometry
import geopy
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib as mpl
from sklearn.preprocessing import Normalizer
#grid up residuals
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
from keras import layers
from keras import optimizers

topdir = '/Users/aklimasewski/Documents/'

folder_path = topdir + 'models/gridtargets_ANN/modelgriddedtargets1/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
'''
grids data targets per period, normalizes by path length, finds average normalized residual per grid cell
creates ANN model including the average path term per period as an input feature
comment gridding if csv files already saved
'''
period = [10,7.5,5,4,3,2,1,0.5,0.2,0.1]

df, lon, lat = create_grid(latmin=30,latmax=38,lonmin=-121,lonmax=-115,dx=1.0)

train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain= topdir + 'data/cybertrainyeti10_residfeb.csv', nametest=topdir + 'data/cybertestyeti10_residfeb.csv', n=6)

hypoR, sitelat, sitelon, evlat, evlon, target, gridded_targetsnorm_list, gridded_counts = grid_data(train_data1, train_targets1 , df=df)     
hypoR_test, sitelat_test, sitelon_test, evlat_test, evlon_test, target_test, gridded_targetsnorm_list_test, gridded_counts_test = grid_data(test_data1, test_targets1, df=df)    

gridded_mean,gridded_mean_test= mean_grid_save(gridded_targetsnorm_list,gridded_targetsnorm_list_test,gridded_counts,gridded_counts_test,df,folder_path)

gridded_plots(gridded_mean, gridded_counts, period, lat, lon, evlon, evlat, sitelon, sitelat, folder_path + 'traingrid/')
gridded_plots(gridded_mean_test, gridded_counts_test, period, lat, lon, evlon, evlat, sitelon, sitelat, folder_path = folder_path + 'testgrid/')

df = pd.read_csv(folder_path + 'griddedvalues_train.csv')
path_target_sum = avgpath_resid(df, folder_path, savename = 'train')
df = pd.read_csv(folder_path + 'griddedvalues_test.csv')
path_target_sum_test = avgpath_resid(df, folder_path, savename = 'test')

folder_pathmod = '/Users/aklimase/Documents/USGS/models/gridtargets_ANN/modelgriddedtargets1/' + 'ANN/'
if not os.path.exists(folder_pathmod):
    os.makedirs(folder_pathmod)
    
train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimase/Documents/USGS/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimase/Documents/USGS/data/cybertestyeti10_residfeb.csv', n = 12)
train_data1 = np.append(train_data1,path_target_sum, axis=1)
test_data1 = np.append(test_data1,path_target_sum_test, axis=1)
feature_names = np.append(feature_names, ['T10','T7.5','T5','T4','T3','T2','T1','T0.5','T0.2','T0.1'], axis=0)                      

transform_method = 'Norm'
x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_pathmod)
             
numlayers = 1
units= [50]
epochs = 15
resid, resid_test, pre_train, pre_test = create_ANN(x_train, y_train, x_test, y_test, feature_names, numlayers, units, epochs, transform_method, folder_pathmod)

plot_resid(resid, resid_test, folder_pathmod)

Rindex = np.where(feature_names == 'Rrup')[0][0]
predict_epistemic_allT = []
predict_epistemic_train_allT = []
plot_outputs(folder_path, pre_test, predict_epistemic_allT, pre_train, predict_epistemic_train_allT, x_train, y_train, x_test, y_test, Rindex, period, feature_names)
