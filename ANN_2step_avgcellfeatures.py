#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:38:58 2020

@author: aklimase

2 ANNs, first 12 feature model, second gridded target residuals with features of grid midpoints (determined size of grid)
"""
import sys
import os
sys.path.append(os.path.abspath('/Users/aklimasewski/Documents/pythoncode_nonergodic_ANN'))
from preprocessing import transform_dip, readindata, transform_data
from build_ANN import create_ANN
from model_plots import obs_pre, plot_resid, plot_outputs
from grid_funcs import gridded_plots, create_grid, grid_data, mean_grid_save
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from tensorflow import random
random.set_seed(1)
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

from keras import layers
from keras import optimizers
# import gc
import seaborn as sns; 
sns.set(style="ticks", color_codes=True)
from keras.models import Sequential
import os
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
# import tensorflow as tf
import tensorflow_probability as tfp

# For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float64')

sns.reset_defaults()
sns.set_style('whitegrid')
sns.set_context('talk')
sns.set_context(context='talk',font_scale=0.7)



#%%
topdir = '/Users/aklimasewski/Documents/'
folder_path = topdir + 'models/2step_ANN_avgcellfeatures/model12/'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
transform_method = 'Norm'
epochs = 15
batch_size = 264
numlayers = 1
units = [50]

train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain= topdir + 'data/cybertrainyeti10_residfeb.csv', nametest=topdir + 'data/cybertestyeti10_residfeb.csv', n=12)
x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_path)

resid_train, resid_test, pre_train, pre_test = create_ANN(x_train, y_train, x_test, y_test, feature_names, numlayers, units, epochs, transform_method, folder_path)
 
period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]
plot_resid(resid_train, resid_test, folder_path)

'''
2nd ANN of gridded residuals
'''

#2d ANN of gridded residuals
folder_path = topdir + 'model_results/2step_ANN_avgcellfeatures/modelgriddedresiduals1/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

df, lon, lat = create_grid(dx = 1.0)

train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain= topdir + 'data/cybertrainyeti10_residfeb.csv', nametest=topdir + 'data/cybertestyeti10_residfeb.csv', n=6)

hypoR, sitelat, sitelon, evlat, evlon, target, gridded_targetsnorm_list, gridded_counts = grid_data(train_data1, train_targets1 = resid_train, df=df)     
hypoR_test, sitelat_test, sitelon_test, evlat_test, evlon_test, target_test, gridded_targetsnorm_list_test, gridded_counts_test = grid_data(test_data1, train_targets1 = resid_test, df=df)    

#%%
    
gridded_mean= mean_grid_save(gridded_targetsnorm_list,gridded_counts,df,folder_path,name='train_dx_1')
gridded_mean_test= mean_grid_save(gridded_targetsnorm_list_test,gridded_counts_test,df,folder_path,name='test_dx_1')

gridded_plots(gridded_mean, gridded_counts, period, lat, lon, evlon, evlat, sitelon, sitelat, folder_path)
#%%
y_train = gridded_mean
y_test = gridded_mean_test

# x_test
x_train = df.drop(['polygon'], axis=1)
x_test = df.drop(['polygon'], axis=1)

feature_names = ['latmid','lonmid']
transform_method = 'Norm'
x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_path)

resid_train, resid_test, pre_train, pre_test= create_ANN(x_train, y_train, x_test, y_test, feature_names, numlayers, units, epochs, transform_method, folder_pathmod)
 
period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]
plot_resid(resid_train, resid_test, folder_path)


