#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 13:56:16 2020

@author: aklimasewski
"""


import sys
import os
sys.path.append(os.path.abspath('/Users/aklimasewski/Documents/python_code_nonergodic'))
from preprocessing import transform_dip, readindata, transform_data, add_az, add_midpoint, add_locfeat
from model_plots import plot_resid, obs_pre
from grid_funcs import grid_points, create_grid_square, create_grid, grid_data
from build_ANN import create_ANN
from readNGA import readindataNGA, add_locfeatNGA, add_midpointNGA, add_azNGA
import numpy as np
import pandas as pd
from keras.models import Sequential
import matplotlib as mpl
import matplotlib.pyplot as plt
from build_ANN import create_ANN
from keras import layers
from keras import optimizers
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import pyproj
from pyproj import Geod
from sklearn.preprocessing import PowerTransformer


folder_path = '/Users/aklimasewski/Documents/model_results/ANN_celllocfeat/grid_27km_2step/'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
transform_method = 'Norm' #function or text
epochs = 20
n = 13
#or n = 6, 4
az = True
unit_est = 2*(n+10)+1
numlayers = 1
units= [20]

folder_pathmod = folder_path + 'n4_ANN13_gridmidpoints_az_20ep_1_20/'
if not os.path.exists(folder_pathmod):
    os.makedirs(folder_pathmod)

#create grid of polygons in a dataframe
# df, lon, lat = create_grid_square(latmin=31.75,latmax=40.0,lonmin=-124,lonmax=-115.3,dx=.3,dy=0.25)
#27km
df, lon, lat = create_grid_square(latmin=33,latmax=36.0,lonmin=-120.5,lonmax=-115.7,dx=.3,dy=0.25)

#includes all data except outside .2 percentile (tst and train)
# df, lon, lat = create_grid_square(latmin=33,latmax=36.0,lonmin=-120.5,lonmax=-115.7,dx=.3,dy=0.25)
#10km
# df, lon, lat = create_grid_square(latmin=33,latmax=36.0,lonmin=-120.5,lonmax=-115.7,dx=.11,dy=0.09)

#50km
# df, lon, lat = create_grid_square(latmin=33,latmax=36.0,lonmin=-120.5,lonmax=-115.7,dx=.55,dy=0.45)
#%%
# train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = n)
# train_data1, test_data1, feature_names = add_locfeat(train_data1,test_data1, feature_names)
# train_data1, test_data1, feature_names = add_midpoint(train_data1,test_data1, feature_names)

# #NGA data
# filename = '/Users/aklimasewski/Documents/data/NGA_mag2_9.csv'
# nga_data1, nga_targets1, feature_names = readindataNGA(filename,n)
# nga_data1, feature_names = add_locfeatNGA(filename, nga_data1, feature_names)
# nga_data1, feature_names = add_midpointNGA(filename, nga_data1,feature_names)
# grid_points(nga_data1,df,name='nga',folder_path=folder_path)

# #read in cell data
# cells = pd.read_csv(folder_path + 'gridpointslatlon_train.csv',header = 0,index_col=0)
# cells_test = pd.read_csv(folder_path + 'gridpointslatlon_test.csv',header = 0,index_col=0)
# cells_nga = pd.read_csv(folder_path + 'gridpointslatlon_nga.csv',header = 0,index_col=0)

# #%%

# train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = n)
# train_data1,test_data1, feature_names = add_az(train_data1,test_data1, feature_names)

# filenamenga = '/Users/aklimasewski/Documents/data/NGA_mag2_9.csv'
# nga_data1, nga_targets1, feature_names = readindataNGA(filenamenga,n)
# nga_data1, feature_names = add_azNGA(filenamenga, nga_data1, feature_names)

# #add the cell features
# train_data1 = np.concatenate([train_data1,cells], axis = 1)
# test_data1 = np.concatenate([test_data1,cells_test], axis = 1)
# feature_names = np.concatenate([feature_names,['eventlat','eventlon','midlat','midlon','sitelat','sitelon',]], axis = 0)

# #split nga into training and testing
# #%%
from sklearn.model_selection import train_test_split
# nga_data1 = np.concatenate([nga_data1,cells_nga], axis = 1)
# ngatrain, ngatest, ngatrain_targets, ngatest_targets = train_test_split(nga_data1,nga_targets1, test_size=0.2, random_state=1)

# #combine nga train and test
# train_data1 = np.concatenate([train_data1,ngatrain], axis = 0)
# test_data1 = np.concatenate([test_data1,ngatest], axis = 0)
# train_targets1 = np.concatenate([train_targets1,ngatrain_targets], axis = 0)
# test_targets1 = np.concatenate([test_targets1,ngatest_targets], axis = 0)
# #transform
# x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_pathmod)
# #ANN
# resid, resid_test, pre_train, pre_test = create_ANN(x_train, y_train, x_test, y_test, feature_names, numlayers, units, epochs, transform_method, folder_pathmod)

# period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]
# plot_resid(resid, resid_test, folder_pathmod)

#%%



#2 step model
train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = n)
train_data1,test_data1, feature_names = add_az(train_data1,test_data1, feature_names)

filenamenga = '/Users/aklimasewski/Documents/data/NGA_mag2_9.csv'
nga_data1, nga_targets1, feature_names = readindataNGA(filenamenga,n)
nga_data1, feature_names = add_azNGA(filenamenga, nga_data1, feature_names)

cells = pd.read_csv(folder_path + 'gridpointslatlon_train.csv',header = 0,index_col=0)
cells_test = pd.read_csv(folder_path + 'gridpointslatlon_test.csv',header = 0,index_col=0)
cells_nga = pd.read_csv(folder_path + 'gridpointslatlon_nga.csv',header = 0,index_col=0)

ngatrain, ngatest, ngatrain_targets, ngatest_targets, ngacells_train, ngacells_test = train_test_split(nga_data1,nga_targets1,cells_nga, test_size=0.2, random_state=1)

# ngatrain, ngatest, ngatrain_targets, ngatest_targets = train_test_split(nga_data1,nga_targets1, test_size=0.2, random_state=1)

train_data1 = np.concatenate([train_data1,ngatrain], axis = 0)
test_data1 = np.concatenate([test_data1,ngatest], axis = 0)

train_targets1 = np.concatenate([train_targets1,ngatrain_targets], axis = 0)
test_targets1 = np.concatenate([test_targets1,ngatest_targets], axis = 0)

x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_pathmod)

# numlayers = 1
# units= [50]
resid, resid_test, pre_train, pre_test = create_ANN(x_train, y_train, x_test, y_test, feature_names, numlayers, units, epochs, transform_method, folder_pathmod)

period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]
plot_resid(resid, resid_test, folder_pathmod)

###############################################################################
#second ANN
folder_pathmod = folder_path + 'n4_ANN2_gridmidpoints_pt_30ep_1_10/'
if not os.path.exists(folder_pathmod):
    os.makedirs(folder_pathmod)
    
train_targets1 = resid
test_targets1 = resid_test

#split cells with 
# ngatrain, ngatest = train_test_split(cells_nga, test_size=0.2, random_state=1)
# train_data1 = np.concatenate([cells,ngacells_train], axis = 0)
# test_data1 = np.concatenate([cells_test,ngacells_test], axis = 0)

train_data1 = np.asarray(cells)
test_data1 = np.asarray(cells_test)

transform_method = PowerTransformer()
feature_names = np.asarray(['eventlat','eventlon','midlat','midlon','sitelat','sitelon',])

x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_pathmod)
epochs = 30
numlayers = 1
units= [10]
resid, resid_test, pre_train, pre_test = create_ANN(x_train, y_train, x_test, y_test, feature_names, numlayers, units, epochs, transform_method, folder_pathmod)

period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]
plot_resid(resid, resid_test, folder_pathmod)


#####


