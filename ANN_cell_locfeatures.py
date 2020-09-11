#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 17:30:17 2020

@author: aklimasewski

grids record station locations, midpoints, and event locations and saves in csv, each record has three new features of cell location or id
functions to
    - create an ANN include cells as features per record
    - create a 2 step model with a base model first and a second ANN trained on first residuals and with cell features per record
"""

import sys
import os
sys.path.append(os.path.abspath('/Users/aklimasewski/Documents/python_code_nonergodic'))
from preprocessing import transform_dip, readindata, transform_data, add_az, add_midpoint, add_locfeat
from model_plots import plot_resid, obs_pre
from grid_funcs import grid_points, create_grid_square, create_grid, grid_data, plot_counts
from readNGA import readindataNGA, transform_dataNGA,add_locfeatNGA,add_azNGA,add_midpointNGA
from build_ANN import create_ANN
from readNGA import readindataNGA, add_locfeatNGA, add_midpointNGA
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

folder_path = '/Users/aklimasewski/Documents/model_results/ANN_celllocfeat/grid_27km/'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
transform_method = 'Norm' #function or text
epochs = 13
n = 13
az = True

'''
read in data to grid if needed
create grid of polygons in a dataframe
lat, lon are midpoint list of each grid cell
comment out and read in csvs if they already exist
'''

# includes all data except outside .2 percentile (tst and train)
# 27
df, lon, lat = create_grid_square(latmin=33,latmax=36.0,lonmin=-120.5,lonmax=-115.7,dx=.3,dy=0.25)
# # 10km
# df, lon, lat = create_grid_square(latmin=33,latmax=36.0,lonmin=-120.5,lonmax=-115.7,dx=.11,dy=0.09)
# # 50km
# df, lon, lat = create_grid_square(latmin=33,latmax=36.0,lonmin=-120.5,lonmax=-115.7,dx=.55,dy=0.45)

# read in training data and add 
train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = n)
train_data1,test_data1, feature_names = add_locfeat(train_data1,test_data1, feature_names)
train_data1,test_data1, feature_names = add_midpoint(train_data1,test_data1, feature_names)

# grid NGA data if needed, comment if notgrid_points(train_data1,df,name='train',folder_path=folder_path)
grid_points(test_data1,df,name='test',folder_path=folder_path)


filename = '/Users/aklimasewski/Documents/data/NGA_mag2_9.csv'
nga_data1, nga_targets1, feature_names = readindataNGA(filename,n)
nga_data1, feature_names = add_locfeatNGA(filename, nga_data1, feature_names)
nga_data1, feature_names = add_midpointNGA(filename, nga_data1,feature_names)
grid_points(nga_data1,df,name='nga',folder_path=folder_path)

# read in counts and plot
counts_train = pd.read_csv(folder_path + 'counts_train.csv',header = 0,index_col=0)
counts_test = pd.read_csv(folder_path + 'counts_test.csv',header = 0,index_col=0)
counts_nga = pd.read_csv(folder_path + 'counts_nga.csv',header = 0,index_col=0)

plot_counts(np.asarray(counts_nga),data = nga_data1,name='nga',folder_path =folder_path)
plot_counts(np.asarray(counts_train),data = train_data1, name='train', folder_path =folder_path)
plot_counts(np.asarray(counts_test),data = test_data1,name='test',folder_path =folder_path)


def ANN_gridpoints(folder_pathmod, epochs=50, numlayers = 1, units= [20]):
    '''
    ANN with cell locations as additional features

    Parameters
    ----------
    folder_pathmod: path for saving png files
    epochs: number of training epochs
    numlayers: integer number of hidden layers
    units: array of number of units for hidden layers

    Returns
    -------
    None.
    creates ANNS and saves model files and figures
    '''
    
    cells = pd.read_csv(folder_path + 'gridpointslatlon_train.csv',header = 0,index_col=0)
    cells_test = pd.read_csv(folder_path + 'gridpointslatlon_test.csv',header = 0,index_col=0)
    
    if not os.path.exists(folder_pathmod):
        os.makedirs(folder_pathmod1)
        
    transform_method = 'Norm' #function or text
    n = 13
    
    train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = n)
    train_data1,test_data1, feature_names = add_az(train_data1,test_data1, feature_names)
    
    # add the cell features
    train_data1 = np.concatenate([train_data1,cells], axis = 1)
    test_data1 = np.concatenate([test_data1,cells_test], axis = 1)
    feature_names = np.concatenate([feature_names,['eventlat','eventlon','midlat','midlon','sitelat','sitelon',]], axis = 0)
    
    x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_pathmod)
    
    resid, resid_test, pre_train, pre_test = create_ANN(x_train, y_train, x_test, y_test, feature_names, numlayers, units, epochs, transform_method, folder_pathmod)
    
    plot_resid(resid, resid_test, folder_pathmod1)


folder_pathmod1 = folder_path + 'n4_ANN13_20ep_1_20/'
folder_pathmod2 = folder_path + 'n4_ANN2_30ep_1_10/'

def ANN_2step(folder_pathmod1, folder_pathmod2, epochs1=50, epochs2=50, numlayers1 = 1,numlayers2 = 1, units1= [20],units2= [20]):
    '''
    2 ANNs: 1st is the base ANN, 2nd ANN uses 1st model residuals as targets and cell location features

    Parameters
    ----------
    folder_pathmod1: path for saving png files for the first ANN
    folder_pathmod2: path for saving png files for the second ANN
    epochs1: number of training epochs for the first ANN
    epochs2: number of training epochs for the second ANN
    numlayers1: integer number of hidden layers for the first ANN
    numlayers2: integer number of hidden layers for the second ANN
    units1: array of number of units for hidden layers for first ANN
    units2: array of number of units for hidden layers for second ANN

    Returns
    -------
    None.
    creates two ANNS and saves model files and figures
    '''
    from sklearn.preprocessing import PowerTransformer

    if not os.path.exists(folder_pathmod1):
        os.makedirs(folder_pathmod1)

    # read in training, testing, and cell data
    train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = n)
    train_data1,test_data1, feature_names = add_az(train_data1,test_data1, feature_names)
    
    cells = pd.read_csv(folder_path + 'gridpointslatlon_train.csv',header = 0,index_col=0)
    cells_test = pd.read_csv(folder_path + 'gridpointslatlon_test.csv',header = 0,index_col=0)
    
    x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_pathmod1)
    
    resid, resid_test, pre_train, pre_test = create_ANN(x_train, y_train, x_test, y_test, feature_names, numlayers1, units1, epochs1, transform_method, folder_pathmod1)
    
    period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]
    plot_resid(resid, resid_test, folder_pathmod1)
    
    # second ANN
    if not os.path.exists(folder_pathmod2):
        os.makedirs(folder_pathmod2)
        
    train_targets1 = resid
    test_targets1 = resid_test
    
    train_data1 = np.asarray(cells)
    test_data1 = np.asarray(cells_test)
    
    transform_method = PowerTransformer()
    feature_names = np.asarray(['eventlat','eventlon','midlat','midlon','sitelat','sitelon',])
    
    x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_pathmod2)

    resid, resid_test, pre_train, pre_test = create_ANN(x_train, y_train, x_test, y_test, feature_names, numlayers2, units2, epochs2, transform_method, folder_pathmod2)
    
    period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]
    plot_resid(resid, resid_test, folder_pathmod2)


