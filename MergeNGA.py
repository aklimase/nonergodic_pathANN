#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 13:56:16 2020

@author: aklimasewski

methods for train/test split, merging NGA data with cybershake training and testing data
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

def mergeNGAdata_cells(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', filenamenga= '/Users/aklimasewski/Documents/data/NGA_mag2_9.csv', n=13):
    '''
    Read in NGA data file, train test split and merge with cybershake data
    includes the gridded cell locations as features

    Parameters
    ----------
    nametrain: path for cybershake training data csv
    nametest: path for cybershake testing data csv
    filenamenga: integer number of hidden layers
    n: number of model input features

    Returns
    -------
    train_data1: numpy array of training features
    test_data1: numpy array of testing features
    train_targets1: numpy array of training targets
    test_targets1: numpy array of testing targets
    feature_names: numpy array feature names
    '''
    
    from sklearn.model_selection import train_test_split
        
    cells = pd.read_csv(folder_path + 'gridpointslatlon_train.csv',header = 0,index_col=0)
    cells_test = pd.read_csv(folder_path + 'gridpointslatlon_test.csv',header = 0,index_col=0)
    cells_nga = pd.read_csv(folder_path + 'gridpointslatlon_nga.csv',header = 0,index_col=0)

    train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = n)
    train_data1,test_data1, feature_names = add_az(train_data1,test_data1, feature_names)
    
    
    nga_data1, nga_targets1, feature_names = readindataNGA(filenamenga,n)
    nga_data1, feature_names = add_azNGA(filenamenga, nga_data1, feature_names)
    nga_data1 = np.concatenate([nga_data1,cells_nga], axis = 0)

    ngatrain, ngatest, ngatrain_targets, ngatest_targets = train_test_split(nga_data1,nga_targets1,test_size=0.2, random_state=1)
        
    feature_names = np.concatenate([feature_names,['eventlat','eventlon','midlat','midlon','sitelat','sitelon',]], axis = 0)

    train_data1 = np.concatenate([train_data1,cells], axis = 1)
    test_data1 = np.concatenate([test_data1,cells_test], axis = 1)
    
    train_data1 = np.concatenate([train_data1,ngatrain], axis = 0)
    test_data1 = np.concatenate([test_data1,ngatest], axis = 0)

    train_targets1 = np.concatenate([train_targets1,ngatrain_targets], axis = 0)
    test_targets1 = np.concatenate([test_targets1,ngatest_targets], axis = 0)
    
    return train_data1, test_data1, train_targets1, test_targets1, feature_names

def mergeNGAdata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', filenamenga= '/Users/aklimasewski/Documents/data/NGA_mag2_9.csv', n=13):
    '''
    Read in NGA data file, train test split and merge with cybershake data

    Parameters
    ----------
    nametrain: path for cybershake training data csv
    nametest: path for cybershake testing data csv
    filenamenga: integer number of hidden layers
    n: number of model input features

    Returns
    -------
    train_data1: numpy array of training features
    test_data1: numpy array of testing features
    train_targets1: numpy array of training features
    test_targets1: numpy array of testing features
    feature_names: numpy array feature names
    '''
    
    from sklearn.model_selection import train_test_split

    train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = n)
    train_data1,test_data1, feature_names = add_az(train_data1,test_data1, feature_names)
    
    # filenamenga = '/Users/aklimasewski/Documents/data/NGA_mag2_9.csv'
    nga_data1, nga_targets1, feature_names = readindataNGA(filenamenga,n)
    nga_data1, feature_names = add_azNGA(filenamenga, nga_data1, feature_names)
        
    # ngatrain, ngatest, ngatrain_targets, ngatest_targets = train_test_split(nga_data1,nga_targets1, test_size=0.2, random_state=1)
    ngatrain, ngatest, ngatrain_targets, ngatest_targets = train_test_split(nga_data1,nga_targets1,test_size=0.2, random_state=1)

    train_data1 = np.concatenate([train_data1,ngatrain], axis = 0)
    test_data1 = np.concatenate([test_data1,ngatest], axis = 0)

    train_data1 = np.concatenate([train_data1,ngatrain], axis = 0)
    test_data1 = np.concatenate([test_data1,ngatest], axis = 0)

    train_targets1 = np.concatenate([train_targets1,ngatrain_targets], axis = 0)
    test_targets1 = np.concatenate([test_targets1,ngatest_targets], axis = 0)
    
    return train_data1, test_data1, train_targets1, test_targets1, feature_names

def mergeNGAcells(folder_path):
    '''
    Read in NGA cell file, train test split and merge with cybershake cells

    Parameters
    ----------
    folder_pathmod: path for reading cell data csv

    Returns
    -------
    train_data1: numpy array of training features
    test_data1: numpy array of testing features
    feature_names: numpy array feature names
    '''
    from sklearn.model_selection import train_test_split
        
    cells = pd.read_csv(folder_path + 'gridpointslatlon_train.csv',header = 0,index_col=0)
    cells_test = pd.read_csv(folder_path + 'gridpointslatlon_test.csv',header = 0,index_col=0)
    cells_nga = pd.read_csv(folder_path + 'gridpointslatlon_nga.csv',header = 0,index_col=0)
    
    ngacells_train, ngacells_test = train_test_split(cells_nga, test_size=0.2, random_state=1)

    train_data1 = np.concatenate([cells, ngacells_train], axis = 0)
    test_data1 = np.concatenate([cells_test, ngacells_test], axis = 0)
    feature_names = np.asarray(['eventlat','eventlon','midlat','midlon','sitelat','sitelon',])

    return train_data1, test_data1, feature_names

    
transform_method = 'Norm' #function or text

folder_pathmod = folder_path + 'n4_ANN13_gridmidpoints_az_20ep_1_20/'
def ANN_gridpoints(folder_pathmod, epochs=50, numlayers = 1, units= [20]):
    '''
    ANN with cell locations as additional features
    trained and tested with dataset including NGA data

    Parameters
    ----------
    folder_pathmod: path for saving files
    epochs: number of training epochs
    numlayers: integer number of hidden layers
    units: array of number of units for hidden layers

    Returns
    -------
    None.
    creates ANN and saves model files and figures
    '''
    
    if not os.path.exists(folder_pathmod):
        os.makedirs(folder_pathmod)

    train_data1, test_data1, train_targets1, test_targets1, feature_names = mergeNGAdata_cells(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', filenamenga= '/Users/aklimasewski/Documents/data/NGA_mag2_9.csv', n=13)
     
    x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_pathmod)
    
    resid, resid_test, pre_train, pre_test = create_ANN(x_train, y_train, x_test, y_test, feature_names, numlayers, units, epochs, transform_method, folder_pathmod)
    
    period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]
    plot_resid(resid, resid_test, folder_pathmod)

