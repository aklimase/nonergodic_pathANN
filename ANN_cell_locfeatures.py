#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 17:30:17 2020

@author: aklimasewski

grids record station locations, midpoints, and event locations and saves in csv, each record has three new features of cell location or id
"""

import sys
import os
sys.path.append(os.path.abspath('/Users/aklimasewski/Documents/python_code_nonergodic'))
from preprocessing import transform_dip, readindata, transform_data, add_az, add_midpoint, add_locfeat
from model_plots import plot_resid, obs_pre
from grid_funcs import grid_points, create_grid_square, create_grid, grid_data
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
#or n = 6, 4
az = True
unit_est = 2*(n+10)+1


#create grid of polygons in a dataframe
#lat, lon are midpoint list of each grid cell
# df, lon, lat = create_grid_square(latmin=31.75,latmax=40.0,lonmin=-124,lonmax=-115.3,dx=.3,dy=0.25)
# df, lon, lat = create_grid_square(latmin=33,latmax=36.0,lonmin=-120.5,lonmax=-115.7,dx=.3,dy=0.25)

#includes all data except outside .2 percentile (tst and train)
#27
df, lon, lat = create_grid_square(latmin=33,latmax=36.0,lonmin=-120.5,lonmax=-115.7,dx=.3,dy=0.25)
#10km
# df, lon, lat = create_grid_square(latmin=33,latmax=36.0,lonmin=-120.5,lonmax=-115.7,dx=.11,dy=0.09)

#50km
# df, lon, lat = create_grid_square(latmin=33,latmax=36.0,lonmin=-120.5,lonmax=-115.7,dx=.55,dy=0.45)

train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = n)

train_data1,test_data1, feature_names = add_locfeat(train_data1,test_data1, feature_names)

train_data1,test_data1, feature_names = add_midpoint(train_data1,test_data1, feature_names)

# grid_points(train_data1,df,name='train',folder_path=folder_path)
# grid_points(test_data1,df,name='test',folder_path=folder_path)
#%%
#NGA data
# filename = '/Users/aklimasewski/Documents/data/NGA_mag2_9.csv'
# nga_data1, nga_targets1, feature_names = readindataNGA(filename,n)
# nga_data1, feature_names = add_locfeatNGA(filename, nga_data1, feature_names)
# nga_data1, feature_names = add_midpointNGA(filename, nga_data1,feature_names)
# grid_points(nga_data1,df,name='nga',folder_path=folder_path)

#%%
# # # counts
def plot_counts(gridded_counts,data, name):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    colname = ['event','midpoint','site']
    
    sitelat = data[:,13]
    sitelon = data[:,14]
    evlat = data[:,15]
    evlon = data[:,16]
    midlat = data[:,17]
    midlon = data[:,18]
    
    for i in range(len(gridded_counts[0])):
        # Z = gridded_counts[:,i].reshape(len(lat)-1,len(lon)-1).T
        Z = gridded_counts[:,i].reshape(len(lon)-1,len(lat)-1).T
        
        cbound = np.max(np.abs(Z))
        cmap = mpl.cm.get_cmap('Greens')
        normalize = mpl.colors.Normalize(vmin=0, vmax=cbound)
        colors = [cmap(normalize(value)) for value in Z]
        s_m = mpl.cm.ScalarMappable(cmap = cmap, norm=normalize)
        s_m.set_array([])
        
        fig, ax = plt.subplots(figsize = (10,8))
        plt.pcolormesh(lon, lat, Z, cmap = cmap, norm = normalize) 
        plt.scatter(evlon,evlat,marker = '*', s=0.2, c = 'gray', label = 'event', alpha = 0.02)
        plt.scatter(sitelon,sitelat,marker = '^',s=0.2, c = 'black', label = 'site', alpha = 0.02)
        plt.xlim(min(lon),max(lon))
        plt.ylim(min(lat),max(lat))
        plt.title(colname)
        plt.legend(loc = 'lower left')
        
        fig.subplots_adjust(right=0.75)
        cbar = plt.colorbar(s_m, orientation='vertical')
        cbar.set_label(colname[i] + ' counts', fontsize = 20)
        plt.savefig(folder_path + colname[i] + name + '_counts.png')
        plt.show()
        
        plt.close('all')

# counts_train = pd.read_csv(folder_path + 'counts_train.csv',header = 0,index_col=0)
# counts_test = pd.read_csv(folder_path + 'counts_test.csv',header = 0,index_col=0)

# counts_nga = pd.read_csv(folder_path + 'counts_nga.csv',header = 0,index_col=0)
# plot_counts(np.asarray(counts_nga),data = nga_data1,name='nga')

# plot_counts(np.asarray(counts_train),data = train_data1, name='train')
# plot_counts(np.asarray(counts_test),data = test_data1,name='test')


#%%
#with 2d midpoints
cells = pd.read_csv(folder_path + 'gridpointslatlon_train.csv',header = 0,index_col=0)
cells_test = pd.read_csv(folder_path + 'gridpointslatlon_test.csv',header = 0,index_col=0)

folder_pathmod = folder_path + 'n4_ANN13_gridmidpoints_az_20ep_1_20/'

if not os.path.exists(folder_pathmod):
    os.makedirs(folder_pathmod)
    
transform_method = 'Norm' #function or text
epochs = 20
n = 13
#or n = 6, 4
az = True
unit_est = 2*(n+10)+1
# az = True

train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = n)

if az == True:
    train_data1,test_data1, feature_names = add_az(train_data1,test_data1, feature_names)

#add the cell features
train_data1 = np.concatenate([train_data1,cells], axis = 1)
test_data1 = np.concatenate([test_data1,cells_test], axis = 1)
feature_names = np.concatenate([feature_names,['eventlat','eventlon','midlat','midlon','sitelat','sitelon',]], axis = 0)

# train_data1 = train_data1[0:10000]
# test_data1 = test_data1[0:10000]

x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_pathmod)

#%%
numlayers = 1
units= [20]
resid, resid_test, pre_train, pre_test = create_ANN(x_train, y_train, x_test, y_test, feature_names, numlayers, units, epochs, transform_method, folder_pathmod)

period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]
plot_resid(resid, resid_test, folder_pathmod)

#%% 
# #with cell numbers

# cells = pd.read_csv(folder_path + 'gridpoints_train.csv',header = 0,index_col=0)
# cells_test = pd.read_csv(folder_path + 'gridpoints_test.csv',header = 0,index_col=0)

# folder_pathmod = folder_path + 'ANN13_gridnum_az_40ep_20_20/'

# if not os.path.exists(folder_pathmod):
#     os.makedirs(folder_pathmod)
    

# train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = n)

# #add the location features
# train_data1 = np.concatenate([train_data1,cells], axis = 1)
# test_data1 = np.concatenate([test_data1,cells_test], axis = 1)
# feature_names = np.concatenate([feature_names,['event','mid','site']], axis = 0)


# if az == True:
#     train_data1,test_data1, feature_names = add_az(train_data1,test_data1, feature_names)

# x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_pathmod)

# resid, resid_test,pre_train, pre_test = create_ANN(x_train, y_train, x_test, y_test, feature_names, numlayers, units, epochs, transform_method, folder_pathmod)
 

# period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]
# plot_resid(resid, resid_test, folder_pathmod)

#%%
#compare with NGA data
from readNGA import readindataNGA, transform_dataNGA,add_locfeatNGA,add_azNGA,add_midpointNGA

filename = '/Users/aklimasewski/Documents/data/NGAWest2region_clean.csv'

nga_data1, nga_targets1, feature_names = readindataNGA(filename,n=n)

nga_data1,feature_names = add_locfeatNGA(nga_data1,feature_names)
nga_data1,feature_names = add_midpointNGA(nga_data1,feature_names)

if az == True:
    nga_data1, feature_names = add_azNGA(nga_data1,feature_names)


grid_points(nga_data1,df,name='nga',folder_path=folder_path)

counts_nga = pd.read_csv(folder_path + 'counts_nga.csv',header = 0,index_col=0)

plot_counts(np.asarray(counts_nga),data = nga_data1, name='nga')
# 
####### add cell features
nga_data1, nga_targets1, feature_names = readindataNGA(filename,n=n)
if az == True:
    nga_data1, feature_names = add_azNGA(nga_data1,feature_names)

cells = pd.read_csv(folder_path + 'gridpointslatlon_nga.csv',header = 0,index_col=0)
nga_data1 = np.concatenate([nga_data1,cells], axis = 1)
feature_names = np.concatenate([feature_names,['eventlat','eventlon','midlat','midlon','sitelat','sitelon',]], axis = 0)

x_train, y_train, x_nga, y_nga, x_range, x_train_raw,  x_nga_raw = transform_data(transform_method, train_data1, nga_data1, train_targets1, nga_targets1, feature_names, folder_path)
# x_train, y_train, x_nga, y_nga, x_range, x_train_raw,  x_nga_raw = transform_data(transform_method, train_data1, nga_data1, train_targets1, nga_targets1, feature_names, folder_path)

import keras
model =  keras.models.load_model(folder_pathmod + 'model/')
pre_nga = model.predict(x_nga)
resid_nga = np.asarray(nga_targets1) - pre_nga


period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]
plot_resid(resid, resid_nga, folder_pathmod)

