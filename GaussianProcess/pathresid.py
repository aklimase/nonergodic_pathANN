#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 08:50:22 2020

@author: aklimase
grids targets and takes the average per cell, multiplies each path through the average cell as a new feature “average path residual”

"""

import sys
import os
sys.path.append(os.path.abspath('/Users/aklimasewski/Documents/nonergodic_ANN'))
from preprocessing import transform_dip, readindata, transform_data
from model_plots import gridded_plots, obs_pre, plot_resid, plot_outputs
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from tensorflow import random
random.set_seed(1)
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import Normalizer
from keras import layers
from keras import optimizers
# import gc
import seaborn as sns; 
sns.set(style="ticks", color_codes=True)
from keras.models import Sequential
import os
import cartopy
import shapely
import geopy
import geopy.distance
import shapely.wkt
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import random

tf.keras.backend.set_floatx('float64')

sns.reset_defaults()
sns.set_style('whitegrid')
sns.set_context('talk')
sns.set_context(context='talk',font_scale=0.7)

topdir = '/Users/aklimasewski/Documents/'

folder_path = topdir + 'models/gridtargets_ANN/modelgriddedtargets_1/'
df = pd.read_csv(folder_path + 'griddedtargets_dx_1.csv')

# gridcells = df['polygon']
# targets = df[['T10','T7.5','T5','T4','T3','T2','T1','T0.5','T0.2','T0.1']]
# list_wkt = df['polygon']
# list_polygons =  [shapely.wkt.loads(poly) for poly in list_wkt]

path_target_sum = avgpath_resid(df, folder_path, savename = 'train')

# ###
# n = 6
# train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimase/Documents/USGS/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimase/Documents/USGS/data/cybertestyeti10_residfeb.csv', n = n)

# hypoR = train_data1[:,0]
# sitelat = train_data1[:,1]
# sitelon = train_data1[:,2]
# evlat = train_data1[:,3]
# evlon = train_data1[:,4]
# target = train_targets1[:]

# # normtarget = target / hypoR[:, np.newaxis]
# # gridded_targetsnorm_list = [ [] for _ in range(df.shape[0]) ]

# path_target_sum = np.zeros((len(hypoR),10))#length of number of records
# # gridded_counts = np.zeros(df.shape[0])
# # lenlist = []

# #loop through each record     
# for i in range(len(sitelat)):                       
#     line = [(evlon[i], evlat[i]), (sitelon[i], sitelat[i])]
#     path=shapely.geometry.LineString(line)
#     #loop through each grid cell
#     if (i % 1000) == 0:
#         print('record: ', str(i))
#     pathsum = 0
#     for j in range(len(list_polygons)):
#         # shapely_poly = df['polygon'][j].split('(')[2].split(')')[0]
#         # polygon_points = [(lon[i], lat[j]), (lon[i], lat[j+1]), (lon[i+1], lat[j+1]), (lon[i+1], lat[j]), (lon[i], lat[j])]
#         shapely_poly = shapely.geometry.Polygon(list_polygons[j])
#         if path.intersects(shapely_poly) == True:
#             shapely_line = shapely.geometry.LineString(line)
#             intersection_line = list(shapely_poly.intersection(shapely_line).coords)
#             if len(intersection_line)== 2:
#                 coords_1 = (intersection_line[0][1], intersection_line[0][0])
#                 coords_2 = (intersection_line[1][1], intersection_line[1][0])
#                 length=geopy.distance.distance(coords_1, coords_2).km
                
#                 pathsum += length*np.asarray(targets.iloc[j])
        
#     path_target_sum[i] = (pathsum)          

                
                
# # dictout = {['T10','T7.5','T5','T4','T3','T2','T1','T0.5','T0.2','T0.1']:path_target_sum}
# df_out = pd.DataFrame(path_target_sum, columns=['T10','T7.5','T5','T4','T3','T2','T1','T0.5','T0.2','T0.1'])   
# # df_save.append(df2)
# df_out.to_csv(folder_path + 'avgrecord_targets_dx_1.csv')            

#

#%%

df = pd.read_csv(folder_path + 'griddedtargets_dx_1.csv')


path_target_sum_test = avgpath_resid(df, folder_path, savename = 'test')


# gridcells = df['polygon']
# targets = df[['T10test','T7.5test','T5test','T4test','T3test','T2test','T1test','T0.5test','T0.2test','T0.1test']]
# list_wkt = df['polygon']
# list_polygons =  [shapely.wkt.loads(poly) for poly in list_wkt]

# hypoR = test_data1[:,0]
# sitelat = test_data1[:,1]
# sitelon = test_data1[:,2]
# evlat = test_data1[:,3]
# evlon = test_data1[:,4]
# target = test_targets1[:]

# # normtarget = target / hypoR[:, np.newaxis]
# # gridded_targetsnorm_list = [ [] for _ in range(df.shape[0]) ]

# path_target_sum_test = np.zeros((len(hypoR),10))#length of number of records
# # gridded_counts = np.zeros(df.shape[0])
# # lenlist = []
# # 
# #loop through each record     
# for i in range(len(sitelat)):                       
#     line = [(evlon[i], evlat[i]), (sitelon[i], sitelat[i])]
#     path=shapely.geometry.LineString(line)
#     #loop through each grid cell
#     if (i % 1000) == 0:
#         print('record: ', str(i))
#     pathsum = 0
#     for j in range(len(list_polygons)):
#         # shapely_poly = df['polygon'][j].split('(')[2].split(')')[0]
#         # polygon_points = [(lon[i], lat[j]), (lon[i], lat[j+1]), (lon[i+1], lat[j+1]), (lon[i+1], lat[j]), (lon[i], lat[j])]
#         shapely_poly = shapely.geometry.Polygon(list_polygons[j])
#         if path.intersects(shapely_poly) == True:
#             shapely_line = shapely.geometry.LineString(line)
#             intersection_line = list(shapely_poly.intersection(shapely_line).coords)
#             if len(intersection_line)== 2:
#                 coords_1 = (intersection_line[0][1], intersection_line[0][0])
#                 coords_2 = (intersection_line[1][1], intersection_line[1][0])
#                 length=geopy.distance.distance(coords_1, coords_2).km
                
#                 pathsum += length*np.asarray(targets.iloc[j])
        
#     path_target_sum_test[i] = (pathsum)          

                
# # dictout = {['T10','T7.5','T5','T4','T3','T2','T1','T0.5','T0.2','T0.1']:path_target_sum}
# df_out = pd.DataFrame(path_target_sum, columns=['T10','T7.5','T5','T4','T3','T2','T1','T0.5','T0.2','T0.1'])   
# # df_save.append(df2)
# df_out.to_csv(folder_path + 'avgrecord_targets_test_dx_1.csv')            



#%%
#load csvs
folder_path = '/Users/aklimase/Documents/USGS/models/gridtargets_ANN/modelgriddedtargets_1/' + 'ANNfigs/'

train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimase/Documents/USGS/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimase/Documents/USGS/data/cybertestyeti10_residfeb.csv', n = 12)
train_data1 = np.append(train_data1,path_target_sum, axis=1)
test_data1 = np.append(test_data1,path_target_sum_test, axis=1)
feature_names = np.append(feature_names, ['T10','T7.5','T5','T4','T3','T2','T1','T0.5','T0.2','T0.1'], axis=0)
                          

transform_method = 'Norm'
x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_path)
             
numlayers = 1
units= [50]
resid, resid_test, pre_train, pre_test = create_ANN(x_train, y_train, x_test, y_test, feature_names, numlayers, units, epochs, transform_method, folder_pathmod)


period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]
plot_resid(resid, resid_test, folder_pathmod)


Rindex = np.where(feature_names == 'Rrup')[0][0]
predict_epistemic_allT = []
predict_epistemic_train_allT = []
plot_outputs(folder_path, pre_test, predict_epistemic_allT, pre_Train, predict_epistemic_train_allT, x_train, y_train, x_test, y_test, Rindex, period, feature_names)






                