#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 16:38:02 2020

@author: aklimasewski
"""
import sys
import os
sys.path.append(os.path.abspath('/Users/aklimase/Documents/nonergodic_ANN'))
from preprocessing import transform_dip, readindata, transform_data, create_grid, grid_data
from model_plots import gridded_plots, plot_resid, obs_pre

import shapely
import shapely.geometry
import geopy
import geopy.distance
import numpy as np
from preprocessing import transform_dip, readindata, transform_data
import matplotlib.pyplot as plt
import shapely.geometry
import pandas as pd
from matplotlib import cm
import matplotlib as mpl
from sklearn.preprocessing import Normalizer
import random
import keras
from keras.models import Sequential
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import Normalizer
#grid up residuals
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
from keras import layers
from keras import optimizers


#%%
topdir = '/Users/aklimase/Documents/USGS/'

folder_path = topdir + 'models/gridtargets_ANN/modelgriddedtargets_1/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

df, lon, lat = create_grid(dx = 0.1)
nsamples = 100000

train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain= topdir + 'data/cybertrainyeti10_residfeb.csv', nametest=topdir + 'data/cybertestyeti10_residfeb.csv', n=6)

hypoR, sitelat, sitelon, evlat, evlon, target, gridded_targetsnorm_list, gridded_counts = grid_data(train_data1, train_targets1 , df=df, nsamples = nsamples)     
hypoR_test, sitelat_test, sitelon_test, evlat_test, evlon_test, target_test, gridded_targetsnorm_list_test, gridded_counts_test = grid_data(test_data1, test_targets1, df=df, nsamples = nsamples)    

#%%

#find mean of norm residual
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
    
#find mean of norm residual
gridded_targetsnorm_list_test = np.asarray(gridded_targetsnorm_list_test)

griddednorm_mean_test=np.zeros((len(gridded_targetsnorm_list_test),10))
for i in range(len(gridded_targetsnorm_list_test)):
    # for j in range(10):
    griddednorm_mean_test[i] = np.mean(gridded_targetsnorm_list_test[i],axis=0)

#find the cells with no paths (nans)
nan_ind=np.argwhere(np.isnan(griddednorm_mean_test)).flatten()
# set nan elements for empty array
for i in nan_ind:
    griddednorm_mean_test[i] = 0
    
period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]

#write gridded data to a file
#latmid, lonmid, target, counts
#save in arrays for file
df_save = df
meandict = {'T' + str(period[i]): griddednorm_mean[:,i] for i in range(len(period))}
meandict_test = {'T' + str(period[i]) + 'test': griddednorm_mean_test[:,i] for i in range(len(period))}
d2 = {'griddedcounts': gridded_counts, 'griddedcountstest': gridded_counts_test}
d2.update(meandict)
d2.update(meandict_test)
df2 = pd.DataFrame(data=d2)   
# df_save.append(df2)
df_save=pd.concat([df_save,df2],axis=1)
df_save.to_csv(folder_path + 'griddedtargets_dx_1.csv')


gridded_plots(griddednorm_mean, gridded_counts, period, lat, lon, evlon, evlat, sitelon, sitelat, folder_path + 'traingrid/')
gridded_plots(griddednorm_mean_test, gridded_counts_test, period, lat, lon, evlon, evlat, sitelon, sitelat, folder_path = folder_path + 'testgrid/')

exec(open('/Users/aklimase/Documents/USGS/nonergodic_pathANN/pathresid.py').read())















# #%%
# y_train = griddednorm_mean
# y_test = griddednorm_mean_test






# #%%

# #ANN with grid
# y_train = griddednorm_mean
# y_test = griddednorm_mean_test

# # x_test
# x_train = df.drop(['polygon','counts'], axis=1)

# transform = Normalizer()
# aa=transform.fit(x_train)
# train_data=aa.transform(x_train)
# # test_data=aa.transform(x_test)

# batch_size = 264

# def build_model():
#     model = Sequential()
#     model.add(layers.Dense(train_data.shape[1],activation='sigmoid', input_shape=(train_data.shape[1],)))
#     # model.add(layers.Dense(10))
#     # model.add(RBFLayer(10, 2))

#     #no gP layer
#     model.add(layers.Dense(10))

#     model.compile(optimizer=optimizers.Adam(lr=0.01),loss='mse',metrics=['mae','mse']) 
#     return model


# model=build_model()

# #fit the model
# history=model.fit(train_data,y_train,epochs=10,batch_size=batch_size,verbose=1)

# # mae_history=history.history['val_mae']
# mae_history_train=history.history['mae']
# # test_mse_score,test_mae_score,tempp=model.evaluate(x_test,y_test)

# pre = model.predict(train_data)
# # r = np.asarray(y_train)-pre.flatten()
# r = (y_train)-pre


# period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]

# ##################
# resid = r
# resid_test = []
# pre_test = []
# y_test = []
# plot_resid(resid, resid_test, pre, pre_test, y_train, y_test, folder_path)




# obs_pre(y_train, y_test, pre, pre_test, folder_path)




# #%%


# from keras.layers import Layer
# from keras import backend as K

# class RBFLayer(Layer):
#     def __init__(self, units, gamma, **kwargs):
#         super(RBFLayer, self).__init__(**kwargs)
#         self.units = units
#         self.gamma = K.cast_to_floatx(gamma)

#     def build(self, input_shape):
# #         print(input_shape)
# #         print(self.units)
#         self.mu = self.add_weight(name='mu',
#                                   shape=(int(input_shape[1]), self.units),
#                                   initializer='uniform',
#                                   trainable=True)
#         super(RBFLayer, self).build(input_shape)

#     def call(self, inputs):
#         diff = K.expand_dims(inputs) - self.mu
#         l2 = K.sum(K.pow(diff, 2), axis=1)
#         res = K.exp(-1 * self.gamma * l2)
#         return res

#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.units)




