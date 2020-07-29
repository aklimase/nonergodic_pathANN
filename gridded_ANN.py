#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 16:38:02 2020

@author: aklimasewski
"""
import sys
import os
sys.path.append(os.path.abspath('/Users/aklimasewski/Documents/nonergodic_ANN'))
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


folder_path = '/Users/aklimase/Documents/USGS/models/gridded_ANN/model12/'

nsamples = 500

train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimase/Documents/USGS/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimase/Documents/USGS/data/cybertestyeti10_residfeb.csv', n=6)
#start with defaults
df, lon, lat = create_grid(dx = 0.5)

hypoR, sitelat, sitelon, evlat, evlon, target, gridded_targetsnorm_list, gridded_counts = grid_data(train_data1, train_targets1, df, nsamples = nsamples)           
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


#add gridded mean and gridded counts to df
df['counts'] = gridded_counts
period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]

gridded_plots(griddednorm_mean, gridded_counts, period, lat, lon, evlon, evlat, sitelon, sitelat, folder_path)

# for i in range(len(griddednorm_mean.T)):
#     T= period[i]
#     g = griddednorm_mean.T[i]
#     Z = g.reshape(len(lat)-1,len(lon)-1)
    
#     cbound = max(np.abs(g))
#     cbound = 0.12

#     cmap = mpl.cm.get_cmap('seismic')
#     normalize = mpl.colors.Normalize(vmin=-1*cbound, vmax=cbound)
#     colors = [cmap(normalize(value)) for value in Z]
#     s_m = mpl.cm.ScalarMappable(cmap = cmap, norm=normalize)
#     s_m.set_array([])
        
#     fig, ax = plt.subplots(figsize = (10,8))
#     plt.pcolormesh(lon, lat, Z, cmap = cmap, norm = normalize) 
#     plt.scatter(evlon,evlat,marker = '*', s=1, c = 'gray', label = 'event')
#     plt.scatter(sitelon,sitelat,marker = '^',s=1, c = 'black', label = 'site')
#     plt.xlim(min(lon),max(lon))
#     plt.ylim(min(lat),max(lat))
#     plt.title('T ' + str(T) + ' s')
#     plt.legend(loc = 'lower left')
    
#     fig.subplots_adjust(right=0.75)
#     cbar = plt.colorbar(s_m, orientation='vertical')
#     cbar.set_label(r'average normalized residual (resid/km)', fontsize = 20)
#     plt.savefig(folder_path + 'normresid_T_' + str(T) + '.png')
#     plt.show()
    

# # counts
# Z = gridded_counts.reshape(len(lat)-1,len(lon)-1)

# cbound = max(np.abs(gridded_counts))
# cmap = mpl.cm.get_cmap('Greens')
# normalize = mpl.colors.Normalize(vmin=0, vmax=cbound)
# colors = [cmap(normalize(value)) for value in Z]
# s_m = mpl.cm.ScalarMappable(cmap = cmap, norm=normalize)
# s_m.set_array([])

# fig, ax = plt.subplots(figsize = (10,8))
# plt.pcolormesh(lon, lat, Z, cmap = cmap, norm = normalize) 
# plt.scatter(evlon,evlat,marker = '*', s=1, c = 'gray', label = 'event')
# plt.scatter(sitelon,sitelat,marker = '^',s=1, c = 'black', label = 'site')
# plt.xlim(min(lon),max(lon))
# plt.ylim(min(lat),max(lat))
# plt.title('T ' + str(T) + ' s')
# plt.legend(loc = 'lower left')

# fig.subplots_adjust(right=0.75)
# cbar = plt.colorbar(s_m, orientation='vertical')
# cbar.set_label(r'paths per cell', fontsize = 20)
# plt.savefig(folder_path + 'pathcounts.png')
# plt.show()


#%%
y_train = griddednorm_mean

# x_test
x_train = df.drop(['polygon','counts'], axis=1)

transform = Normalizer()
aa=transform.fit(x_train)
train_data=aa.transform(x_train)
# test_data=aa.transform(x_test)

batch_size = 264

def build_model():
    model = Sequential()
    model.add(layers.Dense(train_data.shape[1],activation='sigmoid', input_shape=(train_data.shape[1],)))
    # model.add(layers.Dense(10))
    # model.add(RBFLayer(10, 2))

    #no gP layer
    model.add(layers.Dense(10))

    model.compile(optimizer=optimizers.Adam(lr=0.01),loss='mse',metrics=['mae','mse']) 
    return model


model=build_model()

#fit the model
history=model.fit(train_data,y_train,epochs=10,batch_size=batch_size,verbose=1)

# mae_history=history.history['val_mae']
mae_history_train=history.history['mae']
# test_mse_score,test_mae_score,tempp=model.evaluate(x_test,y_test)

pre = model.predict(train_data)
# r = np.asarray(y_train)-pre.flatten()
r = (y_train)-pre


period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]

##################
resid = r
resid_test = []
pre_test = []
y_test = []
plot_resid(resid, resid_test, pre, pre_test, y_train, y_test, folder_path)

# diff=np.std(r,axis=0)
# # diffmean=np.mean(y_train-mean_x_train_allT,axis=0)
# f22=plt.figure('Difference Std of residuals vs Period')
# plt.semilogx(period,diff,label='Training ')
# # plt.semilogx(period,difftest,label='Testing')
# plt.xlabel('Period')
# plt.ylabel('Total Standard Deviation')
# plt.legend()
# plt.savefig(folder_path + 'resid_T.png')
# plt.show()

# diffmean=np.mean(r,axis=0)
# # diffmeantest=np.mean(resid_test-mean_x_test_allT,axis=0)
# f22=plt.figure('Difference Std of residuals vs Period')
# plt.semilogx(period,diffmean,label='Training ')
# # plt.semilogx(period,diffmeantest,label='Testing')
# plt.xlabel('Period')
# plt.ylabel('Mean residual')
# plt.legend()
# plt.savefig(folder_path + 'mean_T.png')
# plt.show()


obs_pre(y_train, y_test, pre, pre_test, folder_path)

# for i in range(10):
#     T= period[i]
#     y = pre.T[i]
#     x = y_train.T[i]
#     plt.figure(figsize = (6,6))
#     lim = np.max(np.asarray([abs(x), abs(y)]).flatten())
#     plt.scatter(x,y,s=1)
#     plt.xlabel('observed')
#     plt.ylabel('predicted')
#     plt.title('T ' + str(T) + ' s')
#     plt.xlim(-1*lim, lim)
#     plt.ylim(-1*lim, lim)
#     plt.savefig(folder_path + 'obs_pre_T_' + str(T) + '.png')
#     plt.show()








#%%


from keras.layers import Layer
from keras import backend as K

class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
#         print(input_shape)
#         print(self.units)
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)




