#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 18:03:06 2020

@author: aklimasewski

ANN with only lat,lon of station and event trained on residuals

includes kernel class and optimization
"""
import sys
import os
sys.path.append(os.path.abspath('/Users/aklimasewski/Documents/python_code_nonergodic'))
from preprocessing import transform_dip, readindata, transform_data, add_az, add_locfeat
from model_plots import plot_resid, obs_pre, plot_outputs, plot_rawinputs
from build_ANN import create_ANN
from readNGA import readindataNGA, transform_dataNGA,add_locfeatNGA,add_azNGA
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from tensorflow import random
random.set_seed(1)
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler

from keras import layers
from keras import optimizers
# import gc
import seaborn as sns; 
sns.set(style="ticks", color_codes=True)
from keras.models import Sequential
import os
import cartopy
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import random
import pyproj
from pyproj import Geod

import tensorflow_probability as tfp

# For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float64')

sns.reset_defaults()
sns.set_style('whitegrid')
sns.set_context('talk')
sns.set_context(context='talk',font_scale=0.7)


#%%

folder_path = '/Users/aklimasewski/Documents/model_results/base/ANNbase_nga_20ep_50hidden/'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
transform_method = 'Norm' #function or text
epochs =20
n = 13
#or n = 6, 4
az = True
unit_est = 2*(n+10)+1

train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = n)


# #add the location features
# train_data1_4, test_data1_4, train_targets1_4, test_targets1_4, feature_names_4 = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = 4)
# train_data1 = np.concatenate([train_data1,train_data1_4], axis = 1)
# test_data1 = np.concatenate([test_data1,test_data1_4], axis = 1)
# feature_names = np.concatenate([feature_names,feature_names_4], axis = 0)

# train_data1,test_data1, feature_names = add_locfeat(train_data1,test_data1, feature_names)

#add azimuth
############
if az == True:
    train_data1,test_data1, feature_names = add_az(train_data1,test_data1, feature_names)

#%%
filenamenga = '/Users/aklimasewski/Documents/data/NGA_mag2_9.csv'

nga_data1, nga_targets1, feature_names = readindataNGA(filenamenga,n)
nga_data1, feature_names = add_azNGA(filenamenga, nga_data1, feature_names)

from sklearn.model_selection import train_test_split
ngatrain, ngatest, ngatrain_targets, ngatest_targets = train_test_split(nga_data1,nga_targets1, test_size=0.2, random_state=1)

# #combine nga train and test
train_data1 = np.concatenate([train_data1,ngatrain], axis = 0)
test_data1 = np.concatenate([test_data1,ngatest], axis = 0)
train_targets1 = np.concatenate([train_targets1,ngatrain_targets], axis = 0)
test_targets1 = np.concatenate([test_targets1,ngatest_targets], axis = 0)


#%%
    
    
    
#transform data
x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_path)

#%%

# create sequential model with positional inputs and predict 
# try sequential with VGP layer
# For numeric stability, set the default floating-point dtype to float64
numlayers = 1
units= [50]
resid_train, resid_test, pre_train, pre_test = create_ANN(x_train, y_train, x_test, y_test, feature_names, numlayers, units, epochs, transform_method, folder_path)

period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]
plot_resid(resid_train, resid_test, folder_path)

mean_x_test_allT = pre_test
mean_x_train_allT = pre_train
predict_epistemic_allT = []
predict_epistemic_train_allT = []

Rindex = np.where(feature_names == 'Rrup')[0][0]

# plot_outputs(folder_path, mean_x_test_allT, predict_epistemic_allT, mean_x_train_allT, predict_epistemic_train_allT, x_train, y_train, x_test, y_test, Rindex, period, feature_names)
# plot_rawinputs(x_raw = x_train_raw, mean_x_allT = mean_x_train_allT, y=y_train, feature_names=feature_names, period = period, folder_path = folder_path + 'train/')
# plot_rawinputs(x_raw = x_test_raw, mean_x_allT = mean_x_test_allT, y=y_test, feature_names=feature_names, period = period, folder_path = folder_path + 'test/')

# obs_pre(y_train, y_test, pre_train, pre_test, period, folder_path)

#write training predictions to a file
# df_out = pd.DataFrame(resid_train, columns=['T10','T7.5','T5','T4','T3','T2','T1','T0.5','T0.2','T0.1'])   
# df_out.to_csv(folder_path + 'ANNresiduals_train.csv')   

# df_outtest = pd.DataFrame(resid_test, columns=['T10','T7.5','T5','T4','T3','T2','T1','T0.5','T0.2','T0.1'])   
# df_out.to_csv(folder_path + 'ANNresiduals_test.csv')   


#%%
#compare to NGA data
# filename = '/Users/aklimasewski/Documents/data/NGAWest2region_clean.csv'
filenamenga = '/Users/aklimasewski/Documents/data/NGA_mag2_9.csv'

nga_data1, nga_targets1, feature_names = readindataNGA(filenamenga,n)
nga_data1, feature_names = add_azNGA(filenamenga, nga_data1, feature_names)
# nga_data1,feature_names = add_locfeatNGA(filenamenga,nga_data1,feature_names)

# if az == True:
    # nga_data1, feature_names = add_azNGA(nga_data1,feature_names)
    
train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = n)
train_data1,test_data1, feature_names = add_az(train_data1,test_data1, feature_names)

folder_pathNGA = folder_path + 'NGAtest/'
if not os.path.exists(folder_pathNGA):
    os.makedirs(folder_pathNGA)

# x_nga, y_nga, x_ngarange, x_nga_raw = transform_dataNGA(transform_method, nga_data1, nga_targets1, feature_names, folder_path)
x_train, y_train, x_nga, y_nga, x_range, x_train_raw,  x_nga_raw = transform_data(transform_method, train_data1, nga_data1, train_targets1, nga_targets1, feature_names, folder_pathNGA)

#load model and fit
import keras
loadedmodel =  keras.models.load_model(folder_path + 'model/')

pre_nga = loadedmodel.predict(x_nga)

# model_avg = np.asarray(pd.read_csv('/Users/aklimasewski/Documents/data/NGAopenquake_predictions_lng.csv',index_col=0))
# nga_resid = np.log(nga_targets1) - model_avg

# #use average of base models for residual
resid_nga = np.asarray(nga_targets1) - pre_nga

pre_train = loadedmodel.predict(x_train)
resid_train = np.asarray(train_targets1) - pre_train

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]

diff=np.std(resid_train,axis=0)
difftest=np.std(resid_nga,axis=0)
f22=plt.figure('Difference Std of residuals vs Period')
plt.semilogx(period,diff,label='Training ')
plt.semilogx(period,difftest,c='green',label='NGA')
plt.xlabel('Period')
plt.ylabel('Total Standard Deviation')
plt.legend()
# plt.ylim(.25,.85)
plt.savefig(folder_pathNGA + 'resid_T.png')
plt.show()

diffmean=np.mean(resid_train,axis=0)
diffmeantest=np.mean(resid_nga,axis=0)
f22=plt.figure('Difference Std of residuals vs Period')
plt.semilogx(period,diffmean,label='Training')
plt.semilogx(period,diffmeantest,c='green',label='NGA')
plt.xlabel('Period')
plt.ylabel('Mean residual')
plt.legend()
plt.savefig(folder_pathNGA + 'mean_T.png')
plt.show()
plt.close('all')

mean_x_train_allT = pre_nga
plot_rawinputs(x_raw = x_nga_raw, mean_x_allT = mean_x_train_allT, y=y_nga, feature_names=feature_names, period = period, folder_path = folder_pathNGA)

