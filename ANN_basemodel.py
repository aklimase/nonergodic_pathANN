#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 18:03:06 2020

@author: aklimasewski

creates base ANN, saves model details files, plots, model outputs
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
from sklearn.model_selection import train_test_split

# For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float64')
sns.reset_defaults()
sns.set_style('whitegrid')
sns.set_context('talk')
sns.set_context(context='talk',font_scale=0.7)

# set path for saving files
folder_path = '/Users/aklimasewski/Documents/model_results/base/ANNbase_nga_20ep_50hidden/'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# set up model parameters
transform_method = 'Norm' # function or text
# number of model parameters n=12, n=13 (base with and without Xi)
n = 13
az = True
numlayers = 1
units= [50]
epochs = 20

train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = n)

if az == True:
    train_data1,test_data1, feature_names = add_az(train_data1,test_data1, feature_names)

'''
if including NGA data read in, split into training and testing, and concatenate
'''

filenamenga = '/Users/aklimasewski/Documents/data/NGA_mag2_9.csv'
nga_data1, nga_targets1, feature_names = readindataNGA(filenamenga, n)
nga_data1, feature_names = add_azNGA(filenamenga, nga_data1, feature_names)

# split into training and testing
ngatrain, ngatest, ngatrain_targets, ngatest_targets = train_test_split(nga_data1,nga_targets1, test_size=0.2, random_state=1)

# combine nga train and test
train_data1 = np.concatenate([train_data1,ngatrain], axis = 0)
test_data1 = np.concatenate([test_data1,ngatest], axis = 0)
train_targets1 = np.concatenate([train_targets1,ngatrain_targets], axis = 0)
test_targets1 = np.concatenate([test_targets1,ngatest_targets], axis = 0)

# transform data
x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_path)

# build and train the ANN
resid_train, resid_test, pre_train, pre_test = create_ANN(x_train, y_train, x_test, y_test, feature_names, numlayers, units, epochs, transform_method, folder_path)

period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]
plot_resid(resid_train, resid_test, folder_path)

mean_x_test_allT = pre_test
mean_x_train_allT = pre_train
predict_epistemic_allT = []
predict_epistemic_train_allT = []

Rindex = np.where(feature_names == 'Rrup')[0][0]

# comment out 
plot_outputs(folder_path, mean_x_test_allT, predict_epistemic_allT, mean_x_train_allT, predict_epistemic_train_allT, x_train, y_train, x_test, y_test, Rindex, period, feature_names)
plot_rawinputs(x_raw = x_train_raw, mean_x_allT = mean_x_train_allT, y=y_train, feature_names=feature_names, period = period, folder_path = folder_path + 'train/')
plot_rawinputs(x_raw = x_test_raw, mean_x_allT = mean_x_test_allT, y=y_test, feature_names=feature_names, period = period, folder_path = folder_path + 'test/')
# 
obs_pre(y_train, y_test, pre_train, pre_test, period, folder_path)

# write training predictions to a file
df_out = pd.DataFrame(resid_train, columns=['T10','T7.5','T5','T4','T3','T2','T1','T0.5','T0.2','T0.1'])   
df_out.to_csv(folder_path + 'ANNresiduals_train.csv')   

df_outtest = pd.DataFrame(resid_test, columns=['T10','T7.5','T5','T4','T3','T2','T1','T0.5','T0.2','T0.1'])   
df_out.to_csv(folder_path + 'ANNresiduals_test.csv')   
