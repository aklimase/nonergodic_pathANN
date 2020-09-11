#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 13:19:45 2020

@author: aklimasewski

build a model that takes cybershake data, input to 12 feature ANN,input those residuals to a spatial ANN

use spatial kernel in second model
"""

import sys
import os
sys.path.append(os.path.abspath('/Users/aklimasewski/Documents/python_code_nonergodic'))
from preprocessing import transform_dip, readindata, transform_data
from model_plots import plot_resid
from build_ANN import create_ANN
from pprint import pprint
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
import cartopy
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

'''
first ANN with the base model features
'''
topdir = '/Users/aklimasewski/Documents/'
folder_path = topdir + 'model_results/2step_ANN/model13/'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
transform_method = 'Norm'
epochs = 15

train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain= topdir + 'data/cybertrainyeti10_residfeb.csv', nametest=topdir + 'data/cybertestyeti10_residfeb.csv', n=13)
x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_path)

numlayers = 1
units= [20]

resid_train, resid_test, pre_train, pre_test = create_ANN(x_train, y_train, x_test, y_test, feature_names, numlayers, units, epochs, transform_method, folder_path)

period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]
plot_resid(resid_train, resid_test, folder_path)
#%%

'''
second ANN with residuals as targets and event station lat lon as features
'''

transform_method =  'Norm'
epochs = 25

folder_path = topdir + 'models/2step_ANN/model4_residuals_25ep_2_20/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain=topdir + 'data/cybertrainyeti10_residfeb.csv', nametest=topdir + 'data/cybertestyeti10_residfeb.csv', n=4)

#redefine targets
train_targets1 = resid_train
test_targets1 = resid_test

x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw  = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_path)

numlayers = 2
units= [20,20]

resid_train2, resid_test2, pre_train, pre_test = create_ANN(x_train, y_train, x_test, y_test, feature_names, numlayers, units, epochs, transform_method, folder_path)

period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]
plot_resid(resid_train2, resid_test2, folder_path)
