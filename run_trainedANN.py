#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 17:21:36 2020

@author: aklimasewski
"""
import sys
import os
sys.path.append(os.path.abspath('/Users/aklimasewski/Documents/python_code_nonergodic'))
from preprocessing import transform_dip, readindata, transform_data, add_az, add_locfeat
from model_plots import plot_resid, obs_pre, plot_outputs, plot_rawinputs
from build_ANN import create_ANN
from readNGA import readindataNGA, transform_dataNGA,add_locfeatNGA,add_azNGA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from tensorflow import random
random.set_seed(1)

# import gc
import seaborn as sns
sns.set(style="ticks", color_codes=True)
import os
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import random
import keras

# For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float64')
sns.reset_defaults()
sns.set_style('whitegrid')
sns.set_context('talk')
sns.set_context(context='talk',font_scale=0.7)

# path of trained model files
folder_path = '/Users/aklimasewski/Documents/model_results/base/ANNbase_nga_20ep_50hidden/'
folder_pathNGA = folder_path + 'NGAtest/'
if not os.path.exists(folder_pathNGA):
    os.makedirs(folder_pathNGA)
    
n = 13
az = True
transform_method = 'Norm'

# compare to NGA data
filenamenga = '/Users/aklimasewski/Documents/data/NGA_mag2_9.csv'

nga_data1, nga_targets1, feature_names = readindataNGA(filenamenga,n)
nga_data1, feature_names = add_azNGA(filenamenga, nga_data1, feature_names)
# nga_data1,feature_names = add_locfeatNGA(filenamenga,nga_data1,feature_names)

if az == True:
    nga_data1, feature_names = add_azNGA(nga_data1,feature_names)
    
# read in cyber shake trainineg and testing data
train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = n)
train_data1,test_data1, feature_names = add_az(train_data1,test_data1, feature_names)

x_train, y_train, x_nga, y_nga, x_range, x_train_raw,  x_nga_raw = transform_data(transform_method, train_data1, nga_data1, train_targets1, nga_targets1, feature_names, folder_pathNGA)

# load model and fit
loadedmodel =  keras.models.load_model(folder_path + 'model/')

pre_nga = loadedmodel.predict(x_nga)
resid_nga = np.asarray(nga_targets1) - pre_nga

pre_train = loadedmodel.predict(x_train)
resid_train = np.asarray(train_targets1) - pre_train

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

