#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 13:41:36 2020

@author: aklimasewski
"""


#data histograms

import sys
import os
sys.path.append(os.path.abspath('/Users/aklimasewski/Documents/python_code_nonergodic'))
from preprocessing import transform_dip, readindata, transform_data, add_az, add_locfeat
from model_plots import plot_resid, obs_pre, plot_outputs, plot_rawinputs
from build_ANN import create_ANN
from base_gmpe import gmpe_avg
from readNGA import readindataNGA, transform_dataNGA,add_locfeatNGA,add_azNGA
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

sns.reset_defaults()
sns.set_style('whitegrid')
sns.set_context('talk')
sns.set_context(context='talk',font_scale=0.8)

folder_path = '/Users/aklimasewski/Documents/data/histograms/'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    os.makedirs(folder_path + 'cybershake/')
    os.makedirs(folder_path + 'NGA_SA/')
    os.makedirs(folder_path + 'NGAopenquake/')
    os.makedirs(folder_path + 'NGAresid/')

    
period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]

#load in training and testing data
n=13
train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = n)

for i in range(len(period)):
    plt.figure(figsize =(8,6))
    plt.hist(train_targets1[:,i],bins=50)
    plt.title('Cybershake train targets T= '+ str(period[i]) + ' s')
    plt.xlabel('target values')
    plt.ylabel('counts')
    plt.savefig(folder_path + 'cybershake/T_' + str(period[i]) + '.png')
    plt.show()

filename = '/Users/aklimasewski/Documents/data/NGAWest2region_clean.csv'
#load NGA data (targets in g?)
nga_data1, nga_targets1, feature_names = readindataNGA(filename,n=n)

for i in range(len(period)):
    plt.figure(figsize =(8,6))
    plt.hist(nga_targets1[:,i],bins=80)
    plt.title('NGA targets T= '+ str(period[i]) + ' s in g?')
    plt.xlabel('target values')
    plt.ylabel('counts')
    xmax = np.percentile(nga_targets1[:,i],98)
    plt.xlim(0,xmax)
    plt.savefig(folder_path + 'NGA_SA/g_T_' + str(period[i]) + '.png')
    plt.show()
    
    plt.figure(figsize =(8,6))
    plt.hist(np.log(nga_targets1[:,i]),bins=40)
    plt.title('NGA targets T= '+ str(period[i]) + ' s in lng?')
    plt.xlabel('target values')
    plt.ylabel('counts')
    # xmax = np.percentile(np.log(9.81*nga_targets1[:,i]),98)
    # plt.xlim(0,xmax)
    plt.savefig(folder_path + 'NGA_SA/lng_T_' + str(period[i]) + '.png')
    plt.show()

    plt.figure(figsize =(8,6))
    plt.hist(9.81*nga_targets1[:,i],bins=80)
    plt.title('NGA targets T= '+ str(period[i]) + ' s in m/s2?')
    plt.xlabel('target values')
    plt.ylabel('counts')
    xmax = np.percentile(9.81*nga_targets1[:,i],98)
    plt.xlim(0,xmax)
    plt.savefig(folder_path + 'NGA_SA/ms_T_' + str(period[i]) + '.png')
    plt.show()
    
#openquake predictions of NGA data in ln(g)
# model_avg = gmpe_avg(nga_data1)
# df_out = pd.DataFrame(model_avg, columns=['T10','T7.5','T5','T4','T3','T2','T1','T0.5','T0.2','T0.1'])   
# df_out.to_csv('/Users/aklimasewski/Documents/data/NGAopenquake_predictions_lng.csv')  

model_avg = np.asarray(pd.read_csv('/Users/aklimasewski/Documents/data/NGAopenquake_predictions_lng.csv',index_col=0))
for i in range(len(period)):
    plt.figure(figsize =(8,6))
    plt.hist(model_avg[:,i],bins=40)
    plt.title('NGA openquake predictions T= '+ str(period[i]) + ' s in ln g?')
    plt.xlabel('target values')
    plt.ylabel('counts')
    # xmax = np.percentile(model_avg[:,i],98)
    # plt.xlim(0,xmax)
    plt.savefig(folder_path + 'NGAopenquake/T_' + str(period[i]) + '.png')
    plt.show()
    
    plt.figure(figsize =(8,6))
    plt.hist(np.exp(model_avg[:,i]),bins=40)
    plt.title('NGA openquake predictions T= '+ str(period[i]) + ' s in g?')
    plt.xlabel('target values')
    plt.ylabel('counts')
    xmax = np.percentile(np.exp(model_avg[:,i]),98)
    plt.xlim(0,xmax)
    plt.savefig(folder_path + 'NGAopenquake/g_T_' + str(period[i]) + '.png')
    plt.show()


nga_resid = np.log(nga_targets1) - model_avg

for i in range(len(period)):
    plt.figure(figsize =(8,6))
    plt.hist(nga_resid[:,i],bins=40)
    plt.title('NGA GM residuals T= '+ str(period[i]) + ' s in ln g?')
    plt.xlabel('target values')
    plt.ylabel('counts')
    # xmax = np.percentile(nga_resid[:,i],98)
    # plt.xlim(0,xmax)
    plt.savefig(folder_path + 'NGAresid/T_' + str(period[i]) + '.png')
    plt.show()






