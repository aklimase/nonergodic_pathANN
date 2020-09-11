#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 17:27:09 2020

@author: aklimasewski
GMmap

reads in cybershake data, finds list of unique source ids
plot two panel "map" to compare ANN predictions and base model predictions spatial
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath('/Users/aklimasewski/Documents/python_code_nonergodic'))
from preprocessing import transform_dip, readindata, transform_data, add_locfeat, add_az #, create_grid, grid_data
from model_plots import plot_resid, obs_pre
from build_ANN import create_ANN
import numpy as np
import openquake

# path of model to be plotted
folder_pathmod =  '/Users/aklimasewski/Documents/model_results/base/ANN18_xi_normalizer_40ep_50hidden/'

def unique_events(df):
    '''
    sorts through dataframe and creates list of unique events with 'Source_ID' and 'Rupture_ID' 
    
    Parameters
    ----------
    df: dataframe from cybershake csv file
    
    Returns
    -------
    eventid_unique: list of string source id and rupture id
    '''
    
    # pick out ev ids
    print(df['Source_ID'], df['Rupture_ID'])#, dftrain['Rup_Var_ID'])
    
    eventid = []
    for i in range(len(df)):
        eventid.append(str(df['Source_ID'][i]) + '_' + str(df['Rupture_ID'][i])) #+ '_' + str(dftrain['Rup_Var_ID'][i]))
    eventid_unique = list(set(eventid))
    return eventid_unique


nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv'
nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv'

dftrain = pd.read_pickle(nametrain) 
dftest = pd.read_pickle(nametest)
dftrain = dftrain.reset_index()
dftest = dftest.reset_index()

eventid_unique  = unique_events(dftrain)
print('number of unique events: ' + len(eventid_unique))

'''
set the sourceid and ruptureid parameters (from eventid_unique)
reads in training and testing data and finds location paramters for events
'''
# choose event for plotting
sourceid = 148
ruptureid = 1

# find all instances of events (on all stations and grab stations and predictions at each station)
eventids = dftrain.loc[(dftrain['Source_ID'] == sourceid) & (dftrain['Rupture_ID']== ruptureid )]# & (dftrain['Rup_Var_ID']==55)]
inds = eventids.index.values

# observed and predicted ground motions
dftrain_obspre = pd.read_csv(folder_pathmod + 'train_obs_pre.csv', index_col=0)
dftest_obspre  = pd.read_csv(folder_pathmod + 'test_obs_pre.csv', index_col=0)

# split into obs and predicted
colnames = dftrain_obspre.columns

target_obs = dftrain_obspre[colnames[0:10]].iloc[inds]
target_pre = dftrain_obspre[colnames[10:20]].iloc[inds]

n = 13
train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = n)

# add the location features
train_data1,test_data1, feature_names = add_locfeat(train_data1,test_data1, feature_names)
train_data1,test_data1, feature_names = add_az(train_data1,test_data1, feature_names)

data = train_data1

# location parameters for plots
sitelat = data[:,13]
sitelon = data[:,14]
evlat = data[:,15]
evlon = data[:,16]

sitelat_ev = data[:,13][inds]
sitelon_ev = data[:,14][inds]
evlat_ev = data[:,15][inds]
evlon_ev = data[:,16][inds]

period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]

for i in range(len(period)):
    obs = np.asarray(target_obs)[:,i]
    pre = np.asarray(target_pre)[:,i]

    cbound = np.max(np.abs(obs))
    cmap = mpl.cm.get_cmap('seismic')
    normalize = mpl.colors.Normalize(vmin=-1*cbound, vmax=cbound)
    colors_obs = [cmap(normalize(value)) for value in obs]
    colors_pre = [cmap(normalize(value)) for value in pre]

    s_m = mpl.cm.ScalarMappable(cmap = cmap, norm=normalize)
    s_m.set_array([])
    
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12,6))
    plt.suptitle('Period ' + str(period[i]) + ' s, Source ID ' + str(sourceid) + ' Rup ID ' + str(ruptureid))
    ax1.set_title('observation')
    ax1.scatter(evlon,evlat,marker = '*', s=0.2, c = 'gray', label = 'event', alpha = 0.02)
    ax1.scatter(sitelon,sitelat,marker = '^',s=0.2, c = 'black', label = 'site', alpha = 0.02)
    ax1.scatter(sitelon_ev, sitelat_ev, s=50, c = colors_obs)
    ax1.scatter(evlon_ev, evlat_ev, s=50, marker = '*',c = 'black')
    ax1.set_xlim(-120,-116.5)
    ax1.set_ylim(33,35.5)
   
    ax2.set_title('ANN prediction')
    ax2.scatter(evlon,evlat,marker = '*', s=0.2, c = 'gray', label = 'event', alpha = 0.02)
    ax2.scatter(sitelon,sitelat,marker = '^',s=0.2, c = 'black', label = 'site', alpha = 0.02)
    ax2.scatter(sitelon_ev, sitelat_ev, s=50, c = colors_pre)    
    plt.xlim(-120,-116.5)
    plt.ylim(33,35.5)
    plt.legend(loc = 'lower left')
    
    plt.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
    cbar = plt.colorbar(s_m, orientation='vertical',cax=cbar_ax)
    plt.savefig(folder_pathmod + 'map_' + str(period[i]) +'.png')
    plt.show()

    
    