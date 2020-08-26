#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 17:27:09 2020

@author: aklimasewski
shakemap

plot two panel "shakemap" to compare ANN predictions and base model predictions
"""


#plot stations by ground motions predicted, actual and base

#for one event, plot predictions at stations

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath('/Users/aklimasewski/Documents/nonergodic_python_code_nonergodic'))
from preprocessing import transform_dip, readindata, transform_data, create_grid, grid_data
from model_plots import gridded_plots, plot_resid, obs_pre
from build_ANN import create_ANN
import numpy as np
import openquake

#use pandas to match event identifiers
# folder_pathmod + 'test_obs_pre.csv
folder_path = '/Users/aklimasewski/Documents/models/create_grid_2/'

# folder_pathmod = folder_path + 'ANN13_gridmidpoints_az_40ep_1_40/'
folder_pathmod = folder_path + 'ANN13_gridmidpoints_az_13ep_1_50/'
#full dataset
nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv'
nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv'

dftrain = pd.read_pickle(nametrain) 
dftest = pd.read_pickle(nametest)

dftrain = dftrain.reset_index()


#pick out ev ids
print(dftrain['Source_ID'], dftrain['Rupture_ID'])#, dftrain['Rup_Var_ID'])
#make a set for all events

eventid = []

for i in range(len(dftrain)):
    eventid.append(str(dftrain['Source_ID'][i]) + '_' + str(dftrain['Rupture_ID'][i]) + '_' + str(dftrain['Rup_Var_ID'][i]))
eventid_unique = list(set(eventid))

#find all instances of events (on all stations and grab stations and predictions at each station)
#source id 1-288, rupture id 1-1296, rup var 1-604
eventids = dftrain.loc[(dftrain['Source_ID'] == 265) & (dftrain['Rupture_ID']== 81) & (dftrain['Rup_Var_ID']==55)]
inds = eventids.index.values
# read in csvs of ytrain and prediction

#observed and predicted ground motions
dftrain_obspre = pd.read_csv(folder_pathmod + 'train_obs_pre.csv', index_col=0)
dftest_obspre  = pd.read_csv(folder_pathmod + 'test_obs_pre.csv', index_col=0)

#split into obs and predicted
colnames = dftrain_obspre.columns

target_obs = dftrain_obspre[colnames[0:10]].iloc[inds]
target_pre = dftrain_obspre[colnames[10:20]].iloc[inds]


#%%
#base models
#write a function
# import openquake.hazardlib.gsim.abrahamson_silva_2008 as abrahamson_silva_2008
# import openquake.hazardlib.gsim.abrahamson_silva_2008 as gmpeASK

# import openquake.hazardlib.gsim.boore_atkinson_2008 as boore_atkinson_2008
# import openquake.hazardlib.gsim.boore_atkinson_2008 as gmpeBSSA

# import openquake.hazardlib.gsim.campbell_bozorgnia_2008 as campbell_bozorgnia_2008
# import openquake.hazardlib.gsim.campbell_bozorgnia_2008 as gmpeCB

# import openquake.hazardlib.gsim.chiou_youngs_2008 as chiou_youngs_2008
# import openquake.hazardlib.gsim.chiou_youngs_2008 as gmpeCY

# import openquake.hazardlib.gsim.boore_atkinson_2008 as gmpeBSSA




#%% 

#open trained model
keras.models.load_model(folderpath_mod + "model")
#%%


n = 13
train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = n)
#add the location features
train_data1_4, test_data1_4, train_targets1_4, test_targets1_4, feature_names_4 = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = 4)
train_data1 = np.concatenate([train_data1,train_data1_4], axis = 1)
test_data1 = np.concatenate([test_data1,test_data1_4], axis = 1)
feature_names = np.concatenate([feature_names,feature_names_4], axis = 0)


data = train_data1

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
    plt.suptitle('Period ' + str(period[i]) + ' s')
    ax1.set_title('observation')
    ax1.scatter(evlon,evlat,marker = '*', s=0.2, c = 'gray', label = 'event', alpha = 0.02)
    ax1.scatter(sitelon,sitelat,marker = '^',s=0.2, c = 'black', label = 'site', alpha = 0.02)
    ax1.scatter(sitelon_ev, sitelat_ev, s=50, c = colors_obs)
    ax1.scatter(evlon_ev, evlat_ev, s=50, marker = '*',c = 'black')
    ax1.set_xlim(-120,-116.5)
    ax1.set_ylim(33,35.5)
   
    ax2.set_title('prediction')
    ax2.scatter(evlon,evlat,marker = '*', s=0.2, c = 'gray', label = 'event', alpha = 0.02)
    ax2.scatter(sitelon,sitelat,marker = '^',s=0.2, c = 'black', label = 'site', alpha = 0.02)
    ax2.scatter(sitelon_ev, sitelat_ev, s=50, c = colors_pre)    
    
    plt.xlim(-120,-116.5)
    plt.ylim(33,35.5)
    # plt.title(colname)
    plt.legend(loc = 'lower left')
    
    fig.subplots_adjust(right=0.8)
    cbar = plt.colorbar(s_m, orientation='vertical')
    # cbar.set_label(colname[i] + ' counts', fontsize = 20)
    plt.savefig(folder_pathmod + 'shakemap_' + str(period[i]) +'.png')
    plt.show()

    
    