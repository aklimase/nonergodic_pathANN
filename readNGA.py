#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:23:21 2020

@author: aklimasewski
"""
import numpy as np
import pandas as pd


name = '/Users/aklimasewski/Documents/data/Updated_NGA_West2_Flatfile_RotD50_d050_public_version.csv'

dfnga = pd.read_csv(name) 


#separate dataframe to pandas series (1 col of df)

# distrain=dfnga["Site_Rupture_Dist"]
# z10train=dfnga["z10"]
# z25train=dfnga["z25"]

Mwtrain= dfnga["Earthquake Magnitude"]

vs30train=np.array(dfnga["Vs30 (m/s) selected for analysis"])

lattrain=dfnga["Station Latitude"]
longtrain=dfnga["Station Longitude"]
hypolattrain=dfnga["Hypocenter Latitude (deg)"]
hypolontrain=dfnga["Hypocenter Longitude (deg)"]
hypodepthtrain=dfnga["Hypocenter Depth (km)"]
raketrain=dfnga["Rake Angle (deg)"]
diptrain=dfnga["Dip (deg)"]
striketrain=dfnga["Strike (deg)"]+180
widthtrain=dfnga["Fault Rupture Width (km)"]

#Outputs (col per period)
periodnames = ['T0.100S','T0.200S','T0.500S','T1.000S','T2.000S','T3.000S','T4.000S','T5.000S','T7.500S','T10.000S']
residtesttemp=dfnga.loc[:, periodnames]
train_targets1=residtesttemp.values

# lengthtrain=dfnga["Length"]
rjbtrain=dfnga["Joyner-Boore Dist. (km)"]
rxtrain=dfnga["Rx"]
# rytrain=dftrain["ry"]
hypodistrain=dfnga["hypodistance"]
# Utrain=dftrain["U"]
# Ttrain=dftrain["T"]
# xitrain=dftrain["X"]
startdepthtrain=dfnga["Start_Depth"]
    
    

#defualt, cybershake ANN, n = 12
# elif n ==12: 
#     train_data1 = np.column_stack([Mwtrain,distrain,vs30train,z10train,z25train,raketrain,diptrain,hypodepthtrain, widthtrain,
#                             rjbtrain,rxtrain,startdepthtrain])
#     test_data1 = np.column_stack([Mwtest,distest,vs30test,z10test,z25test,raketest,diptest, hypodepthtest, widthtest,
#                           rjbtest,rxtest,startdepthtest])

#     feature_names=np.asarray(['Mw','Rrup','Vs30', 'Z1.0', 'Z2.5', 'Rake','Dip','Hypo_depth', 'Width',
#             'Rjb','Rx','Ztor',])
