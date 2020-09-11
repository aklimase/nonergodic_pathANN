#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:23:21 2020

@author: aklimasewski

read in NGA data and calculate 4 base gmpe average
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from base_gmpe import gmpe_avg
import sys
import os
sys.path.append(os.path.abspath('/Users/aklimasewski/Documents/python_code_nonergodic'))
from preprocessing import transform_dip

def saveNGAtargets(filename = '/Users/aklimasewski/Documents/data/Updated_NGA_West2_Flatfile_RotD50_d050_public_version.csv'):
    '''
    
    Parameters
    ----------
    filename: name of NGA file

    Returns
    saves filtered data in csv
    '''
    
    from base_gmpe import gmpe_avg
    
    # filename = '/Users/aklimasewski/Documents/data/Updated_NGA_West2_Flatfile_RotD50_d050_public_version.csv'
    dfnga = pd.read_csv(filename,index_col=0)
    
    periodnames = ['T10.000S','T7.500S','T5.000S','T4.000S','T3.000S','T2.000S','T1.000S','T0.200S','T0.500S','T0.100S']
    residtesttemp=dfnga.loc[:, periodnames]
    nga_GM=residtesttemp.values
    period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]
    
    Mwnga= dfnga["Earthquake Magnitude"]
    vs30nga=np.array(dfnga["Vs30 (m/s) selected for analysis"])
    latnga=dfnga["Station Latitude"]
    longnga=dfnga["Station Longitude"]
    hypolatnga=dfnga["Hypocenter Latitude (deg)"]
    hypolonnga=dfnga["Hypocenter Longitude (deg)"]
    hypodepthnga=dfnga["Hypocenter Depth (km)"]
    rakenga=dfnga["Rake Angle (deg)"]
    dipnga=dfnga["Dip (deg)"]
    strikenga=dfnga["Strike (deg)"]+180
    widthnga=dfnga["Fault Rupture Width (km)"]
    
    rjbnga=dfnga["Joyner-Boore Dist. (km)"]
    rxnga=dfnga["Rx"]
    hypodisnga=dfnga["HypD (km)"]
    distnga=dfnga["ClstD (km)"]
    startdepthnga=dfnga["Depth to Top Of Fault Rupture Model"]
    
    xinga = dfnga["X"]
        
    # use avg cyershake values
    z10nga=dfnga["Northern CA/Southern CA - S4 Z1 (m)"]
    z25nga=dfnga["Northern CA/Southern CA - S4 Z2.5 (m)"]
    
    dipnga,rxnga =  transform_dip(np.array(dipnga),np.array(rxnga))
    
    ztest = np.column_stack([Mwnga,distnga,vs30nga,z10nga,z25nga,rakenga,dipnga,hypodepthnga, widthnga,
                                    rjbnga,rxnga,startdepthnga, xinga,latnga,longnga,hypolatnga,hypolonnga])
        
    # feature_names=np.asarray(['Mw','Rrup','Vs30', 'Z1.0', 'Z2.5', 'Rake','Dip','Hypo_depth', 'Width',
                    # 'Rjb','Rx','Ztor','Xi','latnga','longnga','hypolatnga','hypolonnga',])
    feature_names=np.asarray(['Earthquake Magnitude','ClstD (km)','Vs30 (m/s) selected for analysis', 'Northern CA/Southern CA - S4 Z1 (m)', 'Northern CA/Southern CA - S4 Z2.5 (m)', 'Rake Angle (deg)','Dip (deg)','Hypocenter Depth (km)', 'Fault Rupture Width (km)','Joyner-Boore Dist. (km)','Rx','Depth to Top Of Fault Rupture Model','X','Station Latitude','Station Longitude','Hypocenter Latitude (deg)','Hypocenter Longitude (deg)',])
    
    lus1=dfnga["Lowest Usable Freq - H1 (Hz)"]
    lus2=dfnga["Lowest Usable Freq - H2 (H2)"]
    mm=np.max([lus1,lus2],axis=0)
    mmt=1./mm
    
    for k in range(nga_GM.shape[0]):
        indices = [i for i,v in enumerate(period >= mmt[k]) if v]
    nonnan=np.zeros(nga_GM.shape[1])
    
    for k in range(nga_GM.shape[1]):  
        nonnan[k]=np.count_nonzero(~np.isnan(nga_GM[:,k]))
        
    f1=plt.figure('Records at frequency2',figsize=(5,5))
    plt.semilogx(period,nonnan)
    plt.xlabel('Period (s)')
    plt.ylabel('Count')
    plt.title('Number of Accurate Records (PEER)')
    plt.grid()
    
    #'Mw','Rrup','Vs30', 'Z1.0' all greater than -200?
    for i in range(0,4+1):
        index=(ztest[:,i]>-200)
        ztest=ztest[index]
        print(ztest.shape)
        nga_GM=nga_GM[index]
        print (i,ztest.shape,i)
        
    #first SA values greater than 0
    index=(nga_GM[:,0]>0.0)
    ztest=ztest[index]
    nga_GM=nga_GM[index]
    print(ztest.shape,6)
    
    #seconds SA values greater than 0
    index=(nga_GM[:,1]>0.0)
    ztest=ztest[index]
    nga_GM=nga_GM[index]
    print(ztest.shape,7)
    
    #added magnitude above 5.9 here
    index=(ztest[:,0]>2.9)
    ztest=ztest[index]
    nga_GM=nga_GM[index]
    print(ztest.shape,7)
    
    #distance max of 200km
    index=(ztest[:,1]<200)
    ztest=ztest[index]
    nga_GM=nga_GM[index]
    print(ztest.shape,7)
    
    #vs30=500-710
    index=(ztest[:,2]>300)# & (ztest[:,2]<710))
    ztest=ztest[index]
    nga_GM=nga_GM[index]
    print(ztest.shape,7)
    
    #z10max=900
    index=(ztest[:,3]<900)
    ztest=ztest[index]
    nga_GM=nga_GM[index]
    print(ztest.shape,7)
    
    #z25max=5350
    index=(ztest[:,4]<5350)
    ztest=ztest[index]
    nga_GM=nga_GM[index]
    print(ztest.shape,7)
    
    index=(ztest[:,5]>5)
    ztest=ztest[index]
    nga_GM=nga_GM[index]
    print(ztest.shape,13)
    
    #lat= 33.45402,35.21083
    index=((ztest[:,13]>= 33.45) & (ztest[:,13]<=35.21))
    ztest=ztest[index]
    nga_GM=nga_GM[index]
    print(ztest.shape,7)
    
    #lon= -120.8561,   -116.4977
    index=((ztest[:,14]>= -120.8561) & (ztest[:,14]<= -116.4977))
    ztest=ztest[index]
    nga_GM=nga_GM[index]
    print(ztest.shape,7)
    
    for i in range(0,len(period)):
        index=(nga_GM[:,i]>-200)
        ztest=ztest[index]
        print(ztest.shape)
        nga_GM=nga_GM[index]
        print (i,ztest.shape,i)
    
    # in units of g
    nga_GM = np.log(9.81*nga_GM)#now in units of ln(m/s2)
    model_avg = gmpe_avg(ztest)
    NGA_targets = nga_GM - model_avg
    
    ztestdf =  pd.DataFrame(data=ztest, columns=feature_names)
    GMdf = pd.DataFrame(data=nga_GM, columns=periodnames)
    ngatargetdf = pd.DataFrame(data=NGA_targets, columns=[(periodnames[i] + 'resid') for i in range(len(periodnames))])
    #save to df
    dfsave = pd.concat([ztestdf,GMdf,ngatargetdf],axis=1)
    dfsave.to_csv('/Users/aklimasewski/Documents/data/NGA_mag2_9.csv')

#%%

def readindataNGA(filename,n=13):
    '''
    
    Parameters
    ----------
    filename: name of NGA file
    n: number of model parameters

    Returns
    nga_data1:  numpy array of nga features
    nga_targets1: numpy array of nga targets
    feature_names: list of feature names
    '''
    
    import sys
    import os
    sys.path.append(os.path.abspath('/Users/aklimasewski/Documents/python_code_nonergodic'))
    from preprocessing import transform_dip
    import pandas as pd
        
    # name = '/Users/aklimasewski/Documents/data/Updated_NGA_West2_Flatfile_RotD50_d050_public_version.csv'
    dfnga = pd.read_csv(filename,index_col=0) 

        # latmin=33,latmax=36.0,lonmin=-120.5,lonmax=-115.7
    # if region == True:
    #     stlat = (dfnga["Station Latitude"] >=33) & (dfnga["Station Latitude"] <=36)
    #     stlon = (dfnga["Station Longitude"] >=-120.5) & (dfnga["Station Longitude"] <=-115.7)
    #     evlat = (dfnga["Hypocenter Latitude (deg)"] >=33) & (dfnga["Hypocenter Latitude (deg)"] <=36)
    #     evlon = (dfnga["Hypocenter Longitude (deg)"] >=-120.5) & (dfnga["Hypocenter Longitude (deg)"] <=-115.7)
    #     dfnga = dfnga[(stlat & stlon & evlat & evlon)]        
    
    Mwnga= dfnga["Earthquake Magnitude"]
    
    vs30nga=np.array(dfnga["Vs30 (m/s) selected for analysis"])
    
    latnga=dfnga["Station Latitude"]
    longnga=dfnga["Station Longitude"]
    hypolatnga=dfnga["Hypocenter Latitude (deg)"]
    hypolonnga=dfnga["Hypocenter Longitude (deg)"]
    hypodepthnga=dfnga["Hypocenter Depth (km)"]
    rakenga=dfnga["Rake Angle (deg)"]
    dipnga=dfnga["Dip (deg)"]
    # strikenga=dfnga["Strike (deg)"]+180
    widthnga=dfnga["Fault Rupture Width (km)"]
    
    #targets
    periodnames = ['T10.000Sresid','T7.500Sresid','T5.000Sresid','T4.000Sresid','T3.000Sresid','T2.000Sresid','T1.000Sresid','T0.200Sresid','T0.500Sresid','T0.100Sresid']
    residtesttemp=dfnga.loc[:, periodnames]
    nga_targets1=residtesttemp.values
    
    # lengthtrain=dfnga["Length"]
    rjbnga=dfnga["Joyner-Boore Dist. (km)"]
    rxnga=dfnga["Rx"]
    # rytrain=dftrain["ry"]
    # hypodisnga=dfnga["HypD (km)"]
    distnga=dfnga["ClstD (km)"]

    startdepthnga=dfnga["Depth to Top Of Fault Rupture Model"]
    
    xinga = dfnga["X"]
        
    # distnga = hypodisnga
    #use avg cyershake values
    z10nga=dfnga["Northern CA/Southern CA - S4 Z1 (m)"]
    z25nga=dfnga["Northern CA/Southern CA - S4 Z2.5 (m)"]
    
    dipnga,rxnga =  transform_dip(np.array(dipnga),np.array(rxnga))
    
    
    if n ==12:
        nga_data1 = np.column_stack([Mwnga,distnga,vs30nga,z10nga,z25nga,rakenga,dipnga,hypodepthnga, widthnga,
                                    rjbnga,rxnga,startdepthnga])
        
        feature_names=np.asarray(['Mw','Rrup','Vs30', 'Z1.0', 'Z2.5', 'Rake','Dip','Hypo_depth', 'Width',
                    'Rjb','Rx','Ztor',])
    elif n==13:
        nga_data1 = np.column_stack([Mwnga,distnga,vs30nga,z10nga,z25nga,rakenga,dipnga,hypodepthnga, widthnga,
                                    rjbnga,rxnga,startdepthnga, xinga])
        
        feature_names=np.asarray(['Mw','Rrup','Vs30', 'Z1.0', 'Z2.5', 'Rake','Dip','Hypo_depth', 'Width',
                    'Rjb','Rx','Ztor','Xi',])
    
    #omit all records without targets
    # rows, cols = np.where(nga_targets1 == (-999.0))
    # rows = list(set(rows))
    # nga_targets1 = np.delete(nga_targets1,rows,axis=0)
    # nga_data1 = np.delete(nga_data1,rows,axis=0)
    
    #NGA targets
    
    return nga_data1, nga_targets1, feature_names


def add_locfeatNGA(filename, train_data1,feature_names):
    
    '''
    add station and hypocenter lat, lon
    
    Parameters
    ----------
    filename: name of NGA file
    n: number of model parameters

    Returns
    nga_data1:  numpy array of nga features
    nga_targets1: numpy array of nga targets
    feature_names: list of feature names
    '''
    import numpy as np
    import pandas as pd

    dfnga = pd.read_csv(filename) 
    
    latnga=dfnga["Station Latitude"]
    longnga=dfnga["Station Longitude"]
    hypolatnga=dfnga["Hypocenter Latitude (deg)"]
    hypolonnga=dfnga["Hypocenter Longitude (deg)"]

    train_data1_4 = np.column_stack([latnga, longnga, hypolatnga, hypolonnga])
    feature_names_4=np.asarray(['stlat', 'stlon', 'hypolat','hypolon'])
    
    train_data1 = np.concatenate([train_data1,train_data1_4], axis = 1)
    feature_names = np.concatenate([feature_names,feature_names_4], axis = 0)
    
    return train_data1, feature_names


def add_azNGA(filename, train_data1,feature_names):
    '''
    add forward aziumth as feature
    
    Parameters
    ----------
    filename: name of NGA file
    n: number of model parameters

    Returns
    nga_data1:  numpy array of nga features
    nga_targets1: numpy array of nga targets
    feature_names: list of feature names
    '''
    #calculats forward azimuth between event and station and adds to training and testing data
    import pyproj
    import numpy as np
    
    geodesic = pyproj.Geod(ellps='WGS84')
    
    # filename = '/Users/aklimasewski/Documents/data/NGAWest2region.csv'
    # filename = '/Users/aklimasewski/Documents/data/NGAWest2region_clean.csv'
    
    dfnga = pd.read_csv(filename) 
    
    latnga=dfnga["Station Latitude"]
    longnga=dfnga["Station Longitude"]
    hypolatnga=dfnga["Hypocenter Latitude (deg)"]
    hypolonnga=dfnga["Hypocenter Longitude (deg)"]

    train_data1_4 = np.column_stack([latnga, longnga, hypolatnga, hypolonnga])
    feature_names_4=np.asarray(['stlat', 'stlon', 'hypolat','hypolon'])
        
    #station lat lon and event lat lon
    az12,az21,distance = geodesic.inv(train_data1_4[:,3],train_data1_4[:,2],train_data1_4[:,1],train_data1_4[:,0])

    #add the path features
    train_data1 = np.concatenate([train_data1,az12.reshape(len(az12),1)], axis = 1)
    feature_names = np.concatenate([feature_names,np.asarray(['forward_az'])], axis = 0)
    
    return train_data1, feature_names

def add_midpointNGA(filename, train_data1, feature_names):
    '''
    add path midpoin lat, lon as feature
    
    Parameters
    ----------
    filename: name of NGA file
    n: number of model parameters

    Returns
    nga_data1:  numpy array of nga features
    nga_targets1: numpy array of nga targets
    feature_names: list of feature names
    '''
    
    #calculated midpoint lat, lon between event and station and adds to training and testing data
    import numpy as np
    
    # filename = '/Users/aklimasewski/Documents/data/NGAWest2region.csv'
    # filename = '/Users/aklimasewski/Documents/data/NGAWest2region_clean.csv'
    
    dfnga = pd.read_csv(filename) 
    
    latnga=dfnga["Station Latitude"]
    longnga=dfnga["Station Longitude"]
    hypolatnga=dfnga["Hypocenter Latitude (deg)"]
    hypolonnga=dfnga["Hypocenter Longitude (deg)"]

    train_data1_4 = np.column_stack([latnga, longnga, hypolatnga, hypolonnga])
    feature_names_4=np.asarray(['stlat', 'stlon', 'hypolat','hypolon'])
    
    #station lat lon and event lat lon
    midpoint = np.asarray([(train_data1_4[:,0]+train_data1_4[:,2])/2.,(train_data1_4[:,1]+train_data1_4[:,3])/2.]).T

    #add the path features
    train_data1 = np.concatenate([train_data1,midpoint], axis = 1)
    feature_names = np.concatenate([feature_names,np.asarray(['midpointlat','midpointlon'])], axis = 0)
    
    return train_data1, feature_names
