#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:57:47 2020

@author: aklimasewski

functions for preprocessings cybershake data for ANNs
"""

def transform_dip(dip,rx):
    '''
    transforms cybershake dips and Rx
    
    Parameters
    ----------
    dip: numpy array of cybershake fault dips
    rx: numpy array of rx
    
    Returns
    -------
    transformed inputs as numpy arrays
    '''
    for i in range(len(dip)):
        if (dip[i] > 30):
            rx[i]=rx[i]*(90.-dip[i])/45.
        else:
            rx[i]=rx[i]*60./45.
            
    #return the transformed arrays
    return dip, rx

def readindata(nametrain, nametest, n=13):
    '''
    takes training and testing files and creates numpy arrays formatted for Kyle's code
    
    Parameters
    ----------
    nametrain: path of the training data file
    nametest: path of the testing data file
    n: number of input variables, corresponds to specific model
    
    Returns
    -------
    train_data1: numpy array of training features
    test_data1: numpy array of testing features
    train_targets1: numpy array of PSA of training data for 10 periods
    test_targets1: numpy array of PSA of testing data for 10 periods
    feature_names: numpy array feature names
    
    '''
    import numpy as np
    import pandas as pd


    dftrain = pd.read_pickle(nametrain) 
    dftest = pd.read_pickle(nametest)
    print(dftrain.shape)
    print(dftest.shape)
    
    # separate dataframe to pandas series (1 col of df)
    Mwtrain= dftrain["Mag"]
    distrain=dftrain["Site_Rupture_Dist"]
    vs30train=np.array(dftrain["vs30"])
    z10train=dftrain["z10"]
    z25train=dftrain["z25"]
    lattrain=dftrain["CS_Site_Lat"]
    longtrain=dftrain["CS_Site_Lon"]
    periodtrain=dftrain["siteperiod"]
    hypolattrain=dftrain["Hypocenter_Lat"]
    hypolontrain=dftrain["Hypocenter_Lon"]
    hypodepthtrain=dftrain["Hypocenter_Depth"]
    raketrain=dftrain["Rake_y"]
    diptrain=dftrain["Dip_y"]
    striketrain=dftrain["Strike_y"]+180
    widthtrain=dftrain["Width"]
    # Outputs (col per period)
    residtesttemp=dftrain.loc[:, 'IM_Value':'IM175']
    train_targets1=residtesttemp.values

    lengthtrain=dftrain["Length"]
    rjbtrain=dftrain["rjb"]
    rxtrain=dftrain["rx"]
    rytrain=dftrain["ry"]
    hypodistrain=dftrain["hypodistance"]
    Utrain=dftrain["U"]
    Ttrain=dftrain["T"]
    xitrain=dftrain["xi"]
    startdepthtrain=dftrain["Start_Depth"]
    
    # same with testdata
    Mwtest= dftest["Mag"]
    distest=dftest["Site_Rupture_Dist"]
    vs30test=np.array(dftest["vs30"])
    z10test=dftest["z10"]
    z25test=dftest["z25"]
    lattest=dftest["CS_Site_Lat"]
    longtest=dftest["CS_Site_Lon"]
    periodtest=dftest["siteperiod"]
    hypolattest=dftest["Hypocenter_Lat"]
    hypolontest=dftest["Hypocenter_Lon"]
    hypodepthtest=dftest["Hypocenter_Depth"]
    raketest=dftest["Rake_y"]
    diptest=dftest["Dip_y"]
    striketest=dftest["Strike_y"]+180
    widthtest=dftest["Width"]
    residtesttemp1=dftest.loc[:, 'IM_Value':'IM175']
    test_targets1=residtesttemp1.values
    lengthtest=dftest["Length"]
    rjbtest=dftest["rjb"]
    rxtest=dftest["rx"]
    rytest=dftest["ry"]
    hypodistest=dftest["hypodistance"]
    Utest=dftest["U"]
    Ttest=dftest["T"]
    xitest=dftest["xi"]
    startdepthtest=dftest["Start_Depth"]
    
    # diptrain, diptest, rxtrain, rxtest = transform_dip(diptrain=np.array(diptrain),diptest=np.array(diptest),rxtrain=np.array(rxtrain),rxtest=np.array(rxtest))
    diptrain, rxtrain= transform_dip(np.array(diptrain),np.array(rxtrain))
    diptest, rxtest= transform_dip(np.array(diptest),np.array(rxtest))

    
    if n == 6:
        train_data1 = np.column_stack([hypodistrain, lattrain, longtrain, hypolattrain, hypolontrain, hypodepthtrain])
        test_data1 = np.column_stack([hypodistest, lattest, longtest, hypolattest, hypolontest, hypodepthtest])
        feature_names=np.asarray(['hypoR','stlat', 'stlon', 'hypolat','hypolon', 'hypodepth'])
    
    elif n == 4:
        train_data1 = np.column_stack([lattrain, longtrain, hypolattrain, hypolontrain])
        test_data1 = np.column_stack([lattest, longtest, hypolattest, hypolontest])
        feature_names=np.asarray(['stlat', 'stlon', 'hypolat','hypolon'])
   
    # defualt, cybershake ANN, n = 12
    elif n ==12: 
        train_data1 = np.column_stack([Mwtrain,distrain,vs30train,z10train,z25train,raketrain,diptrain,hypodepthtrain, widthtrain,
                                rjbtrain,rxtrain,startdepthtrain])
        test_data1 = np.column_stack([Mwtest,distest,vs30test,z10test,z25test,raketest,diptest, hypodepthtest, widthtest,
                              rjbtest,rxtest,startdepthtest])

        feature_names=np.asarray(['Mw','Rrup','Vs30', 'Z1.0', 'Z2.5', 'Rake','Dip','Hypo_depth', 'Width',
                'Rjb','Rx','Ztor',])
    elif n==13:
        train_data1 = np.column_stack([Mwtrain,distrain,vs30train,z10train,z25train,raketrain,diptrain,hypodepthtrain, widthtrain,
                                rjbtrain,rxtrain,startdepthtrain, xitrain])
        test_data1 = np.column_stack([Mwtest,distest,vs30test,z10test,z25test,raketest,diptest, hypodepthtest, widthtest,
                              rjbtest,rxtest,startdepthtest, xitest])

        feature_names=np.asarray(['Mw','Rrup','Vs30', 'Z1.0', 'Z2.5', 'Rake','Dip','Hypo_depth', 'Width',
                'Rjb','Rx','Ztor','xi',])


    return train_data1, test_data1, train_targets1, test_targets1, feature_names



def add_locfeat(train_data1,test_data1, feature_names):
    '''
    takes training and testing arrays and adds station and hypocenter latitude and longitude
    
    Parameters
    ----------
    train_data1: numpy array of training features
    test_data1: numpy array of testing features
    feature_names: numpy array feature names
    
    Returns
    -------
    train_data1: numpy array of training features
    test_data1: numpy array of testing features
    feature_names: numpy array feature names
    
    '''
    #calculats forward azimuth between event and station and adds to training and testing data
    import numpy as np
    
    train_data1_4, test_data1_4, train_targets1_4, test_targets1_4, feature_names_4 = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = 4)
    train_data1 = np.concatenate([train_data1,train_data1_4], axis = 1)
    test_data1 = np.concatenate([test_data1,test_data1_4], axis = 1)
    feature_names = np.concatenate([feature_names,feature_names_4], axis = 0)
    
    return train_data1, test_data1, feature_names

def add_az(train_data1,test_data1, feature_names):
    '''
    takes training and testing arrays and adds forward azimuth
    
    Parameters
    ----------
    train_data1: numpy array of training features
    test_data1: numpy array of testing features
    feature_names: numpy array feature names
    
    Returns
    -------
    train_data1: numpy array of training features
    test_data1: numpy array of testing features
    feature_names: numpy array feature names
    
    '''
    # calculats forward azimuth between event and station and adds to training and testing data
    import pyproj
    import numpy as np
    
    geodesic = pyproj.Geod(ellps='WGS84')
    
    train_data1_4, test_data1_4, train_targets1_4, test_targets1_4, feature_names_4 = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = 4)
    
    # station lat lon and event lat lon
    az12,az21,distance = geodesic.inv(train_data1_4[:,3],train_data1_4[:,2],train_data1_4[:,1],train_data1_4[:,0])
    az12_test,az21_test,distance_test = geodesic.inv(test_data1_4[:,3],test_data1_4[:,2],test_data1_4[:,1],test_data1_4[:,0])


    # add the path features
    train_data1 = np.concatenate([train_data1,az12.reshape(len(az12),1)], axis = 1)
    test_data1 = np.concatenate([test_data1,az12_test.reshape(len(az12_test),1)], axis = 1)
    feature_names = np.concatenate([feature_names,np.asarray(['forward_az'])], axis = 0)
    
    return train_data1, test_data1, feature_names

def add_midpoint(train_data1,test_data1, feature_names):
    '''
    takes training and testing arrays and adds forward azimuth
    
    Parameters
    ----------
    train_data1: numpy array of training features
    test_data1: numpy array of testing features
    feature_names: numpy array feature names
    
    Returns
    -------
    train_data1: numpy array of training features
    test_data1: numpy array of testing features
    feature_names: numpy array feature names
    
    '''
    # calculated midpoint lat, lon between event and station and adds to training and testing data
    import numpy as np
    
    train_data1_4, test_data1_4, train_targets1_4, test_targets1_4, feature_names_4 = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = 4)
    
    # station lat lon and event lat lon
    midpoint = np.asarray([(train_data1_4[:,0]+train_data1_4[:,2])/2.,(train_data1_4[:,1]+train_data1_4[:,3])/2.]).T
    midpoint_test = np.asarray([(test_data1_4[:,0]+test_data1_4[:,2])/2.,(test_data1_4[:,1]+test_data1_4[:,3])/2.]).T

    # add the path features
    train_data1 = np.concatenate([train_data1,midpoint], axis = 1)
    test_data1 = np.concatenate([test_data1,midpoint_test], axis = 1)
    feature_names = np.concatenate([feature_names,np.asarray(['midpointlat','midpointlon'])], axis = 0)
    
    return train_data1, test_data1, feature_names


def transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_path):
    '''
    uses a sklearn transformation function to transform data for the ANN and creates hisogram of transformed variables
    transform_method = 'Norm' is custom normalization functions
    
    Parameters
    ----------
    transform_method: name of transformation function (ex. Normalizer(), StandardScaler())
    train_data1: numpy array of training data
    test_data1: numpy array of testing data
    train_targets1: numpy array of training targets
    test_targets1: numpy array of testing targets
    feature_names: numpy array of feature names for histogram names
    folder_path: path to save histogram as .pngs
    
    Returns
    -------
    x_train: numpy array of transformed training data
    y_train: numpy array of training targets
    x_test: numpy array of transformed testing data 
    y_test: numpy array of testing targets
    x_range: 2D list of min, max for all transformed training features
    x_train_raw: numpy array of untransformed train data
    x_test_raw: numpy array of untransformed test data
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    
    if transform_method == 'Norm':
        keep1=np.max(train_data1,axis=0)
        keep2=np.min(train_data1,axis=0)
        keep3=np.mean(train_data1,axis=0)
        train_data = 2./(np.max(train_data1,axis=0)-np.min(train_data1,axis=0))*(train_data1-np.mean(train_data1,axis=0))
        test_data = 2./(keep1-keep2)*(test_data1-keep3)

    else:
        transform = transform_method
        aa=transform.fit(train_data1[:,:])
        train_data=aa.transform(train_data1)
        test_data=aa.transform(test_data1)
    
    # plot transformed features
    for i in range(len(train_data[0])):
        plt.figure(figsize =(8,8))
        plt.title('transformed feature: ' + str(feature_names[i]))
        plt.hist(train_data[:,i])
        plt.savefig(folder_path + 'histo_transformedfeature_' + str(feature_names[i]) + '.png')
        plt.show()
    
    train_targets = train_targets1
    test_targets = test_targets1
    
    y_test = test_targets
    y_train = train_targets
    
    x_train = train_data
    x_test = test_data
    
    x_train_raw = train_data1
    x_test_raw = test_data1
    
    x_range = [[min(train_data.T[i]) for i in range(len(train_data[0]))],[max(train_data.T[i]) for i in range(len(train_data[0]))]]

    return(x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw)


