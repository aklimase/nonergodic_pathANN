#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:57:47 2020

@author: aklimasewski

functions for preprocessings ANN data
"""

def transform_dip(diptrain,diptest,rxtrain,rxtest):
    '''
    transforms cybershake dips and Rx
    
    diptrain: numpy array of cybershake fault dips of training data
    diptest: numpy array of cybershake fault dips of testing data
    rxtrain: numpy array of rx train
    rxtest: numpy array of rx test
    
    returns transformed inputs as numpy arrays
    '''
    for i in range(len(diptrain)):
        if diptrain[i]>30:
            rxtrain[i]=rxtrain[i]*(90-diptrain[i])/45
        else:
            rxtrain[i]=rxtrain[i]*60/45
            
    for i in range(len(diptest)): 
        if diptest[i]>30:
            rxtest[i]=rxtest[i]*(90-diptest[i])/45
        else:
            rxtest[i]=rxtest[i]*60/45    
    #return the transformed arrays
    return diptrain, diptest, rxtrain, rxtest


def readindata(nametrain, nametest, n):
    '''
    takes training and testing files and creates numpy arrays formatted for Kyle's code
    
    nametrain: path of the training data file
    nametest: path of the testing data file
    n: number of input variables, corresponds to specific model
    
    returns
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
    
    
    #separate dataframe to pandas series (1 col of df)
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
    #Outputs (col per period)
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
    
    #same with testdata
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
    
    diptrain, diptest, rxtrain, rxtest = transform_dip(diptrain=np.array(diptrain),diptest=np.array(diptest),rxtrain=np.array(rxtrain),rxtest=np.array(rxtest))
    
    if n == 6:
        train_data1 = np.column_stack([hypodistrain, lattrain, longtrain, hypolattrain, hypolontrain, hypodepthtrain])
        test_data1 = np.column_stack([hypodistest, lattest, longtest, hypolattest, hypolontest, hypodepthtest])
        feature_names=np.asarray(['hypoR','stlat', 'stlon', 'hypolat','hypolon', 'hypodepth'])
    
    elif n == 4:
        train_data1 = np.column_stack([lattrain, longtrain, hypolattrain, hypolontrain])
        test_data1 = np.column_stack([lattest, longtest, hypolattest, hypolontest])
        feature_names=np.asarray(['stlat', 'stlon', 'hypolat','hypolon'])
   
    #defualt, cybershake ANN, n = 12
    else: 
        train_data1 = np.column_stack([Mwtrain,distrain,vs30train,z10train,z25train,raketrain,diptrain,hypodepthtrain, widthtrain,
                                rjbtrain,rxtrain,startdepthtrain])
        test_data1 = np.column_stack([Mwtest,distest,vs30test,z10test,z25test,raketest,diptest, hypodepthtest, widthtest,
                              rjbtest,rxtest,startdepthtest])

        feature_names=np.asarray(['Mw','Rrup','Vs30', 'Z1.0', 'Z2.5', 'Rake','Dip','Hypo_depth', 'Width',
                'Rjb','Rx','Ztor',])

    return train_data1, test_data1, train_targets1, test_targets1, feature_names


def transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_path):
    '''
    uses a sklearn transformation function to transform data for the ANN and creates hisogram of transformed variables
    
    transform_method: name of transformation function (ex. Normalizer(), StandardScaler())
    train_data1: numpy array of training data
    test_data1: numpy array of testing data
    train_targets1: numpy array of training targets
    test_targets1: numpy array of testing targets
    feature_names: numpy array of feature names for histogram names
    folder_path: path to save histogram as .pngs
    
    Returns
    x_train: numpy array of transformed training data
    y_train: numpy array of training targets (not transformed)
    x_test: numpy array of transformed testing data 
    y_test: numpy array of testing targets (not transformed)
    x_range: 2D list of min, max for all transformed training features
    x_train_raw: numpy array of untransformed train data
    x_test_raw: numpy array of untransformed test data
    
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    
    transform = transform_method
    aa=transform.fit(train_data1[:,:])
    train_data=aa.transform(train_data1)
    test_data=aa.transform(test_data1)
    
    #plot transformed features
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


def create_grid(latmin=32,latmax=37.5,lonmin=-121,lonmax=-115.5,dx=0.05):
    '''
    latmin: float minimum value of grid latitude, default 32 N
    latmax: float maximum value of grid latitude, default 37.5 N
    lonmin: float minimum value of grid longitude, default -121 W
    lonmax: float maximum value of grid longitude, default -115.5 W
        DESCRIPTION. The default is -115.5.
    dx: float grid spacing in degrees default is 0.05.

    Returns
    df: pandas dataframe of shapely polgons and midpoint of each grid cell in lat, lon
    lon: 1D numpy array of longitude grid vertices
    lat: 1D numpy array of latitude grid vertices
    '''

    import numpy as np
    import shapely
    import pandas as pd
    
    dx=0.1
    lon = np.arange(-121,-115.5, dx)
    lat = np.arange(32, 37.5, dx)
    
    latmid = []
    lonmid = []
    polygons = []
    for i in range(len(lon)-1):
        for j in range(len(lat)-1):
            polygon_points = [(lon[i], lat[j]), (lon[i], lat[j+1]), (lon[i+1], lat[j+1]), (lon[i+1], lat[j]), (lon[i], lat[j])]
            shapely_poly = shapely.geometry.Polygon(polygon_points)
            polygons.append(shapely_poly)
            latmid.append((lat[j]+lat[j+1])/2.)
            lonmid.append((lon[i]+lon[i+1])/2.)
               
    d = {'polygon': polygons, 'latmid': latmid, 'lonmid': lonmid}
    df = pd.DataFrame(data=d)    
    return df, lon, lat
    
def grid_data(train_data1, train_targets1, df, nsamples = 5000):
    '''
    train_data1: numpy array of training data for gridding
    train_targets1: numpy array of testing targets for gridding
    df: pandas dataframe of shapely polgons and midpoint of each grid cell in lat, lon
    nsamples: number of samples to randomly choose (for fast testing)
    
    Returns
    hypoR: numpy array of hypocentral distance for sample
    sitelat: numpy array of site latitude for sample
    sitelon: numpy array of site longitude for sample
    evlat: numpy array of event latitude for sample
    evlon: numpy array of event longitude for sample
    target: numpy array of targets for sample
    gridded_targetsnorm_list: 2D list of targets normalized by path length and multiplied by distance per cell
    gridded_counts: 2D list of path counts per grid cell
    '''
    import shapely
    import numpy as np
    import geopy
    import random
    
    randindex = random.sample(range(0, len(train_data1)), nsamples)
    
    hypoR = train_data1[:,0][randindex]
    sitelat = train_data1[:,1][randindex]
    sitelon = train_data1[:,2][randindex]
    evlat = train_data1[:,3][randindex]
    evlon = train_data1[:,4][randindex]
    target = train_targets1[:][randindex]
    
    normtarget = target / hypoR[:, np.newaxis]
    gridded_targetsnorm_list = [ [] for _ in range(df.shape[0]) ]
    
    gridded_counts = np.zeros(df.shape[0])
    lenlist = []
    
    #loop through each record     
    for i in range(len(sitelat)):                         
        line = [(evlon[i], evlat[i]), (sitelon[i], sitelat[i])]
        path=shapely.geometry.LineString(line)
        #loop through each grid cell
        for j in range(len(df)):
            shapely_poly = df['polygon'][j]
            if path.intersects(shapely_poly) == True:
                shapely_line = shapely.geometry.LineString(line)
                intersection_line = list(shapely_poly.intersection(shapely_line).coords)
                if len(intersection_line)== 2:
                    coords_1 = (intersection_line[0][1], intersection_line[0][0])
                    coords_2 = (intersection_line[1][1], intersection_line[1][0])
                    length=geopy.distance.distance(coords_1, coords_2).km
                    gridded_targetsnorm_list[j].append(normtarget[i]*length)          
                    gridded_counts[j] += 1
                    lenlist.append(length)
                
    return hypoR, sitelat, sitelon, evlat, evlon, target, gridded_targetsnorm_list, gridded_counts
    



