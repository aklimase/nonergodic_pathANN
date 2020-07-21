#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:57:47 2020

@author: aklimasewski
"""


def transform_dip(diptrain,diptest,rxtrain,rxtest):
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
    import numpy as np
    import pandas as pd


#Read in datasets
# nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv'
# nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv'
    
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
    
    #histogram of output (1 period of IM175)
    # plt.hist(train_targets1[:,3],100)
    
    lengthtrain=dftrain["Length"]
    rjbtrain=dftrain["rjb"]
    rxtrain=dftrain["rx"]
    rytrain=dftrain["ry"]
    hypodistrain=dftrain["hypodistance"]
    Utrain=dftrain["U"]
    Ttrain=dftrain["T"]
    xitrain=dftrain["xi"]
    startdepthtrain=dftrain["Start_Depth"]
    
    
    #### same with testdata
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
   
    
    else: #12
        train_data1 = np.column_stack([Mwtrain,distrain,vs30train,z10train,z25train,raketrain,diptrain,hypodepthtrain, widthtrain,
                                rjbtrain,rxtrain,startdepthtrain])
        test_data1 = np.column_stack([Mwtest,distest,vs30test,z10test,z25test,raketest,diptest, hypodepthtest, widthtest,
                              rjbtest,rxtest,startdepthtest])

        feature_names=np.asarray(['Mw','Rrup','Vs30', 'Z1.0', 'Z2.5', 'Rake','Dip','Hypo_depth', 'Width',
                'Rjb','Rx','Ztor',])



    return train_data1, test_data1, train_targets1, test_targets1, feature_names



# train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv')



def transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_path):
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
    
    # # Rindex = np.where(feature_names == 'hypoR')[0][0]
    # Rindex = np.where(feature_names == 'hypoR')[0][0]

    # Rtrain = x_train_raw[:,Rindex:Rindex+1]
    # Rtest = x_test_raw[:,Rindex:Rindex+1]

    return(x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw)




# x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw, Rindex, Rtrain, Rtest = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names)
