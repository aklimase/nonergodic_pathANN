#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 15:18:00 2020

@author: aklimase
"""

from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(1)

import numpy as np
import matplotlib.pyplot as plt
# from scipy.fftpack import fft
# import obspy
# from obspy.core import Trace,Stream,UTCDateTime
# from numpy.linalg import inv, qr
import pandas as pd
# from pandas import Series, DataFrame
# import time
# from itertools import compress
from sklearn.preprocessing import power_transform
from sklearn.preprocessing import PowerTransformer
import keras
import tensorflow as tf
from keras import layers
from keras import optimizers
from keras.layers import Dropout
# import tensorflow as tf
# import re
# import pickle
# from openquake.hazardlib.gsim.boore_2014 import BooreEtAl2014
# import pygmm
# import glob
from keras.callbacks import EarlyStopping
# import gc
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from keras.models import Sequential

#Read in datasets
nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv'
nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv'

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
plt.hist(train_targets1[:,3],100)

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



#########################


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

diptrain, diptest, rxtrain, rxtest = transform_dip(diptrain=np.array(diptrain),diptest=np.array(diptest),rxtrain=np.array(rxtrain),rxtest=np.array(rxtest))
###############################


#put together arrays of features for ANN
train_data1 = np.column_stack([Mwtrain,distrain,vs30train,z10train,z25train,raketrain,diptrain,hypodepthtrain, widthtrain,
                               rjbtrain,rxtrain,startdepthtrain])
test_data1 = np.column_stack([Mwtest,distest,vs30test,z10test,z25test,raketest,diptest, hypodepthtest, widthtest,
                             rjbtest,rxtest,startdepthtest])

feature_names=['Mw','Rrup','Vs30', 'Z1.0', 'Z2.5', 'Rake','Dip','Hypo_depth', 'Width',
               'Rjb','Rx','Ztor',]

### what is z?
ztestcyber=train_data1
ztestcybertest=test_data1

#various periods datasets?
#period=[10,7.5,5,4,3,2,1,.5,.2,.1]
period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]
# period=[10,7.5,5,3,2,1]
period=np.array(period)



#preprocessing transform inputs data to be guassian shaped
pt = PowerTransformer()
aa=pt.fit(train_data1[:,:])
train_data=aa.transform(train_data1)
test_data=aa.transform(test_data1)

train_targets = train_targets1
test_targets = test_targets1



def build_model():
    #model=models.Sequential()
    model = Sequential()
    #model.add(Dropout(0.0,seed=1))
    model.add(layers.Dense(6,activation='sigmoid', input_shape=(train_data.shape[1],)))

    model.add(Dropout(rate=0.0, trainable =True))#trainint=true v mismatch
    
    model.add(layers.Dense(train_targets.shape[1])) #add sigmoid aciivation functio? (only alues betwen 0 and 1)

    model.compile(optimizer=optimizers.Adam(lr=2e-3),loss='mse',metrics=['mae','mse']) 
    #model.compile(optimizer='adam',loss='mse',metrics=['mae']) 
    return model


model=build_model()

es = EarlyStopping(monitor='val_mean_absolute_error', mode='max', min_delta=10)



# history=model.fit(train_data1,train_targets1,validation_data=(test_data1,test_targets1),epochs=13,batch_size=256,verbose=1)
# mae_history=history.history['val_mae']
# mae_history_train=history.history['mae']
# test_mse_score,test_mae_score,tempp=model.evaluate(test_data1,test_targets1)


history=model.fit(train_data,train_targets,validation_data=(test_data,test_targets),epochs=13,batch_size=256,verbose=1)
mae_history=history.history['val_mae']
mae_history_train=history.history['mae']
test_mse_score,test_mae_score,tempp=model.evaluate(test_data,test_targets)




folder_path = '/Users/aklimasewski/Documents/cybershakeANN/'



#make a plotting function
f10=plt.figure('Overfitting Test')
plt.plot(mae_history,label='Testing Data')
plt.plot(mae_history_train,label='Training Data')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('Overfitting Test')
plt.legend()
print(test_mae_score)
plt.grid()
plt.savefig(folder_path + 'resid_T.png')



a=model.predict(test_data)
b=model.predict(train_data)
f1=plt.figure('Earthquake Magnitude normalized Prediction')
plt.plot(train_data[:,0],b[:,1],'.r', label='train data')
plt.plot(test_data[:,0],a[:,1],'.b',label='test data')
plt.xlabel('input')
plt.ylabel('Prediction')
plt.title('Earthquake Magnitude normalized Prediction')
plt.legend()
plt.savefig(folder_path + 'Mag_vs_pre.png')
plt.show()



f11=plt.figure('Joyner-Boore Dist. (km) normalized Prediction')
plt.plot(train_data[:,1],b[:,1],'.r',label='train data')
plt.plot(test_data[:,1],a[:,1],'.b',label='test data')
plt.xlabel('input')
plt.ylabel('Prediction')
plt.ylabel('Prediction')
plt.title('Joyner-Boore Dist. (km) normalized Prediction')
plt.legend()
plt.savefig(folder_path + 'Dist_vs_pre.png')
plt.show()






f1=plt.figure('Earthquake Magnitude normalized Actual')
plt.plot(train_data[:,0],train_targets[:,1],'.r')
#plt.plot(test_data[:,0],test_targets[:,0],'.b')
plt.xlabel('input')
plt.ylabel('Prediction')
plt.title('Earthquake Magnitude normalized Actual')
plt.savefig(folder_path + 'Mag_vs_actual.png')
plt.show()

f11=plt.figure('Joyner-Boore Dist. (km) normalized Actual')
plt.plot(train_data[:,1],train_targets[:,1],'.r')
#plt.plot(test_data[:,1],test_targets[:,1],'.b')
plt.xlabel('input')
plt.ylabel('Prediction')
plt.title('Joyner-Boore Dist. (km) normalized Actual')
plt.savefig(folder_path + 'Dist_vs_actual.png')
plt.show()
 


f212=plt.figure('T = 5.0 s')
plt.hist(train_targets[:,2]-b[:,2],100,label='Training')
plt.hist(test_targets[:,2]-a[:,2],100,label='Testing')
plt.xlabel('Residual ln(Target/Predicted)')
plt.ylabel('Count')
temp1=str(np.std(train_targets[:,2]-b[:,2]))
temp2=str(np.std(test_targets[:,2]-a[:,2]))
temp11=str(np.mean(train_targets[:,2]-b[:,2]))
temp22=str(np.mean(test_targets[:,2]-a[:,2]))
plt.text(-3,3000,   'sigma_train = '+ temp1[0:4])
plt.text(-3,2500,   'sigma_test =' + temp2[0:4])
plt.text(2,3000,   'mean_train =  '+ temp11[0:4])
plt.text(2,2500,   'mean_test = '+ temp22[0:4])
plt.title('Residual ln(Target/Predicted)): T = 5.0 s')
plt.legend()
plt.savefig(folder_path + 'Histo5s.png')
plt.show()


f212=plt.figure('T = 0.5 s')
plt.hist(train_targets[:,4]-b[:,4],100,label='Training')
plt.hist(test_targets[:,4]-a[:,4],100,label='Testing')
plt.xlabel('Residual ln(Target/Predicted)')
plt.ylabel('Count')
temp1=str(np.std(train_targets[:,4]-b[:,4]))
temp2=str(np.std(test_targets[:,4]-a[:,4]))
temp11=str(np.mean(train_targets[:,4]-b[:,4]))
temp22=str(np.mean(test_targets[:,4]-a[:,4]))
plt.text(1.2,6000,   'sigma_train = '+ temp1[0:4])
plt.text(1.2,5500,   'sigma_test =' + temp2[0:4])
plt.text(1.2,3000,   'mean_train =  '+ temp11[0:4])
plt.text(1.2,2500,   'mean_test = '+ temp22[0:4])
plt.title('Residual ln(Target/Predicted): T = 0.5 s')
plt.legend()
plt.savefig(folder_path + 'Histo_5s.png')
plt.show()



diff=np.std(train_targets-b,axis=0)
difftest=np.std(test_targets-a,axis=0)
diffmean=np.mean(train_targets-b,axis=0)
f22=plt.figure('Difference Std of residuals vs Period')
plt.semilogx(period,diff,label='Training ')
plt.semilogx(period,difftest,label='Testing')
plt.xlabel('Period')
plt.ylabel('Total Standard Deviation')
plt.savefig(folder_path + 'Stddev.png')
plt.show()




diffmean=np.mean(train_targets-b,axis=0)
diffmeantest=np.mean(test_targets-a,axis=0)
f22=plt.figure('Difference Std of residuals vs Period')
plt.semilogx(period,diffmean,label='Training ')
plt.semilogx(period,diffmeantest,label='Testing')
plt.xlabel('Period')
plt.ylabel('Mean residual (target - prediction)')
plt.legend()
plt.savefig(folder_path + 'mean_T.png')
plt.show()









