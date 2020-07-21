#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 08:58:44 2020

@author: kwithers
"""

from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(seed)

import numpy as np

import matplotlib.pyplot as plt
from scipy.fftpack import fft
import obspy
from obspy.core import Trace,Stream,UTCDateTime
from numpy.linalg import inv, qr
import pandas as pd
from pandas import Series, DataFrame
import time
from itertools import compress
from sklearn.preprocessing import power_transform
from sklearn.preprocessing import PowerTransformer

import keras

import tensorflow as tf

from tf.keras import layers
from keras import optimizers
from keras.layers import Dropout
import tensorflow as tf
import re
import pickle
#from openquake.hazardlib.gsim.boore_2014 import BooreEtAl2014
import pygmm
import glob
from keras.callbacks import EarlyStopping
import gc
import seaborn as sns; sns.set(style="ticks", color_codes=True)



nametrain='/Users/kwithers/march/cybertrainyeti10_residfeb.csv'
nametest='/Users/kwithers/march/cybertestyeti10_residfeb.csv'


dftrain = pd.read_pickle(nametrain) 
dftest = pd.read_pickle(nametest)


df1 = pd.read_csv('/Users/kwithers/march/lonlatcyber_TableToExcel3.csv')

print(dftrain.shape)
print(dftest.shape)

dftrain=pd.merge_asof(dftrain.sort_values('CS_Site_Lat'), df1.sort_values('CS_Site_Lat'), on='CS_Site_Lat', direction='nearest')
dftest=pd.merge_asof(dftest.sort_values('CS_Site_Lat'), df1.sort_values('CS_Site_Lat'), on='CS_Site_Lat', direction='nearest')


#df = pd.read_csv('/Users/kwithers/Downloads/Updated_NGA_West2_Flatfile_RotD50_d050_public_version.csv')
#df1 = pd.read_csv('/Users/kwithers/Downloads/lonlatcyber_TableToExcel3.csv')
#dftrain=dftrain.merge(df1, left_on='CS_Site_Lat', right_on='lat')
#dftest=dftest.merge(df1, left_on='CS_Site_Lat', right_on='lat')
#pause

plt.close('all')
Mwtrain= dftrain["Mag"]
distrain=dftrain["Site_Rupture_Dist"]
vs30train=np.array(dftrain["vs30"])


#index=(vs30train<500)
#vs30train[index]=500


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

residtesttemp=dftrain.loc[:, 'IM_Value':'IM175']
#train_targets1=np.log(residtesttemp.values/100)
train_targets1=residtesttemp.values


plt.close('all')
plt.hist(train_targets1[:,3],100)
#pause

lengthtrain=dftrain["Length"]
rjbtrain=dftrain["rjb"]
rxtrain=dftrain["rx"]
rytrain=dftrain["ry"]
hypodistrain=dftrain["hypodistance"]

Utrain=dftrain["U"]
Ttrain=dftrain["T"]
xitrain=dftrain["xi"]


startdepthtrain=dftrain["Start_Depth"]


b1train=dftrain["dist_z1pt0_50m"]
b2train=dftrain["dist_z1pt0_100m"]
b3train=dftrain["dist_z1pt0_200m"]
b4train=dftrain["dist_z1pt0_300m"]
b5train=dftrain["dist_z1pt0_400m"]
b6train=dftrain["dist_z1pt0_500m"]

########################################################
Mwtest= dftest["Mag"]
distest=dftest["Site_Rupture_Dist"]
vs30test=np.array(dftest["vs30"])

#index=(vs30test<500)
#vs30test[index]=500


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


#Rake: -90:180
#Strike:-180:180
#dip: 0-90

residtesttemp1=dftest.loc[:, 'IM_Value':'IM175']
#test_targets1=np.log(residtesttemp1.values/100)
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

b1test=dftest["dist_z1pt0_50m"]
b2test=dftest["dist_z1pt0_100m"]
b3test=dftest["dist_z1pt0_200m"]
b4test=dftest["dist_z1pt0_300m"]
b5test=dftest["dist_z1pt0_400m"]
b6test=dftest["dist_z1pt0_500m"]
######################################################## 


#ztest = np.column_stack([Mwtest,distest,vs30test,aziumuthtest, z1test, z2p5test])
#ztest = np.column_stack([Mwtest,distest,vs30test, z1test, z2p5test,ztt,aziumuthtest, strike,mm,ID])
#ztest = np.column_stack([Mwtest,distest,vs30test, z1test, z2p5test,ztt,aziumuthtest, strike,mm,ID,staLontest,staLattest])
#'CS_Site_Lat','CS_Site_Lon'

#train_data = np.column_stack([Mwtrain,distrain,vs30train,z10train,z25train])
#test_data = np.column_stack([Mwtest,distest,vs30test,z10test,z25test])

#train_data = np.column_stack([Mwtrain,distrain,vs30train,z10train,z25train,raketrain,diptrain,striketrain,hypodepthtrain, lattrain, longtrain, hypolattrain,hypolontrain,widthtrain,
#lengthtrain,rjbtrain,rxtrain,rytrain,hypodistrain,startdepthtrain, b6train]) #Utrain, Ttrain, xitrain,,b1train,b2train,b3train,b4train,b5train,
#
#test_data = np.column_stack([Mwtest,distest,vs30test,z10test,z25test,raketest,diptest,striketest, hypodepthtest, lattest, longtest,hypolattest,hypolontest,widthtest,
#                             lengthtest,rjbtest,rxtest,rytest,hypodistest,startdepthtest, b6test]) #Utest, Ttest, xitest,b1test,b2test,b3test,b4test,b5test,


#train_data = np.column_stack([Mwtrain,distrain,vs30train,z10train,z25train,raketrain,diptrain,striketrain,hypodepthtrain, widthtrain,
#lengthtrain,rjbtrain,rxtrain,rytrain,hypodistrain,startdepthtrain]) #Utrain, Ttrain, xitrain,,b1train,b2train,b3train,b4train,b5train,
#
#test_data = np.column_stack([Mwtest,distest,vs30test,z10test,z25test,raketest,diptest,striketest, hypodepthtest, widthtest,
#                             lengthtest,rjbtest,rxtest,rytest,hypodistest,startdepthtest]) #Utest, Ttest, xitest,b1test,b2test,b3test,b4test,b5test,


#feature_names=['Mw','Rrup','Vs30', 'z1.0', 'z2.5', 'Rake','Dip','Strike', 'Hypo_depth', 'Width',
#               'Length','Rjb','Rx','Ry','Hypo Dis','Depth to Top', 'b6']

#train_data=train_data[index]

#Mwtrain*Mwtrain, np.log(distrain)

diptrain=np.array(diptrain)
diptest=np.array(diptest)
rxtrain=np.array(rxtrain)
rxtest=np.array(rxtest)

rec1=-1
for ij in diptrain:
    rec1=rec1+1
    if diptrain[rec1]>30:
        rxtrain[rec1]=rxtrain[rec1]*(90-diptrain[rec1])/45
    else:
        rxtrain[rec1]=rxtrain[rec1]*60/45

rec1=-1
for ij in diptest:
    rec1=rec1+1    
    if diptest[rec1]>30:
        rxtest[rec1]=rxtest[rec1]*(90-diptest[rec1])/45
    else:
        rxtest[rec1]=rxtest[rec1]*60/45    




train_data1 = np.column_stack([Mwtrain,distrain,vs30train,z10train,z25train,raketrain,diptrain,hypodepthtrain, widthtrain,
                               rjbtrain,rxtrain,startdepthtrain])
#rytrain,
test_data1 = np.column_stack([Mwtest,distest,vs30test,z10test,z25test,raketest,diptest, hypodepthtest, widthtest,
                             rjbtest,rxtest,startdepthtest])




feature_names=['Mw','Rrup','Vs30', 'Z1.0', 'Z2.5', 'Rake','Dip','Hypo_depth', 'Width',
               'Rjb','Rx','Ztor',]


ztestcyber=train_data
ztestcybertest=test_data


#period=[10,7.5,5,4,3,2,1,.5,.2,.1]
period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]
period=[10,7.5,5,3,2,1]
period=np.array(period)


test_data1=test_data





#pause
pt = PowerTransformer()
aa=pt.fit(train_data[:,:])
train_data=aa.transform(train_data)
test_data=aa.transform(test_data)




from keras.models import Sequential


def build_model():
    #model=models.Sequential()
    model = Sequential()
    #model.add(Dropout(0.0,seed=1))
    model.add(layers.Dense(6,activation='sigmoid', input_shape=(train_data.shape[1],)))

    model.add(Dropout(rate=0.0, training=True))
    
    model.add(layers.Dense(train_targets.shape[1])) #add sigmoid aciivation functio? (only alues betwen 0 and 1)

    model.compile(optimizer=optimizers.Adam(lr=2e-3),loss='mse',metrics=['mae','mse']) 
    #model.compile(optimizer='adam',loss='mse',metrics=['mae']) 
    return model


model=build_model()

es = EarlyStopping(monitor='val_mean_absolute_error', mode='max', min_delta=10)



history=model.fit(train_data,train_targets,validation_data=(test_data,test_targets),epochs=13,batch_size=256,verbose=1)
mae_history=history.history['val_mae']
mae_history_train=history.history['mae']
test_mse_score,test_mae_score,tempp=model.evaluate(test_data,test_targets)

f10=plt.figure('Overfitting Test')
plt.plot(mae_history,label='Testing Data')
plt.plot(mae_history_train,label='Training Data')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('Overfitting Test')
plt.legend()
print(test_mae_score)
plt.grid()


a=model.predict(test_data)
b=model.predict(train_data)
f1=plt.figure('Earthquake Magnitude normalized Prediction')
plt.plot(train_data[:,0],b[:,1],'.r', label='train data')
plt.plot(test_data[:,0],a[:,1],'.b',label='test data')
plt.xlabel('input')
plt.ylabel('Prediction')
plt.title('Earthquake Magnitude normalized Prediction')
plt.show()
plt.legend()

f11=plt.figure('Joyner-Boore Dist. (km) normalized Prediction')
plt.plot(train_data[:,1],b[:,1],'.r',label='train data')
plt.plot(test_data[:,1],a[:,1],'.b',label='test data')
plt.xlabel('input')
plt.ylabel('Prediction')
plt.ylabel('Prediction')
plt.title('Joyner-Boore Dist. (km) normalized Prediction')
plt.show()
plt.legend()



f1=plt.figure('Earthquake Magnitude normalized Actual')
plt.plot(train_data[:,0],train_targets[:,1],'.r')
#plt.plot(test_data[:,0],test_targets[:,0],'.b')
plt.xlabel('input')
plt.ylabel('Prediction')
plt.title('Earthquake Magnitude normalized Actual')
plt.show()

f11=plt.figure('Joyner-Boore Dist. (km) normalized Actual')
plt.plot(train_data[:,1],train_targets[:,1],'.r')
#plt.plot(test_data[:,1],test_targets[:,1],'.b')
plt.xlabel('input')
plt.ylabel('Prediction')
plt.title('Joyner-Boore Dist. (km) normalized Actual')
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



diff=np.std(train_targets-b,axis=0)
difftest=np.std(test_targets-a,axis=0)
diffmean=np.mean(train_targets-b,axis=0)
f22=plt.figure('Difference Std of residuals vs Period')
plt.semilogx(period,diff,label='Training ')
plt.semilogx(period,difftest,label='Testing')
plt.xlabel('Period')
plt.ylabel('Total Standard Deviation')


with open('diffb1.txt', 'wb') as filehandle:  
    # store the data as binary data stream
    pickle.dump(diff, filehandle)

plt.xlabel('Period')
plt.ylabel('Total Standard Deviation ')
plt.title('Standard Deviation of Residuals vs Period')




gmpeBSSAdata=np.zeros([period.shape[0]])
gmpeASKdata=np.zeros([period.shape[0]])
gmpeCBdata=np.zeros([period.shape[0]])
gmpeCYdata=np.zeros([period.shape[0]])

gmpeBSSAstd=np.zeros([period.shape[0]])
gmpeASKstd=np.zeros([period.shape[0]])
gmpeCBstd=np.zeros([period.shape[0]])
gmpeCYstd=np.zeros([period.shape[0]])

        
dx = base.DistancesContext()
dx.rjb=    np.array([ztest[i,9]])



#dx.rjb = np.logspace(-1, 2, 10)
# Magnitude and rake
rx = base.RuptureContext()
rx.mag = np.array([ztest[i,0]])
rx.rake = np.array([ztest[i,5]])
rx.hypo_depth = np.array([ztest[i,7]])
# Vs30
sx = base.SitesContext()
sx.vs30 = np.array([ztest[i,2]])
sx.vs30measured = 0

dx.rrup=np.array([ztest[i,1]])
rx.ztor=np.array([ztest[i,11]])
rx.dip=np.array([ztest[i,6]])
rx.width=np.array([ztest[i,8]])
dx.rx=np.array([rxkeep[i]])
dx.ry0=np.array([0])
sx.z1pt0= np.array([ztest[i,3]])
sx.z2pt5=np.array([ztest[i,4]])

# Evaluate GMPE
#Unit of measure for Z1.0 is [m] (ASK)
#lmean, lsd = gmpeASK.get_mean_and_stddevs(sx, rx, dx, imt.PGV(), stddev_types)
i=0
#for period1 in period:
for ii in range(0,6):
    sx.vs30measured = 0
    period1=period[ii]
    gmpeBSSAdata[ii], g = gmpeBSSA.get_mean_and_stddevs(sx, rx, dx, imt.SA(period1), stddev_types)
    gmpeBSSAstd[ii]=g[0][0]
    
    gmpeCBdata[ii], g = gmpeCB.get_mean_and_stddevs(sx, rx, dx, imt.SA(period1), stddev_types)
    gmpeCBstd[ii]=g[0][0]
    
    gmpeCYdata[ii], g = gmpeCY.get_mean_and_stddevs(sx, rx, dx, imt.SA(period1), stddev_types)
    gmpeCYstd[ii]=g[0][0]
    
    sx.vs30measured = [0]
    gmpeASKdata[ii], g = gmpeASK.get_mean_and_stddevs(sx, rx, dx, imt.SA(period1), stddev_types)
    gmpeASKstd[ii]=g[0][0]
        



boore=gmpeBSSAstd
plt.semilogx(period,boore,label='BSSA')

boore=gmpeASKstd
plt.semilogx(period,boore,label='ASK')

boore=gmpeCBstd
plt.semilogx(period,boore,label='CB')

boore=gmpeCYstd
plt.semilogx(period,boore,label='CY')


plt.legend()
plt.grid()

with open('diffg.txt', 'wb') as filehandle:  
    # store the data as binary data stream
    pickle.dump(boore, filehandle)



f22=plt.figure('Diff of Residuals vs period3 ')
ggg=train_targets-b
for i in range(6):
    plt.semilogx(period[i]*np.ones(train_data.shape[0]),ggg[:,i],'.b',label='Residuals',alpha=.2)   
    plt.plot([period[i],period[i]],[-diff[i],diff[i]],color='r')
plt.semilogx(period,diffmean,linestyle='--', marker='o', color='g',label='Mean')
plt.xlabel('Period')
plt.ylabel('Residuals')
plt.title('Residuals, Standard Deviation, and Mean vs Period')
#plt.legend()
plt.grid()
plt.show

f22=plt.figure('Diff of Residuals vs period2 ')
ggg=train_targets-b
#for i in range(85):
#    plt.semilogx(period[i]*np.ones(2400),ggg[:,i],'.b',label='Residuals',alpha=.2)   
plt.semilogx(period,diffmean,linestyle='--', marker='o', color='g',label='Mean')
#plt.semilogx.errorbar(period, np.zeros(period.size), yerr=diff,color='r',label='Std',fmt='.')
#plt.semilogx(period,diff,label='Std')
plt.xlabel('Period')
plt.ylabel('Residuals')
plt.title('Residuals')
plt.legend()
plt.grid()




def garson(A, B):
    """
    Computes Garson's algorithm
    A = matrix of weights of input-hidden layer (rows=input & cols=hidden)
    B = vector of weights of hidden-output layer
    """
    B = np.diag(B)

    # connection weight through the different hidden node
    cw = np.dot(A, B)

    # weight through node (axis=0 is column; sum per input feature)
    cw_h = abs(cw).sum(axis=0)

    # relative contribution of input neuron to outgoing signal of each hidden neuron
    # sum to find relative contribution of input neuron
    rc = np.divide(abs(cw), abs(cw_h))
    rc = rc.sum(axis=1)

    # normalize to 100% for relative importance
    ri = rc / rc.sum()
    return(ri)
    
def connection_weights(A, B):
    """
    Computes Connection weights algorithm
    A = matrix of weights of input-hidden layer (rows=input & cols=hidden)
    B = matrix of weights of hidden-output layer (rows=hidden & cols=output)
    """    
    cw = np.abs(np.dot(A, B))

    # normalize to 100% for relative importance
    ri = cw / cw.sum()
    return(ri)    
    
weights=model.get_weights()
weights1=weights[0]
weights2=weights[1]
weights3=weights[2]

depend=[]
for i in range(6):
#    depend.append(garson(weights1,weights3[:,i]))
    depend.append(garson(weights1,weights2))
depend=np.array(depend)

#depend=np.abs(connection_weights(weights1, weights3))
#depend=(connection_weights(weights1, weights3))
#plt.close('all')
cw = np.abs(np.dot(weights1, weights3))
summ1=np.sum(cw,axis=0)
ri = cw/summ1
depend=ri
#,figsize=(7,7)
f22=plt.figure('Significance of parame')
for i in range(0,train_data.shape[1]):
    plt.semilogx(period,100*depend[i,:],label=feature_names[i])

plt.xlabel('Period')
plt.ylabel('Significance (%)')
plt.title('Significance of Parameters (%)')
#plt.legend()
plt.grid()
plt.xlim([0.1,10])

#sepctral acdleation at 1 sec versus period
predictgmpe=np.zeros(300)
for i in np.arange(0,300):
    m=pygmm.BooreStewartSeyhanAtkinson2014(mag=6, dist_jb=i, dip=90, v_s30=393,depth_1_0 =0.329,depth_2_5 =1.642,depth_tor=6.8)
    m.interp_ln_stds(periods=period)
    predictgmpe[i]=m.spec_accels[67]
 #m.periods[67]
#ztest = np.column_stack([Mwtest,distest,vs30test, z1test, z2p5test,ztt,aziumuthtest])
gmpe_data=np.zeros([300,6])


#array([    6.35102006,   120.5014957 ,   393.94584814,    88.67761032,
#         329.90647564,  1425.55618911])
gmpe_data[:,0]=6.0
gmpe_data[:,1]=np.arange(0,300)
gmpe_data[:,2]=393.
#gmpe_data[:,3]=90
gmpe_data[:,3]=329
gmpe_data[:,4]=1642
gmpe_data[:,5]=6.8
#gmpe_data[:,6]=0


