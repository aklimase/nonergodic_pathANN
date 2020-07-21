#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 13:17:31 2020

@author: aklimasewski
"""


#tn

import tensorflow as tf
import tensorflow_probability as tfp

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

tfd = tfp.distributions



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



# #preprocessing transform inputs data to be guassian shaped
# pt = PowerTransformer()
# aa=pt.fit(train_data1[:,:])
# train_data=aa.transform(train_data1)
# test_data=aa.transform(test_data1)

# train_targets = train_targets1
# test_targets = test_targets1



# def build_model():
#     #model=models.Sequential()
#     model = Sequential()
#     #model.add(Dropout(0.0,seed=1))
#     model.add(layers.Dense(6,activation='sigmoid', input_shape=(train_data.shape[1],)))

#     model.add(Dropout(rate=0.0, trainable =True))#trainint=true v mismatch
    
#     model.add(layers.Dense(train_targets.shape[1])) #add sigmoid aciivation functio? (only alues betwen 0 and 1)

#     model.compile(optimizer=optimizers.Adam(lr=2e-3),loss='mse',metrics=['mae','mse']) 
#     #model.compile(optimizer='adam',loss='mse',metrics=['mae']) 
#     return model


# model=build_model()


# def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
#   n = kernel_size + bias_size
#   c = np.log(np.expm1(1.))
#   return tf.keras.Sequential([
#       tfp.layers.VariableLayer(2 * n, dtype=dtype),
#       tfp.layers.DistributionLambda(lambda t: tfd.Independent(
#           tfd.Normal(loc=t[..., :n],
#                      scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
#           reinterpreted_batch_ndims=1)),
#   ])




# def prior_trainable(kernel_size, bias_size=0, dtype=None):
#   n = kernel_size + bias_size
#   return tf.keras.Sequential([
#       tfp.layers.VariableLayer(n, dtype=dtype),
#       tfp.layers.DistributionLambda(lambda t: tfd.Independent(
#           tfd.Normal(loc=t, scale=1),
#           reinterpreted_batch_ndims=1)),
#   ])




# def build_model():
#     #model=models.Sequential()
#     model = Sequential()
#     #model.add(Dropout(0.0,seed=1))
#     model.add(layers.Dense(6,activation='sigmoid', input_shape=(train_data.shape[1],)))

#     # model.add(Dropout(rate=0.0, trainable =True))#trainint=true v mismatch
#     model.add(layers.Dense(train_targets.shape[1])) #add sigmoid aciivation functio? (only alues betwen 0 and 1)

#     # model.add(layers.Dense(train_targets.shape[1])) #add sigmoid aciivation functio? (only alues betwen 0 and 1)

#     # model.add(tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)))
#     model.add(tfp.layers.DistributionLambda(lambda t: tfd.Gamma(loc=t[..., :1],scale=1)))
    
#     model.compile(optimizer=optimizers.Adam(lr=2e-3),loss='mse',metrics=['mae','mse']) 
#     #model.compile(optimizer='adam',loss='mse',metrics=['mae']) 
#     return model





# model=build_model()






class RBFKernelFn(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(RBFKernelFn, self).__init__(**kwargs)
    dtype = kwargs.get('dtype', None)

    self._amplitude = self.add_variable(
            initializer=tf.constant_initializer(0),
            dtype=dtype,
            name='amplitude')
    
    self._length_scale = self.add_variable(
            initializer=tf.constant_initializer(0),
            dtype=dtype,
            name='length_scale')

  def call(self, x):
    # Never called -- this is just a layer so it can hold variables
    # in a way Keras understands.
    return x

  @property
  def kernel(self):
    return tfp.math.psd_kernels.ExponentiatedQuadratic(
      amplitude=tf.nn.softplus(0.1 * self._amplitude),
      length_scale=tf.nn.softplus(5. * self._length_scale)
    )


#take first input var and first period for simplification
x = train_data1.T[0][0:1000]
y = train_targets1.T[0][0:1000]


# pt = PowerTransformer()
# aa=pt.fit(x)
# train_data=aa.transform(x)
# test_data=aa.transform(x)

# train_targets = train_targets1
# test_targets = test_targets1






x_range = [min(x), max(x)]

#idk what this number means
num_inducing_points = 40
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=[1], dtype=x.dtype),
    tf.keras.layers.Dense(1, kernel_initializer='ones', use_bias=False), tfp.layers.VariationalGaussianProcess(
        num_inducing_points=num_inducing_points,
        kernel_provider=RBFKernelFn(dtype=x.dtype),
        event_shape=[1],
        inducing_index_points_initializer=tf.constant_initializer(
            np.linspace(*x_range, num=num_inducing_points,
                        dtype=x.dtype)[..., np.newaxis]),
        unconstrained_observation_noise_variance_initializer=(
            tf.constant_initializer(
                np.log(np.expm1(1.)).astype(x.dtype))),
    ),
])

# Do inference.
batch_size = 32
# loss = lambda y, rv_y: rv_y.variational_loss(
    # y, kl_weight=np.array(batch_size, x.dtype) / x.shape[0])
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss='mse')
model.fit(x, y, batch_size=batch_size, epochs=10, verbose=True)



x_tst = test_data.T[0]
# Make predictions.
yhats = [model(x_tst) for _ in range(100)]





















es = EarlyStopping(monitor='val_mean_absolute_error', mode='max', min_delta=10)



history=model.fit(train_data1,train_targets1,validation_data=(test_data1,test_targets1),epochs=13,batch_size=256,verbose=1)
mae_history=history.history['val_mae']
mae_history_train=history.history['mae']
test_mse_score,test_mae_score,tempp=model.evaluate(test_data1,test_targets1)







# [print(np.squeeze(w.numpy())) for w in model.weights];
# yhat = model.predict(test_targets1)
# assert isinstance(yhat, tfd.Distribution)






preds = [model.predict(test_data1) for _ in range(len(test_data1))]
mean = np.array([pred.mean() for pred in preds]).mean(axis=0)
stddev = np.array([pred.stddev()**2 + (pred.mean() - mean)**2 for pred in preds])


# model.fit(x, y, epochs=1000, verbose=False);

# # Profit.
# [print(np.squeeze(w.numpy())) for w in model.weights];
# yhat = model(x_tst)
# assert isinstance(yhat, tfd.Distribution)






