#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:25:56 2020

@author: aklimasewski
"""
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from tensorflow import random
random.set_seed(1)
from sklearn.preprocessing import PowerTransformer


import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

import tensorflow_probability as tfp

sns.reset_defaults()
#sns.set_style('whitegrid')
#sns.set_context('talk')
sns.set_context(context='talk',font_scale=0.7)


tfd = tfp.distributions



###############
#recreate the demo with one input and one output only

# %%

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


def readindata(nametrain, nametest):
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
    
    diptrain, diptest, rxtrain, rxtest = transform_dip(diptrain=np.array(diptrain),diptest=np.array(diptest),rxtrain=np.array(rxtrain),rxtest=np.array(rxtest))
    
    #put together arrays of features for ANN
    train_data1 = np.column_stack([Mwtrain,distrain,vs30train,z10train,z25train,raketrain,diptrain,hypodepthtrain, widthtrain,
                                   rjbtrain,rxtrain,startdepthtrain])
    test_data1 = np.column_stack([Mwtest,distest,vs30test,z10test,z25test,raketest,diptest, hypodepthtest, widthtest,
                                 rjbtest,rxtest,startdepthtest])
    
    feature_names=['Mw','Rrup','Vs30', 'Z1.0', 'Z2.5', 'Rake','Dip','Hypo_depth', 'Width',
                   'Rjb','Rx','Ztor',]
    return train_data1, test_data1, train_targets1, test_targets1, feature_names



train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv')
#Read in datasets
# nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv'
# nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv'
#%%
#########################

#preprocessing transform inputs data to be guassian shaped
pt = PowerTransformer()
aa=pt.fit(train_data1[:,:])
train_data=aa.transform(train_data1)
test_data=aa.transform(test_data1)

train_targets = train_targets1
test_targets = test_targets1

y_test = test_targets1.T[0:2]
y_test = y_test.T

y = train_targets1.T[0:2]
y = y.T

x_range = [[min(train_data.T[i]) for i in range(len(train_data[0]))],[max(train_data.T[i]) for i in range(len(train_data[0]))]]


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
     return tfp.math.psd_kernels.MaternOneHalf(
       amplitude=tf.nn.softplus(2.5 * self._amplitude),
       length_scale=tf.nn.softplus(10. * self._length_scale)
     )



# For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float64')

# Build model.
# points to sample your data range
num_inducing_points = 40

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(train_data.shape[1],),dtype='float32'),
    tf.keras.layers.Dense(12, kernel_initializer='zeros', use_bias=False),
    tfp.layers.VariationalGaussianProcess(
        num_inducing_points=num_inducing_points,
        kernel_provider=RBFKernelFn(),
        event_shape=[2],
        # inducing_index_points_initializer=tf.constant_initializer(np.linspace((min(train_data.T[0]),min(train_data.T[1]),min(train_data.T[2])),(max(train_data.T[0]),max(train_data.T[1]),max(train_data.T[2])),num_inducing_points,dtype='float32')),
        #change initializer dim for multiple outputs
        inducing_index_points_initializer=tf.constant_initializer([np.linspace(*x_range,num_inducing_points,dtype='float32'),np.linspace(*x_range,num_inducing_points,dtype='float32')]),
        unconstrained_observation_noise_variance_initializer=(
            tf.constant_initializer(0.1))
            # tf.constant_initializer(0.1)),
    ),
])


batch_size=264

# batch_size = 64
loss = lambda y, rv_y: rv_y.variational_loss(
    y, kl_weight=np.array(batch_size) / train_data.shape[0])

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=loss,metrics=['mae','mse'])


history=model.fit(train_data,y,validation_data=(test_data,y_test),epochs=10,batch_size=264,verbose=1)
mae_history=history.history['val_mae']
mae_history_train=history.history['mae']
test_mse_score,test_mae_score,tempp=model.evaluate(test_data,y_test)


yhat = model.predict(test_data)

yhat = np.ndarray.flatten(yhat)








#######



# #print simple test
# #10 examples
# stest = np.asarray([test_data1[1175] for i in range(10)])
# #vary distances for plotting
# dist_test = [1,5,10,20,40,50,75,100,120,150]
# for i in range(len(stest)):
#     stest[i][1] = dist_test[i]


# yhat = model(stest)

# plt.figure(figsize=[6, 1.5])  # inches
# # plt.plot(x, y, 'b.', label='observed');
# #sample 5 different models
# num_samples = 5
# for i in range(num_samples):
#   sample_ = yhat.sample().numpy()
#   xaxis = dist_test
#   plt.plot(xaxis,sample_[..., 0].T,'r',linewidth=0.9,label='ensemble means' if i == 0 else None);
# plt.xticks(np.linspace(0,160, num=8));
# plt.show()

plt.figure()
plt.scatter(test_data1.T[1],y_test,label='Testing Data', s = 1)
plt.scatter(test_data1.T[1],yhat,label='Model predictions',  s = 1)
plt.xlabel('Distance (km)')
plt.ylabel('Prediction')
plt.title('predictions vs distance (km)')
plt.grid()
plt.show()



f10=plt.figure('Overfitting Test')
plt.plot(mae_history,label='Testing Data')
plt.plot(mae_history_train,label='Training Data')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('Overfitting Test')
plt.legend()
print(test_mae_score)
plt.grid()
plt.show()




a=model.predict(test_data)
b=model.predict(train_data)
f1=plt.figure('Earthquake Magnitude normalized Prediction')
plt.plot(train_data[:,0],b,'.r', label='train data')
plt.plot(test_data[:,0],a,'.b',label='test data')
plt.xlabel('input')
plt.ylabel('Prediction')
plt.title('Earthquake Magnitude normalized Prediction')
plt.legend()
plt.show()


f11=plt.figure('Joyner-Boore Dist. (km) normalized Prediction')
plt.plot(train_data[:,1],b,'.r',label='train data')
plt.plot(test_data[:,1],a,'.b',label='test data')
plt.xlabel('input')
plt.ylabel('Prediction')
plt.ylabel('Prediction')
plt.title('Joyner-Boore Dist. (km) normalized Prediction')
plt.legend()
plt.show()














# y, x, _ = load_dataset()

plt.figure(figsize=[6, 1.5])  # inches
plt.plot(train_data.T[1], y, 'b.', label='observed');

num_samples = 2
for i in range(num_samples):
  sample_ = yhat.sample().numpy()
  plt.scatter(test_data.T[1],
           sample_[..., 0].T,
           'r');

# plt.ylim(-0.,17);
# plt.yticks(np.linspace(0, 15, 4)[1:]);
plt.xticks(np.linspace(*x_range, num=9));

ax=plt.gca();
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['left'].set_smart_bounds(True)
#ax.spines['bottom'].set_smart_bounds(True)
plt.legend(loc='center left', fancybox=True, framealpha=0., bbox_to_anchor=(1.05, 0.5))
plt.show()
# plt.savefig('/tmp/fig5.png', bbox_inches='tight', dpi=300)



# # Build model.
# model = tf.keras.Sequential([
#   tf.keras.layers.Dense(1),
#   tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
# ])

# # Do inference.
# model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=negloglik)
# model.fit(x, y, epochs=1000, verbose=False);

# # Profit.
# [print(np.squeeze(w.numpy())) for w in model.weights];
# yhat = model(x_tst)
# assert isinstance(yhat, tfd.Distribution)




# # Build model.
# model = tf.keras.Sequential([
#   tfp.layers.DenseVariational(1 + 1, posterior_mean_field, prior_trainable, kl_weight=1/x.shape[0]),
#   tfp.layers.DistributionLambda(
#       lambda t: tfd.Normal(loc=t[..., :1],
#                            scale=1e-3 + tf.math.softplus(0.01 * t[...,1:]))),
# ])

# # Do inference.
# model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=negloglik)
# model.fit(x, y, epochs=1000, verbose=False);

# # Profit.
# [print(np.squeeze(w.numpy())) for w in model.weights];
# yhat = model(x_tst)
# assert isinstance(yhat, tfd.Distribution)


#%%
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.regularizers import l2


from tensorflow.keras import backend as K
import tensorflow as tf


#aleatoric uncertainty using aleatoric loss function

n = 100
x_func = np.linspace(-4,4,100)
y_func = x_func

x_train = np.random.uniform(-3, -2, n)
y_train = x_train + np.random.randn(*x_train.shape)*0.5

x_train = np.concatenate([x_train, np.random.uniform(2, 3, n)])
y_train = np.concatenate([y_train, x_train[n:] + np.random.randn(*x_train[n:].shape)*0.1])
x_test = np.linspace(-5,5,100)

fig, ax = plt.subplots(1,1,figsize=(10,5))
ax.scatter(x_train, y_train, label='training data', s = 3)
ax.plot(x_func, y_func, ls='--', label='real function', color='green')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.set_title('Data with uncertainty')
plt.show()

# aleatoric- predicted with loss function (output 2D)
# aleatoric loss function
def aleatoric_loss(y_true, y_pred):
    N = y_true.shape[0]
    se = K.pow((y_true[:,0]-y_pred[:,0]),2)
    inv_std = K.exp(-y_pred[:,1])
    mse = K.mean(inv_std*se)
    reg = K.mean(y_pred[:,1])
    return 0.5*(mse + reg)

# def nll_gaussian(y_pred_mean,y_pred_sd,y_test):
#     ## element wise square
#     square = tf.square(y_pred_mean - y_test)## preserve the same shape as y_pred.shape
#     ms = tf.add(tf.divide(square,y_pred_sd), tf.log(y_pred_sd))
#     ## axis = -1 means that we take mean across the last dimension 
#     ## the output keeps all but the last dimension
#     ## ms = tf.reduce_mean(ms,axis=-1)
#     ## return scalar
#     ms = tf.reduce_mean(ms)
#     return(ms)

# def nll_gaussian(y_true, y_pred):
#     ## element wise square
#     square = tf.square(y_pred[:,0] - y_true[:,0])## preserve the same shape as y_pred.shape
#     ms = tf.add(tf.divide(square,y_pred[:,1]), np.log(y_pred[:,1]))
#     ## axis = -1 means that we take mean across the last dimension 
#     ## the output keeps all but the last dimension
#     ## ms = tf.reduce_mean(ms,axis=-1)
#     ## return scalar
#     ms = tf.reduce_mean(ms)
#     return(ms)



def architecture(layers_shape, input_dim, output_dim, dropout_proba, reg, act='relu', verbose=False):
    inputs = Input(shape=(input_dim,))
    hidden = Dense(layers_shape[0], activation=act,
                   kernel_regularizer=l2(reg))(inputs)
    for i in range(len(layers_shape)-1):
        if dropout_proba > 0:
          hidden = Dropout(dropout_proba)(hidden, training=True)
        hidden = Dense(layers_shape[i+1], activation=act, kernel_regularizer=l2(reg))(hidden)
    if dropout_proba > 0:
      hidden = Dropout(dropout_proba)(hidden, training=True)
    outputs = Dense(output_dim, kernel_regularizer=l2(reg))(hidden) 
    model = Model(inputs, outputs)
    if verbose:
      model.summary()
    return model
  
model_aleatoric = architecture(layers_shape=[5], 
                                            input_dim= 1, output_dim=2, 
                                            dropout_proba=0, reg=0, 
                                            act='relu', verbose=1)
model_aleatoric.compile(optimizer='rmsprop', 
                                     loss=nll_gaussian, metrics=['mae'])
model_aleatoric.fit(x_train, y_train, 
                                 batch_size=20, epochs=500, shuffle=True, verbose=1)


# plot aleatoric uncertainty
fig, ax = plt.subplots(1,1,figsize=(10,5))
x_test=np.linspace(-5,5,100)
y_test=np.linspace(-5,5,100)
p =  model_aleatoric.predict(x_test)
predict_mean, predict_al = p[:,0],p[:,1]

aleatoric_std = np.exp(0.5*predict_al)
ax.scatter(x_train, y_train, s=3, label='train data')
ax.plot(x_test, y_test, ls='--', label='test data', color='green')
ax.errorbar(x_test, predict_mean, yerr=aleatoric_std, fmt='.', label='aleatory uncertainty', color='orange')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()





#%%
#epistemic uncertianty from model distribution
# neg log likelihood loss function

tf.keras.backend.set_floatx('float64')

# optimized_kernel = tfk.MaternOneHalf(amplitude_var, length_scale_var)



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
     return tfk.MaternOneHalf(
       amplitude=amplitude_var,
       length_scale=length_scale_var)



num_inducing_points = 40
x_range = [min(x_train), max(x_train)]

# observation_noise_variance = tfp.util.TransformedVariable(
#                                     initial_value=1,
#                                     bijector=tfb.Chain([tfb.Shift(np.float64(1e-6)), tfb.Softplus()]),
#                                     name='observation_noise_variance') 

# # Build model
# model = tf.keras.Sequential([
#     tf.keras.layers.InputLayer(input_shape=[1]),
#     # tf.keras.layers.Dense(5, kernel_initializer='ones', use_bias=False),
#     # tf.keras.layers.Dense(1, kernel_initializer='ones', use_bias=False),
#     tfp.layers.VariationalGaussianProcess(
#         num_inducing_points=num_inducing_points,
#         kernel_provider=RBFKernelFn(),
#         event_shape=[1],

#         # inducing_index_points_initializer=(
#             # np.asarray([np.linspace(*x_range,num_inducing_points),np.linspace(*x_range,num_inducing_points)])[..., np.newaxis]),
#       # 
#         inducing_index_points_initializer=tf.constant_initializer(
#             np.asarray(np.linspace(*x_range,num_inducing_points))[..., np.newaxis]),

#         # unconstrained_observation_noise_variance_initializer=(None),
#         unconstrained_observation_noise_variance_initializer=(
#             tf.constant_initializer(observation_noise_variance_var._value().numpy())),
#                 # np.log(np.expm1(1.)).astype(x_train.dtype))),
#     ),
# ])

def architecture(layers_shape, input_dim, output_dim, dropout_proba, reg, act='relu', verbose=False):
    inputs = Input(shape=(input_dim,))
    hidden = Dense(layers_shape[0], activation=act,
                   kernel_regularizer=l2(reg))(inputs)
    for i in range(len(layers_shape)-1):
        if dropout_proba > 0:
          hidden = Dropout(dropout_proba)(hidden, training=True)
        hidden = Dense(layers_shape[i+1], activation=act, kernel_regularizer=l2(reg))(hidden)
    if dropout_proba > 0:
      hidden = Dropout(dropout_proba)(hidden, training=True)
    # outputs = Dense(output_dim, kernel_regularizer=l2(reg))(hidden) 
    outputs = tfp.layers.VariationalGaussianProcess(output_dim, kernel_provider=RBFKernelFn(dtype=x_train.dtype),inducing_index_points_initializer=(np.asarray([np.linspace(*x_range,num_inducing_points),np.linspace(*x_range,num_inducing_points)])[..., np.newaxis]),)(hidden)
    #     num_inducing_points=num_inducing_points,
    #     kernel_provider=RBFKernelFn(dtype=x_train.dtype),
    #     event_shape=[2],
    #     # inducing_index_points_initializer=tf.constant_initializer(
    #         # np.linspace(*x_range, num=num_inducing_points,
    #                     # dtype=x_train.dtype)[..., np.newaxis]),
    #     inducing_index_points_initializer=(
    #         np.asarray([np.linspace(*x_range,num_inducing_points),np.linspace(*x_range,num_inducing_points)])[..., np.newaxis]),
       


    #     # unconstrained_observation_noise_variance_initializer=(
    #         # tf.constant_initializer(
    #             # np.log(np.expm1(1.)).astype(x_train.dtype))),
    # ),

    
    model = Model(inputs, outputs)
    if verbose:
      model.summary()
    return model
  
model_aleatoric = architecture(layers_shape=[5], 
                                            input_dim= 1, output_dim=2, 
                                            dropout_proba=0, reg=0, 
                                            act='relu', verbose=1)




model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=[1], dtype=x_train.dtype),
    tf.keras.layers.Dense(1, kernel_initializer='ones', use_bias=False),
    tfp.layers.VariationalGaussianProcess(
        num_inducing_points=num_inducing_points,
        kernel_provider=RBFKernelFn(dtype=x_train.dtype),
        event_shape=[2],
        # inducing_index_points_initializer=tf.constant_initializer(
            # np.linspace(*x_range, num=num_inducing_points,
                        # dtype=x_train.dtype)[..., np.newaxis]),
        inducing_index_points_initializer=(
            np.asarray([np.linspace(*x_range,num_inducing_points),np.linspace(*x_range,num_inducing_points)])[..., np.newaxis]),
       


        # unconstrained_observation_noise_variance_initializer=(
            # tf.constant_initializer(
                # np.log(np.expm1(1.)).astype(x_train.dtype))),
    ),
])


print(model.summary())

# # #loss
batch_size = 32
loss = lambda y, rv_y: rv_y.variational_loss(
    y, kl_weight=np.array(batch_size, x_train.dtype) / x_train.shape[0])

# @tf.function(autograph=False, experimental_compile=False)
# def target_log_prob(amplitude, length_scale, observation_noise_variance):
#   return gp_joint_model.log_prob({
#       'amplitude': amplitude,
#       'length_scale': length_scale,
#       'observation_noise_variance': observation_noise_variance,
#       'observations': observations_
#   })
# loss = -target_log_prob(amplitude_var, length_scale_var,
#                             observation_noise_variance_var)

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=aleatoric_loss,metrics=['mae','mse'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=100, verbose=True)



model_aleatoric.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=aleatoric_loss,metrics=['mae','mse'])

model_aleatoric.fit(x_train, y_train, batch_size=batch_size, epochs=100, verbose=True)



predict_mean = []
predict_al = []
predict_epistemic = []
for i in range(100):
    p = np.array(model.predict(x_test)) 
    mean = p[:,0]
    predict_mean.append(mean)

mean_x_test = np.mean(predict_mean, axis = 0)
predict_epistemic = np.std(predict_mean, axis = 0)

# plot aleatoric uncertainty
fig, ax = plt.subplots(1,1,figsize=(10,5))

ax.scatter(x_train, y_train, s=3, label='train data')
ax.plot(x_test, y_test, ls='--', label='test data', color='green')

ax.errorbar(x_test, mean_x_test, yerr=predict_epistemic, fmt='.', label='epistemic uncertainty', color='pink')

ax.set_xlabel('x')

ax.set_ylabel('y')
ax.legend()


#%%
# GaussianProcessRegressionModel  for epistemic uncertainty with trainable kernal params
#

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()


# Configure plot defaults
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#666666'



# Generate training data with a known noise level (we'll later try to recover
# this value from the data).
NUM_TRAINING_POINTS = 200
observation_index_points_, observations_ = (x_train.reshape((200, 1)),y_train)



def build_gp(amplitude, length_scale, observation_noise_variance):
  """Defines the conditional dist. of GP outputs, given kernel parameters."""

  # Create the covariance kernel, which will be shared between the prior (which we
  # use for maximum likelihood training) and the posterior (which we use for
  # posterior predictive sampling)
  # kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)
  
  kernel = tfp.math.psd_kernels.MaternOneHalf(amplitude, length_scale)

  # Create the GP prior distribution, which we will use to train the model
  # parameters.
  return tfd.GaussianProcess(
      kernel=kernel,
      index_points=observation_index_points_,
      observation_noise_variance=observation_noise_variance)

gp_joint_model = tfd.JointDistributionNamed({
    'amplitude': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'length_scale': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'observations': build_gp,
})


x = gp_joint_model.sample()
lp = gp_joint_model.log_prob(x)

print("sampled {}".format(x))
print("log_prob of sample: {}".format(lp))


# Create the trainable model parameters, which we'll subsequently optimize.
# Note that we constrain them to be strictly positive.

constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

amplitude_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='amplitude',
    dtype=np.float64)

length_scale_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='length_scale',
    dtype=np.float64)

observation_noise_variance_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='observation_noise_variance_var',
    dtype=np.float64)

trainable_variables = [v.trainable_variables[0] for v in 
                       [amplitude_var,
                       length_scale_var,
                       observation_noise_variance_var]]


@tf.function(autograph=False, experimental_compile=False)
def target_log_prob(amplitude, length_scale, observation_noise_variance):
  return gp_joint_model.log_prob({
      'amplitude': amplitude,
      'length_scale': length_scale,
      'observation_noise_variance': observation_noise_variance,
      'observations': observations_
  })


# Now we optimize the model parameters.
num_iters = 1000
optimizer = tf.optimizers.Adam(learning_rate=.01)

# Store the likelihood values during training, so we can plot the progress
lls_ = np.zeros(num_iters, np.float64)
for i in range(num_iters):
  with tf.GradientTape() as tape:
    loss = -target_log_prob(amplitude_var, length_scale_var,
                            observation_noise_variance_var)
  grads = tape.gradient(loss, trainable_variables)
  optimizer.apply_gradients(zip(grads, trainable_variables))
  lls_[i] = loss

print('Trained parameters:')
print('amplitude: {}'.format(amplitude_var._value().numpy()))
print('length_scale: {}'.format(length_scale_var._value().numpy()))
print('observation_noise_variance: {}'.format(observation_noise_variance_var._value().numpy()))


# Having trained the model, we'd like to sample from the posterior conditioned
# on observations. We'd like the samples to be at points other than the training
# inputs.
predictive_index_points_ = x_test
# Reshape to [200, 1] -- 1 is the dimensionality of the feature space.
predictive_index_points_ = predictive_index_points_[..., np.newaxis]

optimized_kernel = tfk.MaternOneHalf(amplitude_var, length_scale_var)
gprm = tfd.GaussianProcessRegressionModel(
    kernel=optimized_kernel,
    index_points=predictive_index_points_,
    observation_index_points=observation_index_points_,
    observations=observations_,
    observation_noise_variance=observation_noise_variance_var,
    predictive_noise_variance=0.)

# Create op to draw  50 independent samples, each of which is a *joint* draw
# from the posterior at the predictive_index_points_. Since we have 200 input
# locations as defined above, this posterior distribution over corresponding
# function values is a 200-dimensional multivariate Gaussian distribution!
num_samples = 50
samples = gprm.sample(num_samples)



predict_mean = []
predict_al = []
predict_epistemic = []
for i in range(num_samples):
    p = samples[i]
    mean =  np.asarray(p)
    predict_mean.append(mean)

mean_x_test = np.mean(predict_mean, axis = 0).flatten()
predict_epistemic = np.std(predict_mean, axis = 0).flatten()



# plot epistemic uncertainty
fig, ax = plt.subplots(1,1,figsize=(10,5))
ax.scatter(x_train, y_train, s=3, label='train data')
ax.plot(x_test, y_test, ls='--', label='test data', color='green')
ax.errorbar(x_test, mean_x_test, yerr=predict_epistemic, fmt='.', label='epistemic uncertainty', color='pink')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()





