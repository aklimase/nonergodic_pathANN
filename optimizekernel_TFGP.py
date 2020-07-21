#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:50:44 2020

@author: aklimasewski


toy data set from main data
optimize the kernel
sequential model with VGP layer
"""


#try epistemic uncertainty '


from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from tensorflow import random
random.set_seed(1)
from sklearn.preprocessing import PowerTransformer
from keras import layers
from keras import optimizers
# import gc
import seaborn as sns; 
sns.set(style="ticks", color_codes=True)
from keras.models import Sequential


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
    #starting with just lat,lon of station and event    
    train_data1 = np.column_stack([Mwtrain, lattrain, longtrain, hypolattrain, hypolontrain])
    test_data1 = np.column_stack([Mwtest, lattest, longtest, hypolattest, hypolontest])

    
    feature_names=['Mw','r','stlat', 'stlon', 'hypolat','hypolon']
    return train_data1, test_data1, train_targets1, test_targets1, feature_names



train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv')
#Read in datasets
# nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv'
# nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv'
#%%
#preprocessing transform inputs data to be guassian shaped
pt = PowerTransformer()
aa=pt.fit(train_data1[:,:])
train_data=aa.transform(train_data1)
test_data=aa.transform(test_data1)

train_targets = train_targets1[0:5000]
test_targets = test_targets1[0:5000]

y_test = test_targets1.T[0:1]
y_train = train_targets1.T[0:1]

x_range = [[min(train_data.T[i]) for i in range(len(train_data[0]))],[max(train_data.T[i]) for i in range(len(train_data[0]))]]

x_train = train_data[0:5000]
x_test = test_data[0:5000]

num_inducing_points = 40
tfb = tfp.bijectors
tfk = tfp.math.psd_kernels

# observations = y_train.reshape(len(y_train),1)
# index_points = np.asarray([np.linspace(*x_range,num_inducing_points)])
# index_points = x_train
#%%
#optimize kernel

# Generate training data with a known noise level (we'll later try to recover
# this value from the data).

def build_gp(amplitude, length_scale, observation_noise_variance):
  """Defines the conditional dist. of GP outputs, given kernel parameters."""

  # Create the covariance kernel, which will be shared between the prior (which we
  # use for maximum likelihood training) and the posterior (which we use for
  # posterior predictive sampling)
  
  #can also add kernels here
  kernel = tfp.math.psd_kernels.MaternOneHalf(amplitude, length_scale)

  # Create the GP prior distribution, which we will use to train the model
  # parameters.
  return tfd.GaussianProcess(
      kernel=kernel,
      index_points=index_points,
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
       'observations': observations
  })


# Now we optimize the model parameters.
num_iters = 1000
optimizer = tf.optimizers.Adam(learning_rate=.01)

# Store the likelihood values during training, so we can plot the progress
lls_ = np.zeros(num_iters, np.float64)
for i in range(num_iters):
  with tf.GradientTape() as tape:
    loss = -target_log_prob(amplitude_var, length_scale_var,observation_noise_variance_var)
    print(loss)
  grads = tape.gradient(loss, trainable_variables)
  optimizer.apply_gradients(zip(grads, trainable_variables))
  lls_[i] = loss

#plot loss
plt.figure(figsize=(12, 4))
plt.plot(lls_)
plt.xlabel("Training iteration")
plt.ylabel("Log marginal likelihood")
plt.show()


print('Trained parameters:')
print('amplitude: {}'.format(amplitude_var._value().numpy()))
print('length_scale: {}'.format(length_scale_var._value().numpy()))
print('observation_noise_variance: {}'.format(observation_noise_variance_var._value().numpy()))

optimized_kernel = tfk.MaternOneHalf(amplitude_var, length_scale_var)
noise = observation_noise_variance_var._value().numpy()

#class for optimized kernel
class OptKernelFn(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(OptKernelFn, self).__init__(**kwargs)
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






#%%
#optimize kernel

# Generate training data with a known noise level (we'll later try to recover
# this value from the data).

#this works with 1 input
#higher output dimensions?
observations_ = train_targets[:, 0:1].flatten()

event_size = 2

observation_index_points_ = x_train[:, 0:1]
# observation_index_points_ = x_train[:, 0:2]

observations_ = y_train.flatten()
event_size = 6
observation_index_points_ = x_train
# observation_index_points_ = x_train[:, 0:2]

def build_gp(amplitude, length_scale, observation_noise_variance):
  """Defines the conditional dist. of GP outputs, given kernel parameters."""

  # Create the covariance kernel, which will be shared between the prior (which we
  # use for maximum likelihood training) and the posterior (which we use for
  # posterior predictive sampling)
  kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)
  #tfk.MaternOneHalf(

  # Create the GP prior distribution, which we will use to train the model
  # parameters.
  return tfd.GaussianProcess(
      kernel=kernel,
      index_points=observation_index_points_,
      observation_noise_variance=observation_noise_variance)

gp_joint_model = tfd.JointDistributionNamed({
    'amplitude': tfd.LogNormal(loc=tf.Variable(tf.zeros(event_size,dtype=np.float64)),
        scale=tfp.util.TransformedVariable(
            tf.ones([event_size, 1],dtype=np.float64),bijector=tfb.Exp())),
    'length_scale': tfd.LogNormal(loc=tf.Variable(tf.zeros(event_size,dtype=np.float64)),
        scale=tfp.util.TransformedVariable(
            tf.ones([event_size, 1],dtype=np.float64),bijector=tfb.Exp())),
    'observation_noise_variance': tfd.LogNormal(loc=tf.Variable(tf.zeros(event_size,dtype=np.float64)),
        scale=tfp.util.TransformedVariable(
            tf.ones([event_size, 1],dtype=np.float64),bijector=tfb.Exp())),
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
    initial_value=tf.ones(event_size,dtype=np.float64),
    bijector=constrain_positive,
    name='amplitude',
    dtype=np.float64)

length_scale_var = tfp.util.TransformedVariable(
    initial_value=tf.ones(event_size,dtype=np.float64),
    bijector=constrain_positive,
    name='length_scale',
    dtype=np.float64)

observation_noise_variance_var = tfp.util.TransformedVariable(
    initial_value=tf.ones(event_size,dtype=np.float64),
    bijector=constrain_positive,
    name='observation_noise_variance_var',
    dtype=np.float64)

trainable_variables = [v.trainable_variables[0] for v in 
                       [amplitude_var,
                       length_scale_var,
                       observation_noise_variance_var]]


# Use `tf.function` to trace the loss for more efficient evaluation.
@tf.function(autograph=False, experimental_compile=False)
def target_log_prob(amplitude, length_scale, observation_noise_variance):
  return gp_joint_model.log_prob({
      'amplitude': amplitude,
      'length_scale': length_scale,
      'observation_noise_variance': observation_noise_variance,
      'observations': observations_
  })

# Now we optimize the model parameters.
num_iters = 20
optimizer = tf.optimizers.Adam(learning_rate=.01)

# Store the likelihood values during training, so we can plot the progress
lls_ = np.zeros((num_iters,2), np.float64)
for i in range(num_iters):
  with tf.GradientTape() as tape:
    loss = -target_log_prob(amplitude_var, length_scale_var,
                            observation_noise_variance_var)
    print(loss)
  grads = tape.gradient(loss, trainable_variables)
  optimizer.apply_gradients(zip(grads, trainable_variables))
  lls_[i] = loss[0]

print('Trained parameters:')
print('amplitude: {}'.format(amplitude_var._value().numpy()))
print('length_scale: {}'.format(length_scale_var._value().numpy()))
print('observation_noise_variance: {}'.format(observation_noise_variance_var._value().numpy()))


#plot loss
plt.figure(figsize=(12, 4))
plt.plot(lls_)
plt.xlabel("Training iteration")
plt.ylabel("Log marginal likelihood")
plt.show()


optimized_kernel = tfk.MaternOneHalf(amplitude_var, length_scale_var)
noise = observation_noise_variance_var._value().numpy()

#class for optimized kernel
class OptKernelFn(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(OptKernelFn, self).__init__(**kwargs)
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





#%%



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
y_train = train_targets1.T[0:2]

x_range = [[min(train_data.T[i]) for i in range(len(train_data[0]))],[max(train_data.T[i]) for i in range(len(train_data[0]))]]

x_train = train_data
x_test = test_data


#now use optimized kernel in ann

# For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float64')

batch_size = 32
num_inducing_points = 40
loss = lambda y, rv_y: rv_y.variational_loss(
    y, kl_weight=np.array(batch_size) / train_data.shape[0])


def build_model():
    #model=models.Sequential()
    model = Sequential()
    model.add(layers.Dense(5,activation='sigmoid', input_shape=(train_data.shape[1],)))

    # model.add(Dropout(rate=0.5, trainable =True))#trainint=true v mismatch
    
    model.add(tfp.layers.VariationalGaussianProcess(
        num_inducing_points=num_inducing_points,
        kernel_provider=RBFKernelFn(),
        # kernel=optimized_kernel,
        event_shape=[2],#outputshape
        # inducing_index_points_initializer=tf.constant_initializer(np.linspace(*x_range,num_inducing_points,dtype='float32')),
        inducing_index_points_initializer=tf.constant_initializer(np.linspace(*x_range,num_inducing_points,dtype='float32'),np.linspace(*x_range,num_inducing_points,dtype='float32')),

        unconstrained_observation_noise_variance_initializer=(
            tf.constant_initializer(noise))))

    model.compile(optimizer=optimizers.Adam(lr=0.01),loss='mse',metrics=['mae','mse']) 
    #model.compile(optimizer='adam',loss='mse',metrics=['mae']) 
    return model


model=build_model()


#fit the model
history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=32,verbose=1)
mae_history=history.history['val_mae']
mae_history_train=history.history['mae']
test_mse_score,test_mae_score,tempp=model.evaluate(test_data,y_test)


model.predict(x_test)

#####

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


