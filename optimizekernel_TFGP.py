#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:50:44 2020

@author: aklimasewski


toy data set from main data
optimize the kernel
sequential model with VGP layer for epistemic uncertainty
"""

import sys
import os
sys.path.append(os.path.abspath('/Users/aklimasewski/Documents/nonergodic_ANN'))
from preprocessing import transform_dip, readindata, transform_data, create_grid, grid_data

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

nametrain='/Users/aklimase/Documents/USGS/data/cybertrainyeti10_residfeb.csv'
nametest='/Users/aklimase/Documents/USGS/data/cybertestyeti10_residfeb.csv'
train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain, nametest, n=12)

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
index_points = x_train
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
     # 'observations': observations
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


