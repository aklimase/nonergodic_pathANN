#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:25:56 2020

@author: aklimasewski

includes variational gaussian process layer
ANN outputs distribution of models
calling model.predict pulls one prediction
"""
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from tensorflow import random
random.set_seed(1)
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import Normalizer

import sys
import os
sys.path.append(os.path.abspath('/Users/aklimase/Documents/USGS/nonergodic_ANN'))
from preprocessing import transform_dip, readindata, transform_data, create_grid, grid_data
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

import tensorflow_probability as tfp

sns.reset_defaults()
#sns.set_style('whitegrid')
#sns.set_context('talk')
sns.set_context(context='talk',font_scale=0.7)

tfd = tfp.distributions

###############
#recreate the demo

folder_path  = '/Users/aklimase/Documents/USGS/models/VGPlayer/'

nametrain='/Users/aklimase/Documents/USGS/data/cybertrainyeti10_residfeb.csv'
nametest='/Users/aklimase/Documents/USGS/data/cybertestyeti10_residfeb.csv'
train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain, nametest, n=12)

#%%
transform_method = Normalizer()
x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_path)

y_test = y_test[:,0:2]

y = y_train[:,0:2]

x_range = [[min(x_train.T[i]) for i in range(len(x_train[0]))],[max(x_train.T[i]) for i in range(len(x_train[0]))]]

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
    tf.keras.layers.InputLayer(input_shape=(x_train.shape[1],),dtype='float32'),
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
    y, kl_weight=np.array(batch_size) / x_train.shape[0])

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=loss,metrics=['mae','mse'])


history=model.fit(x_train,y,validation_data=(x_test,y_test),epochs=10,batch_size=264,verbose=1)
mae_history=history.history['val_mae']
mae_history_train=history.history['mae']
test_mse_score,test_mae_score,tempp=model.evaluate(x_test,y_test)

yhat = model.predict(x_test)

yhat = np.ndarray.flatten(yhat)



