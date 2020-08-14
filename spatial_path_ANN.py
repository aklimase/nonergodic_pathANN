#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 18:03:06 2020

@author: aklimasewski

ANN with only lat,lon of station and event trained on residuals

includes kernel class and optimization
"""
import sys
import os
sys.path.append(os.path.abspath('/Users/aklimase/Documents/nonergodic_ANN'))
from preprocessing import transform_dip, readindata, transform_data
from model_plots import gridded_plots, plot_resid, obs_pre, plot_outputs, plot_rawinputs

from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from tensorflow import random
random.set_seed(1)
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import Normalizer
from keras import layers
from keras import optimizers
# import gc
import seaborn as sns; 
sns.set(style="ticks", color_codes=True)
from keras.models import Sequential
import os
import cartopy
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import random

import tensorflow_probability as tfp

# For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float64')

sns.reset_defaults()
sns.set_style('whitegrid')
sns.set_context('talk')
sns.set_context(context='talk',font_scale=0.7)

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tfp.math.psd_kernels

#%%

folder_path = '/Users/aklimase/Documents/USGS/models/ANN13_VGP4/ANN13/'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
transform_method = PowerTransformer()
epochs = 15
n = 13
#or n = 6, 4

train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimase/Documents/USGS/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimase/Documents/USGS/data/cybertestyeti10_residfeb.csv', n = n)

# #add the location features
# train_data1_4, test_data1_4, train_targets1_4, test_targets1_4, feature_names_4 = readindata(nametrain='/Users/aklimase/Documents/USGS/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimase/Documents/USGS/data/cybertestyeti10_residfeb.csv', n = 4)
# train_data1 = np.concatenate([train_data1,train_data1_4], axis = 1)
# test_data1 = np.concatenate([test_data1,test_data1_4], axis = 1)
# feature_names = np.concatenate([feature_names,feature_names_4], axis = 0)


x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_path)
Rindex = np.where(feature_names == 'Rrup')[0][0]
#%%

# create sequential model with positional inputs and predict 
# try sequential with VGP layer
# For numeric stability, set the default floating-point dtype to float64

batch_size = 264
num_inducing_points = 40
loss = lambda y, rv_y: rv_y.variational_loss(
    y, kl_weight=np.array(batch_size) / x_train.shape[0])


def build_model():
    #model=models.Sequential()
    model = Sequential()
    model.add(layers.Dense(x_train.shape[1],activation='sigmoid', input_shape=(x_train.shape[1],)))

    #no gP layer
    model.add(layers.Dense(y_train.shape[1]))
    # model.add(tfp.layers.VariationalGaussianProcess(
    #     num_inducing_points=num_inducing_points,
    #     # kernel_provider=RBFKernelFn(),
    #     kernel_provider=RBFKernelFn_opt(),

    #     # kernel_provider=TestKernelFn(),
    #     # kernel=optimized_kernel,
    #     event_shape=[y_train.shape[1]],#outputshape
    #     inducing_index_points_initializer=tf.constant_initializer(y_train.shape[1]*[np.linspace(*x_range,num_inducing_points,dtype='float64')]),
    #     unconstrained_observation_noise_variance_initializer=(
    #                     tf.constant_initializer(np.array(0.1).astype(x_train.dtype)))))

    model.compile(optimizer=optimizers.Adam(lr=0.01),loss='mse',metrics=['mae','mse']) 
    #model.compile(optimizer='adam',loss='mse',metrics=['mae']) 
    return model

model=build_model()

#fit the model
history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=epochs,batch_size=batch_size,verbose=1)

mae_history=history.history['val_mae']
mae_history_train=history.history['mae']
test_mse_score,test_mae_score,tempp=model.evaluate(x_test,y_test)
#dataframe for saving purposes
hist_df = pd.DataFrame(history.history)

f10=plt.figure('Overfitting Test')
plt.plot(mae_history_train,label='Training Data')
plt.plot(mae_history,label='Testing Data')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('Overfitting Test')
plt.legend()
print(test_mae_score)
plt.grid()
plt.savefig(folder_path + 'error.png')
plt.show()

# pr1edict_mean = []
# predict_epistemic = []

# predict_mean_train = []
# predict_epistemic_train = []

pre_test = np.array(model.predict(x_test))
pre = np.array(model.predict(x_train))


   #test data
mean_x_test_allT = pre_test
predict_epistemic_allT = np.zeros(pre_test.shape)

#training data
mean_x_train_allT = pre
predict_epistemic_train_allT = np.zeros(pre.shape)

resid = y_train-mean_x_train_allT
resid_test = y_test-mean_x_test_allT
# pre = predict_mean_train
# pre_test = predict_mean

period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]
plot_resid(resid, resid_test, folder_path)
plot_outputs(folder_path, mean_x_test_allT, predict_epistemic_allT, mean_x_train_allT, predict_epistemic_train_allT, x_train, y_train, x_test, y_test, Rindex, period, feature_names)
plot_rawinputs(x_raw = x_train_raw, mean_x_allT = mean_x_train_allT, y=y_train, feature_names=feature_names, period = period, folder_path = folder_path + 'train/')
plot_rawinputs(x_raw = x_test_raw, mean_x_allT = mean_x_test_allT, y=y_test, feature_names=feature_names, period = period, folder_path = folder_path + 'test/')

obs_pre(y_train, y_test, pre, pre_test, period, folder_path)

diff=np.std(y_train-mean_x_train_allT,axis=0)
difftest=np.std(y_test-mean_x_test_allT,axis=0)

#write model details to a file
file = open(folder_path + 'model_details.txt',"w+")
file.write('number training samples ' + str(len(x_train)) + '\n')
file.write('number testing samples ' + str(len(x_test)) + '\n')
file.write('data transformation method ' + str(transform_method) + '\n')
file.write('input feature names ' +  str(feature_names)+ '\n')
file.write('number of epochs ' +  str(epochs)+ '\n')
# file.write('number kernel optimization samples ' + str(num_kernelopt_samples) + '\n')
# file.write('kernel name ' + str(kernel.name) + '\n')
# file.write('kernel trainable params ' + str(gp.trainable_variables) + '\n')
model.summary(print_fn=lambda x: file.write(x + '\n'))
file.write('model fit history' + str(hist_df.to_string) + '\n')
file.write('stddev train' + str(diff) + '\n')
file.write('stddev test' + str(difftest) + '\n')
file.close()

#write training predictions to a file
df_out = pd.DataFrame(resid, columns=['T10','T7.5','T5','T4','T3','T2','T1','T0.5','T0.2','T0.1'])   
df_out.to_csv(folder_path + 'ANNresiduals_train.csv')   

df_outtest = pd.DataFrame(resid_test, columns=['T10','T7.5','T5','T4','T3','T2','T1','T0.5','T0.2','T0.1'])   
df_out.to_csv(folder_path + 'ANNresiduals_test.csv')   

#%%
class RBFKernelFn(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(RBFKernelFn, self).__init__(**kwargs)
    dtype = kwargs.get('dtype', None)

    self._amplitude = self.add_variable(
            initializer=tf.constant([1.0,1.0,1.0,1.0,1.0]),
            dtype=dtype,
            name='amplitude')
    
    self._length_scale = self.add_variable(
            initializer=tf.constant([0.0,0.0,0.0,0.0,0.0]),
            dtype=dtype,
            name='length_scale')

  def call(self, x):
    # Never called -- this is just a layer so it can hold variables
    # in a way Keras understands.
    return x

  @property
  def kernel(self):
    return tfp.math.psd_kernels.MaternOneHalf(
      amplitude=tf.nn.softplus(tf.Variable(10. * tf.ones(5, dtype=tf.float64)) * self._amplitude),
      length_scale=tf.nn.softplus(tf.Variable(3. * tf.ones(5, dtype=np.float64)) * self._length_scale)
    )

#%%o

def Optimize_kernelparams(train_data1):
    
    #kernel optimization
    #to speed up, try randomly picking 1000 training samples
    num_kernelopt_samples = 1000
    rand_ind= np.random.randint(0,len(train_data1),num_kernelopt_samples )
    observed_index_points = train_data1[rand_ind]
    observed_values = train_targets1[rand_ind].T
    
    
    
    kernel = tfp.math.psd_kernels.MaternOneHalf(
        amplitude=tf.Variable(tf.ones(10,dtype=np.float64), dtype=np.float64, name='amplitude'),
        length_scale=tf.Variable(tf.ones(10,dtype=np.float64), dtype=np.float64, name='length_scale'))
    
    gp = tfd.GaussianProcess(kernel, observed_index_points)
    
    optimizer = tf.optimizers.Adam()

    @tf.function
    def optimize():
      with tf.GradientTape() as tape:
        loss = -gp.log_prob(observed_values)
      grads = tape.gradient(loss, gp.trainable_variables)
      optimizer.apply_gradients(zip(grads, gp.trainable_variables))
      return loss
    
    
    for i in range(1000):
      neg_log_likelihood = optimize()
      if i % 100 == 0:
        print("Step {}: NLL = {}".format(i, neg_log_likelihood))
    print("Final NLL = {}".format(neg_log_likelihood))
    
    #set amp and length for rbf function
    print(gp.trainable_variables)
    amplitude_opt,length_scale_opt = gp.trainable_variables
    amplitude_opt = amplitude_opt.numpy()
    length_scale_opt = length_scale_opt.numpy()
    
    return amplitude_opt,length_scale_opt

# amplitude_opt,length_scale_opt = Optimize_kernelparams(train_data1)

class RBFKernelFn_opt(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(RBFKernelFn_opt, self).__init__(**kwargs)
    dtype = kwargs.get('dtype', None)

    self._amplitude = self.add_variable(
            initializer=tf.constant_initializer(0),
            dtype=dtype,
            name='amplitude')
    
    self._length_scale = self.add_variable(
            # initializer=tf.keras.initializers(tf.constant([0.0,0.0,0.0,0.0,0.0])),
            initializer=tf.constant_initializer(0),
            dtype=dtype,
            name='length_scale')

  def call(self, x):
    # Never called -- this is just a layer so it can hold variables
    # in a way Keras understands.
    return x

  @property
  def kernel(self):
    return tfp.math.psd_kernels.MaternFiveHalves(
      # amplitude=tf.nn.softplus([0.1,0.1,0.1,0.1,0.1] * self._amplitude),
      # length_scale=tf.nn.softplus([1.0,1.0,1.0,1.0,1.0] * self._length_scale), feature_ndims=5
      amplitude=(tf.nn.softplus(amplitude_opt* self._amplitude)),
      length_scale=(tf.nn.softplus(length_scale_opt* self._length_scale)))






