#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 13:19:45 2020

@author: aklimasewski


build a model that takes cybershake data, input to 12 feature ANN,input those residuals to a spatial ANN

use spatial kernel in second model
"""
from preprocessing import transform_dip, readindata, transform_data
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

folder_path = '/Users/aklimasewski/Documents/2step_ANN/model12/'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
transform_method = PowerTransformer()
epochs = 15
batch_size = 264


train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n=12)
x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_path)


#12 feature ANN
def build_model():
    #model=models.Sequential()
    model = Sequential()
    model.add(layers.Dense(x_train.shape[1],activation='sigmoid', input_shape=(x_train.shape[1],)))

    #no gP layer
    model.add(layers.Dense(y_train.shape[1]))
    model.compile(optimizer=optimizers.Adam(lr=0.01),loss='mse',metrics=['mae','mse']) 
    return model


model=build_model()


#fit the model
history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=epochs,batch_size=batch_size,verbose=1)

mae_history=history.history['val_mae']
mae_history_train=history.history['mae']
test_mse_score,test_mae_score,tempp=model.evaluate(x_test,y_test)
#dataframe for saving purposes
hist_df = pd.DataFrame(history.history)


#plotting

f10=plt.figure('Overfitting Test')
plt.plot(mae_history,label='Testing Data')
plt.plot(mae_history_train,label='Training Data')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('Overfitting Test')
plt.legend()
print(test_mae_score)
plt.grid()
plt.savefig(folder_path + 'error1.png')
plt.show()



p = np.array(model.predict(x_test))
predict_mean= p
    
p = np.array(model.predict(x_train))
predict_mean_train = p

   #test data
mean_x_test_allT = np.mean(predict_mean, axis = 0)
predict_epistemic_allT = np.std(predict_mean, axis = 0)

#training data
mean_x_train_allT = np.mean(predict_mean_train, axis = 0)
predict_epistemic_train_allT = np.std(predict_mean_train, axis = 0)

period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]

diff=np.std(y_train-mean_x_train_allT,axis=0)
difftest=np.std(y_test-mean_x_test_allT,axis=0)
# diffmean=np.mean(y_train-mean_x_train_allT,axis=0)
f22=plt.figure('Difference Std of residuals vs Period')
plt.semilogx(period,diff,label='Training ')
plt.semilogx(period,difftest,label='Testing')
plt.xlabel('Period')
plt.ylabel('Total Standard Deviation')
plt.legend()
plt.savefig(folder_path + 'resid_T.png')
plt.show()

diffmean=np.mean(y_train-mean_x_train_allT,axis=0)
diffmeantest=np.mean(y_test-mean_x_test_allT,axis=0)
f22=plt.figure('Difference Std of residuals vs Period')
plt.semilogx(period,diffmean,label='Training ')
plt.semilogx(period,diffmeantest,label='Testing')
plt.xlabel('Period')
plt.ylabel('Mean residual')
plt.legend()
plt.savefig(folder_path + 'mean_T.png')
plt.show()

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

#new targets
resid_test = predict_mean
resid_train  = predict_mean_train

#%%
transform_method = Normalizer()
epochs = 15
batch_size = 264
folder_path = '/Users/aklimasewski/Documents/2step_ANN/model4/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n=4)
x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw  = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_path)







def build_model():
    #model=models.Sequential()
    model = Sequential()
    model.add(layers.Dense(x_train.shape[1],activation='sigmoid', input_shape=(x_train.shape[1],)))

    #no gP layer
    model.add(layers.Dense(resid_train.shape[1]))

    model.compile(optimizer=optimizers.Adam(lr=0.01),loss='mse',metrics=['mae','mse']) 
    #model.compile(optimizer='adam',loss='mse',metrics=['mae']) 
    return model


model=build_model()



#fit the model
history=model.fit(x_train,resid_train,validation_data=(x_test,resid_test),epochs=epochs,batch_size=batch_size,verbose=1)

mae_history=history.history['val_mae']
mae_history_train=history.history['mae']
test_mse_score,test_mae_score,tempp=model.evaluate(x_test,resid_test)
#dataframe for saving purposes
hist_df = pd.DataFrame(history.history)

f10=plt.figure('Overfitting Test')
plt.plot(mae_history,label='Testing Data')
plt.plot(mae_history_train,label='Training Data')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('Overfitting Test')
plt.legend()
print(test_mae_score)
plt.grid()
plt.savefig(folder_path + 'error_2.png')
plt.show()




p = np.array(model.predict(x_test))
predict_mean2= p
    
p = np.array(model.predict(x_train))
predict_mean_train2 = p

   #test data
mean_x_test_allT = np.mean(predict_mean2, axis = 0)
predict_epistemic_allT = np.std(predict_mean2, axis = 0)

#training data
mean_x_train_allT = np.mean(predict_mean_train2, axis = 0)
predict_epistemic_train_allT = np.std(predict_mean_train2, axis = 0)

period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]

##################
# target1_resid = (resid_train - mean_x_train_all) + 
# target2 = resid_test

target2_residual = resid_test - mean_x_test_allT
target1_residual = 

# target1_residual = resid_test +
# target2_residual + mean_x_test_allT
# 


diff=np.std(resid_train - mean_x_train_allT,axis=0)
difftest=np.std(resid_test-mean_x_test_allT,axis=0)
# diffmean=np.mean(y_train-mean_x_train_allT,axis=0)
f22=plt.figure('Difference Std of residuals vs Period')
plt.semilogx(period,diff,label='Training ')
plt.semilogx(period,difftest,label='Testing')
plt.xlabel('Period')
plt.ylabel('Total Standard Deviation')
plt.legend()
plt.savefig(folder_path + 'resid_T.png')
plt.show()

diffmean=np.mean(resid_train-mean_x_train_allT,axis=0)
diffmeantest=np.mean(resid_test-mean_x_test_allT,axis=0)
f22=plt.figure('Difference Std of residuals vs Period')
plt.semilogx(period,diffmean,label='Training ')
plt.semilogx(period,diffmeantest,label='Testing')
plt.xlabel('Period')
plt.ylabel('Mean residual')
plt.legend()
plt.savefig(folder_path + 'mean_T.png')
plt.show()

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




#%%



def Optimize_kernelparams(train_data1):
    
    #kernel optimization
    #to speed up, try randomly picking 1000 training samples
    num_kernelopt_samples = 500
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
    
    
    for i in range(500):
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

amplitude_opt,length_scale_opt = Optimize_kernelparams(train_data1)

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
#%%




Q = 10 # nr of terms in the sum
max_iters = 1000
D = len(y_train[0])
def create_model(hypers):
    f = np.clip(hypers[:Q], 0, 5)
    weights = np.ones(Q) / Q
    lengths = hypers[Q:]

    kterms = []
    for i in range(Q):
        # rbf = gpflow.kernels.RBF(D, lengthscales=lengths[i], variance=1./Q)
        mat =  tfp.math.psd_kernels.MaternOneHalf(feature_ndims=D, amplitude=1./Q, length_scale=lengths[i])
        transformation_fn=lambda x, _: tf.exp(x)
        # rbf.lengthscales.transform = gpflow.transforms.Exp()
        mat = tfp.math.psd_kernels.FeatureTransformed(mat, transformation_fn, validate_args=False, parameters=None,name='FeatureTransformed')
        
        cos = tfp.math.psd_kernels.Linear(feature_ndims=D, bias_variance=None, slope_variance=None)
        transformation_fn=lambda x, _: tf.cos(x)
        cos = tfp.math.psd_kernels.FeatureTransformed(cos, transformation_fn, validate_args=False, parameters=None,name='FeatureTransformed')

        kterms.append(mat * cos)

    k = np.sum(kterms) + tfp.math.psd_kernels.Linear(D) #+ gpflow.kernels.Bias(D)
    # m = gpflow.gpr.GPR(X_train, Y_train, kern=k)
    return k

k = create_model(np.ones((2*Q,)))




