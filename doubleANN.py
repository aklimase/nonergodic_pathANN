#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 13:19:45 2020

@author: aklimasewski


build a model that takes cybershake data, input to 12 feature ANN,input those residuals to a spatial ANN

use spatial kernel in second model
"""
import sys
import os
sys.path.append(os.path.abspath('/Users/aklimasewski/Documents/nonergodic_ANN'))
from preprocessing import transform_dip, readindata, transform_data

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
from sklearn.preprocessing import StandardScaler

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
# import tensorflow as tf
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
topdir = '/Users/aklimase/Documents/USGS/'
folder_path = topdir + 'models/2step_ANN/model12/'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
transform_method = PowerTransformer()
epochs = 15
batch_size = 264


train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain= topdir + 'data/cybertrainyeti10_residfeb.csv', nametest=topdir + 'data/cybertestyeti10_residfeb.csv', n=12)
x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_path)


#12 feature ANN
def build_model():
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
#predictions
resid_test = predict_mean
resid_train  = predict_mean_train

#residuals
resid_test = y_test-mean_x_test_allT
resid_train = y_train-mean_x_train_allT
#%%
transform_method =  StandardScaler()
epochs = 15
batch_size = 264
folder_path = topdir + 'models/2step_ANN/model4_residuals/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain=topdir + 'data/cybertrainyeti10_residfeb.csv', nametest=topdir + 'data/cybertestyeti10_residfeb.csv', n=4)
#redefine targets
x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw  = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_path)

def build_model():
    #model=models.Sequential()
    model = Sequential()
    model.add(layers.Dense(x_train.shape[1],activation='sigmoid', input_shape=(x_train.shape[1],)))
    # model.add(RBFLayer(10, 2))

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
from keras.layers import Layer
from keras import backend as K

class RBFLayer(Layer):

    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
#         print(input_shape)
#         print(self.units)
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
