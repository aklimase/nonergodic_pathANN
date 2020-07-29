#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:38:58 2020

@author: aklimase

2 ANNs, first 12 feature model, second gridded target residuals
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

#%%
folder_path = topdir + 'models/2step_ANN/modelgriddedresiduals/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
#new targets
resid_test = y_test-mean_x_test_allT
resid_train = y_train-mean_x_train_allT

df, lon, lat = create_grid(dx = 0.05)
nsamples = 1000

hypoR, sitelat, sitelon, evlat, evlon, target, gridded_targetsnorm_list, gridded_counts = grid_data(train_data1, train_targets1 = resid_train, df=df, nsamples = nsamples)     
hypoR_test, sitelat_test, sitelon_test, evlat_test, evlon_test, target_test, gridded_targetsnorm_list_test, gridded_counts_test = grid_data(test_data1, train_targets1 = resid_test, df=df, nsamples = nsamples)    

             
#%%

#find mean of norm residual
gridded_targetsnorm_list = np.asarray(gridded_targetsnorm_list)

griddednorm_mean=np.zeros((len(gridded_targetsnorm_list),10))
for i in range(len(gridded_targetsnorm_list)):
    # for j in range(10):
    griddednorm_mean[i] = np.mean(gridded_targetsnorm_list[i],axis=0)

#find the cells with no paths (nans)
nan_ind=np.argwhere(np.isnan(griddednorm_mean)).flatten()
# set nan elements for empty array
for i in nan_ind:
    griddednorm_mean[i] = 0
    
#find mean of norm residual
gridded_targetsnorm_list_test = np.asarray(gridded_targetsnorm_list_test)

griddednorm_mean_test=np.zeros((len(gridded_targetsnorm_list_test),10))
for i in range(len(gridded_targetsnorm_list_test)):
    # for j in range(10):
    griddednorm_mean_test[i] = np.mean(gridded_targetsnorm_list_test[i],axis=0)

#find the cells with no paths (nans)
nan_ind=np.argwhere(np.isnan(griddednorm_mean_test)).flatten()
# set nan elements for empty array
for i in nan_ind:
    griddednorm_mean_test[i] = 0

y_train = griddednorm_mean
y_test = griddednorm_mean_test

# x_test
x_train = df.drop(['polygon','counts'], axis=1)

transform = Normalizer()
aa=transform.fit(x_train)
train_data=aa.transform(x_train)
test_data=aa.transform(x_test)

batch_size = 264

def build_model():
    model = Sequential()
    model.add(layers.Dense(train_data.shape[1],activation='sigmoid', input_shape=(train_data.shape[1],)))
    # model.add(layers.Dense(10))
    # model.add(RBFLayer(10, 2))

    #no gP layer
    model.add(layers.Dense(10))

    model.compile(optimizer=optimizers.Adam(lr=0.01),loss='mse',metrics=['mae','mse']) 
    return model


model=build_model()

#fit the model
history=model.fit(train_data,y_train,epochs=10,batch_size=batch_size,verbose=1)

# mae_history=history.history['val_mae']
mae_history_train=history.history['mae']
# test_mse_score,test_mae_score,tempp=model.evaluate(x_test,y_test)

pre = model.predict(train_data)
r = (y_train)-pre
pre_test = model.predict(test_data)
r_test = (y_test)-pre

period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]

##################

diff=np.std(r,axis=0)
difftest=np.mean(r_test,axis=0)
f22=plt.figure('Difference Std of residuals vs Period')
plt.semilogx(period,diff,label='Training ')
plt.semilogx(period,difftest,label='Testing')
plt.xlabel('Period')
plt.ylabel('Total Standard Deviation')
plt.legend()
plt.savefig(folder_path + 'resid_T.png')
plt.show()

diffmean=np.mean(r,axis=0)
diffmeantest=np.mean(r_test,axis=0)
f22=plt.figure('Difference Std of residuals vs Period')
plt.semilogx(period,diffmean,label='Training')
plt.semilogx(period,diffmeantest,label='Testing')
plt.xlabel('Period')
plt.ylabel('Mean residual')
plt.legend()
plt.savefig(folder_path + 'mean_T.png')
plt.show()

for i in range(10):
    T= period[i]
    y = pre.T[i]
    x = y_train.T[i]
    y_test = pre_test.T[i]
    x_test = y_test.T[i]
    plt.figure(figsize = (6,6))
    lim = np.max(np.asarray([abs(x), abs(y)]).flatten())
    plt.scatter(x,y,s=1,label='Training')
    plt.scatter(x_test,y_test,s=1,label='Testing')
    plt.xlabel('observed')
    plt.ylabel('predicted')
    plt.title('T ' + str(T) + ' s')
    plt.xlim(-1*lim, lim)
    plt.ylim(-1*lim, lim)
    plt.savefig(folder_path + 'obs_pre_T_' + str(T) + '.png')
    plt.show()

###next use grided predictions to add back to original targets

