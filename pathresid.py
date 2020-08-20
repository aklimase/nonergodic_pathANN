#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 08:50:22 2020

@author: aklimase
"""

import sys
import os
sys.path.append(os.path.abspath('/Users/aklimase/Documents/USGS/nonergodic_ANN'))
from preprocessing import transform_dip, readindata, transform_data
from model_plots import gridded_plots, obs_pre, plot_resid, plot_outputs
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
import shapely
import geopy
import geopy.distance
import shapely.wkt
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import random

# import tensorflow_probability as tfp

# For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float64')

sns.reset_defaults()
sns.set_style('whitegrid')
sns.set_context('talk')
sns.set_context(context='talk',font_scale=0.7)

#read in gridded residual file

topdir = '/Users/aklimasewski/Documents/'

# folder_path = topdir + 'models/2step_ANN/modelgriddedresiduals_1/'
folder_path = topdir + 'models/gridtargets_ANN/modelgriddedtargets_1/'
df = pd.read_csv(folder_path + 'griddedtargets_dx_1.csv')

gridcells = df['polygon']
targets = df[['T10','T7.5','T5','T4','T3','T2','T1','T0.5','T0.2','T0.1']]
list_wkt = df['polygon']
list_polygons =  [shapely.wkt.loads(poly) for poly in list_wkt]

###
n = 6
train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimase/Documents/USGS/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimase/Documents/USGS/data/cybertestyeti10_residfeb.csv', n = n)


hypoR = train_data1[:,0]
sitelat = train_data1[:,1]
sitelon = train_data1[:,2]
evlat = train_data1[:,3]
evlon = train_data1[:,4]
target = train_targets1[:]

# normtarget = target / hypoR[:, np.newaxis]
# gridded_targetsnorm_list = [ [] for _ in range(df.shape[0]) ]

path_target_sum = np.zeros((len(hypoR),10))#length of number of records
# gridded_counts = np.zeros(df.shape[0])
# lenlist = []
# 
#loop through each record     
for i in range(len(sitelat)):                       
    line = [(evlon[i], evlat[i]), (sitelon[i], sitelat[i])]
    path=shapely.geometry.LineString(line)
    #loop through each grid cell
    if (i % 1000) == 0:
        print('record: ', str(i))
    pathsum = 0
    for j in range(len(list_polygons)):
        # shapely_poly = df['polygon'][j].split('(')[2].split(')')[0]
        # polygon_points = [(lon[i], lat[j]), (lon[i], lat[j+1]), (lon[i+1], lat[j+1]), (lon[i+1], lat[j]), (lon[i], lat[j])]
        shapely_poly = shapely.geometry.Polygon(list_polygons[j])
        if path.intersects(shapely_poly) == True:
            shapely_line = shapely.geometry.LineString(line)
            intersection_line = list(shapely_poly.intersection(shapely_line).coords)
            if len(intersection_line)== 2:
                coords_1 = (intersection_line[0][1], intersection_line[0][0])
                coords_2 = (intersection_line[1][1], intersection_line[1][0])
                length=geopy.distance.distance(coords_1, coords_2).km
                
                pathsum += length*np.asarray(targets.iloc[j])
        
    path_target_sum[i] = (pathsum)          

                
                
# dictout = {['T10','T7.5','T5','T4','T3','T2','T1','T0.5','T0.2','T0.1']:path_target_sum}
df_out = pd.DataFrame(path_target_sum, columns=['T10','T7.5','T5','T4','T3','T2','T1','T0.5','T0.2','T0.1'])   
# df_save.append(df2)
df_out.to_csv(folder_path + 'avgrecord_targets_dx_1.csv')            

#

#%%

df = pd.read_csv(folder_path + 'griddedtargets_dx_1.csv')

gridcells = df['polygon']
targets = df[['T10test','T7.5test','T5test','T4test','T3test','T2test','T1test','T0.5test','T0.2test','T0.1test']]
list_wkt = df['polygon']
list_polygons =  [shapely.wkt.loads(poly) for poly in list_wkt]


hypoR = test_data1[:,0]
sitelat = test_data1[:,1]
sitelon = test_data1[:,2]
evlat = test_data1[:,3]
evlon = test_data1[:,4]
target = test_targets1[:]

# normtarget = target / hypoR[:, np.newaxis]
# gridded_targetsnorm_list = [ [] for _ in range(df.shape[0]) ]

path_target_sum_test = np.zeros((len(hypoR),10))#length of number of records
# gridded_counts = np.zeros(df.shape[0])
# lenlist = []
# 
#loop through each record     
for i in range(len(sitelat)):                       
    line = [(evlon[i], evlat[i]), (sitelon[i], sitelat[i])]
    path=shapely.geometry.LineString(line)
    #loop through each grid cell
    if (i % 1000) == 0:
        print('record: ', str(i))
    pathsum = 0
    for j in range(len(list_polygons)):
        # shapely_poly = df['polygon'][j].split('(')[2].split(')')[0]
        # polygon_points = [(lon[i], lat[j]), (lon[i], lat[j+1]), (lon[i+1], lat[j+1]), (lon[i+1], lat[j]), (lon[i], lat[j])]
        shapely_poly = shapely.geometry.Polygon(list_polygons[j])
        if path.intersects(shapely_poly) == True:
            shapely_line = shapely.geometry.LineString(line)
            intersection_line = list(shapely_poly.intersection(shapely_line).coords)
            if len(intersection_line)== 2:
                coords_1 = (intersection_line[0][1], intersection_line[0][0])
                coords_2 = (intersection_line[1][1], intersection_line[1][0])
                length=geopy.distance.distance(coords_1, coords_2).km
                
                pathsum += length*np.asarray(targets.iloc[j])
        
    path_target_sum_test[i] = (pathsum)          

                
# dictout = {['T10','T7.5','T5','T4','T3','T2','T1','T0.5','T0.2','T0.1']:path_target_sum}
df_out = pd.DataFrame(path_target_sum, columns=['T10','T7.5','T5','T4','T3','T2','T1','T0.5','T0.2','T0.1'])   
# df_save.append(df2)
df_out.to_csv(folder_path + 'avgrecord_targets_test_dx_1.csv')            



#%%
#load csvs
folder_path = '/Users/aklimase/Documents/USGS/models/gridtargets_ANN/modelgriddedtargets_1/' + 'ANNfigs/'

train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimase/Documents/USGS/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimase/Documents/USGS/data/cybertestyeti10_residfeb.csv', n = 12)
train_data1 = np.append(train_data1,path_target_sum, axis=1)
test_data1 = np.append(test_data1,path_target_sum_test, axis=1)
feature_names = np.append(feature_names, ['T10','T7.5','T5','T4','T3','T2','T1','T0.5','T0.2','T0.1'], axis=0)
                          

transform_method = PowerTransformer()
x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_path)
             

batch_size = 264

def build_model():
    model = Sequential()
    model.add(layers.Dense(x_train.shape[1],activation='sigmoid', input_shape=(x_train.shape[1],)))
    # model.add(layers.Dense(10))
    # model.add(RBFLayer(10, 2))

    #no gP layer
    # model.add(layers.Dense(10))
    model.add(layers.Dense(y_train.shape[1]))

    model.compile(optimizer=optimizers.Adam(lr=0.01),loss='mse',metrics=['mae','mse']) 
    return model


model=build_model()

epochs = 10
#fit the model
# history=model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,verbose=1)
history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=batch_size,verbose=1)


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





pre = model.predict(x_train)
r = (y_train)-pre
pre_test = model.predict(x_test)
r_test = (y_test)-pre_test
         

period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]

plot_resid(resid = r, resid_test = r_test, folder_path = folder_path)
obs_pre(y_train, y_test, pre, pre_test, period, folder_path)








mean_x_test_allT = pre_test
predict_epistemic_allT = np.zeros((len(x_test), 10))

#training data
mean_x_train_allT = pre
predict_epistemic_train_allT = np.zeros((len(x_train), 10))

diff=np.std(y_train-mean_x_train_allT,axis=0)
difftest=np.std(y_test-mean_x_test_allT,axis=0)
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



period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]

Rindex = np.where(feature_names == 'Rrup')[0][0]

# plot_resid(resid = y_train-mean_x_train_allT, resid_test = y_test-mean_x_test_allT, folder_path = folder_path)
plot_outputs(folder_path, mean_x_test_allT, predict_epistemic_allT, mean_x_train_allT, predict_epistemic_train_allT, x_train, y_train, x_test, y_test, Rindex, period, feature_names)






                