#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 17:30:17 2020

@author: aklimasewski
"""

import sys
import os
sys.path.append(os.path.abspath('/Users/aklimase/Documents/nonergodic_ANN'))
from preprocessing import transform_dip, readindata, transform_data, create_grid, grid_data
from model_plots import gridded_plots, plot_resid, obs_pre
import numpy as np
import pandas as pd
from keras.models import Sequential
import matplotlib as mpl
import matplotlib.pyplot as plt

from keras import layers
from keras import optimizers
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

folder_path = '/Users/aklimasewski/Documents/models/create_grid_1/'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
transform_method = 'Norm' #function or text
epochs = 13
n = 13
#or n = 6, 4
az = True
unit_est = 2*(n+10)+1


#create grid
df, lon, lat = create_grid(latmin=32,latmax=37.5,lonmin=-121,lonmax=-115.5,dx=.25)

train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = n)

#add the location features
train_data1_4, test_data1_4, train_targets1_4, test_targets1_4, feature_names_4 = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = 4)
train_data1 = np.concatenate([train_data1,train_data1_4], axis = 1)
test_data1 = np.concatenate([test_data1,test_data1_4], axis = 1)
feature_names = np.concatenate([feature_names,feature_names_4], axis = 0)

midpoint = np.asarray([(train_data1[:,13]+train_data1[:,15])/2.,(train_data1[:,14]+train_data1[:,16])/2.]).T
midpoint_test = np.asarray([(test_data1[:,13]+test_data1[:,15])/2.,(test_data1[:,14]+test_data1[:,16])/2.]).T

#add the path features
train_data1 = np.concatenate([train_data1,midpoint], axis = 1)
test_data1 = np.concatenate([test_data1,midpoint_test], axis = 1)
feature_names = np.concatenate([feature_names,np.asarray(['midpointlat','midpointlon'])], axis = 0)
# ######
    
#%%
def grid_points(data,df,name):
    import shapely
    import shapely.geometry
    import numpy as np
    import geopy
    import geopy.distance
    from shapely.geometry import Point

    
    sitelat = data[:,13]
    sitelon = data[:,14]
    evlat = data[:,15]
    evlon = data[:,16]
    midlat = data[:,17]
    midlon = data[:,18]
    
    
    gridded_num = np.zeros((len(sitelat),3))#event, mid, site'
    gridded_mid = np.zeros((len(sitelat),6))#event, mid, site'
    
    gridded_counts = np.zeros((df.shape[0],3))
    
    
    #loop through each record     
    for i in range(len(sitelat)):    
        event = shapely.geometry.Point(evlon[i], evlat[i])
        mid = shapely.geometry.Point(midlon[i], midlat[i])
        site = shapely.geometry.Point(sitelon[i], sitelat[i])
        #loop through each grid cell
        #add a 1 for the column if event, mid, site in the cell
        if (i % 1000) == 0:
        	print('record: ', str(i))
        for j in range(len(df)):
            shapely_poly = df['polygon'][j]
            if event.within(shapely_poly) == True:
                gridded_mid[i,0:2] = [df['latmid'][j],df['lonmid'][j]]
                gridded_num[i,0] = j
                gridded_counts[j,0] += 1
            if mid.within(shapely_poly) == True:
                gridded_mid[i,2:4] = [df['latmid'][j],df['lonmid'][j]]
                gridded_num[i,1] = j
                gridded_counts[j,1] +=1
            if site.within(shapely_poly) == True:
                gridded_mid[i,4:6] = [df['latmid'][j],df['lonmid'][j]]
                gridded_num[i,2] = j
                gridded_counts[j,2] += 1
    
    df_out = pd.DataFrame(gridded_mid, columns=['eventlat','eventlon','midlat','midlon','sitelat','sitelon'])   
    df_out.to_csv(folder_path + 'gridpointslatlon_' + name + '.csv')   
    
    df_out = pd.DataFrame(gridded_num, columns=['event','mid','site'])   
    df_out.to_csv(folder_path + 'gridpoints_' + name + '.csv')
    
    df_out = pd.DataFrame(gridded_counts, columns=['event','mid','site'])   
    df_out.to_csv(folder_path + 'counts_' + name + '.csv')

grid_points(train_data1,df,name='train')
grid_points(test_data1,df,name='test')

#%%
# # import matplotlib as mpl
# # import matplotlib.pyplot as plt
# # #make figs

# # # counts
def plot_points(gridded_counts,name):
    colname = ['event','midpoint','site']
    
    for i in range(len(gridded_counts[0])):
        Z = gridded_counts[:,i].reshape(len(lat)-1,len(lon)-1)
        
        cbound = np.max(np.abs(Z))
        cmap = mpl.cm.get_cmap('Greens')
        normalize = mpl.colors.Normalize(vmin=0, vmax=cbound)
        colors = [cmap(normalize(value)) for value in Z]
        s_m = mpl.cm.ScalarMappable(cmap = cmap, norm=normalize)
        s_m.set_array([])
        
        fig, ax = plt.subplots(figsize = (10,8))
        plt.pcolormesh(lon, lat, Z, cmap = cmap, norm = normalize) 
        plt.scatter(evlon,evlat,marker = '*', s=1, c = 'gray', label = 'event')
        plt.scatter(sitelon,sitelat,marker = '^',s=1, c = 'black', label = 'site')
        plt.xlim(min(lon),max(lon))
        plt.ylim(min(lat),max(lat))
        plt.title(colname)
        plt.legend(loc = 'lower left')
        
        fig.subplots_adjust(right=0.75)
        cbar = plt.colorbar(s_m, orientation='vertical')
        cbar.set_label(colname[i] + ' counts', fontsize = 20)
        plt.savefig(folder_path + colname[i] + name + '_counts.png')
        plt.show()
        
        plt.close('all')

counts_train = pd.read_csv(folder_path + 'counts_train.csv',header = 0,index_col=0)
counts_test = pd.read_csv(folder_path + 'counts_test.csv',header = 0,index_col=0)

plot_points(np.asarray(counts_train),name='train')
plot_points(np.asarray(counts_test),name='test')

#%%
#with 2d midpoints
cells = pd.read_csv(folder_path + 'gridpointslatlon_train.csv',header = 0,index_col=0)
cells_test = pd.read_csv(folder_path + 'gridpointslatlon_test.csv',header = 0,index_col=0)

folder_pathmod = folder_path + 'ANN13_gridmidpoints/'

if not os.path.exists(folder_pathmod):
    os.makedirs(folder_pathmod)
    
transform_method = 'Norm' #function or text
epochs = 13
n = 13
#or n = 6, 4
az = True
unit_est = 2*(n+10)+1

train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = n)

#add the location features
# train_data1_4, test_data1_4, train_targets1_4, test_targets1_4, feature_names_4 = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = 4)
train_data1 = np.concatenate([train_data1,cells], axis = 1)
test_data1 = np.concatenate([test_data1,cells_test], axis = 1)
# feature_names = np.concatenate([feature_names,['event','mid','site']], axis = 0)
feature_names = np.concatenate([feature_names,['eventlat','eventlon','midlat','midlon','sitelat','sitelon',]], axis = 0)


x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_pathmod)

Rindex = np.where(feature_names == 'Rrup')[0][0]

batch_size = 256


def build_model():
    #model=models.Sequential()
    model = Sequential()
    # model.add(layers.Dense(6,activation='sigmoid', input_shape=(x_train.shape[1],)))
    model.add(layers.Dense(10,activation='sigmoid', input_shape=(x_train.shape[1],)))
    # model.add(layers.Dense(50)) #add sigmoid aciivation functio? (only alues betwen 0 and 1)
    # model.add(layers.Dense(50)) #add sigmoid aciivation functio? (only alues betwen 0 and 1)

    model.add(layers.Dense(y_train.shape[1])) #add sigmoid aciivation functio? (only alues betwen 0 and 1)

    model.compile(optimizer=optimizers.Adam(lr=2e-3),loss='mse',metrics=['mae','mse']) 
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
plt.savefig(folder_pathmod + 'error.png')
plt.show()

# pr1edict_mean = []
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

diff=np.std(y_train-mean_x_train_allT,axis=0)
difftest=np.std(y_test-mean_x_test_allT,axis=0)
#write model details to a file
file = open(folder_pathmod + 'model_details.txt',"w+")
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


# pre = predict_mean_train
# pre_test = predict_mean

period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]
plot_resid(resid, resid_test, folder_pathmod)

#%% 
#with cell numbers

cells = pd.read_csv(folder_path + 'gridpoints_train.csv',header = 0,index_col=0)
cells_test = pd.read_csv(folder_path + 'gridpoints_test.csv',header = 0,index_col=0)

folder_pathmod = folder_path + 'ANN13_gridnum/'

if not os.path.exists(folder_pathmod):
    os.makedirs(folder_pathmod)
    
transform_method = 'Norm' #function or text
epochs = 13
n = 13
#or n = 6, 4
az = True
unit_est = 2*(n+10)+1

train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = n)

#add the location features
# train_data1_4, test_data1_4, train_targets1_4, test_targets1_4, feature_names_4 = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = 4)
train_data1 = np.concatenate([train_data1,cells], axis = 1)
test_data1 = np.concatenate([test_data1,cells_test], axis = 1)
feature_names = np.concatenate([feature_names,['event','mid','site']], axis = 0)

x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_pathmod)

Rindex = np.where(feature_names == 'Rrup')[0][0]

batch_size = 256


def build_model():
    #model=models.Sequential()
    model = Sequential()
    # model.add(layers.Dense(6,activation='sigmoid', input_shape=(x_train.shape[1],)))
    model.add(layers.Dense(10,activation='sigmoid', input_shape=(x_train.shape[1],)))
    # model.add(layers.Dense(50)) #add sigmoid aciivation functio? (only alues betwen 0 and 1)
    # model.add(layers.Dense(50)) #add sigmoid aciivation functio? (only alues betwen 0 and 1)

    model.add(layers.Dense(y_train.shape[1])) #add sigmoid aciivation functio? (only alues betwen 0 and 1)

    model.compile(optimizer=optimizers.Adam(lr=2e-3),loss='mse',metrics=['mae','mse']) 
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
plt.savefig(folder_pathmod + 'error.png')
plt.show()

# pr1edict_mean = []
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

diff=np.std(y_train-mean_x_train_allT,axis=0)
difftest=np.std(y_test-mean_x_test_allT,axis=0)
#write model details to a file
file = open(folder_pathmod + 'model_details.txt',"w+")
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


# pre = predict_mean_train
# pre_test = predict_mean

period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]
plot_resid(resid, resid_test, folder_pathmod)