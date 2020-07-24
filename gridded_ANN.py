#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 16:38:02 2020

@author: aklimasewski
"""

import shapely
import shapely.geometry
import geopy
import geopy.distance
import numpy as np
from preprocessing import transform_dip, readindata, transform_data
import matplotlib.pyplot as plt
import shapely.geometry
import pandas as pd
from matplotlib import cm
import matplotlib as mpl
from sklearn.preprocessing import Normalizer
import random
import keras
from keras.models import Sequential
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import Normalizer
#grid up residuals
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
from keras import layers
from keras import optimizers

folder_path = '/Users/aklimasewski/Documents/gridded_ANN/10Tgrid_rand20000/'

nsamples = 5000

def create_grid(latmin=32,latmax=37.5,lonmin=-121,lonmax=-115.5,dx=0.1):
    dx=0.1
    lon = np.arange(-121,-115.5, dx)
    lat = np.arange(32, 37.5, dx)
    
    latmid = []
    lonmid = []
    polygons = []
    for i in range(len(lon)-1):
        for j in range(len(lat)-1):
            polygon_points = [(lon[i], lat[j]), (lon[i], lat[j+1]), (lon[i+1], lat[j+1]), (lon[i+1], lat[j]), (lon[i], lat[j])]
            shapely_poly = shapely.geometry.Polygon(polygon_points)
            polygons.append(shapely_poly)
            latmid.append((lat[j]+lat[j+1])/2.)
            lonmid.append((lon[i]+lon[i+1])/2.)
               
    d = {'polygon': polygons, 'latmid': latmid, 'lonmid': lonmid}
    df = pd.DataFrame(data=d)    
    return df
    
#choose random subset for fast testing
def grid_data(train_data1, train_targets1, df, nsamples = 5000):
    randindex = random.sample(range(0, len(train_data1)), nsamples)
    
    hypoR = train_data1[:,0][randindex]
    sitelat = train_data1[:,1][randindex]
    sitelon = train_data1[:,2][randindex]
    evlat = train_data1[:,3][randindex]
    evlon = train_data1[:,4][randindex]
    target = train_targets1[:][randindex]
    
    normtarget = target / hypoR[:, np.newaxis]
    gridded_targetsnorm_list = [ [] for _ in range(df.shape[0]) ]
    
    gridded_counts = np.zeros(df.shape[0])
    lenlist = []
    
    #loop through each record     
    for i in range(len(sitelat)):                         
        line = [(evlon[i], evlat[i]), (sitelon[i], sitelat[i])]
        path=shapely.geometry.LineString(line)
        #loop through each grid cell
        for j in range(len(df)):
            shapely_poly = df['polygon'][j]
            if path.intersects(shapely_poly) == True:
                shapely_line = shapely.geometry.LineString(line)
                intersection_line = list(shapely_poly.intersection(shapely_line).coords)
                if len(intersection_line)== 2:
                    coords_1 = (intersection_line[0][1], intersection_line[0][0])
                    coords_2 = (intersection_line[1][1], intersection_line[1][0])
                    length=geopy.distance.distance(coords_1, coords_2).km
                    gridded_targetsnorm_list[j].append(normtarget[i]*length)          
                    gridded_counts[j] += 1
                    lenlist.append(length)
                
        return hypoR, sitelat, sitelon, evlat, evlon, target, gridded_targetsnorm_list, gridded_counts
    
    
train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n=6)
#start with defaults
df = create_grid()

hypoR, sitelat, sitelon, evlat, evlon, target, gridded_targetsnorm_list, gridded_counts = grid_data(train_data1, train_targets1, df, nsamples = 5000)           
#%%
# #test data
# randindex = random.sample(range(0, len(test_data1)), 5000)

# hypoR = test_data1[:,0][randindex]
# sitelat = test_data1[:,1][randindex]
# sitelon = test_data1[:,2][randindex]
# evlat = test_data1[:,3][randindex]
# evlon = test_data1[:,4][randindex]
# # target = train_targets1[:,5][randindex]
# target_test = test_targets1[:][randindex]

# # normtarget = target/hypoR
# normtarget_test = target_test / hypoR[:, np.newaxis]
# # gridded_targets_sum = np.zeros(df.shape[0])
# gridded_targetsnorm_list_test = [ [] for _ in range(5000) ]
# # gridded_targets_list = [ np.ones(10) for _ in range(df.shape[0]) ]

# gridded_counts_test = np.zeros(5000)
# lenlist = []

# #loop through each record     
# for i in range(len(sitelat)):                         
#     line = [(evlon[i], evlat[i]), (sitelon[i], sitelat[i])]
#     path=shapely.geometry.LineString(line)
#     #loop through each grid cell
#     for j in range(len(df)):
#         shapely_poly = polygons[j]
#         if path.intersects(shapely_poly) == True:
#             shapely_line = shapely.geometry.LineString(line)
#             intersection_line = list(shapely_poly.intersection(shapely_line).coords)
#             if len(intersection_line)== 2:
#                 coords_1 = (intersection_line[0][1], intersection_line[0][0])
#                 coords_2 = (intersection_line[1][1], intersection_line[1][0])
#                 length=geopy.distance.distance(coords_1, coords_2).km

#                 gridded_targetsnorm_list_test[j].append(normtarget_test[i]*length)          

#                 gridded_counts_test[j] += 1
#                 #lenlist.append(length)
                
# #find mean of norm residual
# gridded_targetsnorm_list_test = np.asarray(gridded_targetsnorm_list_test)

# griddednorm_mean_test=np.zeros((len(gridded_targetsnorm_list_test),10))
# for i in range(len(gridded_targetsnorm_list_test)):
#     griddednorm_mean_test[i] = np.mean(gridded_targetsnorm_list_test[i],axis=0)

# #find the cells with no paths (nans)
# nan_ind=np.argwhere(np.isnan(griddednorm_mean_test)).flatten()
# # set nan elements for empty array
# # not sureif this isbest orto leaveas a nan
# for i in nan_ind:
#     griddednorm_mean_test[i] =0


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
# not sureif this isbest orto leaveas a nan
for i in nan_ind:
    griddednorm_mean[i] =0


#add gridded mean and gridded counts to df
df['counts'] = gridded_counts
period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]


for i in range(len(griddednorm_mean.T)):
    T= period[i]
    g = griddednorm_mean.T[i]
    Z = g.reshape(len(lat)-1,len(lon)-1)
    
    cbound = max(np.abs(g))
    cbound = 0.12

    cmap = mpl.cm.get_cmap('seismic')
    normalize = mpl.colors.Normalize(vmin=-1*cbound, vmax=cbound)
    colors = [cmap(normalize(value)) for value in Z]
    s_m = mpl.cm.ScalarMappable(cmap = cmap, norm=normalize)
    s_m.set_array([])
        
    fig, ax = plt.subplots(figsize = (10,8))
    plt.pcolormesh(lon, lat, Z, cmap = cmap, norm = normalize) 
    plt.scatter(evlon,evlat,marker = '*', s=1, c = 'gray', label = 'event')
    plt.scatter(sitelon,sitelat,marker = '^',s=1, c = 'black', label = 'site')
    plt.xlim(min(lon),max(lon))
    plt.ylim(min(lat),max(lat))
    plt.title('T ' + str(T) + ' s')
    plt.legend(loc = 'lower left')
    
    fig.subplots_adjust(right=0.75)
    cbar = plt.colorbar(s_m, orientation='vertical')
    cbar.set_label(r'average normalized residual (resid/km)', fontsize = 20)
    plt.savefig(folder_path + 'normresid_T_' + str(T) + '.png')
    plt.show()
    

# counts
Z = gridded_counts.reshape(len(lat)-1,len(lon)-1)

cbound = max(np.abs(gridded_counts))
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
plt.title('T ' + str(T) + ' s')
plt.legend(loc = 'lower left')

fig.subplots_adjust(right=0.75)
cbar = plt.colorbar(s_m, orientation='vertical')
cbar.set_label(r'paths per cell', fontsize = 20)
plt.savefig(folder_path + 'pathcounts.png')
plt.show()


#%%
y_train = griddednorm_mean

x_test
x_train = df.drop(['polygon','counts'], axis=1)

transform = Normalizer()
aa=transform.fit(x_train)
train_data=aa.transform(x_train)
# test_data=aa.transform(x_test)

batch_size = 264

def build_model():
    model = Sequential()
    model.add(layers.Dense(train_data.shape[1],activation='sigmoid', input_shape=(train_data.shape[1],)))
    # model.add(layers.Dense(10))

    #no gP layer
    model.add(layers.Dense(10))

    model.compile(optimizer=optimizers.Adam(lr=0.01),loss='mse',metrics=['mae','mse']) 
    return model


model=build_model()

#fit the model
history=model.fit(train_data,y_train,epochs=100,batch_size=batch_size,verbose=1)

# mae_history=history.history['val_mae']
mae_history_train=history.history['mae']
# test_mse_score,test_mae_score,tempp=model.evaluate(x_test,y_test)

pre = model.predict(train_data)
# r = np.asarray(y_train)-pre.flatten()
r = (y_train)-pre

for i in range(10):
    T= period[i]
    y = pre.T[i]
    x = y_train.T[i]
    plt.figure(figsize = (6,6))
    lim = np.max(np.asarray([abs(x), abs(y)]).flatten())
    plt.scatter(x,y,s=1)
    plt.xlabel('observed')
    plt.xlabel('predicted')
    plt.title('T ' + str(T) + ' s')
    plt.xlim(-1*lim, lim)
    plt.ylim(-1*lim, lim)
    plt.savefig(folder_path + 'obs_pre_T_' + str(T) + '.png')
    plt.show()
