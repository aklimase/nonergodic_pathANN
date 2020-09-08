#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:55:14 2020

@author: aklimasewski
"""

folder_path = '/Users/aklimasewski/Documents/model_results/base/crossedcols/'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
transform_method = 'Norm' #function or text
epochs =20
n = 13
#or n = 6, 4
az = True
unit_est = 2*(n+10)+1

train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = n)


train_data1,test_data1, feature_names = add_az(train_data1,test_data1, feature_names)
    


#%%
    
#transform data
x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_path)

#%%

feature_columns = []

# numeric cols
for name in feature_names:
  feature_columns.append(tf.feature_column.numeric_column(name))
  
  
  
  # #add the location features
train_data1_4, test_data1_4, train_targets1_4, test_targets1_4, feature_names_4 = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = 4)

training_examples = pd.DataFrame(data=train_data1_4, columns=feature_names_4)


x_train = np.concatenate((x_train,train_data1_4),axis=1)
y_train = np.concatenate((y_train,train_targets1_4),axis=1)
x_test = np.concatenate((x_test,test_data1_4),axis=1)
y_test = np.concatenate((y_test,test_targets1_4),axis=1)



def get_quantile_based_boundaries(feature_values, num_buckets):
  boundaries = np.arange(1.0, num_buckets) / num_buckets
  quantiles = feature_values.quantile(boundaries)
  return [quantiles[q] for q in quantiles.keys()]

longitude = tf.feature_column.numeric_column("longitude")

bucketized_longitude = tf.feature_column.bucketized_column(
   longitude, boundaries=get_quantile_based_boundaries(
    training_examples['stlon'], 10))

latitude = tf.feature_column.numeric_column("latitude")

bucketized_latitude = tf.feature_column.bucketized_column(
    latitude, boundaries=get_quantile_based_boundaries(
      training_examples["stlat"], 10))

long_x_lat = tf.feature_column.crossed_column(
  set([bucketized_longitude, bucketized_latitude]), hash_bucket_size=1000) 


feature_columns.append(tf.feature_column.indicator_column(long_x_lat))

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

ds = tf.data.Dataset.from_tensor_slices(x_train, feaure_columns))

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dropout(.1),
  layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

    
    #fit the model
history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=epochs,verbose=1)
    




# create sequential model with positional inputs and predict 
# try sequential with VGP layer
# For numeric stability, set the default floating-point dtype to float64
numlayers = 1
units= [50]
resid_train, resid_test, pre_train, pre_test = create_ANN(x_train, y_train, x_test, y_test, feature_names, numlayers, units, epochs, transform_method, folder_path)

period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]
plot_resid(resid_train, resid_test, folder_path)

#%%%



import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath('/Users/aklimasewski/Documents/python_code_nonergodic'))
from preprocessing import transform_dip, readindata, transform_data, add_az
from model_plots import plot_resid, obs_pre

from keras import optimizers
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
#!pip install tensorflow==2.0.0-alpha0
import tensorflow as tf

from tensorflow import feature_column
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
tf.keras.backend.set_floatx('float64')

folder_path = '/Users/aklimasewski/Documents/model_results/base/crossedcols/'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
transform_method = 'Norm' #function or text
epochs =20
n = 13
train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = n)
train_data1,test_data1, feature_names = add_az(train_data1,test_data1, feature_names)

#load in location data for crossed features
train_data1_4, test_data1_4, train_targets1_4, test_targets1_4, feature_names_4 = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = 4)
# loctraindf = pd.DataFrame(data=train_data1_4, columns=feature_names_4)
# loctestdf = pd.DataFrame(data=test_data1_4, columns=feature_names_4)
# loctrain_ds = df_to_datasetnotargets(loctraindf)
# loctest_ds = df_to_datasetnotargets(loctestdf)
train_data1 = np.concatenate([train_data1,train_data1_4], axis = 1)
test_data1 = np.concatenate([test_data1,test_data1_4], axis = 1)


traindf = pd.DataFrame(data=train_data1,columns=feature_names)
testdf = pd.DataFrame(data=test_data1,columns=feature_names)

periodnames = ['T10.000S','T7.500S','T5.000S','T4.000S','T3.000S','T2.000S','T1.000S','T0.200S','T0.500S','T0.100S']
traintargetsdf = pd.DataFrame(data=train_targets1,columns=periodnames)
testtargetsdf = pd.DataFrame(data=test_targets1,columns=periodnames)

print(len(traindf), 'train examples')
print(len(testdf), 'test examples')


def df_to_dataset(traindf,traintargetsdf, shuffle=True, batch_size=256):
  dataframe = traindf.copy()
  labels = traintargetsdf
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

# tensorflow data structure
train_ds = df_to_dataset(traindf,traintargetsdf)
test_ds = df_to_dataset(testdf,testtargetsdf)


#input pipeline
for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  # print('A batch of ages:', feature_batch['Age'])
  print('A batch of targets:', label_batch)


def Norm(col):
    keep1=tf.reduce_max(col)
    keep2=tf.reduce_min(col)
    keep3=tf.reduce_mean(col)
    norm_data = 2.0/(keep1-keep2)*(col-keep3)
    return norm_data

feature_columns = []
# feature_layer_inputs = {}
# 
# numeric cols
# doesn't hold data
for header in feature_names:
  feature_columns.append(feature_column.numeric_column(header))#,normalizer_fn=Norm))
  
#crossed columns

# #load in location data for crossed features
# train_data1_4, test_data1_4, train_targets1_4, test_targets1_4, feature_names_4 = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = 4)
# loctraindf = pd.DataFrame(data=train_data1_4, columns=feature_names_4)
# loctestdf = pd.DataFrame(data=test_data1_4, columns=feature_names_4)
# loctrain_ds = df_to_datasetnotargets(loctraindf)
# loctest_ds = df_to_datasetnotargets(loctestdf)

def get_quantile_based_boundaries(feature_values, num_buckets):
  boundaries = np.arange(1.0, num_buckets) / num_buckets
  quantiles = feature_values.quantile(boundaries)
  return [quantiles[q] for q in quantiles.keys()]

stlon = tf.feature_column.numeric_column('stlon')
bucketized_longitude = tf.feature_column.bucketized_column(
    stlon, boundaries=get_quantile_based_boundaries(
    loctraindf['stlon'], 10))

stlat = tf.feature_column.numeric_column("stlat")
bucketized_latitude = tf.feature_column.bucketized_column(
    stlat, boundaries=get_quantile_based_boundaries(
      loctraindf["stlat"], 10))

long_x_lat = tf.feature_column.crossed_column(
  set([bucketized_longitude, bucketized_latitude]), hash_bucket_size=1000) 

#add crossed feature to columns
long_x_lat = feature_column.indicator_column(long_x_lat)
feature_columns.append(long_x_lat)










feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

batch_size = 256

def build_model():
    model = tf.keras.Sequential()
    model.add(feature_layer)
    model.add(layers.Dense(20,activation='sigmoid', input_shape=(14,)))
    model.add(layers.Dense(10)) #add sigmoid aciivation functio? (only alues betwen 0 and 1)    
    model.compile(optimizer=optimizers.Adam(lr=2e-3),loss='mse',metrics=['mae','mse']) 
    #model.compile(optimizer='adam',loss='mse',metrics=['mae']) 
    return model

model=build_model()

#fit the model
history=model.fit(train_ds, validation_data=test_ds, epochs=5,batch_size=batch_size,verbose=1)

    
    
    
    
    
    
    
    
    
    








model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=10)
  
  
  # feature_layer_inputs[header] = tf.keras.Input(shape=(1,), name=header)

# def df_to_datasetnotargets(dataframe):
#     dataframe = dataframe.copy()
#     # dataframe[periodnames] = targetsdf
#     # targetsdf['target'] = np.where(dataframe['AdoptionSpeed']==4, 0, 1)

#     ds = tf.data.Dataset.from_tensor_slices((dataframe.to_dict(orient='list')))
#     return ds

# #load in location data for crossed features
# train_data1_4, test_data1_4, train_targets1_4, test_targets1_4, feature_names_4 = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = 4)
# loctraindf = pd.DataFrame(data=train_data1_4, columns=feature_names_4)
# loctestdf = pd.DataFrame(data=test_data1_4, columns=feature_names_4)
# loctrain_ds = df_to_datasetnotargets(loctraindf)
# loctest_ds = df_to_datasetnotargets(loctestdf)

# def get_quantile_based_boundaries(feature_values, num_buckets):
#   boundaries = np.arange(1.0, num_buckets) / num_buckets
#   quantiles = feature_values.quantile(boundaries)
#   return [quantiles[q] for q in quantiles.keys()]

# stlon = tf.feature_column.numeric_column('stlon')
# bucketized_longitude = tf.feature_column.bucketized_column(
#    stlon, boundaries=get_quantile_based_boundaries(
#     loctraindf['stlon'], 10))

# stlat = tf.feature_column.numeric_column("stlat")
# bucketized_latitude = tf.feature_column.bucketized_column(
#     stlat, boundaries=get_quantile_based_boundaries(
#       loctraindf["stlat"], 10))

# long_x_lat = tf.feature_column.crossed_column(
#   set([bucketized_longitude, bucketized_latitude]), hash_bucket_size=1000) 

# #add crossed feature to columns
# long_x_lat = feature_column.indicator_column(long_x_lat)
# feature_columns.append(long_x_lat)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
  feature_layer,
  tf.keras.layers.Dense(20, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.sigmoid)
])

model.compile(optimizer='adam',metrics=['mse'])
model.fit(train_ds)





#######################



feature_layer_inputs['long_x_lat'] = tf.keras.Input(shape=(1,), name='long_x_lat')


feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
feature_layer_outputs = feature_layer(feature_layer_inputs)

x = layers.Dense(128, activation='relu')(feature_layer_outputs)
x = layers.Dense(64, activation='relu')(x)

baggage_pred = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=[v for v in feature_layer_inputs.values()], outputs=baggage_pred)

model.compile(optimizer='adam',
              loss='binary_crossentropy',iu
              metrics=['accuracy'])

model.fit(train_ds)








#%%%



import numpy as np
import pandas as pd

#!pip install tensorflow==2.0.0-alpha0
import tensorflow as tf

from tensorflow import feature_column
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL, nrows = 10000)
dataframe.head()

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

# Define method to create tf.data dataset from Pandas Dataframe
# This worked with tf 2.0 but does not work with tf 2.2
def df_to_dataset_tf_2_0(dataframe, label_column, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    #labels = dataframe.pop(label_column)
    labels = dataframe[label_column]

    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

def df_to_dataset(dataframe, label_column, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop(label_column)
    #labels = dataframe[label_column]

    ds = tf.data.Dataset.from_tensor_slices((dataframe.to_dict(orient='list'), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


batch_size = 5 # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, label_column = 'target', batch_size=batch_size)
val_ds = df_to_dataset(val,label_column = 'target',  shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, label_column = 'target', shuffle=False, batch_size=batch_size)

age = feature_column.numeric_column("age")

feature_columns = []
feature_layer_inputs = {}

# numeric cols
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
  feature_columns.append(feature_column.numeric_column(header))
  feature_layer_inputs[header] = tf.keras.Input(shape=(1,), name=header)

# bucketized cols
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35])
feature_columns.append(age_buckets)

# indicator cols
thal = feature_column.categorical_column_with_vocabulary_list(
      'thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)
feature_layer_inputs['thal'] = tf.keras.Input(shape=(1,), name='thal', dtype=tf.string)

# embedding cols
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# crossed cols
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)



feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
feature_layer_outputs = feature_layer(feature_layer_inputs)

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dropout(.1),
  layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=10)



