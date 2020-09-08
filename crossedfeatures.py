#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:55:14 2020

@author: aklimasewski
"""



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
import matplotlib as mpl
import matplotlib.pyplot as plt

folder_path = '/Users/aklimasewski/Documents/model_results/crossedcols/'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
folder_pathmod = folder_path + 'crossedlatlon_linear_bucket1000_10_30ep/'

if not os.path.exists(folder_pathmod):
    os.makedirs(folder_pathmod)

epochs = 30
hash_bucket_size = 1000
bucketsize = 10

n = 13
train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = n)
train_data1,test_data1, feature_names = add_az(train_data1,test_data1, feature_names)

#load in location data for crossed features
train_data1_4, test_data1_4, train_targets1_4, test_targets1_4, feature_names_4 = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = 4)

train_data1 = np.concatenate([train_data1,train_data1_4], axis = 1)
test_data1 = np.concatenate([test_data1,test_data1_4], axis = 1)
feature_names = np.concatenate([feature_names,feature_names_4])

#%%


# #separate latitude and longiutde into length 2 vectors for input
# stloc = [(train_data1_4[i,0],train_data1_4[i,1]) for i in range(train_data1_4.shape[0])]
# evloc = [(train_data1_4[i,2],train_data1_4[i,3]) for i in range(train_data1_4.shape[0])]
# # train_data1 = np.concatenate([train_data1,stloc,[evloc], axis = 1)
# # test_data1 = np.concatenate([test_data1,test_data1_4], axis = 1)
# feature_names = np.concatenate([feature_names,['stloc','evloc']])

# locdf = pd.DataFrame({'stloc':stloc, 'evloc':evloc})    

#%%

traindf = pd.DataFrame(data=train_data1,columns=feature_names)
testdf = pd.DataFrame(data=test_data1,columns=feature_names)

periodnames = ['T10.000S','T7.500S','T5.000S','T4.000S','T3.000S','T2.000S','T1.000S','T0.200S','T0.500S','T0.100S']
traintargetsdf = pd.DataFrame(data=train_targets1,columns=periodnames)
testtargetsdf = pd.DataFrame(data=test_targets1,columns=periodnames)

print(len(traindf), 'train examples')
print(len(testdf), 'test examples')

def df_to_dataset(traindf,traintargetsdf, shuffle=True, batch_size=256):
    # batch_size = traindf.shape[0]
    dataframe = traindf.copy()
    labels = traintargetsdf
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

# tensorflow data structure
train_ds = df_to_dataset(traindf,traintargetsdf, shuffle=False)
test_ds = df_to_dataset(testdf,testtargetsdf, shuffle=False)


#input pipeline
for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  # print('A batch of ages:', feature_batch['Age'])
  print('A batch of targets:', label_batch)

#%%
def get_normalization_parameters(traindf, features):
    """Get the normalization parameters (E.g., mean, std) for traindf for 
    features. We will use these parameters for training, eval, and serving."""

    def score_params(column):
        trainmean = traindf[column].mean()
        trainmax = traindf[column].max()
        trainmin = traindf[column].min()

        return {'mean': trainmean, 'max': trainmax, 'min': trainmin}

    normalization_parameters = {}
    for column in features:
        normalization_parameters[column] = score_params(column)
    return normalization_parameters

def make_norm(mean, maximum, minimum):
    def normcol(col):
        norm_func = 2.0/(maximum-minimum)*(col-mean)
        return norm_func
    return normcol

column_params = get_normalization_parameters(traindf,feature_names)

feature_columns = []

# numeric cols
# doesn't hold data
for header in feature_names:#[0:14]:
    normparams = column_params[header]
    mean = normparams['mean']
    maximum = normparams['max']
    minimum = normparams['min']
    normalizer_fn = make_norm(mean, maximum, minimum)
    feature_columns.append(feature_column.numeric_column(header,normalizer_fn=normalizer_fn))
    
# #length 2 of locations
# for header in feature_names[14:18]:
#     feature_columns.append(feature_column.numeric_column(header,normalizer_fn=PowerTransformer()))

  
#%%
#crossed columns

def get_quantile_based_boundaries(feature_values, num_buckets):
  boundaries = np.arange(1.0, num_buckets) / num_buckets
  quantiles = feature_values.quantile(boundaries)
  return [quantiles[q] for q in quantiles.keys()]

stlon = tf.feature_column.numeric_column('stlon')
bucketized_stlongitude = tf.feature_column.bucketized_column(
    stlon, boundaries=get_quantile_based_boundaries(
    traindf['stlon'], bucketsize))

stlat = tf.feature_column.numeric_column("stlat")
bucketized_stlatitude = tf.feature_column.bucketized_column(
    stlat, boundaries=get_quantile_based_boundaries(
      traindf["stlat"], bucketsize))

stlong_x_lat = tf.feature_column.crossed_column(set([bucketized_stlongitude, bucketized_stlatitude]), hash_bucket_size=hash_bucket_size) 

#add crossed feature to columns
stlong_x_lat = feature_column.indicator_column(stlong_x_lat)
feature_columns.append(stlong_x_lat)

evlon = tf.feature_column.numeric_column('hypolon')
bucketized_evlongitude = tf.feature_column.bucketized_column(
    evlon, boundaries=get_quantile_based_boundaries(
    traindf['hypolon'], bucketsize))

evlat = tf.feature_column.numeric_column("hypolat")
bucketized_evlatitude = tf.feature_column.bucketized_column(
    evlat, boundaries=get_quantile_based_boundaries(
      traindf["hypolat"], bucketsize))

evlong_x_lat = tf.feature_column.crossed_column(set([bucketized_evlongitude, bucketized_evlatitude]), hash_bucket_size=hash_bucket_size) 

#add crossed feature to columns
evlong_x_lat = feature_column.indicator_column(evlong_x_lat)
feature_columns.append(evlong_x_lat)
#%%

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

batch_size = 256
# input_shape = train_ds.shape

def build_model():
    model = tf.keras.Sequential()
    model.add(feature_layer)
    model.add(layers.Dense(50,activation='sigmoid'))#, input_shape=(18,)))
    model.add(layers.Dense(10)) #add sigmoid aciivation functio? (only alues betwen 0 and 1)    
    model.compile(optimizer=optimizers.Adam(lr=2e-3),loss='mse',metrics=['mae','mse']) 
    #model.compile(optimizer='adam',loss='mse',metrics=['mae']) 
    return model

model=build_model()

#fit the model
history=model.fit(train_ds, validation_data=test_ds, epochs=epochs,verbose=1)#, batch_size=batch_size)

mae_history=history.history['val_mae']
mae_history_train=history.history['mae']
test_mse_score,test_mae_score,tempp=model.evaluate(test_ds)
#dataframe for saving purposes
hist_df = pd.DataFrame(history.history)

f10=plt.figure('Overfitting Test')
plt.plot(mae_history_train,label='Training Data')
plt.plot(mae_history,label='Testing Data')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('Overfitting Test')
plt.legend()
plt.grid()
plt.savefig(folder_pathmod + 'error.png')
plt.show()

# pr1edict_mean = []
pre_test = np.array(model.predict(test_ds))
pre_train = np.array(model.predict(train_ds))

#test data
mean_x_test_allT = pre_test
# predict_epistemic_allT = np.zeros(pre_test.shape)

#training data
mean_x_train_allT = pre_train
# predict_epistemic_train_allT = np.zeros(pre.shape)

resid_train =train_targets1-mean_x_train_allT
resid_test = test_targets1-mean_x_test_allT

diff=np.std(resid_train,axis=0)
difftest=np.std(resid_test,axis=0)
#write model details to a file
file = open(folder_pathmod + 'model_details.txt',"w+")
# file.write('number training samples ' + str(len(train_ds)) + '\n')
# file.write('number testing samples ' + str(len(x_test)) + '\n')
# file.write('data transformation method ' + str(transform_method) + '\n')
file.write('input feature names ' +  str(feature_names)+ '\n')
file.write('number of epochs ' +  str(epochs)+ '\n')
model.summary(print_fn=lambda x: file.write(x + '\n'))
file.write('model fit history' + str(hist_df.to_string) + '\n')
file.write('stddev train' + str(diff) + '\n')
file.write('stddev test' + str(difftest) + '\n')
file.close()


period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]
plot_resid(resid_train, resid_test, folder_pathmod)

# obs_pre(train_targets1, test_targets1, pre_train, pre_test, period, folder_pathmod)

    
    
    
    
    
#%%%


#%%%



# import numpy as np
# import pandas as pd

# #!pip install tensorflow==2.0.0-alpha0
# import tensorflow as tf

# from tensorflow import feature_column
# from tensorflow import keras
# from tensorflow.keras import layers
# from sklearn.model_selection import train_test_split

# URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
# dataframe = pd.read_csv(URL, nrows = 10000)
# dataframe.head()

# train, test = train_test_split(dataframe, test_size=0.2)
# train, val = train_test_split(train, test_size=0.2)
# print(len(train), 'train examples')
# print(len(val), 'validation examples')
# print(len(test), 'test examples')

# # Define method to create tf.data dataset from Pandas Dataframe
# # This worked with tf 2.0 but does not work with tf 2.2
# def df_to_dataset_tf_2_0(dataframe, label_column, shuffle=True, batch_size=32):
#     dataframe = dataframe.copy()
#     #labels = dataframe.pop(label_column)
#     labels = dataframe[label_column]

#     ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
#     if shuffle:
#         ds = ds.shuffle(buffer_size=len(dataframe))
#     ds = ds.batch(batch_size)
#     return ds

# def df_to_dataset(dataframe, label_column, shuffle=True, batch_size=32):
#     dataframe = dataframe.copy()
#     labels = dataframe.pop(label_column)
#     #labels = dataframe[label_column]

#     ds = tf.data.Dataset.from_tensor_slices((dataframe.to_dict(orient='list'), labels))
#     if shuffle:
#         ds = ds.shuffle(buffer_size=len(dataframe))
#     ds = ds.batch(batch_size)
#     return ds


# batch_size = 5 # A small batch sized is used for demonstration purposes
# train_ds = df_to_dataset(train, label_column = 'target', batch_size=batch_size)
# val_ds = df_to_dataset(val,label_column = 'target',  shuffle=False, batch_size=batch_size)
# test_ds = df_to_dataset(test, label_column = 'target', shuffle=False, batch_size=batch_size)

# age = feature_column.numeric_column("age")

# feature_columns = []
# feature_layer_inputs = {}

# # numeric cols
# for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
#   feature_columns.append(feature_column.numeric_column(header))
#   feature_layer_inputs[header] = tf.keras.Input(shape=(1,), name=header)

# # bucketized cols
# age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35])
# feature_columns.append(age_buckets)

# # indicator cols
# thal = feature_column.categorical_column_with_vocabulary_list(
#       'thal', ['fixed', 'normal', 'reversible'])
# thal_one_hot = feature_column.indicator_column(thal)
# feature_columns.append(thal_one_hot)
# feature_layer_inputs['thal'] = tf.keras.Input(shape=(1,), name='thal', dtype=tf.string)

# # embedding cols
# thal_embedding = feature_column.embedding_column(thal, dimension=8)
# feature_columns.append(thal_embedding)

# # crossed cols
# crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
# crossed_feature = feature_column.indicator_column(crossed_feature)
# feature_columns.append(crossed_feature)



# feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
# feature_layer_outputs = feature_layer(feature_layer_inputs)

# model = tf.keras.Sequential([
#   feature_layer,
#   layers.Dense(128, activation='relu'),
#   layers.Dense(128, activation='relu'),
#   layers.Dropout(.1),
#   layers.Dense(1)
# ])

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# model.fit(train_ds,
#           validation_data=val_ds,
#           epochs=10)



