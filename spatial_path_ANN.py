#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 18:03:06 2020

@author: aklimasewski

ANN with only lat,lon of station and event trained on residuals
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


###############

#%%

folder_path = '/Users/aklimasewski/Documents/12featuremodel/ANN_12test/'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
transform_method = PowerTransformer()
epochs = 15
n = 12
#or n = 6, 4

# %%
def transform_dip(diptrain,diptest,rxtrain,rxtest):
    for i in range(len(diptrain)):
        if diptrain[i]>30:
            rxtrain[i]=rxtrain[i]*(90-diptrain[i])/45
        else:
            rxtrain[i]=rxtrain[i]*60/45
            
    for i in range(len(diptest)): 
        if diptest[i]>30:
            rxtest[i]=rxtest[i]*(90-diptest[i])/45
        else:
            rxtest[i]=rxtest[i]*60/45    
    #return the transformed arrays
    return diptrain, diptest, rxtrain, rxtest


def readindata(nametrain, nametest, n):
#Read in datasets
# nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv'
# nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv'
    
    dftrain = pd.read_pickle(nametrain) 
    dftest = pd.read_pickle(nametest)
    print(dftrain.shape)
    print(dftest.shape)
    
    
    #separate dataframe to pandas series (1 col of df)
    Mwtrain= dftrain["Mag"]
    distrain=dftrain["Site_Rupture_Dist"]
    vs30train=np.array(dftrain["vs30"])
    z10train=dftrain["z10"]
    z25train=dftrain["z25"]
    lattrain=dftrain["CS_Site_Lat"]
    longtrain=dftrain["CS_Site_Lon"]
    periodtrain=dftrain["siteperiod"]
    hypolattrain=dftrain["Hypocenter_Lat"]
    hypolontrain=dftrain["Hypocenter_Lon"]
    hypodepthtrain=dftrain["Hypocenter_Depth"]
    raketrain=dftrain["Rake_y"]
    diptrain=dftrain["Dip_y"]
    striketrain=dftrain["Strike_y"]+180
    widthtrain=dftrain["Width"]
    #Outputs (col per period)
    residtesttemp=dftrain.loc[:, 'IM_Value':'IM175']
    train_targets1=residtesttemp.values
    
    #histogram of output (1 period of IM175)
    plt.hist(train_targets1[:,3],100)
    
    lengthtrain=dftrain["Length"]
    rjbtrain=dftrain["rjb"]
    rxtrain=dftrain["rx"]
    rytrain=dftrain["ry"]
    hypodistrain=dftrain["hypodistance"]
    Utrain=dftrain["U"]
    Ttrain=dftrain["T"]
    xitrain=dftrain["xi"]
    startdepthtrain=dftrain["Start_Depth"]
    
    
    #### same with testdata
    Mwtest= dftest["Mag"]
    distest=dftest["Site_Rupture_Dist"]
    vs30test=np.array(dftest["vs30"])
    z10test=dftest["z10"]
    z25test=dftest["z25"]
    lattest=dftest["CS_Site_Lat"]
    longtest=dftest["CS_Site_Lon"]
    periodtest=dftest["siteperiod"]
    hypolattest=dftest["Hypocenter_Lat"]
    hypolontest=dftest["Hypocenter_Lon"]
    hypodepthtest=dftest["Hypocenter_Depth"]
    raketest=dftest["Rake_y"]
    diptest=dftest["Dip_y"]
    striketest=dftest["Strike_y"]+180
    widthtest=dftest["Width"]
    residtesttemp1=dftest.loc[:, 'IM_Value':'IM175']
    test_targets1=residtesttemp1.values
    lengthtest=dftest["Length"]
    rjbtest=dftest["rjb"]
    rxtest=dftest["rx"]
    rytest=dftest["ry"]
    hypodistest=dftest["hypodistance"]
    Utest=dftest["U"]
    Ttest=dftest["T"]
    xitest=dftest["xi"]
    startdepthtest=dftest["Start_Depth"]
    
    diptrain, diptest, rxtrain, rxtest = transform_dip(diptrain=np.array(diptrain),diptest=np.array(diptest),rxtrain=np.array(rxtrain),rxtest=np.array(rxtest))
    
    #put together arrays of features for ANN
    #starting with just lat,lon of station and event    
    
    if n == 6:
        train_data1 = np.column_stack([hypodistrain, lattrain, longtrain, hypolattrain, hypolontrain, hypodepthtrain])
        test_data1 = np.column_stack([hypodistest, lattest, longtest, hypolattest, hypolontest, hypodepthtest])
        feature_names=np.asarray(['hypoR','stlat', 'stlon', 'hypolat','hypolon', 'hypodepth'])
    
    
    else: #12
        train_data1 = np.column_stack([Mwtrain,distrain,vs30train,z10train,z25train,raketrain,diptrain,hypodepthtrain, widthtrain,
                                rjbtrain,rxtrain,startdepthtrain])
        test_data1 = np.column_stack([Mwtest,distest,vs30test,z10test,z25test,raketest,diptest, hypodepthtest, widthtest,
                              rjbtest,rxtest,startdepthtest])

        feature_names=np.asarray(['Mw','Rrup','Vs30', 'Z1.0', 'Z2.5', 'Rake','Dip','Hypo_depth', 'Width',
                'Rjb','Rx','Ztor',])


    return train_data1, test_data1, train_targets1, test_targets1, feature_names



train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = n)

#%%

#normalize data
#preprocessing transform inputs data to be guassian shaped
# pt = PowerTransformer()
# aa=pt.fit(train_data1[:,:])
# train_data=aa.transform(train_data1)
# test_data=aa.transform(test_data1)

def transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_path):

    transform = transform_method
    aa=transform.fit(train_data1[:,:])
    train_data=aa.transform(train_data1)
    test_data=aa.transform(test_data1)
    
    #plot transformed features
    for i in range(len(train_data[0])):
        plt.figure(figsize =(8,8))
        plt.title('transformed feature: ' + str(feature_names[i]))
        plt.hist(train_data[:,i])
        plt.savefig(folder_path + 'histo_transformedfeature_' + str(feature_names[i]) + '.png')
        plt.show()
    
    train_targets = train_targets1
    test_targets = test_targets1
    
    y_test = test_targets
    y_train = train_targets
    
    x_train = train_data
    x_test = test_data
    
    x_train_raw = train_data1
    x_test_raw = test_data1
    
    x_range = [[min(train_data.T[i]) for i in range(len(train_data[0]))],[max(train_data.T[i]) for i in range(len(train_data[0]))]]
    
    # Rindex = np.where(feature_names == 'hypoR')[0][0]

    return(x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw)




x_train, y_train, x_test, y_test, x_range, x_train_raw,  x_test_raw = transform_data(transform_method, train_data1, test_data1, train_targets1, test_targets1, feature_names, folder_path)
# x_train = train_data[rand_ind_train]


#%%


# create sequential model with positional inputs and predict 

#try sequential with VGP layer

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


#plotting

# folder_path = '/Users/aklimasewski/Documents/pathresid_ANN_GPlayer_opt_kernel_fulldata/'
# folder_path = '/Users/aklimasewski/Documents/pathresid_ANN_noGPlayer_10000_randsamples/'
# folder_path = '/Users/aklimasewski/Documents/pathresid_ANN_noGPlayer_norm/'
# 
f10=plt.figure('Overfitting Test')
plt.plot(mae_history,label='Testing Data')
plt.plot(mae_history_train,label='Training Data')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('Overfitting Test')
plt.legend()
print(test_mae_score)
plt.grid()
plt.savefig(folder_path + 'error.png')
plt.show()



predict_mean = []
predict_epistemic = []

predict_mean_train = []
predict_epistemic_train = []
for i in range(2):
    print(i)
    p = np.array(model.predict(x_test))
    mean = p
    predict_mean.append(mean)
    
    p = np.array(model.predict(x_train))
    mean = p
    predict_mean_train.append(mean)

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


# diff=np.std(y_train-mean_x_train_allT,axis=0)
# difftest=np.std(y_test-mean_x_test_allT,axis=0)
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

if n == 12:
    Rindex = np.where(feature_names == 'Rrup')[0][0]
    Rtrain = x_train_raw[:,Rindex:Rindex+1]
    Rtest = x_test_raw[:,Rindex:Rindex+1]
elif n== 6:
    Rindex = np.where(feature_names == 'hypoR')[0][0]
    Rtrain = x_train_raw[:,Rindex:Rindex+1]
    Rtest = x_test_raw[:,Rindex:Rindex+1]
#loop per period
#for ...
folderlist = ['T10s','T7_5s','T5s','T4s','T3s','T2s','T1s','T_5s','T_2s','T_1s']


for j in range(0,3):#len(period)):

    y_ind = j
    
    if not os.path.exists(folder_path + folderlist[j]):
        os.makedirs(folder_path + folderlist[j])
    
    
    #each column is prediction for a period
     
    mean_x_test = mean_x_test_allT[:,y_ind:y_ind+1].flatten()
    predict_epistemic= predict_epistemic_allT[:,y_ind:y_ind+1].flatten()
        
    mean_x_train = mean_x_train_allT[:,y_ind:y_ind+1].flatten()
    predict_epistemic_train = predict_epistemic_train_allT[:,y_ind:y_ind+1].flatten()
    

    # plot epistemic uncertainty
    fig, axes = plt.subplots(2,1,figsize=(10,8))
    # ax.scatter(Rtrain, y_train, s=1, label='train data')
    plt.title('T = ' + str(period[y_ind]) + ' s')
    ylim = max(np.abs(y_test[:,y_ind]))
    axes[0].set_ylim(-1*ylim,ylim)
    axes[1].set_ylim(-1*ylim,ylim)
    axes[0].errorbar(Rtest, mean_x_test, yerr=predict_epistemic, fmt='.', label='predictions epistemic uncertainty', color='pink', alpha = 0.5, markeredgecolor='red')
    axes[1].scatter(Rtest, y_test[:,y_ind], s=1, label='test targets', color='green')
    axes[1].set_xlabel('R (km)')
    axes[0].set_ylabel('prediction')
    axes[1].set_ylabel('target')
    axes[0].legend(loc = 'upper left')
    axes[1].legend(loc = 'upper left')
    plt.savefig(folder_path + folderlist[j] + '/Epistemic_vs_R.png')
    plt.show()
    

    f1=plt.figure('Hypo dist normalized Prediction')
    plt.errorbar(x_test[:,Rindex], mean_x_test, yerr=predict_epistemic, fmt='.', label='test with epistemic uncertainty', color='pink', alpha = 0.5)
    plt.plot(x_train[:,Rindex],mean_x_train,'.r', label='train data')
    # plt.errorbar(x_test[:,1], mean_x_test, yerr=predict_epistemic, fmt='.', label='test with epistemic uncertainty', color='pink', alpha = 0.5)
    # plt.plot(x_train[:,1],mean_x_train,'.r', label='train data')
    plt.xlabel('dist input (normalized)')
    plt.ylabel('Prediction')
    plt.title('Hypo dist normalized Prediction T = ' + str(period[y_ind]) + ' s')
    plt.legend(loc = 'upper left')
    plt.savefig(folder_path + folderlist[j] + '/norm_dist_vs_pre.png')
    plt.show()
    
    
    f1=plt.figure('Hypo dist normalized Actual',figsize=(8,8))
    plt.plot(x_train[:,Rindex],y_train[:,y_ind],'.r', label='train data')
    plt.plot(x_test[:,Rindex],y_test[:,y_ind],'.b', label='test data')
    # plt.plot(x_train[:,1],y_train[:,y_ind],'.r', label='train data')
    # plt.plot(x_test[:,1],y_test[:,y_ind],'.b', label='test data')
    # plt.xlabel('input')
    plt.ylabel('target')
    plt.xlabel('input normalized distance')
    plt.title('hypo dist normalized Actual')
    plt.legend(loc = 'upper left')
    plt.savefig(folder_path + folderlist[j] + '/norm_dist_vs_actual.png')
    plt.show()
    
    # f1=plt.figure('Mag normalized Actual')
    # plt.plot(x_train[:,Mindex],y_train[:,y_ind],'.r')
    # plt.plot(x_test[:,Mindex],y_test[:,y_ind],'.b')
    # # plt.plot(x_train[:,1],y_train[:,y_ind],'.r')
    # # plt.plot(x_test[:,1],y_test[:,y_ind],'.b')
    # plt.xlabel('input normalized mag')
    # plt.ylabel('target')
    # plt.title('Earthquake Magnitude normalized Actual')
    # plt.savefig(folder_path + folderlist[j] + '/norm_mag_vs_actual.png')
    # plt.show()
    
    
    
    #title = 'T = 5.0 s'
    # f212, ax=plt.figure()
    f212, ax = plt.subplots(1,1,figsize=(8,8))
    ax.hist(y_train[:,y_ind]-mean_x_train,100,label='Training')
    ax.hist(y_test[:,y_ind]-mean_x_test,100,label='Testing')
    plt.xlabel('Residual ln(Target/Predicted)')
    plt.ylabel('Count')
    temp1=str(np.std(y_train[:,y_ind]-mean_x_train))
    temp2=str(np.std(y_test[:,y_ind]-mean_x_test))
    temp11=str(np.mean(y_train[:,y_ind]-mean_x_train))
    temp22=str(np.mean(y_test[:,y_ind]-mean_x_test))
    ax.text(0.7,0.60,'sigma_train = '+ temp1[0:4],transform=ax.transAxes)
    ax.text(0.7,0.65,'sigma_test =' + temp2[0:4], transform=ax.transAxes)
    ax.text(0.7,0.70,'mean_train =  '+ temp11[0:4],transform=ax.transAxes)
    ax.text(0.7,0.75,'mean_test = '+ temp22[0:4],transform=ax.transAxes)
    plt.title('Residual ln(Target/Predicted)): T = ' + str(period[y_ind]) + ' s')
    plt.legend()
    plt.savefig(folder_path + folderlist[j] + '/Histo.png')
    plt.show()
    

    




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


#%%o




# tf.keras.backend.set_floatx('float64')
# batch_size = 10
# num_inducing_points = 10
# loss = lambda y, rv_y: rv_y.variational_loss(
#     y, kl_weight=np.array(batch_size) / train_data.shape[0])


# def build_model():
#     #model=models.Sequential()
#     model = Sequential()
#     model.add(layers.Dense(x_train.shape[1],activation='sigmoid', input_shape=(x_train.shape[1],)))

#     #no gP layer
#     # model.add(layers.Dense(train_targets.shape[1]))
#     model.add(tfp.layers.VariationalGaussianProcess(
#         num_inducing_points=num_inducing_points,
#         kernel_provider=RBFKernelFn_opt(),
#         # kernel_provider=TestKernelFn(),
#         # kernel=optimized_kernel,
#         event_shape=[y_train.shape[1]],#outputshape
#         inducing_index_points_initializer=tf.constant_initializer(y_train.shape[1]*[np.linspace(*x_range,num_inducing_points,dtype='float64')]),
#         unconstrained_observation_noise_variance_initializer=(
#             tf.constant_initializer(0.1))))

#     model.compile(optimizer=optimizers.Adam(lr=0.01),loss='mse',metrics=['mae','mse']) 
#     #model.compile(optimizer='adam',loss='mse',metrics=['mae']) 
#     return model


# model=build_model()


# #fit the model
# history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=15,batch_size=batch_size,verbose=1)


# # # Create the trainable variables in our model. Constrain to be strictly
# # # positive by wrapping in softplus and adding a little nudge away from zero.
# # amplitude = (np.finfo(np.float64).tiny +
# #              tf.nn.softplus(
# #                  tf.Variable(10. * tf.ones(2, dtype=tf.float64)),
# #                  name='amplitude'))
# # length_scale = (np.finfo(np.float64).tiny +
# #                 tf.nn.softplus(
# #                     tf.Variable([300., 30.], dtype=np.float64),
# #                     name='length_scale'))

# #%%

# #test kernel class
# class TestKernelFn(tf.keras.layers.Layer):
#   def __init__(self, **kwargs):
#     super(TestKernelFn, self).__init__(**kwargs)
#     dtype = kwargs.get('dtype', None)

#     self._amplitude = self.add_variable(
#             initializer=tf.constant_initializer(0),
#             dtype=dtype,
#             name='amplitude')
    
#     self._length_scale = self.add_variable(
#             initializer=tf.constant_initializer(0),
#             dtype=dtype,
#             name='length_scale')

#   def call(self, x):
#     # Never called -- this is just a layer so it can hold variables
#     # in a way Keras understands.
#     return x

#   @property
#   def kernel(self):
          
#     eq_indices = [0, 2, 4]
#     periodic_indices = [1, 3,5]
#     eq_kernel = tfp.math.psd_kernels.FeatureTransformed(
#         # tfp.math.psd_kernels.MaternOneHalf(amplitude=tf.constant([0.1 * self._amplitude, 0.1 * self._amplitude, 0.1 * self._amplitude]),
#         tfp.math.psd_kernels.MaternOneHalf(amplitude=tf.constant([2.0, 3.0, 4.0]),
#         length_scale=tf.constant([1.,1.,1.])),

#         # length_scale=tf.constant([5. * self._length_scale, 5. * self._length_scale, 5. * self._length_scale])),
#         # transformation_fn=lambda x: tf.gather(x, eq_indices, axis=-1))
#         transformation_fn=lambda x, _: tf.gather(x, eq_indices, axis=-1))
#     periodic_kernel = tfp.math.psd_kernels.FeatureTransformed(
#         tfp.math.psd_kernels.Linear(),
#         # transformation_fn=lambda x: tf.gather(x, periodic_indices, axis=-1))
#         transformation_fn=lambda x, _: tf.gather(x, periodic_indices, axis=-1))
#     kernel = eq_kernel + periodic_kernel


#     return kernel



# eq_indices = [0, 2, 4]
# periodic_indices = [1, 3]
# eq_kernel = tfp.math.psd_kernels.FeatureTransformed(
#     tfp.math.psd_kernels.MaternOneHalf(amplitude=tf.constant([2.0, 3.0, 4.0]),
#       length_scale=tf.constant([2.0, 3.0, 4.0])),
#     # transformation_fn=lambda x: tf.gather(x, eq_indices, axis=-1))
#     transformation_fn=lambda x, _: tf.gather(x, eq_indices, axis=-1))
# periodic_kernel = tfp.math.psd_kernels.FeatureTransformed(
#     tfp.math.psd_kernels.Linear(),
#     # transformation_fn=lambda x: tf.gather(x, periodic_indices, axis=-1))
#     transformation_fn=lambda x, _: tf.gather(x, periodic_indices, axis=-1))
# kernel = eq_kernel + periodic_kernel

















