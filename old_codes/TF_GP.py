#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 15:45:54 2020

@author: aklimasewski

Reads in testing and training data files
formats ANN inputs
builds and compiles ANN with GP layer
plots some outputs
"""


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import PowerTransformer
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import power_transform
from sklearn.preprocessing import PowerTransformer
import keras
import tensorflow as tf
from keras import layers
from keras import optimizers
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
# import gc
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from keras.models import Sequential
import pickle


sns.reset_defaults()

sns.set_context(context='talk',font_scale=0.7)

tfd = tfp.distributions

model = 'model_16in_latlons'


data_dir = '/Users/aklimasewski/Documents/data/'
fig_dir = '/Users/aklimasewski/Documents/'+model+'/figs/'
# negloglik = lambda y, rv_y: -rv_y.log_prob(y)


###############
#recreated the demo with one input and one output only
#recreated demo with three inputs and one output
#recreated demo with all 12 inputs and one output


##
#to do 
#need to figure out how to adjust kernel to use event and site locations as the only input parameters
#need to output all 10 periods (issue with dimensions of the Variation GP layer)
#what to use as a noise parameter(s), can we optimize this param?
#from returned distribution of models how to find epistemic and aleatory uncertainty

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


def readindata(nametrain, nametest):
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
    # train_data1 = np.column_stack([Mwtrain,distrain,vs30train,z10train,z25train,raketrain,diptrain,hypodepthtrain, widthtrain,
    #                                 rjbtrain,rxtrain,startdepthtrain])
    # test_data1 = np.column_stack([Mwtest,distest,vs30test,z10test,z25test,raketest,diptest, hypodepthtest, widthtest,
    #                               rjbtest,rxtest,startdepthtest])
    
    train_data1 = np.column_stack([Mwtrain,distrain,vs30train,z10train,z25train,raketrain,diptrain,hypodepthtrain, widthtrain,
                                    rjbtrain,rxtrain,startdepthtrain, lattrain, longtrain, hypolattrain, hypolontrain])
    test_data1 = np.column_stack([Mwtest,distest,vs30test,z10test,z25test,raketest,diptest, hypodepthtest, widthtest,
                                  rjbtest,rxtest,startdepthtest, lattest, longtest, hypolattest, hypolontest])

    feature_names=['Mw','Rrup','Vs30', 'Z1.0', 'Z2.5', 'Rake','Dip','Hypo_depth', 'Width',
                   'Rjb','Rx','Ztor',]
    return train_data1, test_data1, train_targets1, test_targets1, feature_names



train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain=data_dir + 'cybertrainyeti10_residfeb.csv', nametest=data_dir + 'cybertestyeti10_residfeb.csv')

# ind_pos = np.random.randint(0,len(train_data1),10000)
# ind_postest = np.random.randint(0,len(test_data1),2000)

# train_data1 = np.asarray([list(train_data1[i]) for i in ind_pos])
# test_data1= np.asarray([list(test_data1[i]) for i in ind_postest])
# train_targets1= np.asarray([list(train_targets1[i]) for i in ind_pos])
# test_targets1= np.asarray([list(test_targets1[i]) for i in ind_postest])

#Read in datasets
# nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv'
# nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv'
#%%
#########################

#preprocessing transform inputs data to be guassian shaped
pt = PowerTransformer()
aa=pt.fit(train_data1[:,:])
train_data=aa.transform(train_data1)
test_data=aa.transform(test_data1)

train_targets = train_targets1
test_targets = test_targets1

#start with first period only
y_test = test_targets1.T[0]
y = train_targets1.T[0]

x_range = [[min(train_data.T[i]) for i in range(len(train_data[0]))],[max(train_data.T[i]) for i in range(len(train_data[0]))]]


class RBFKernelFn(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(RBFKernelFn, self).__init__(**kwargs)
    dtype = kwargs.get('dtype', None)

    self._amplitude = self.add_variable(
            initializer=tf.constant_initializer(0),
            dtype=dtype,
            name='amplitude')
    
    self._length_scale = self.add_variable(
            initializer=tf.constant_initializer(0),
            dtype=dtype,
            name='length_scale')

  def call(self, x):
    # Never called -- this is just a layer so it can hold variables
    # in a way Keras understands.
    return x

  @property
  def kernel(self):
      #can this be modified to use only lat, lon?
     return tfp.math.psd_kernels.MaternOneHalf(
       amplitude=tf.nn.softplus(0.5 * self._amplitude),
       length_scale=tf.nn.softplus(1. * self._length_scale)
     )



# For numeric stability, set the default floating-point dtype to float64
# tf.keras.backend.set_floatx('float32')

batch_size = 264
num_inducing_points = 40
loss = lambda y, rv_y: rv_y.variational_loss(
    y, kl_weight=np.array(batch_size) / train_data.shape[0])


def build_model():
    #model=models.Sequential()
    model = Sequential()
    model.add(layers.Dense(16,activation='sigmoid', input_shape=(train_data.shape[1],)))

    # model.add(Dropout(rate=0.5, trainable =True))#trainint=true v mismatch
    
    model.add(tfp.layers.VariationalGaussianProcess(
        num_inducing_points=num_inducing_points,
        kernel_provider=RBFKernelFn(),
        event_shape=[1],#outputshape
        inducing_index_points_initializer=tf.constant_initializer(np.linspace(*x_range,num_inducing_points,dtype='float32')),
        unconstrained_observation_noise_variance_initializer=(
            tf.constant_initializer(0.1))))

    model.compile(optimizer=optimizers.Adam(lr=2e-3),loss='mse',metrics=['mae','mse']) 
    #model.compile(optimizer='adam',loss='mse',metrics=['mae']) 
    return model


model=build_model()


#fit the model
history=model.fit(train_data,y,validation_data=(test_data,y_test),epochs=15,batch_size=264,verbose=1)
mae_history=history.history['val_mae']
mae_history_train=history.history['mae']
test_mse_score,test_mae_score,tempp=model.evaluate(test_data,y_test)

#write file  with history
with open(fig_dir + 'history', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# history = pickle.load(open(fig_dir + 'history', "rb"))

#returns
yhat = model.predict(test_data)

yhat = np.ndarray.flatten(yhat)




#%%

testi = 100
num_dist = 100
# yhats = [model.predict(test_data) for _ in range(50)]
#######
#grab the first obs
yhat0 = [model.predict(test_data[testi:testi+1]) for i in range(num_dist)]
yhat0 = np.ndarray.flatten(np.asarray(yhat0))
#print simple test
print('mean: ',np.mean(yhat0), 'stdev: ',np.std(yhat0))
print('actual value: ', y_test[testi])
print('prediction samples: ', yhat0)
label = 'mean: ' + str(round(np.mean(yhat0),4)) +'\n' + ' stdev: ' + str(round(np.std(yhat0),4))

fig= plt.figure()
ax = fig.add_subplot(1, 1, 1)
# We can set the number of bins with the `bins` kwarg
ax.hist(yhat0)
ax.axvline(x =y_test[testi], linewidth=4, color='r')
plt.title(str(num_dist) +' samples of predictions index: '+ str(testi))
# plt.text(0.75, 0.75, 'matplotlib', horizontalalignment='center',verticalalignment='center',)
ax.annotate(label,xycoords = 'axes fraction',xy=(0.8,0.8), ha='center', va='bottom')
plt.savefig(fig_dir +'prediction_hist_i_' + str(testi) + '_n_' + str(num_dist))
plt.show()
#%%


####
plt.figure()
plt.scatter(test_data1.T[1],y_test,label='Testing Data', s = 1)
plt.scatter(test_data1.T[1],yhat,label='Model predictions',  s = 1)
plt.xlabel('Distance (km)')
plt.ylabel('Prediction')
plt.title('predictions vs distance (km)')
plt.grid()
plt.legend()
plt.savefig(fig_dir + 'predictionsvsdistance.png')
plt.show()



f10=plt.figure('Overfitting Test')
plt.plot(mae_history,label='Testing Data')
plt.plot(mae_history_train,label='Training Data')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('Overfitting Test')
plt.legend()
print(test_mae_score)
plt.grid()
plt.savefig(fig_dir + 'OverfittingTest.png')
plt.show()



a=model.predict(test_data)
b=model.predict(train_data)
f1=plt.figure('Earthquake Magnitude normalized Prediction')
plt.plot(train_data[:,0],b,'.r', label='train data')
plt.plot(test_data[:,0],a,'.b',label='test data')
plt.xlabel('input')
plt.ylabel('Prediction')
plt.title('Earthquake Magnitude normalized Prediction')
plt.legend()
plt.savefig(fig_dir + 'EarthquakeMagnitudenormalizedPrediction.png')
plt.show()


f11=plt.figure('Joyner-Boore Dist. (km) normalized Prediction')
plt.plot(train_data[:,1],b,'.r',label='train data')
plt.plot(test_data[:,1],a,'.b',label='test data')
plt.xlabel('input')
plt.ylabel('Prediction')
plt.ylabel('Prediction')
plt.title('Joyner-Boore Dist. (km) normalized Prediction')
plt.legend()
plt.savefig(fig_dir + 'Joyner-BooreDistnormalizedPrediction.png')
plt.show()





