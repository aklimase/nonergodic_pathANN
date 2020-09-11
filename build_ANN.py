#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 13:24:48 2020

@author: aklimasewski

building ANN
"""

def create_ANN(x_train, y_train, x_test, y_test, feature_names, numlayers, units, epochs, transform_method, folder_pathmod):
    '''
    build, compiles, and fits ANN model
    saves trained model files with keras
    saves error figure and model details text file
    
    Parameters
    ----------
    x_train: 2d numpy array of transformed training data	
    y_train: 2d numpy array of training targets
    x_test: 2d numpy array of transformed testing data 
    y_test: 2d numpy array of testing targets
    feature_names: array or list of feature names
    numlayers: integer for number of layers
    units: list of hidden units per layer
    epochs: integer number of epochs
    transform_method: name of transformation method of model details file
    folder_pathmod: path for saving png files and model detail text file
    
    Returns
    -------
    resid_train: array of observations - gmpe predictions for training data
    resid_test: array of observations - gmpe predictions for testing data
    pre_train: 2d array of model predictions for training data
    pre_test: 2d array of model predictions for testing data
    '''
    import numpy as np
    import pandas as pd
    from keras.models import Sequential
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from keras import layers
    from keras import optimizers
    import tensorflow.compat.v2 as tf
    tf.enable_v2_behavior()
    import seaborn as sns
    sns.set(style="ticks", color_codes=True)
    sns.reset_defaults()
    sns.set_style('whitegrid')
    sns.set_context('talk')
    sns.set_context(context='talk',font_scale=0.7)
    
    batch_size = 256
    
    def build_model():
        model = Sequential()
        model.add(layers.Dense(units[0],activation='sigmoid', input_shape=(x_train.shape[1],)))
        for i in range(1,numlayers):
            model.add(layers.Dense(units[i])) #add sigmoid aciivation functio? (only alues betwen 0 and 1)
        model.add(layers.Dense(y_train.shape[1])) #add sigmoid aciivation functio? (only alues betwen 0 and 1)    
        model.compile(optimizer=optimizers.Adam(lr=2e-3),loss='mse',metrics=['mae','mse']) 
        return model
    
    model=build_model()
    
    # fit the model
    history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=epochs,batch_size=batch_size,verbose=1)
    
    model.save(folder_pathmod + 'model')
    
    mae_history=history.history['val_mae']
    mae_history_train=history.history['mae']
    test_mse_score,test_mae_score,tempp=model.evaluate(x_test,y_test)
    # dataframe for saving purposes
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
    
    pre_test = np.array(model.predict(x_test))
    pre_train = np.array(model.predict(x_train))
    
    # test data
    mean_x_test_allT = pre_test
    
    # training data
    mean_x_train_allT = pre_train
    
    resid_train = y_train-mean_x_train_allT
    resid_test = y_test-mean_x_test_allT
    
    diff=np.std(y_train-mean_x_train_allT,axis=0)
    difftest=np.std(y_test-mean_x_test_allT,axis=0)
    # write model details to a file
    file = open(folder_pathmod + 'model_details.txt',"w+")
    file.write('number training samples ' + str(len(x_train)) + '\n')
    file.write('number testing samples ' + str(len(x_test)) + '\n')
    file.write('data transformation method ' + str(transform_method) + '\n')
    file.write('input feature names ' +  str(feature_names)+ '\n')
    file.write('number of epochs ' +  str(epochs)+ '\n')
    model.summary(print_fn=lambda x: file.write(x + '\n'))
    file.write('model fit history' + str(hist_df.to_string) + '\n')
    file.write('stddev train' + str(diff) + '\n')
    file.write('stddev test' + str(difftest) + '\n')
    file.close()
    
    # write predictions to a file
    period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]
    cols = ['obs_' + str(period[i]) for i in range(len(period))] + ['pre_' + str(period[i]) for i in range(len(period))]
    out = np.concatenate((y_train, pre_train), axis=1)
    df_out = pd.DataFrame(out, columns=cols)   
    df_out.to_csv(folder_pathmod + 'train_obs_pre.csv')
    
    period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]
    cols = ['obs_' + str(period[i]) for i in range(len(period))] + ['pre_' + str(period[i]) for i in range(len(period))]
    out = np.concatenate((y_test, pre_test), axis=1)
    df_out = pd.DataFrame(out, columns=cols)   
    df_out.to_csv(folder_pathmod + 'test_obs_pre.csv')
    
    return resid_train, resid_test, pre_train, pre_test
