#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 08:51:16 2020

@author: aklimase
"""

import seaborn as sns
sns.set(style="ticks", color_codes=True)
sns.reset_defaults()
sns.set_style('whitegrid')
sns.set_context('talk')
sns.set_context(context='talk',font_scale=0.7)

def plot_resid(resid, resid_test, folder_path):
    '''
    creates plots of mean and standard devations of training and testing
    
    Parameters
    ----------
    resid: array of observations - gmpe predictions for training data
    resid_test: array of observations - gmpe predictions for testing data
    folder_path: path for saving png files

    Returns
    -------
    creates pngs of standard deviation of residuals and average of residuals

    '''
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="ticks", color_codes=True)
    sns.reset_defaults()
    sns.set_style('whitegrid')
    sns.set_context('talk')
    sns.set_context(context='talk',font_scale=0.7)
    
    period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]

    diff=np.std(resid,axis=0)
    difftest=np.std(resid_test,axis=0)
    f22=plt.figure('Difference Std of residuals vs Period')
    plt.semilogx(period,diff,label='Training ')
    plt.semilogx(period,difftest,label='Testing')
    plt.xlabel('Period')
    plt.ylabel('Total Standard Deviation')
    plt.legend()
    plt.ylim(.25,.85)
    plt.savefig(folder_path + 'resid_T.png')
    plt.show()
    
    diffmean=np.mean(resid,axis=0)
    diffmeantest=np.mean(resid_test,axis=0)
    f22=plt.figure('Difference Std of residuals vs Period')
    plt.semilogx(period,diffmean,label='Training')
    plt.semilogx(period,diffmeantest,label='Testing')
    plt.xlabel('Period')
    plt.ylabel('Mean residual')
    plt.legend()
    plt.savefig(folder_path + 'mean_T.png')
    plt.show()
    plt.close('all')
    
def obs_pre(y_train, y_test, pre, pre_test, period, folder_path):
    '''
    creates scatterplots of observed ground motion residuals vs. model predicted ground motion data for training and testing data
    
    Parameters
    ----------
    y_train: 2d numpy array of observed ground motion residuals for training data
    y_test: 2d numpy array of observed ground motion residuals for testing data
    pre: numpy array of model predictions for training data
    pre_test: numpy array of model predictions for testing data
    period: list of periods 
    folder_path: path for saving png files
    
    Returns
    -------
    creates png scatterplots of observed ground motions vs. predicted for each period
    '''
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="ticks", color_codes=True)
    sns.reset_defaults()
    sns.set_style('whitegrid')
    sns.set_context('talk')
    sns.set_context(context='talk',font_scale=0.7)
    
    for i in range(len(period)):
        T= period[i]
        y = pre.T[i]
        x = y_train.T[i]
        y_testplot = pre_test.T[i]
        x_test = y_test.T[i]
        plt.figure(figsize = (6,6))
        lim = np.max(np.asarray([abs(x), abs(y)]).flatten())
        plt.scatter(x,y,s=1,label='Training')
        plt.scatter(x_test,y_testplot,s=1,label='Testing')
        plt.xlabel('observed')
        plt.ylabel('predicted')
        plt.title('T ' + str(T) + ' s')
        plt.xlim(-1*lim, lim)
        plt.ylim(-1*lim, lim)
        plt.legend()
        plt.savefig(folder_path + 'obs_pre_T_' + str(T) + '.png')
        plt.show()
    plt.close('all')


def plot_rawinputs(x_raw, mean_x_allT, y, feature_names, period, folder_path):
    '''
    plots model predictions vs. raw (untransformed) input features
    
    Parameters
    ----------
    x_raw: numpy array of untransformed data
    mean_x_test_allT: 2d array of model predictions for data
    y: 2d array numpy array of targets
    feature_names: array or list of feature names
    period: list of periods
    folder_path: path for saving png files
    
    Returns
    -------
    creates png scatterplots of predicted ground motions vs. each input feature (before transformation)
    '''
    
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import seaborn as sns
    sns.set(style="ticks", color_codes=True)
    sns.reset_defaults()
    sns.set_style('whitegrid')
    sns.set_context('talk')
    sns.set_context(context='talk',font_scale=0.7)

    folderlist = ['T10s','T7_5s','T5s','T4s','T3s','T2s','T1s','T_5s','T_2s','T_1s']
    for j in range(len(period)):
        mean_x_test = mean_x_allT[:,j:j+1].flatten()
        if not os.path.exists(folder_path + folderlist[j]):
            os.makedirs(folder_path + folderlist[j])
        for i in range(len(x_raw[0])):
            fig, axes = plt.subplots(2,1,figsize=(10,8))
            plt.title('T = ' + str(period[j]) + ' s')
            ylim = max(np.abs(y[:,j]))
            axes[0].set_ylim(-1*ylim,ylim)
            axes[1].set_ylim(-1*ylim,ylim)
            axes[0].scatter(x_raw[:,i], mean_x_test,s=1, label='predictions', color='blue')
            axes[1].scatter(x_raw[:,i], y[:,j], s=1, label='targets', color='green')
            axes[1].set_xlabel(feature_names[i])
            axes[0].set_ylabel('prediction')
            axes[1].set_ylabel('target')
            axes[0].legend(loc = 'upper left')
            axes[1].legend(loc = 'upper left')
            plt.savefig(folder_path + folderlist[j] + '/predictions_vs_' + feature_names[i] + '.png')
            plt.show()

def plot_outputs(folder_path, mean_x_test_allT, predict_epistemic_allT, mean_x_train_allT, predict_epistemic_train_allT, x_train, y_train, x_test, y_test, Rindex, period, feature_names):
    '''
    transforms cybershake dips and Rx
    
    Parameters
    ----------
    folder_path: path for saving png files
    mean_x_test_allT: 2d array of model predictions for testing data
    predict_epistemic_allT: 2d array of modeling training episteic uncertainty
    mean_x_train_all: 2d array of model predictions for training data
    predict_epistemic_train_allT:  2d array of model predictions for training data
    x_train: numpy array of transformed training data
    y_train: numpy array of observed ground motion residuals for training data
    y_test: numpy array of observed ground motion residuals for testing data
    x_test: numpy array of transformed testing data 
    Rindex: index in trainind data for distance feature
    period: list of periods
    feature_names: array or list of feature names
    
    Returns
    -------
    creates png scatterplots per period: 
        - predicted ground motion residuals with epistemic uncertainty vs. distance
        - predicted ground motions residuals with epistemic uncertainty vs. normalized distance
        - observed ground motions residuals with epistemic uncertainty vs. normalized distance
        - histograms of model residuals (observed - predicted residuals)
    '''
    
    import os
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="ticks", color_codes=True)
    sns.reset_defaults()
    sns.set_style('whitegrid')
    sns.set_context('talk')
    sns.set_context(context='talk',font_scale=0.7)

    #write frunction for either test or train
    folderlist = ['T10s','T7_5s','T5s','T4s','T3s','T2s','T1s','T_5s','T_2s','T_1s']

    for j in range(len(period)):
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
        plt.title('T = ' + str(period[y_ind]) + ' s')
        ylim = max(np.abs(y_test[:,y_ind]))
        axes[0].set_ylim(-1*ylim,ylim)
        axes[1].set_ylim(-1*ylim,ylim)
        axes[0].errorbar(x_test[:,Rindex], mean_x_test, yerr=predict_epistemic, fmt='.', label='predictions epistemic uncertainty', color='pink', alpha = 0.5, markeredgecolor='red')
        axes[1].scatter(x_test[:,Rindex], y_test[:,y_ind], s=1, label='test targets', color='green')
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
        plt.xlabel('dist input (normalized)')
        plt.ylabel('Prediction')
        plt.title('Hypo dist normalized Prediction T = ' + str(period[y_ind]) + ' s')
        plt.legend(loc = 'upper left')
        plt.savefig(folder_path + folderlist[j] + '/norm_dist_vs_pre.png')
        plt.show()
        
        f1=plt.figure('Hypo dist normalized Actual',figsize=(8,8))
        plt.plot(x_train[:,Rindex],y_train[:,y_ind],'.r', label='train data')
        plt.plot(x_test[:,Rindex],y_test[:,y_ind],'.b', label='test data')
        plt.ylabel('target')
        plt.xlabel('input normalized distance')
        plt.title('hypo dist normalized Actual')
        plt.legend(loc = 'upper left')
        plt.savefig(folder_path + folderlist[j] + '/norm_dist_vs_actual.png')
        plt.show()

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
        plt.close('all')

        
