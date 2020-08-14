#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 08:51:16 2020

@author: aklimase
"""



def plot_resid(resid, resid_test, folder_path):
    '''

    Parameters
    ----------
    resid : TYPE
        DESCRIPTION.
    resid_test : TYPE
        DESCRIPTION.
    folder_path : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]

    diff=np.std(resid,axis=0)
    difftest=np.std(resid_test,axis=0)
    f22=plt.figure('Difference Std of residuals vs Period')
    plt.semilogx(period,diff,label='Training ')
    plt.semilogx(period,difftest,label='Testing')
    plt.xlabel('Period')
    plt.ylabel('Total Standard Deviation')
    plt.legend()
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
    Parameters
    ----------
    
    '''
    
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    for i in range(10):
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
        plt.savefig(folder_path + 'obs_pre_T_' + str(T) + '.png')
        plt.show()
    plt.close('all')

    
    
def gridded_plots(griddednorm_mean, gridded_counts, period, lat, lon, evlon, evlat, sitelon, sitelat, folder_path):
    '''
    Parameters
    ----------
    griddednorm_mean: 2D list of gridded normalized residuals
    gridded_counts: 2D list of path counts per cell
    period:
    lat: list of grid cell latitudes
    lon: list of gtrid cell longitudes
    evlon: 
    evlat
    sitelon
    sitelat
    folder_path
    '''
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    for i in range(len(griddednorm_mean.T)):
        T = period[i]
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
    plt.close('all')

def plot_rawinputs(x_raw, mean_x_allT, y, feature_names, period, folder_path):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    folderlist = ['T10s','T7_5s','T5s','T4s','T3s','T2s','T1s','T_5s','T_2s','T_1s']
    for j in range(len(period)):
        # y_ind = j
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
    Parameters
    ----------
    '''
    import os
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt

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
        # ax.scatter(Rtrain, y_train, s=1, label='train data')
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
        
        # #
        # for i in range(len(x_train)):
        #     fig, axes = plt.subplots(2,1,figsize=(10,8))
        #     plt.title('T = ' + str(period[y_ind]) + ' s')
        #     ylim = max(np.abs(y_test[:,y_ind]))
        #     axes[0].set_ylim(-1*ylim,ylim)
        #     axes[1].set_ylim(-1*ylim,ylim)
        #     axes[0].scatter(x_test[:,i], mean_x_test,s=1, label='predictions', color='blue')
        #     axes[1].scatter(x_test[:,i], y_test[:,y_ind], s=1, label='test targets', color='green')
        #     axes[1].set_xlabel(feature_names[i])
        #     axes[0].set_ylabel('prediction')
        #     axes[1].set_ylabel('target')
        #     axes[0].legend(loc = 'upper left')
        #     axes[1].legend(loc = 'upper left')
        #     plt.savefig(folder_path + folderlist[j] + '/predictions_vs_' + feature_names[i] + '.png')
        #     plt.show()
        
    
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
        plt.close('all')

        
