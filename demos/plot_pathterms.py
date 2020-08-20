#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 13:22:07 2020

@author: aklimasewski
"""


#plot station lat lon and paths
# import folium
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
#color by residual value


train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv')



#sep columns
train_r, train_stlat, train_stlon, train_evlat, train_evlon, train_evdepth = [train_data1[:,i:i+1] for i in range(len(train_data1[0]))]

#choose what period to plot

period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]

y_ind = 1
y=train_targets1[:].T[y_ind]  

for j in range(len(period)):
    y_ind = j
    y=train_targets1[:].T[y_ind]  
    
    #color events by depth or pga
    cbound = max(np.abs(y))
    cmap = mpl.cm.get_cmap('seismic')
    normalize = mpl.colors.Normalize(vmin=-1*cbound, vmax=cbound)
    colors = [cmap(normalize(value)) for value in y]
    s_m = mpl.cm.ScalarMappable(cmap = cmap, norm=normalize)
    s_m.set_array([])
    
    fig = plt.figure(figsize = (12,10))
    plt.title('training path midpoint, T = ' + str(period[y_ind]) + ' s ')
    # plt.scatter(train_stlon,train_stlat, s =1,marker='^', alpha = 0.5,label = 'stations')
    # plt.scatter(train_evlon,train_evlat, s =1,marker='^',alpha = 0.5, label = 'events')
    # for i in range(len(train_evlon)):
        # plt.plot([train_stlon[i],train_evlon[i]], [train_stlat[i],train_evlat[i]], '-', lw = 1, c = colors[i], alpha = 0.5)
    plt.scatter((train_stlon+train_evlon)/2., (train_stlat+train_evlat)/2., s = 2, c = colors, alpha = 0.25)
    
    # plt.legend(loc = 'upper right')
    
    
    fig.subplots_adjust(right=0.75)
    cbar_ax = fig.add_axes([0.85, 0.18, 0.1, 0.63])
    
    plt.ylabel(r'y', fontsize = 20)
    plt.xlabel(r'counts', fontsize = 20)
    plt.yticks(np.arange(-3,3,1))
    
    N, bins, patches = cbar_ax.hist(y, bins=np.arange(-3,3,0.25), orientation='horizontal')
    for bin, patch in zip(bins, patches):
        color = cmap(normalize(bin))
        patch.set_facecolor(color)
    plt.savefig('/Users/aklimasewski/Documents/pathresidual_figs/train_'+ str(period[y_ind]) + 's.png')
    
    plt.show()
# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in range(len(train_evlon)):
#     ax.plot([train_stlon[i],train_evlon[i]],[train_stlat[i],train_evlat[i]],[0,train_evdepth[i]], '-', lw = 1, c = colors[i], alpha = 0.5)
#%%


# #plotting
# f10=plt.figure('Overfitting Test')
# plt.plot(mae_history,label='Testing Data')
# plt.plot(mae_history_train,label='Training Data')
# plt.xlabel('Epoch')
# plt.ylabel('Mean Absolute Error')
# plt.title('Overfitting Test')
# plt.legend()
# print(test_mae_score)
# plt.grid()
# plt.show()

# #1d
# a=model.predict(x_test)
# b=model.predict(x_train)

#10 values (1 per period)
#n x 10 array

y_ind = 0
#each column is prediction for a period
mean_x_test_allT = np.mean(predict_mean, axis = 0)
predict_epistemic_allT = np.std(predict_mean, axis = 0)

mean_x_test = mean_x_test_allT[:,y_ind:y_ind+1].flatten()
predict_epistemic= predict_epistemic_allT[:,y_ind:y_ind+1].flatten()

Rtrain = x_train_raw[0:train_samples,0:1]
Rtest = x_train_raw[0:test_samples,0:1]

f1=plt.figure('Hypo dist normalized Prediction')
# plt.plot(x_train[:,0],mean_x_train,'.r', label='train data')
plt.plot(x_test[:,0],mean_x_test,'.b',label='test data')
plt.xlabel('input')
plt.ylabel('Prediction')
plt.title('Earthquake Magnitude normalized Prediction')
plt.legend()
plt.show()


f1=plt.figure('Hypo dist normalized Actual')
plt.plot(x_train[:,0],y_train[:,0],'.r')
plt.plot(x_test[:,0],y_test[:,0],'.b')
plt.xlabel('input')
plt.ylabel('Prediction')
plt.title('Earthquake Magnitude normalized Actual')
plt.show()


#title = 'T = 5.0 s'
f212=plt.figure()
# plt.hist(y_train[:,0]-mean_x_test,100,label='Training')
plt.hist(y_test[:,0]-mean_x_test,100,label='Testing')
plt.xlabel('Residual ln(Target/Predicted)')
plt.ylabel('Count')
# temp1=str(np.std(y_train[:,0]-b[:,0]))
temp2=str(np.std(y_test[:,0]-mean_x_test))
# temp11=str(np.mean(y_train[:,0]-b[:,0]))
temp22=str(np.mean(y_test[:,0]-mean_x_test))
# plt.text(1,1,'sigma_train = '+ temp1[0:4])
plt.text(1,1,'sigma_test =' + temp2[0:4])
# plt.text(1,1,'mean_train =  '+ temp11[0:4])
plt.text(1,1,'mean_test = '+ temp22[0:4])
plt.title('Residual ln(Target/Predicted)): T = 5.0 s')
plt.legend()
plt.show()

f212=plt.figure('T = 0.5 s')
plt.hist(train_targets[:,4]-b[:,4],100,label='Training')
plt.hist(test_targets[:,4]-a[:,4],100,label='Testing')
plt.xlabel('Residual ln(Target/Predicted)')
plt.ylabel('Count')
temp1=str(np.std(train_targets[:,4]-b[:,4]))
temp2=str(np.std(test_targets[:,4]-a[:,4]))
temp11=str(np.mean(train_targets[:,4]-b[:,4]))
temp22=str(np.mean(test_targets[:,4]-a[:,4]))
plt.text(1.2,6000,   'sigma_train = '+ temp1[0:4])
plt.text(1.2,5500,   'sigma_test =' + temp2[0:4])
plt.text(1.2,3000,   'mean_train =  '+ temp11[0:4])
plt.text(1.2,2500,   'mean_test = '+ temp22[0:4])
plt.title('Residual ln(Target/Predicted): T = 0.5 s')
plt.legend()



diff=np.std(train_targets-b,axis=0)
difftest=np.std(test_targets-a,axis=0)
diffmean=np.mean(train_targets-b,axis=0)
f22=plt.figure('Difference Std of residuals vs Period')
plt.semilogx(period,diff,label='Training ')
plt.semilogx(period,difftest,label='Testing')
plt.xlabel('Period')
plt.ylabel('Total Standard Deviation')
