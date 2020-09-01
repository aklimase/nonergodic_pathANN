#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:38:19 2020

@author: aklimasewski
"""
import numpy as np
# from readNGA import readindataNGA

import openquake
from openquake.hazardlib.gsim.abrahamson_2014 import AbrahamsonEtAl2014   #ASK
from openquake.hazardlib.gsim.campbell_bozorgnia_2014 import CampbellBozorgnia2014  #CB
from openquake.hazardlib.gsim.chiou_youngs_2014 import ChiouYoungs2014   #CY
from openquake.hazardlib.gsim.boore_2014 import BooreEtAl2014   #BSSA 
import openquake.hazardlib.imt as imt
from openquake.hazardlib.const import StdDev
import pandas as pd
from openquake.hazardlib.gsim.base import GMPE
import openquake.hazardlib.gsim.base as base

import matplotlib.pyplot as plt

stddev_types = [StdDev.TOTAL]


filename = '/Users/aklimasewski/Documents/data/NGAWest2region.csv'

# nga_data1, nga_targets1, feature_names = readindataNGA(filename,n=12)
# ztest = nga_data1

period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]

# gmpeBSSAdata=np.zeros((ztest.shape[0], len(period)))
# gmpeASKdata=np.zeros((ztest.shape[0], len(period)))
# gmpeCBdata=np.zeros((ztest.shape[0], len(period)))
# gmpeCYdata=np.zeros((ztest.shape[0], len(period)))

# gmpeBSSAstd=np.zeros((ztest.shape[0],len(period)))
# gmpeASKstd=np.zeros((ztest.shape[0],len(period)))
# gmpeCBstd=np.zeros((ztest.shape[0],len(period)))
# gmpeCYstd=np.zeros((ztest.shape[0],len(period)))

# gmpeASK = AbrahamsonEtAl2014()
# gmpeCB = CampbellBozorgnia2014()
# gmpeCY = ChiouYoungs2014()
# gmpeBSSA = BooreEtAl2014()

# for i in range(ztest.shape[0]):
#     print(i)
#     dx = base.DistancesContext()
#     dx.rjb=np.array([ztest[i,9]])
    
#     #dx.rjb = np.logspace(-1, 2, 10)
#     # Magnitude and rake
#     rx = base.RuptureContext()
#     rx.mag = np.array([ztest[i,0]])
#     rx.rake = np.array([ztest[i,5]])
#     rx.hypo_depth = np.array([ztest[i,7]])
#     # Vs30
#     sx = base.SitesContext()
#     sx.vs30 = np.array([ztest[i,2]])
#     sx.vs30measured = 0
    
#     dx.rrup=np.array([ztest[i,1]])
#     rx.ztor=np.array([ztest[i,11]])
#     rx.dip=np.array([ztest[i,6]])
#     rx.width=np.array([ztest[i,8]])
#     # dx.rx=np.array([rxkeep[i]])
#     dx.rx=np.array([ztest[i,10]])

#     dx.ry0=np.array([0])
#     sx.z1pt0= np.array([ztest[i,3]])
#     sx.z2pt5=np.array([ztest[i,4]])

#     # Evaluate GMPE
#     #Unit of measure for Z1.0 is [m] (ASK)
#     #lmean, lsd = gmpeASK.get_mean_and_stddevs(sx, rx, dx, imt.PGV(), stddev_types)
    
#     #for period1 in period:
#     for ii in range(len(period)):
#         sx.vs30measured = 0
#         period1=period[ii]
#         gmpeBSSAdata[i,ii], g = gmpeBSSA.get_mean_and_stddevs(sx, rx, dx, imt.SA(period1), stddev_types)
#         gmpeBSSAstd[i,ii]=g[0][0]
        
#         gmpeCBdata[i,ii], g = gmpeCB.get_mean_and_stddevs(sx, rx, dx, imt.SA(period1), stddev_types)
#         gmpeCBstd[i,ii]=g[0][0]
        
#         gmpeCYdata[i,ii], g = gmpeCY.get_mean_and_stddevs(sx, rx, dx, imt.SA(period1), stddev_types)
#         gmpeCYstd[i,ii]=g[0][0]
        
#         sx.vs30measured = [0]
#         gmpeASKdata[i,ii], g = gmpeASK.get_mean_and_stddevs(sx, rx, dx, imt.SA(period1), stddev_types)
#         gmpeASKstd[i,ii]=g[0][0]

# #average the models
# #models are ln SA
# # model_avg = np.mean([np.exp(gmpeBSSAdata),np.exp(gmpeCBdata),np.exp(gmpeCYdata),np.exp(gmpeASKdata)],axis=0)
# # resid_NGA = nga_targets1- model_avg

# # model_avg_std = np.mean([gmpeBSSAstd,gmpeCBstd,gmpeCYstd,gmpeASKstd],axis=0)
# # resid_NGA = nga_targets1- model_avg

# #openquake models predict ln(g)
# #convert to linear value and multipple by g, so now in overall value m/s2
# model_avg = np.mean([np.exp(gmpeBSSAdata)*9.81,np.exp(gmpeCBdata)*9.81,np.exp(gmpeCYdata)*9.81,np.exp(gmpeASKdata)*9.81],axis=0)
# resid_NGA = np.log(nga_targets1*9.81) - np.log(model_avg)

# df_out = pd.DataFrame(resid_NGA, columns=['T10','T7.5','T5','T4','T3','T2','T1','T0.5','T0.2','T0.1'])   
# df_out.to_csv('/Users/aklimasewski/Documents/data/NGAmodel_GMresiduals_g.csv')   

# # df_out = pd.DataFrame(model_avg, columns=['T10','T7.5','T5','T4','T3','T2','T1','T0.5','T0.2','T0.1'])   
# # df_out.to_csv('/Users/aklimasewski/Documents/data/NGAopenquake_predictions.csv')   


# plt.figure()
# boore=np.mean(gmpeBSSAstd,axis=0)
# plt.semilogx(period,boore,label='BSSA')

# boore=np.mean(gmpeASKstd,axis=0)
# plt.semilogx(period,boore,label='ASK')

# boore=np.mean(gmpeCBstd,axis=0)
# plt.semilogx(period,boore,label='CB')

# boore=np.mean(gmpeCYstd,axis=0)
# plt.semilogx(period,boore,label='CY')

# plt.legend()
# plt.grid()
# plt.show()

# plt.figure()
# # boore=np.mean(gmpeBSSAstd,axis=0)
# avgresid = np.mean(resid_NGA,axis=0)
# plt.semilogx(period,avgresid )
# plt.legend()
# plt.grid()
# plt.show()


def gmpe_avg(ztest):
    period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]
    
    gmpeBSSAdata=np.zeros((ztest.shape[0], len(period)))
    gmpeASKdata=np.zeros((ztest.shape[0], len(period)))
    gmpeCBdata=np.zeros((ztest.shape[0], len(period)))
    gmpeCYdata=np.zeros((ztest.shape[0], len(period)))
    
    gmpeBSSAstd=np.zeros((ztest.shape[0],len(period)))
    gmpeASKstd=np.zeros((ztest.shape[0],len(period)))
    gmpeCBstd=np.zeros((ztest.shape[0],len(period)))
    gmpeCYstd=np.zeros((ztest.shape[0],len(period)))
    
    gmpeASK = AbrahamsonEtAl2014()
    gmpeCB = CampbellBozorgnia2014()
    gmpeCY = ChiouYoungs2014()
    gmpeBSSA = BooreEtAl2014()
    
    for i in range(ztest.shape[0]):
        # print(i)
        dx = base.DistancesContext()
        dx.rjb=np.array([ztest[i,9]])
        
        #dx.rjb = np.logspace(-1, 2, 10)
        # Magnitude and rake
        rx = base.RuptureContext()
        rx.mag = np.array([ztest[i,0]])
        rx.rake = np.array([ztest[i,5]])
        rx.hypo_depth = np.array([ztest[i,7]])
        # Vs30
        sx = base.SitesContext()
        sx.vs30 = np.array([ztest[i,2]])
        sx.vs30measured = 0
        
        dx.rrup=np.array([ztest[i,1]])
        rx.ztor=np.array([ztest[i,11]])
        rx.dip=np.array([ztest[i,6]])
        rx.width=np.array([ztest[i,8]])
        # dx.rx=np.array([rxkeep[i]])
        dx.rx=np.array([ztest[i,10]])
    
        dx.ry0=np.array([0])
        sx.z1pt0= np.array([ztest[i,3]])
        sx.z2pt5=np.array([ztest[i,4]])
    
        # Evaluate GMPE
        #Unit of measure for Z1.0 is [m] (ASK)
        #lmean, lsd = gmpeASK.get_mean_and_stddevs(sx, rx, dx, imt.PGV(), stddev_types)
        
        #for period1 in period:
        for ii in range(len(period)):
            sx.vs30measured = 0
            period1=period[ii]
            gmpeBSSAdata[i,ii], g = gmpeBSSA.get_mean_and_stddevs(sx, rx, dx, imt.SA(period1), stddev_types)
            gmpeBSSAstd[i,ii]=g[0][0]
            
            gmpeCBdata[i,ii], g = gmpeCB.get_mean_and_stddevs(sx, rx, dx, imt.SA(period1), stddev_types)
            gmpeCBstd[i,ii]=g[0][0]
            
            gmpeCYdata[i,ii], g = gmpeCY.get_mean_and_stddevs(sx, rx, dx, imt.SA(period1), stddev_types)
            gmpeCYstd[i,ii]=g[0][0]
            
            sx.vs30measured = [0]
            gmpeASKdata[i,ii], g = gmpeASK.get_mean_and_stddevs(sx, rx, dx, imt.SA(period1), stddev_types)
            gmpeASKstd[i,ii]=g[0][0]
        
    # np.log ((np.exp(gmpeBSSAdata)+np.exp(gmpeASKdata)+np.exp(gmpeCBdata)+np.exp(gmpeCYdata))/4*9.8)
    
    # model_avg = np.log(np.mean([9.81*np.exp(gmpeBSSAdata),9.81*np.exp(gmpeCBdata),9.81*np.exp(gmpeCYdata),9.81*np.exp(gmpeASKdata)],axis=0))
    model_avg = np.log(9.81*np.exp(np.mean([gmpeBSSAdata,gmpeCBdata,gmpeCYdata,gmpeASKdata],axis=0)))

    #outputs ln m/s2
    return model_avg
        


#%%
