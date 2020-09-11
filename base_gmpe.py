#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:38:19 2020

@author: aklimasewski
"""

filename = '/Users/aklimasewski/Documents/data/NGAWest2region.csv'

# nga_data1, nga_targets1, feature_names = readindataNGA(filename,n=12)
# ztest = nga_data1


def gmpe_avg(ztest):
    '''
    transforms cybershake dips and Rx
    
    Parameters
    ----------
    ztest: 2d numpy array of 12 feature data
    
    Returns
    -------
    model_avg: average of base model predictions in ln m/s2
    '''
    import numpy as np
    import openquake
    from openquake.hazardlib.gsim.abrahamson_2014 import AbrahamsonEtAl2014   #ASK
    from openquake.hazardlib.gsim.campbell_bozorgnia_2014 import CampbellBozorgnia2014  #CB
    from openquake.hazardlib.gsim.chiou_youngs_2014 import ChiouYoungs2014   #CY
    from openquake.hazardlib.gsim.boore_2014 import BooreEtAl2014   #BSSA 
    import openquake.hazardlib.imt as imt
    from openquake.hazardlib.const import StdDev
    import openquake.hazardlib.gsim.base as base
    
    stddev_types = [StdDev.TOTAL]
    
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
            
    model_avg = np.log(9.81*np.exp(np.mean([gmpeBSSAdata,gmpeCBdata,gmpeCYdata,gmpeASKdata],axis=0)))

    #outputs ln m/s2
    return model_avg
