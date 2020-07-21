#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 15:30:26 2020

@author: aklimase
"""

from openquake.hazardlib.gsim import base
import openquake.hazardlib.imt as imt
from openquake.hazardlib.const import StdDev
from openquake.hazardlib.gsim.abrahamson_2014 import AbrahamsonEtAl2014   #ASK
from openquake.hazardlib.gsim.campbell_bozorgnia_2014 import CampbellBozorgnia2014  #CB
from openquake.hazardlib.gsim.chiou_youngs_2014 import ChiouYoungs2014   #CY
from openquake.hazardlib.gsim.boore_2014 import BooreEtAl2014   #BSSA 

import pandas as pd
import numpy as np

#read in files
nametrain='/Users/aklimase/Documents/USGS/python_code/cybertrainyeti10_residfeb.csv'
nametest='/Users/aklimase/Documents/USGS/python_code/cybertestyeti10_residfeb.csv'
dftrain = pd.read_pickle(nametrain) 
dftest = pd.read_pickle(nametest)
print(dftrain.shape)
print(dftest.shape)



####

ztest = dftest
print(ztest.columns)

period=[10,7.5,5,3,2,1]
period=np.array(period)

# period = ztest["siteperiod"]
###

# Type of standard deviation
stddev_types = [StdDev.TOTAL]

# Instantiate the GMPE
#gmpe = Atkinson2010Hawaii()
gmpeASK = AbrahamsonEtAl2014()

#CY: z1.0 is in meters
gmpeASKdata=np.zeros([ztest.shape[0],period.shape[0]])
gmpeASKstd=np.zeros([ztest.shape[0],period.shape[0]])


Mwtest= dftest["Mag"]
distest=dftest["Site_Rupture_Dist"]
vs30test=np.array(dftest["vs30"])

#index=(vs30test<500)
#vs30test[index]=500


z1test=dftest["z10"]
z2p5test=dftest["z25"]
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
    
rjbtest=dftest["rjb"]
rxtest=dftest["rx"]
rytest=dftest["ry"]
hypodistest=dftest["hypodistance"]
depthtotop = dftest["Depthtotop"]

ztest = np.column_stack([Mwtest,distest,vs30test,z1test, z2p5test, raketest, diptest, hypodepthtest, widthtest, rjbtest, rxtest, depthtotop, rytest])

    
    
for i in range(ztest.shape[0]):
    print(i)
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
    for ii in range(0,6):
        sx.vs30measured = 0
        period1=period[ii]
        # gmpeBSSAdata[i,ii], g = gmpeBSSA.get_mean_and_stddevs(sx, rx, dx, imt.SA(period1), stddev_types)
        # gmpeBSSAstd[i,ii]=g[0][0]
        
        # gmpeCBdata[i,ii], g = gmpeCB.get_mean_and_stddevs(sx, rx, dx, imt.SA(period1), stddev_types)
        # gmpeCBstd[i,ii]=g[0][0]
        
        # gmpeCYdata[i,ii], g = gmpeCY.get_mean_and_stddevs(sx, rx, dx, imt.SA(period1), stddev_types)
        # gmpeCYstd[i,ii]=g[0][0]
        
        sx.vs30measured = [0]
        gmpeASKdata[i,ii], g = gmpeASK.get_mean_and_stddevs(sx, rx, dx, imt.SA(period1), stddev_types)
        gmpeASKstd[i,ii]=g[0][0]






# def create_ANN():
    
    

