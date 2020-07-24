
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 09:17:45 2020

@author: kwithers
"""


from openquake.hazardlib.gsim import base
import openquake.hazardlib.imt as imt
from openquake.hazardlib.const import StdDev

from openquake.hazardlib.gsim.abrahamson_2014 import AbrahamsonEtAl2014   #ASK
from openquake.hazardlib.gsim.campbell_bozorgnia_2014 import CampbellBozorgnia2014  #CB
from openquake.hazardlib.gsim.chiou_youngs_2014 import ChiouYoungs2014   #CY
from openquake.hazardlib.gsim.boore_2014 import BooreEtAl2014   #BSSA 


# Type of standard deviation
stddev_types = [StdDev.TOTAL]

# Instantiate the GMPE
#gmpe = Atkinson2010Hawaii()
gmpeASK = AbrahamsonEtAl2014()
gmpeBSSA = BooreEtAl2014()
gmpeCB= CampbellBozorgnia2014()
gmpeCY = ChiouYoungs2014()


#CY: z1.0 is in meters



gmpeBSSAdata=np.zeros([ztest.shape[0],period.shape[0]])
gmpeASKdata=np.zeros([ztest.shape[0],period.shape[0]])
gmpeCBdata=np.zeros([ztest.shape[0],period.shape[0]])
gmpeCYdata=np.zeros([ztest.shape[0],period.shape[0]])

gmpeBSSAstd=np.zeros([ztest.shape[0],period.shape[0]])
gmpeASKstd=np.zeros([ztest.shape[0],period.shape[0]])
gmpeCBstd=np.zeros([ztest.shape[0],period.shape[0]])
gmpeCYstd=np.zeros([ztest.shape[0],period.shape[0]])
    ##ztest = np.column_stack([
    #Mwtest: 0
    #distest : 1
    #vs30test: 2
    #z1test : 3
    #z2p5test: 4
    #rake: 5
    #dip : 6
    #hypodepth : 7
    #width : 8
    #rjb: 9
    #rx: 10
    #depthtotop: 11
    #ry : 12
    
for i in range(ztest.shape[0]):
    
        # Distance
        
    dx = base.DistancesContext()
    dx.rjb=    np.array([ztest[i,9]])
    
    
    
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
    dx.rx=np.array([rxkeep[i]])
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
        gmpeBSSAdata[i,ii], g = gmpeBSSA.get_mean_and_stddevs(sx, rx, dx, imt.SA(period1), stddev_types)
        gmpeBSSAstd[i,ii]=g[0][0]
        
        gmpeCBdata[i,ii], g = gmpeCB.get_mean_and_stddevs(sx, rx, dx, imt.SA(period1), stddev_types)
        gmpeCBstd[i,ii]=g[0][0]
        
        gmpeCYdata[i,ii], g = gmpeCY.get_mean_and_stddevs(sx, rx, dx, imt.SA(period1), stddev_types)
        gmpeCYstd[i,ii]=g[0][0]
        
        sx.vs30measured = [0]
        gmpeASKdata[i,ii], g = gmpeASK.get_mean_and_stddevs(sx, rx, dx, imt.SA(period1), stddev_types)
        gmpeASKstd[i,ii]=g[0][0]
        

rrrr=np.log((np.exp(gmpeBSSAdata)+np.exp(gmpeASKdata)+np.exp(gmpeCBdata)+np.exp(gmpeCYdata))/4*9.8)

residtest=(residtest)-(rrrr)