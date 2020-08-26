#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:38:19 2020

@author: aklimasewski
"""



from openquake.hazardlib.gsim.abrahamson_2014 import AbrahamsonEtAl2014 as ASK
from openquake.hazardlib.gsim.campbell_bozorgnia_2014 import CampbellBozorgnia2014  #CB
from openquake.hazardlib.gsim.chiou_youngs_2014 import ChiouYoungs2014   #CY
from openquake.hazardlib.gsim.boore_2014 import BooreEtAl2014   #BSSA 


from openquake.hazardlib.gsim.base import GMPE
from openquake.hazardlib.gsim.boore_2014 import BooreEtAl2014
from openquake.hazardlib.gsim.campbell_bozorgnia_2014 import (
    CampbellBozorgnia2014)
from openquake.hazardlib.imt import PGA, PGV, SA
from openquake.hazardlib import const


from openquake.hazardlib.gsim.base import GMPE
import openquake.hazardlib.gsim.base as base

period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]

gmpeBSSAdata=np.zeros([period.shape[0]])
gmpeASKdata=np.zeros([period.shape[0]])
gmpeCBdata=np.zeros([period.shape[0]])
gmpeCYdata=np.zeros([period.shape[0]])

gmpeBSSAstd=np.zeros([period.shape[0]])
gmpeASKstd=np.zeros([period.shape[0]])
gmpeCBstd=np.zeros([period.shape[0]])
gmpeCYstd=np.zeros([period.shape[0]])

n=13
train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n = n)
ztest = train_data1

dx = base.DistancesContext()
dx.rjb = np.array([ztest[:,9]])

#dx.rjb = np.logspace(-1, 2, 10)
# Magnitude and rake
rx = base.RuptureContext()
rx.mag = np.array([ztest[:,0]])
rx.rake = np.array([ztest[:,5]])
rx.hypo_depth = np.array([ztest[:,7]])
# Vs30
sx = base.SitesContext()
sx.vs30 = np.array([ztest[:,2]])
sx.vs30measured = 0



dx.rrup=np.array([ztest[:,1]])
rx.ztor=np.array([ztest[:,11]])
rx.dip=np.array([ztest[:,6]])
rx.width=np.array([ztest[:,8]])
# dx.rx=np.array([rxkeep[i]])

dx.ry0=np.array([0])

sx.z1pt0= np.array([ztest[:,3]])
sx.z2pt5=np.array([ztest[:,4]])

import openquake.hazardlib.imt as imt #intensity measure type
stddev_types = openquake.hazardlib.const.StdDev.TOTAL

from openquake.hazardlib.const import StdDev
stddev_types = [StdDev.TOTAL]

# Evaluate GMPE
#Unit of measure for Z1.0 is [m] (ASK)
#lmean, lsd = gmpeASK.get_mean_and_stddevs(sx, rx, dx, imt.PGV(), stddev_types)
i=0
#for period1 in period:
for ii in range(0,6):
    sx.vs30measured = 0
    period1=period[ii]
    gmpeBSSAdata[ii], g = gmpeBSSA.get_mean_and_stddevs(sx, rx, dx, imt.SA(period1), stddev_types)
    gmpeBSSAstd[ii]=g[0][0]
    
    gmpeCBdata[ii], g = gmpeCB.get_mean_and_stddevs(sx, rx, dx, imt.SA(period1), stddev_types)
    gmpeCBstd[ii]=g[0][0]
    
    gmpeCYdata[ii], g = gmpeCY.get_mean_and_stddevs(sx, rx, dx, imt.SA(period1), stddev_types)
    gmpeCYstd[ii]=g[0][0]
    
    sx.vs30measured = [0]
    gmpeASKdata[ii], g = gmpeASK.get_mean_and_stddevs(sx, rx, dx, imt.SA(period1), stddev_types)
    gmpeASKstd[ii]=g[0][0]
        



boore=gmpeBSSAstd
plt.semilogx(period,boore,label='BSSA')

boore=gmpeASKstd
plt.semilogx(period,boore,label='ASK')

boore=gmpeCBstd
plt.semilogx(period,boore,label='CB')

boore=gmpeCYstd
plt.semilogx(period,boore,label='CY')


plt.legend()
plt.grid()