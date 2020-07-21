#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 16:38:02 2020

@author: aklimasewski
"""
import shapely
import geopy
import numpy as np
from preprocessing import transform_dip, readindata, transform_data

#grid up residuals


dx=0.05
lon = np.arange(-119.5,-116.5, dx)
lat = np.arange(33.25, 35.25, dx)
longrid, latgrid = np.meshgrid(lon,lat)
# 

polygons = []
for i in range(len(lon)-1):
    for j in range(len(lat)-1):
        print(lat[j], lon[i], lat[j+1], lon[i+1])
        polygon_points = [(lon[i], lat[j]), (lon[i], lat[j+1]), (lon[i+1], lat[j+1]), (lon[i+1], lat[j]), (lon[i], lat[j])]
        shapely_poly = shapely.geometry.Polygon(polygon_points)
        polygons.append(shapely_poly)
          

train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n=6)
  
sitelat = train_data1[:,1]
sitelon = train_data1[:,2]
evlat = train_data1[:,3]
evlon = train_data1[:,4]

                      
#loop through each record     
for i in range(len(sitelat)):                         
    line = [(evlon[i], evlat[i]), (sitelon[i], sitelat[i])]
    path=shapely.geometry.LineString(line)
    
    for j in range(len(polygons)):
        shapely_poly = polygons[j]
        if path.intersects(shapely_poly) == True:
            shapely_line = shapely.geometry.LineString(line)
            intersection_line = list(shapely_poly.intersection(shapely_line).coords)
                        
                                  
 
    
 
    
 
    
 
    
 
# polygons = 
# for lati in lat:
#     for loni in lon:
#         print(lati, loni)
#         polygon = [(loni, lati), (loni, lati, (l+dx,ll+dx), (l,ll+dx), (l,ll)]

polygons = [(l,ll), (l+dx,ll), (l+dx,ll+dx), (l,ll+dx), (l,ll)]
shapely_poly = shapely.geometry.Polygon(polygon)

train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n=6)

sitelat = train_data1[:,1]
sitelon = train_data1[:,2]
evlat = train_data1[:,3]
evlon = train_data1[:,4]













fig, axs  = plt.subplots(5, 5, figsize=(12,12))


Mw=[6.5,6.75,7,7.25,7.5]
Dip=[90,75,60,45,30.5]

      
irec=-1        
for iii in Mw:
    irec=irec+1
    jrec=-1
    for jjj in Dip:
        jrec=jrec+1
        
        Mwtrain= dftrain["Mag"]
        distrain=dftrain["Site_Rupture_Dist"]
        vs30train=dftrain["vs30"]
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
        
        residtesttemp=dftrain.loc[:, 'IM_Value':'IM175']
        train_targets=np.log(residtesttemp.values/100)
        
        lengthtrain=dftrain["Length"]
        rjbtrain=dftrain["rjb"]
        rxtrain=dftrain["rx"]
        rytrain=dftrain["ry0"]
        hypodistrain=dftrain["hypodistance"]
        
        Utrain=dftrain["U"]
        Ttrain=dftrain["T"]
        xitrain=dftrain["xi"]


        index=(diptrain<(jjj+10))
        diptrain=diptrain[index]
        Mwtrain=Mwtrain[index]
        distrain=distrain[index]
        rxtrain=rxtrain[index]
        rytrain=rytrain[index]
        lattrain=lattrain[index]
        longtrain=longtrain[index]
        hypolontrain=hypolontrain[index]
        hypolattrain=hypolattrain[index]
        
        
        index=(diptrain>(jjj-10))
        diptrain=diptrain[index]
        Mwtrain=Mwtrain[index]
        distrain=distrain[index]
        rxtrain=rxtrain[index]
        rytrain=rytrain[index]
        lattrain=lattrain[index]
        longtrain=longtrain[index]
        hypolontrain=hypolontrain[index]
        hypolattrain=hypolattrain[index]

        index=(Mwtrain<(iii+0.1))
        diptrain=diptrain[index]
        Mwtrain=Mwtrain[index]
        distrain=distrain[index]
        rxtrain=rxtrain[index]
        rytrain=rytrain[index]
        lattrain=lattrain[index]
        longtrain=longtrain[index]
        hypolontrain=hypolontrain[index]
        hypolattrain=hypolattrain[index]
        
        index=(diptrain>(iii-0.1))
        diptrain=diptrain[index]
        Mwtrain=Mwtrain[index]
        distrain=distrain[index]    
        rxtrain=rxtrain[index]
        rytrain=rytrain[index]
        lattrain=lattrain[index]
        longtrain=longtrain[index]
        hypolontrain=hypolontrain[index]
        hypolattrain=hypolattrain[index]        
       
        
        nnn=9600#  +1    
        
        temp1=np.zeros([nnn])
        temp2=np.zeros([nnn])
        temp3=np.zeros([nnn])
        temp4=np.zeros([nnn])
        temp5=np.zeros([nnn])
        
        Latt=[]
        Longg=[]
        #i3=-1
        from shapely.geometry import LineString    
        dx=0.05/2
        for l in np.arange(-119.5,-116.5,dx):
            for ll in np.arange(33.25,35.25,dx):
                Latt.append(ll)
                Longg.append(l)
        
        #for i in dftrain.shape[0]:
        for i in range(hypolattrain.shape[0]):
            print(i,iii,jjj)
            
            
            i3=-1
            
            #loop through grid
            for l in np.arange(-119.5,-116.5,dx):
                for ll in np.arange(33.25,35.25,dx):
                    #Latt.append(ll)
                    #Longg.append(l)
                    i3=i3+1
                    #print(i3)
                    
        #    hypolattrain=dftrain["Hypocenter_Lat"]
        #hypolontrain=dftrain["Hypocenter_Lon"]
        #lattrain=dftrain["CS_Site_Lat"]
        #longtrain=dftrain["CS_Site_Lon"]
        
        
            
                    polygon = [(l,ll), (l+dx,ll), (l+dx,ll+dx), (l,ll+dx), (l,ll)]
                    shapely_poly = shapely.geometry.Polygon(polygon)
        
                    line = [(hypolontrain.values[i], hypolattrain.values[i]), (longtrain.values[i], lattrain.values[i])]
        #line = [(3.5, -2.0000000000000004), (2.0, -1.1102230246251565e-15)]
                    path=shapely.geometry.LineString(line)
                    if path.intersects(shapely_poly) == True:
                        temp1[i3]=temp1[i3]+1
                        
                        
                        shapely_line = shapely.geometry.LineString(line)
                        intersection_line = list(shapely_poly.intersection(shapely_line).coords)
                        
                        if len(intersection_line)== 2:
                            coords_1 = (intersection_line[0][1], intersection_line[0][0])
                            coords_2 = (intersection_line[1][1], intersection_line[1][0])
                            length=geopy.distance.vincenty(coords_1, coords_2).km
                            temp2[i3]=temp2[i3]+length
                            temp3[i3]=temp3[i3]+b[i,2]/length
                            temp4[i3]=temp4[i3]+b[i,4]/length
                            temp5[i3]=temp5[i3]+b[i,7]/length
                            #print(len(intersection_line))
                        else:
                            print(len(intersection_line))
                            length=0
                            temp2[i3]=temp2[i3]+length
                        #print(intersection_line)
        
        
        
        
        r=np.array(temp1).reshape(len(Latt),1)
        from matplotlib  import cm
        

        #fig = plt.figure(figsize=(12,12))
        #axs[0].set_title("Shakemap: Ridgecrest: 7.1, Period = 3 s (g)",fontsize=14)
        axs[irec,jrec].set_title([str(iii),str(jjj)],fontsize=14)
        axs[irec,jrec].set_xlabel("Ry",fontsize=12)
        
        #axs[irec,jrec].set_ylabel("SA (g)",fontsize=12)
        #axs[irec,jrec].grid(True,linestyle='-',color='0.25')
       # axs[irec,jrec].semilogy(temp2[0:nnn],(np.exp(gmpeBSSAdata2)+np.exp(gmpeCBdata2)+np.exp(gmpeCYdata2)+np.exp(gmpeASKdata2))/4,label='GMPE Average',color='k')

       # axs[irec,jrec].fill_between(temp2[0:nnn], np.exp(gmpeBSSAdata2), np.exp(gmpeASKdata2),label='GMPE Range',color='lightslategray')

        
       # axs[irec,jrec].plot(temp2[0:nnn],np.exp(a[:,4])/9.8,label='ANN',color='b')
        
        #axs[irec,jrec].plot([longtrain,lattrain],[hypolontrain,hypolattrain])
        
        
        #axs[irec,jrec].legend()
        #plt.grid()
        #fig = plt.figure(figsize=(6,6))
        #axs[irec,jrec] = fig.add_subplot(111)
        #axs[irec,jrec].set_title("Resolution",fontsize=14)
        #axs[irec,jrec].set_xlabel("Longitude",fontsize=12)
        #axs[irec,jrec].set_ylabel("Latitude",fontsize=12)
        axs[irec,jrec].grid(True,linestyle='-',color='0.75')


# scatter with colormap mapping to z value
        ag=axs[irec,jrec].scatter(np.array(Longg).reshape((len(Latt),1)), np.array(Latt).reshape((len(Latt),1)),s=20,c=r, marker = 'o', cmap = cm.jet );
        Lat = np.loadtxt('/Users/kwithers/cyberLat.txt')
        Lon=  np.loadtxt('/Users/kwithers/cyberLon.txt')
        
        index=(Lon>(-120))
        Lon=Lon[index]
        Lat=Lat[index]
        
        axs[irec,jrec].plot(Lon,Lat,'^',markerfacecolor='k',markeredgecolor='k',markersize=0.5)

        #axs[irec,jrec].colorbar(ag)
        #fig.colorbar(ag, ax=axs[3, 0])
plt.show()
        #ax.grid(True,linestyle='-',color='0.25')


#plt.grid()
#ax.grid(True,linestyle='-',color='0.25')        
#plt.xlabel("Ry",fontsize=12)
#plt.ylabel("SA (g)",fontsize=12)
        


#plt.plot(ztest[:,10], np.exp(residtest[:,4]),'.',label='Data')
#add data to slice plots#




r=np.array(temp1).reshape(len(Latt),1)
from matplotlib  import cm

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax.set_title("Resolution",fontsize=14)
ax.set_title("Resolution",fontsize=14)
ax.set_xlabel("Longitude",fontsize=12)
ax.set_ylabel("Latitude",fontsize=12)
ax.grid(True,linestyle='-',color='0.75')


# scatter with colormap mapping to z value
ag=ax.scatter(np.array(Longg).reshape((len(Latt),1)), np.array(Latt).reshape((len(Latt),1)),s=20,c=r, marker = 'o', cmap = cm.jet );
plt.colorbar(ag)
plt.show()


################################
r=np.array(temp2).reshape(len(Latt),1)
from matplotlib  import cm

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax.set_title("Resolution 2",fontsize=14)
ax.set_title("Resolution 2",fontsize=14)
ax.set_xlabel("Longitude",fontsize=12)
ax.set_ylabel("Latitude",fontsize=12)
ax.grid(True,linestyle='-',color='0.75')


# scatter with colormap mapping to z value
ag=ax.scatter(np.array(Longg).reshape((len(Latt),1)), np.array(Latt).reshape((len(Latt),1)),s=20,c=r, marker = 'o', cmap = cm.jet );
plt.colorbar(ag)
plt.show()
