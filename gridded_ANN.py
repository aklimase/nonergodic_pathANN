#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 16:38:02 2020

@author: aklimasewski
"""
import shapely
import shapely.geometry
import geopy
import geopy.distance
import numpy as np
from preprocessing import transform_dip, readindata, transform_data
import matplotlib.pyplot as plt
import shapely.geometry
import pandas as pd
from matplotlib import cm
import matplotlib as mpl
from sklearn.preprocessing import Normalizer
import random
#grid up residuals


dx=0.1
# lon = np.arange(-119.5,-116.5, dx)
# lat = np.arange(33.25, 35.25, dx)

lon = np.arange(-121,-115.5, dx)
lat = np.arange(32, 37.5, dx)
# longrid, latgrid = np.meshgrid(lon,lat)
# 

latmid = []
lonmid = []
polygons = []
for i in range(len(lon)-1):
    for j in range(len(lat)-1):
        # print(lat[j], lon[i], lat[j+1], lon[i+1])
        polygon_points = [(lon[i], lat[j]), (lon[i], lat[j+1]), (lon[i+1], lat[j+1]), (lon[i+1], lat[j]), (lon[i], lat[j])]
        shapely_poly = shapely.geometry.Polygon(polygon_points)
        polygons.append(shapely_poly)
        latmid.append((lat[j]+lat[j+1])/2.)
        lonmid.append((lon[i]+lon[i+1])/2.)
        
        
# n= len(lon)*len(lat)
# df = pd.DataFrame(np.array([polygons, latmid, lonmid]),
#                    columns=['polygon', 'latmid', 'lonmid'])
         
d = {'polygon': polygons, 'latmid': latmid, 'lonmid': lonmid}
df = pd.DataFrame(data=d)    




train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimasewski/Documents/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimasewski/Documents/data/cybertestyeti10_residfeb.csv', n=6)
  


#choose random subset for fast testing
randindex = random.sample(range(0, len(train_data1)), 1000)
sitelat = train_data1[:,1][randindex]
sitelon = train_data1[:,2][randindex]
evlat = train_data1[:,3][randindex]
evlon = train_data1[:,4][randindex]
target = train_targets1[:,0][randindex]
gridded_targets_sum = np.zeros(df.shape[0])
# gridded_targets_list = np.zeros(shape=(df.shape[0],1))
gridded_targets_list = [ [] for _ in range(df.shape[0]) ]

gridded_counts = np.zeros(df.shape[0])
lenlist = []
            
#loop through each record     
for i in range(len(sitelat)):                         
    line = [(evlon[i], evlat[i]), (sitelon[i], sitelat[i])]
    path=shapely.geometry.LineString(line)
    #loop through each grid cell
    for j in range(len(df)):
        shapely_poly = polygons[j]
        if path.intersects(shapely_poly) == True:
            shapely_line = shapely.geometry.LineString(line)
            intersection_line = list(shapely_poly.intersection(shapely_line).coords)
            if len(intersection_line)== 2:
                coords_1 = (intersection_line[0][1], intersection_line[0][0])
                coords_2 = (intersection_line[1][1], intersection_line[1][0])
                length=geopy.distance.distance(coords_1, coords_2).km
                gridded_targets_sum[j] += (target[i]/length)
                print(gridded_targets_list[j],[target[i]/length]) 
                # gridded_targets_list[j] = np.append(gridded_targets_list[j],[target[i]/length])                # print(target[i],length)
                gridded_targets_list[j].append(target[i]/length)                # print(target[i],length)

                
                gridded_counts[j] += 1
                lenlist.append(length)
            # if gridded_targets_sum[j]<(-100.):
            #         print('index: ', i)
            #         print('length: ', length)
            #         print('target: ', target[i])
            #         print('counts: ', gridded_counts[j])
            #         print('target sum: ', gridded_targets_sum[j])

    
#calculate mean in each cell, unless count is zero
# gridded_mean= np.divide(gridded_targets_sum, gridded_counts, out=np.zeros_like(gridded_targets_sum), where=gridded_counts!=0)

#find mean of norm residual
gridded_targets_list = np.asarray(gridded_targets_list)
gridded_mean=np.asarray([np.mean(gridded_targets_list[i]) for i in range(len(gridded_targets_list))])
#find the cells with no paths (nans)
nan_ind=np.argwhere(np.isnan(gridded_mean)).flatten()
# set nan elements for empty array
# not sureif this isbest orto leaveas a nan
for i in nan_ind:
    gridded_mean[i] =0


#add gridded mean and gridded counts to df
df['counts'] = gridded_counts
df['meantarget'] =gridded_mean

# residuals
# np.asarra
Z = gridded_mean.reshape(len(lat)-1,len(lon)-1)

cbound = max(np.abs(gridded_mean))
cmap = mpl.cm.get_cmap('seismic')
normalize = mpl.colors.Normalize(vmin=-1*cbound, vmax=cbound)
colors = [cmap(normalize(value)) for value in Z]
s_m = mpl.cm.ScalarMappable(cmap = cmap, norm=normalize)
s_m.set_array([])
    
fig, ax = plt.subplots(figsize = (10,8))
plt.pcolormesh(lon, lat, Z, cmap = cmap, norm = normalize) 
plt.scatter(evlon,evlat,marker = '*', s=1, c = 'gray', label = 'event')
plt.scatter(sitelon,sitelat,marker = '^',s=1, c = 'black', label = 'site')
plt.legend(loc = 'lower left')

fig.subplots_adjust(right=0.75)
cbar = plt.colorbar(s_m, orientation='vertical')
cbar.set_label(r'average normalize residual (resid/km)', fontsize = 20)
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
plt.legend(loc = 'lower left')

fig.subplots_adjust(right=0.75)
cbar = plt.colorbar(s_m, orientation='vertical')
cbar.set_label(r'paths per cell', fontsize = 20)
plt.show()




#histogram colorbar
# fig.subplots_adjust(right=0.75)
# cbar_ax = fig.add_axes([0.85, 0.18, 0.1, 0.63])

# plt.ylabel(r'counts per cell', fontsize = 20)
# plt.xlabel(r'path counts', fontsize = 20)

# N, bins, patches = cbar_ax.hist(gridded_counts, orientation='horizontal')
# for bin, patch in zip(bins, patches):
#     color = cmap(normalize(bin))
#     patch.set_facecolor(color)
# plt.show()







transform_method = Normalizer()

aa=transform.fit(train_data1[:,:])
train_data=aa.transform(train_data1)
test_data=aa.transform(test_data1)

# #plot transformed features
# for i in range(len(train_data[0])):
#     plt.figure(figsize =(8,8))
#     plt.title('transformed feature: ' + str(feature_names[i]))
#     plt.hist(train_data[:,i])
#     plt.savefig(folder_path + 'histo_transformedfeature_' + str(feature_names[i]) + '.png')
#     plt.show()

train_targets = train_targets1
test_targets = test_targets1

# y_test = test_targets
y_train = df['meantarget']

x_train = df.drop(['polygon','counts','meantarget'], axis=1)
# x_test = test_data

# x_train_raw = train_data1
# x_test_raw = test_data1



transform = Normalizer()
aa=transform.fit(x_train)
train_data=aa.transform(x_train)
# test_data=aa.transform(x_train)

batch_size = 264

def build_model():
    model = Sequential()
    model.add(layers.Dense(train_data.shape[1],activation='sigmoid', input_shape=(train_data.shape[1],)))

    #no gP layer
    model.add(layers.Dense(1))

    model.compile(optimizer=optimizers.Adam(lr=0.01),loss='mse',metrics=['mae','mse']) 
    return model


model=build_model()

#fit the model
history=model.fit(train_data,y_train,epochs=100,batch_size=batch_size,verbose=1)

# mae_history=history.history['val_mae']
mae_history_train=history.history['mae']
# test_mse_score,test_mae_score,tempp=model.evaluate(x_test,y_test)

pre = model.predict(train_data)
r = np.asarray(y_train)-pre.flatten()

plt.figure()
plt.scatter(np.asarray(y_train),pre.flatten())
plt.xlim()
plt.show()


#%%


# grid predictions
Z = pre.reshape(len(lat)-1,len(lon)-1)

cbound = max(np.abs(pre))
cmap = mpl.cm.get_cmap('seismic')
normalize = mpl.colors.Normalize(vmin=-1*cbound, vmax=cbound)
colors = [cmap(normalize(value)) for value in Z]
s_m = mpl.cm.ScalarMappable(cmap = cmap, norm=normalize)
# s_m.set_array([])
    
fig = plt.figure(figsize = (10,8))
plt.pcolormesh(lon, lat, Z, cmap = cmap, norm = normalize) 
plt.scatter(evlon,evlat,marker = '*', s=1, c = 'gray', label = 'event')
plt.scatter(sitelon,sitelat,marker = '^',s=1, c = 'black', label = 'site')
plt.legend(loc = 'lower left')

fig.subplots_adjust(right=0.75)
cbar_ax = fig.add_axes([0.85, 0.18, 0.1, 0.63])

plt.ylabel(r'counts per cell', fontsize = 20)
plt.xlabel(r'path counts', fontsize = 20)

N, bins, patches = cbar_ax.hist(gridded_counts, orientation='horizontal')
for bin, patch in zip(bins, patches):
    color = cmap(normalize(bin))
    patch.set_facecolor(color)
plt.show()









#%%


Z = np.random.rand(6, 10)
x = np.arange(-0.5, 10, 1)  # len = 11
y = np.arange(4.5, 11, 1)  # len = 7

fig, ax = plt.subplots()
ax.pcolormesh(x, y, Z)
    
    # for j in range(len(polygons)):
    #     shapely_poly = polygons[j]
    #     if path.intersects(shapely_poly) == True:
    #         shapely_line = shapely.geometry.LineString(line)
    #         intersection_line = list(shapely_poly.intersection(shapely_line).coords)
    #         if len(intersection_line)== 2:
    #             print(len(intersection_line))
    #             coords_1 = (intersection_line[0][1], intersection_line[0][0])
    #             coords_2 = (intersection_line[1][1], intersection_line[1][0])
    #             length=geopy.distance.distance(coords_1, coords_2).km
    #             gridded_targets += (target[i]/length)
                

    
 
    
 
polygon_points = [(lon[i], lat[j]), (lon[i], lat[j+1]), (lon[i+1], lat[j+1]), (lon[i+1], lat[j]), (lon[i], lat[j])]

 
    
 
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
