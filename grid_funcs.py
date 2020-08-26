#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:35:56 2020

@author: aklimasewski

contains gridding functions
"""

def create_grid(latmin=32,latmax=37.5,lonmin=-121,lonmax=-115.5,dx=0.05):
    '''
    Parameters
    ----------
    latmin: float minimum value of grid latitude, default 32 N
    latmax: float maximum value of grid latitude, default 37.5 N
    lonmin: float minimum value of grid longitude, default -121 W
    lonmax: float maximum value of grid longitude, default -115.5 W
    dx: float grid spacing in degrees default is 0.05.

    Returns
    df: pandas dataframe of shapely polgons and midpoint of each grid cell in lat, lon
    lon: 1D numpy array of longitude grid vertices
    lat: 1D numpy array of latitude grid vertices
    '''

    import numpy as np
    import shapely
    import shapely.geometry
    import pandas as pd
    import geopy.distance

    
    # dx=0.1
    lon = np.arange(lonmin, lonmax, dx)
    lat = np.arange(latmin, latmax, dx)
    
    latmid = []
    lonmid = []
    polygons = []
    for i in range(len(lon)-1):
        for j in range(len(lat)-1):
            polygon_points = [(lon[i], lat[j]), (lon[i], lat[j+1]), (lon[i+1], lat[j+1]), (lon[i+1], lat[j]), (lon[i], lat[j])]
            shapely_poly = shapely.geometry.Polygon(polygon_points)
            polygons.append(shapely_poly)
            latmid.append((lat[j]+lat[j+1])/2.)
            lonmid.append((lon[i]+lon[i+1])/2.)
               
    d = {'polygon': polygons, 'latmid': latmid, 'lonmid': lonmid}
    df = pd.DataFrame(data=d)    
    return df, lon, lat

def create_grid_square(latmin=32,latmax=37.5,lonmin=-121,lonmax=-115.5,dx=0.24, dy=0.2):
    '''
    Parameters
    ----------
    latmin: float minimum value of grid latitude, default 32 N
    latmax: float maximum value of grid latitude, default 37.5 N
    lonmin: float minimum value of grid longitude, default -121 W
    lonmax: float maximum value of grid longitude, default -115.5 W
    dx: float grid spacing in degrees default is 0.05.

    Returns
    df: pandas dataframe of shapely polgons and midpoint of each grid cell in lat, lon
    lon: 1D numpy array of longitude grid vertices
    lat: 1D numpy array of latitude grid vertices
    '''

    import numpy as np
    import shapely
    import shapely.geometry
    import pandas as pd
    import geopy
    import geopy.distance
    
    # dx=0.1
    lon = np.arange(lonmin, lonmax, dx)
    lat = np.arange(latmin, latmax, dy)
    
    latmidi = round(len(lat)/2.)
    lonmidi = round(len(lon)/2.)
    
    coords_1 = (lat[latmidi],lon[lonmidi])
    coords_2 = (lat[latmidi],lon[lonmidi+1])
    coords_3 = (lat[latmidi+1],lon[lonmidi])

    lengthx=geopy.distance.distance(coords_1,coords_2).km
    lengthy=geopy.distance.distance(coords_1,coords_3).km
    
    print('size estimate x: ' , lengthx, ' km')
    print('size estimate y: ', lengthy, ' km')
    
    latmid = []
    lonmid = []
    polygons = []
    for i in range(len(lon)-1):
        for j in range(len(lat)-1):
            polygon_points = [(lon[i], lat[j]), (lon[i], lat[j+1]), (lon[i+1], lat[j+1]), (lon[i+1], lat[j]), (lon[i], lat[j])]
            shapely_poly = shapely.geometry.Polygon(polygon_points)
            polygons.append(shapely_poly)
            latmid.append((lat[j]+lat[j+1])/2.)
            lonmid.append((lon[i]+lon[i+1])/2.)
               
    d = {'polygon': polygons, 'latmid': latmid, 'lonmid': lonmid}
    df = pd.DataFrame(data=d)    
    return df, lon, lat
    
def grid_data(train_data1, train_targets1, df):
    '''
    Parameters
    ----------
    train_data1: numpy array of training data for gridding
    train_targets1: numpy array of testing targets for gridding
    df: pandas dataframe of shapely polgons and midpoint of each grid cell in lat, lon
    nsamples: number of samples to randomly choose (for fast testing)
    
    Returns
    hypoR: numpy array of hypocentral distance for sample
    sitelat: numpy array of site latitude for sample
    sitelon: numpy array of site longitude for sample
    evlat: numpy array of event latitude for sample
    evlon: numpy array of event longitude for sample
    target: numpy array of targets for sample
    gridded_targetsnorm_list: 2D list of targets normalized by path length and multiplied by distance per cell
    gridded_counts: 2D list of path counts per grid cell
    '''
    import shapely
    import shapely.geometry
    import numpy as np
    import geopy
    import random
    import geopy.distance

    
    # randindex = random.sample(range(0, len(train_data1)), nsamples)
    
    # hypoR = train_data1[:,0][randindex]
    # sitelat = train_data1[:,1][randindex]
    # sitelon = train_data1[:,2][randindex]
    # evlat = train_data1[:,3][randindex]
    # evlon = train_data1[:,4][randindex]
    # target = train_targets1[:][randindex]
    
    
    hypoR = train_data1[:,0]
    sitelat = train_data1[:,1]
    sitelon = train_data1[:,2]
    evlat = train_data1[:,3]
    evlon = train_data1[:,4]
    target = train_targets1[:]
    
    normtarget = target / hypoR[:, np.newaxis]
    gridded_targetsnorm_list = [ [] for _ in range(df.shape[0]) ]
    
    gridded_counts = np.zeros(df.shape[0])
    lenlist = []
    
    #loop through each record     
    for i in range(len(sitelat)):                       
        line = [(evlon[i], evlat[i]), (sitelon[i], sitelat[i])]
        path=shapely.geometry.LineString(line)
        #loop through each grid cell
        if (i % 1000) == 0:
        	print('record: ', str(i))
        for j in range(len(df)):
            shapely_poly = df['polygon'][j]
            if path.intersects(shapely_poly) == True:
                shapely_line = shapely.geometry.LineString(line)
                intersection_line = list(shapely_poly.intersection(shapely_line).coords)
                if len(intersection_line)== 2:
                    coords_1 = (intersection_line[0][1], intersection_line[0][0])
                    coords_2 = (intersection_line[1][1], intersection_line[1][0])
                    length=geopy.distance.distance(coords_1, coords_2).km
                    gridded_targetsnorm_list[j].append(normtarget[i]*length)          
                    gridded_counts[j] += 1
                    lenlist.append(length)
                
    return hypoR, sitelat, sitelon, evlat, evlon, target, gridded_targetsnorm_list, gridded_counts
    




def mean_grid(gridded_targetsnorm_list,gridded_targetsnorm_list_test,gridded_counts,gridded_counts_test, df,folder_path):
    import numpy as np
    import pandas as pd
    
    period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]

    
    #find mean of norm residual
    gridded_targetsnorm_list = np.asarray(gridded_targetsnorm_list)
    
    griddednorm_mean=np.zeros((len(gridded_targetsnorm_list),10))
    for i in range(len(gridded_targetsnorm_list)):
        griddednorm_mean[i] = np.mean(gridded_targetsnorm_list[i],axis=0)
    
    nan_ind=np.argwhere(np.isnan(griddednorm_mean)).flatten()
    for i in nan_ind:
        griddednorm_mean[i] = 0
        
    gridded_targetsnorm_list_test = np.asarray(gridded_targetsnorm_list_test)
    
    griddednorm_mean_test=np.zeros((len(gridded_targetsnorm_list_test),10))
    for i in range(len(gridded_targetsnorm_list_test)):
        griddednorm_mean_test[i] = np.mean(gridded_targetsnorm_list_test[i],axis=0)
    
    #find the cells with no paths (nans)
    nan_ind=np.argwhere(np.isnan(griddednorm_mean_test)).flatten()
    for i in nan_ind:
        griddednorm_mean_test[i] = 0
        
    df_save = df
    meandict = {'T' + str(period[i]): griddednorm_mean[:,i] for i in range(len(period))}
    meandict_test = {'T' + str(period[i]) + 'test': griddednorm_mean_test[:,i] for i in range(len(period))}
    d2 = {'griddedcounts': gridded_counts, 'griddedcountstest': gridded_counts_test}
    d2.update(meandict)
    d2.update(meandict_test)
    df2 = pd.DataFrame(data=d2)   
    # df_save.append(df2)
    df_save=pd.concat([df_save,df2],axis=1)
    df_save.to_csv(folder_path + 'griddedvalues.csv')
    
    return griddednorm_mean, griddednorm_mean_test

def save_gridded_targets(griddednorm_mean,griddednorm_mean_test,gridded_counts,gridded_counts_test,df, folder_path):
    import pandas as pd
    import pandas as pd
    
    period=[10,7.5,5,4,3,2,1,0.5,0.2,0.1]

    
    df_save = df
    meandict = {'T' + str(period[i]): griddednorm_mean[:,i] for i in range(len(period))}
    meandict_test = {'T' + str(period[i]) + 'test': griddednorm_mean_test[:,i] for i in range(len(period))}
    d2 = {'griddedcounts': gridded_counts, 'griddedcountstest': gridded_counts_test}
    d2.update(meandict)
    d2.update(meandict_test)
    df2 = pd.DataFrame(data=d2)   
    # df_save.append(df2)
    df_save=pd.concat([df_save,df2],axis=1)
    df_save.to_csv(folder_path + 'griddedvalues.csv')
    
    
def grid_points(data,df,name, folder_path):
    import pandas as pd
    import shapely
    import shapely.geometry
    import numpy as np
    import geopy
    import geopy.distance
    from shapely.geometry import Point
    
    sitelat = data[:,13]
    sitelon = data[:,14]
    evlat = data[:,15]
    evlon = data[:,16]
    midlat = data[:,17]
    midlon = data[:,18]
    
    gridded_num = np.zeros((len(sitelat),3))#event, mid, site'
    gridded_mid = np.zeros((len(sitelat),6))#event, mid, site'
    
    gridded_counts = np.zeros((df.shape[0],3))
    
    
    #loop through each record     
    for i in range(len(sitelat)):    
        event = shapely.geometry.Point(evlon[i], evlat[i])
        mid = shapely.geometry.Point(midlon[i], midlat[i])
        site = shapely.geometry.Point(sitelon[i], sitelat[i])
        #loop through each grid cell
        #add a 1 for the column if event, mid, site in the cell
        if (i % 1000) == 0:
        	print('record: ', str(i))
        for j in range(len(df)):
            shapely_poly = df['polygon'][j]
            if event.within(shapely_poly) == True:
                gridded_mid[i,0:2] = [df['latmid'][j],df['lonmid'][j]]
                gridded_num[i,0] = j
                gridded_counts[j,0] += 1
            if mid.within(shapely_poly) == True:
                gridded_mid[i,2:4] = [df['latmid'][j],df['lonmid'][j]]
                gridded_num[i,1] = j
                gridded_counts[j,1] +=1
            if site.within(shapely_poly) == True:
                gridded_mid[i,4:6] = [df['latmid'][j],df['lonmid'][j]]
                gridded_num[i,2] = j
                gridded_counts[j,2] += 1
    
    df_out = pd.DataFrame(gridded_mid, columns=['eventlat','eventlon','midlat','midlon','sitelat','sitelon'])   
    df_out.to_csv(folder_path + 'gridpointslatlon_' + name + '.csv')   
    
    df_out = pd.DataFrame(gridded_num, columns=['event','mid','site'])   
    df_out.to_csv(folder_path + 'gridpoints_' + name + '.csv')
    
    df_out = pd.DataFrame(gridded_counts, columns=['event','mid','site'])   
    df_out.to_csv(folder_path + 'counts_' + name + '.csv')
    
    
def avgpath_resid(df, folder_path,savename):
    import shapely
    import shapely.geometry
    from preprocessing import readindata
    import numpy as np
    import geopy
    import geopy.distance
    import pandas as pd
    
    if savename == 'train':
        gridcells = df['polygon']
        targets = df[['T10','T7.5','T5','T4','T3','T2','T1','T0.5','T0.2','T0.1']]
        list_wkt = df['polygon']
        list_polygons =  [shapely.wkt.loads(poly) for poly in list_wkt]
    else:
        gridcells = df['polygon']
        targets = df[['T10test','T7.5test','T5test','T4test','T3test','T2test','T1test','T0.5test','T0.2test','T0.1test']]
        list_wkt = df['polygon']
        list_polygons =  [shapely.wkt.loads(poly) for poly in list_wkt]

    n = 6
    train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimase/Documents/USGS/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimase/Documents/USGS/data/cybertestyeti10_residfeb.csv', n = n)
    
    hypoR = train_data1[:,0]
    sitelat = train_data1[:,1]
    sitelon = train_data1[:,2]
    evlat = train_data1[:,3]
    evlon = train_data1[:,4]
    target = train_targets1[:]

    
    path_target_sum = np.zeros((len(hypoR),10))#length of number of records

    #loop through each record     
    for i in range(len(sitelat)):                       
        line = [(evlon[i], evlat[i]), (sitelon[i], sitelat[i])]
        path=shapely.geometry.LineString(line)
        #loop through each grid cell
        if (i % 1000) == 0:
            print('record: ', str(i))
        pathsum = 0
        for j in range(len(list_polygons)):
            # shapely_poly = df['polygon'][j].split('(')[2].split(')')[0]
            # polygon_points = [(lon[i], lat[j]), (lon[i], lat[j+1]), (lon[i+1], lat[j+1]), (lon[i+1], lat[j]), (lon[i], lat[j])]
            shapely_poly = shapely.geometry.Polygon(list_polygons[j])
            if path.intersects(shapely_poly) == True:
                shapely_line = shapely.geometry.LineString(line)
                intersection_line = list(shapely_poly.intersection(shapely_line).coords)
                if len(intersection_line)== 2:
                    coords_1 = (intersection_line[0][1], intersection_line[0][0])
                    coords_2 = (intersection_line[1][1], intersection_line[1][0])
                    length=geopy.distance.distance(coords_1, coords_2).km
                    
                    pathsum += length*np.asarray(targets.iloc[j])
            
        path_target_sum[i] = (pathsum)          
                    
    # dictout = {['T10','T7.5','T5','T4','T3','T2','T1','T0.5','T0.2','T0.1']:path_target_sum}
    df_out = pd.DataFrame(path_target_sum, columns=['T10','T7.5','T5','T4','T3','T2','T1','T0.5','T0.2','T0.1'])   
    # df_save.append(df2)
    df_out.to_csv(folder_path + 'avgrecord_targets_' + savename+ '.csv')     

    return path_target_sum       

 
def plot_counts(gridded_counts,data, name, folder_path):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    
    colname = ['event','midpoint','site']
    
    sitelat = data[:,13]
    sitelon = data[:,14]
    evlat = data[:,15]
    evlon = data[:,16]
    midlat = data[:,17]
    midlon = data[:,18]
    
    for i in range(len(gridded_counts[0])):
        Z = gridded_counts[:,i].reshape(len(lat)-1,len(lon)-1).T
        
        cbound = np.max(np.abs(Z))
        cmap = mpl.cm.get_cmap('Greens')
        normalize = mpl.colors.Normalize(vmin=0, vmax=cbound)
        colors = [cmap(normalize(value)) for value in Z]
        s_m = mpl.cm.ScalarMappable(cmap = cmap, norm=normalize)
        s_m.set_array([])
        
        fig, ax = plt.subplots(figsize = (10,8))
        plt.pcolormesh(lon, lat, Z, cmap = cmap, norm = normalize) 
        plt.scatter(evlon,evlat,marker = '*', s=0.2, c = 'gray', label = 'event', alpha = 0.02)
        plt.scatter(sitelon,sitelat,marker = '^',s=0.2, c = 'black', label = 'site', alpha = 0.02)
        plt.xlim(min(lon),max(lon))
        plt.ylim(min(lat),max(lat))
        plt.title(colname)
        plt.legend(loc = 'lower left')
        
        fig.subplots_adjust(right=0.75)
        cbar = plt.colorbar(s_m, orientation='vertical')
        cbar.set_label(colname[i] + ' counts', fontsize = 20)
        plt.savefig(folder_path + colname[i] + name + '_counts.png')
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