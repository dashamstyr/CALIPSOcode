import pandas as pan
import CALIPSO_tools as caltools
from CALIPSO_binit import binit
import numpy as np
import matplotlib.pyplot as plt
import time
#from fastbinit import hist_latlon as fasthist
from CALIPSO_hist2d import hist_latlon as slowhist
from CALIPSO_hist2d import hist_class as slow_class


   

if __name__=="__main__":

    from matplotlib.colors import Normalize
    from mpl_toolkits.basemap import Basemap
    from matplotlib import cm
    import os
    
    cmap=cm.YlOrBr
    cmap.set_over('r')
    cmap.set_under('w')
    vmin= 0
    vmax= 20000
    the_norm=Normalize(vmin=vmin,vmax=vmax,clip=False)
    os.chdir('K:\CALIPSO DATA\April 2013\Processed')
    filenames = caltools.get_files('Select files for Lat-Lon Histogramming', filetype = ('.h5','*.h5'))
    
    df_cal = []
    for f in filenames:
        temp = caltools.Calipso(h5file = f,raw = False)
        if not isinstance(df_cal,pan.DataFrame):
            df_cal = temp.VFM
        else:
            df_cal = df_cal.append(temp.VFM)
    
        
    fullLats = df_cal.index.get_level_values('Lats').values
    fullLons = df_cal.index.get_level_values('Lons').values
    
    fullfeatures = df_cal.values
    
    minlat = 30
    maxlat = 55
    
    minlon = 75
    maxlon = -115
    
    numlatbins = 20
    numlonbins = 100
    
    minlon = caltools.to360(minlon)
    maxlon = caltools.to360(maxlon)
    
    fullLons = [caltools.to360(l) for l in fullLons]
        
    bin_lats=binit(minlat,maxlat,numlatbins,-999,-888)
    bin_lons=binit(minlon,maxlon,numlonbins,-999,-888)

    #slow version
    tic=time.clock()
    slowlat_grid,slowlon_grid,slowrad_grid=slowhist(fullLats,fullLons,fullfeatures,bin_lats,bin_lons)
    slowtime=time.clock() - tic

    #fast version
#    tic=time.clock()
#    #lat_grid,lon_grid,rad_grid=fasthist(partLats,partLons,partRads,bin_lats,bin_lons)
#    the_hist=slow_class(fullLats,fullLons,fullfeatures,bin_lats,bin_lons)
#    lat_grid,lon_grid,rad_grid=the_hist.calc_sum()
#    fasttime=time.clock() - tic
#    ## print "slow and fast plus speedup: ",slowtime,fasttime,slowtime/fasttime
#    np.testing.assert_almost_equal(slowrad_grid,rad_grid)
    lon_centers=bin_lons.get_centers()
    lon_centers=[caltools.to180(l) for l in lon_centers]
    lat_centers=bin_lats.get_centers()
    
    lon_edges=bin_lons.get_edges()
    lon_edges=[caltools.to180(l) for l in lon_edges]
    lat_edges=bin_lats.get_edges()
    fig1=plt.figure(1)
    fig1.clf()
    axis1=fig1.add_subplot(111)
    m = Basemap(projection='mill',llcrnrlat=30,urcrnrlat=55,\
            llcrnrlon=75,urcrnrlon=180,resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    # draw parallels and meridians.
    m.drawparallels(np.arange(30.,60.,5.))
    m.drawmeridians(np.arange(75.,195.,15.))
#    m = Basemap(projection='mill',llcrnrlat=lat_edges[0],urcrnrlat=lat_edges[-1],\
#            llcrnrlon=lon_edges[-1],urcrnrlon=lon_edges[0],\
#            resolution='l')
#    m.drawcoastlines()
#    m.drawcountries()
#    m.drawparallels(np.arange(lat_edges[0],lat_edges[-1],5))
#    m.drawmeridians(np.arange(lon_edges[0],lon_edges[-1],15))
    x,y = m(slowlon_grid,slowlat_grid)
    
    csteps = 200
    clevs = np.arange(vmin,vmax,csteps)
    im=axis1.contourf(x,y,slowrad_grid,csteps,cmap=cmap, norm=the_norm)
    axis1.set_title('CALIPSO Aerosol Counts: Dust+Polluted Dust')
    plt.show()


