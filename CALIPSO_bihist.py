
def get_latlonhist(dohist = True, doall= False, histmethod = 'sum', filenames = [],latrange = [], lonrange = [], numlatbins = 20, 
                   numlonbins = 100, savefile = []):
    import pandas as pan
    import numpy as np
    import sys
    
    sys.path.append('C:\Users\dashamstyr\Dropbox\Python_Scripts\GIT_Repos\CALIPSOcode')
    
    import CALIPSO_tools as caltools
    from CALIPSO_binit import binit
#    from CALIPSO_hist2d import hist_latlon as slowhist
    from CALIPSO_hist2d import hist_class as slow_class
    
    if dohist:
        if not filenames:
            if doall:
                filedir = caltools.set_dir('Select Directory for files to histogram')
                os.chdir(filedir)
                filenames = [f for f in os.listdir('.') if (os.path.isfile(f)) & f.endswith('.h5')]
            else:
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
        
        if latrange:
            minlat = latrange[0]
            maxlat = latrange[1]
        else:
            minlat = np.min(fullLats)
            maxlat = np.max(fullLats)
        
        if lonrange:
            minlon = lonrange[0]
            maxlon = lonrange[1]
        else:
            minlon = np.min(fullLons)
            maxlon = np.max(fullLons)
        
        minlon = caltools.to360(minlon)
        maxlon = caltools.to360(maxlon)        
        fullLons = [caltools.to360(l) for l in fullLons]
            
        bin_lats=binit(minlat,maxlat,numlatbins,-999,-888)
        bin_lons=binit(minlon,maxlon,numlonbins,-999,-888)
    
        #slow version
    #    tic=time.clock()
    #    slowlat_grid,slowlon_grid,slowrad_grid=slowhist(fullLats,fullLons,fullfeatures,bin_lats,bin_lons)
    #    slowtime=time.clock() - tic
    
        #fast version
#        lat_grid,lon_grid,rad_grid=fasthist(fullLats,fullLons,fullfeatures,bin_lats,bin_lons)
        the_hist=slow_class(fullLats,fullLons,fullfeatures,bin_lats,bin_lons)
        
        if histmethod == 'mean':
            lat_grid,lon_grid,rad_grid=the_hist.calc_mean()
        elif histmethod == 'sum':
            lat_grid,lon_grid,rad_grid=the_hist.calc_sum()
        
        df_lat = pan.DataFrame(lat_grid)
        df_lon = pan.DataFrame(lon_grid)
        df_rad = pan.DataFrame(rad_grid)
        
        #pad out value on longitude axis to avoid streaks
        
        df_rad.fillna(method='ffill',limit=1, axis=1, inplace=True)
        df_rad.fillna(method='bfill',limit=1, axis=1, inplace=True)
        
        ser_binlats = pan.Series([bin_lats])
        ser_binlons = pan.Series([bin_lons])
        
        if savefile:
            store = pan.HDFStore(savefile)
            store['lat_grid'] = df_lat
            store['lon_grid'] = df_lon
            store['rad_grid'] = df_rad
            store['bin_lats'] = ser_binlats
            store['bin_lons'] = ser_binlons
            store.close()
        
        return lat_grid,lon_grid,rad_grid,bin_lats,bin_lons
    else:
        filename = caltools.get_files('Select pre-processed file for Lat-Lon Histogramming', filetype = ('.h5','*.h5'))
        lat_grid = pan.read_hdf(filename[0],'lat_grid').values
        lon_grid = pan.read_hdf(filename[0],'lon_grid').values
        rad_grid = pan.read_hdf(filename[0],'rad_grid').values
        bin_lats = pan.read_hdf(filename[0],'bin_lats').values[0]
        bin_lons = pan.read_hdf(filename[0],'bin_lons').values[0]
        
        return lat_grid,lon_grid,rad_grid,bin_lats,bin_lons
    
def plot_latlonhist(lat_grid,lon_grid,rad_grid,bin_lats,bin_lons,colorrange=[],
                    lat_window=[],lon_window=[],latstep=[], lonstep=[],cnum=50,
                    dolog=False,plottype = [], savefile=[]):
                        
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.basemap import Basemap
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib import cm
    
    import sys
    
    sys.path.append('C:\Users\dashamstyr\Dropbox\Python_Scripts\GIT_Repos\CALIPSOcode')
    import CALIPSO_tools as caltools
    
    if plottype is 'dust':
        cmap=cm.YlOrBr
        cmap.set_over('r')
        cmap.set_under('w')
    elif plottype is 'smoke':
        cmap=cm.Blues
        cmap.set_over('r')
        cmap.set_under('w')   
    elif plottype is 'continental':
        cmap=cm.Greens
        cmap.set_over('r')
        cmap.set_under('w') 
    else:
        cmap=cm.Binary
        cmap.set_over('r')
        cmap.set_under('w')
        
    if colorrange:
        vmin= colorrange[0]
        vmax= colorrange[1]
    else:
        vmin = 0
        vmax = np.max(rad_grid)
    
    if dolog:
        vmin = np.log(vmin)
        vmax = np.log(vmax)
        rad_grid = np.log(rad_grid)
        
    the_norm=Normalize(vmin=vmin,vmax=vmax,clip=False)
    
    lon_centers=bin_lons.get_centers()
    lon_centers=[caltools.to180(l) for l in lon_centers]
    lat_centers=bin_lats.get_centers()
    
    lon_edges=bin_lons.get_edges()
    lon_edges=[caltools.to180(l) for l in lon_edges]
    lat_edges=bin_lats.get_edges()
    
    if lat_window:
        minlat = lat_window[0]
        maxlat = lat_window[1]
    else:
        minlat = np.min(lat_edges)
        maxlat = np.max(lat_edges)
        
    if lon_window:
        minlon = lon_window[0]
        maxlon = lon_window[1]        
    else:
        minlon = np.min(lon_edges)
        maxlon = np.max(lon_edges)
    
    if not latstep:
        latstep = (maxlat-minlat)/20
    
    if not lonstep:
        lonstep = (caltools.to360(maxlon)-caltools.to360(minlon))/20
        
    lon_0 = caltools.to180(np.mean([caltools.to360(l) for l in [minlon,maxlon]]))
    lat_0 = np.mean([minlat,maxlat])
    
    fig1=plt.figure(1)
    fig1.clf()
    axis1=fig1.add_subplot(111)
    m = Basemap(projection='stere',llcrnrlat=minlat,urcrnrlat=maxlat,\
            llcrnrlon=minlon,urcrnrlon=maxlon,lat_0=lat_0,lon_0=lon_0,resolution='l')
    m.drawcoastlines()
    m.drawcountries()
    # draw parallels and meridians.
    m.drawparallels(np.arange(minlat,maxlat+latstep,latstep))
    m.drawmeridians(np.arange(minlon,maxlon+lonstep,lonstep))
    x,y = m(lon_grid,lat_grid)
    
    im = m.pcolormesh(x,y,rad_grid,cmap=cmap, norm=the_norm)
    
    divider = make_axes_locatable(axis1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    
    cb = plt.colorbar(im, cax=cax)
    the_label=cb.ax.set_ylabel('Counts',rotation=270)
    plt.show()
    
    if savefile:
        plt.savefig(savefile)
        
   

if __name__=="__main__":
    import os

    os.chdir('D:\CALIPSO DATA\Processed-Full')
    
    latrange=[25,55]
    lonrange=[75,-100]
    numlatbins = 30
    numlonbins=175
    
    savefilename = 'D:\CALIPSO DATA\Processed-Full\Calipso_continental_0422-0427-mean.h5'
    figname = savefilename.split('.')[0]+'.png'
    
    lat_grid,lon_grid,rad_grid,bin_lats,bin_lons = get_latlonhist(dohist = False, doall = False, histmethod = 'mean', 
                                                                  filenames = [], latrange=latrange, lonrange=lonrange, 
                                                                  savefile=savefilename)
    
    colorrange = [0,50]
    lat_win = [0,55]
    lon_win = [75,-100]
    lat_step = 15
    lon_step = 15
    plot_type = 'continental'
    
    
    plot_latlonhist(lat_grid,lon_grid,rad_grid,bin_lats,bin_lons,colorrange=colorrange,
                    lat_window=lat_win,lon_window=lon_win,latstep=lat_step, lonstep=lon_step,cnum=50,
                    dolog=False,plottype = plot_type, savefile=figname)
    
    



