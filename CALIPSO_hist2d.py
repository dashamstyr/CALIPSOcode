import numpy as np

def hist_latlon(lats,lons,rads,bin_lats,bin_lons):

    lat_count,lat_index,lowlats,highlats=bin_lats.do_bins(lats)    
    lon_count,lon_index,lowlons,highlons=bin_lons.do_bins(lons)    
    out_vals=np.empty([bin_lats.numbins,bin_lons.numbins],dtype=np.object)
    for row in range(bin_lats.numbins):
        for col in range(bin_lons.numbins):
            out_vals[row,col]=list()

    for data_index in range(lat_index.size):
        grid_row=lat_index[data_index]
        grid_col=lon_index[data_index]
        if grid_row < 0 or grid_col < 0:
            continue
        try:
            out_vals[grid_row,grid_col].append(data_index)
        except:
            print "trouble at: data_index {0:d}, gives {1:d},{2:d}".format(data_index,grid_row,grid_col)

    rad_grid=np.empty_like(out_vals,dtype=np.float)
    lat_grid=np.empty_like(out_vals,dtype=np.float)
    lon_grid=np.empty_like(out_vals,dtype=np.float)
    rows,cols=rad_grid.shape
    rads=rads.ravel()
    for the_row in range(rows):
        for the_col in range(cols):
            rad_list=out_vals[the_row,the_col]
            if len(rad_list)==0:
                rad_grid[the_row,the_col]=np.nan
                lat_grid[the_row,the_col]=np.nan
                lon_grid[the_row,the_col]=np.nan
            else:
                try:
                    rad_vals=np.take(rads,rad_list)
                    lat_vals=np.take(lats,rad_list)
                    lon_vals=np.take(lons,rad_list)
                    rad_grid[the_row,the_col]=np.sum(rad_vals)
                    lat_grid[the_row,the_col]=np.mean(lat_vals)                    
                    lon_grid[the_row,the_col]=np.mean(lon_vals)                    
                except IndexError:
                    print "oops: ",rad_list
    return lat_grid,lon_grid,rad_grid


class hist_class(object):

    def __init__(self,lats,lons,rads,bin_lats,bin_lons):
        self.lats=lats
        self.lons=lons
        self.rads=rads
        self.bin_lats=bin_lats
        self.bin_lons=bin_lons
        self.lat_count,self.lat_index,self.lowlats,self.highlats=bin_lats.do_bins(lats)    
        self.lon_count,self.lon_index,self.lowlons,self.highlons=bin_lons.do_bins(lons)    
        self.out_vals=np.empty([bin_lats.numbins,bin_lons.numbins],dtype=np.object)

    def calc_vals(self):

        numlatbins=self.bin_lats.numbins
        numlonbins=self.bin_lons.numbins

        for row in range(numlatbins):
            for col in range(numlonbins):
                self.out_vals[row,col]=list()
        num_datapts=self.lats.size
        for data_index in range(num_datapts):
            grid_row=self.lat_index[data_index]
            grid_col=self.lon_index[data_index]
            if grid_row < 0 or grid_col < 0:
                continue
            else:
                self.out_vals[grid_row,grid_col].append(data_index)

    def calc_mean(self):
        
        self.calc_vals()
        rad_grid=np.empty_like(self.out_vals,dtype=np.float32)
        lat_grid=np.empty_like(self.out_vals,dtype=np.float32)
        lon_grid=np.empty_like(self.out_vals,dtype=np.float32)
        rows,cols=self.out_vals.shape
        
        for row in range(rows):
            for col in range(cols):
                rad_list=self.out_vals[row,col]
                if len(rad_list)==0:
                    rad_grid[row,col]=np.nan
                    lat_grid[row,col]=np.nan
                    lon_grid[row,col]=np.nan
                else:
                    rad_vals=np.take(self.rads,rad_list)
                    lat_vals=np.take(self.lats,rad_list)
                    lon_vals=np.take(self.lons,rad_list)
                    rad_grid[row,col]=np.mean(rad_vals)
                    lat_grid[row,col]=np.mean(lat_vals)                    
                    lon_grid[row,col]=np.mean(lon_vals)                    

        return np.asarray(lat_grid),np.asarray(lon_grid),np.asarray(rad_grid)

    def calc_sum(self):
        
        self.calc_vals()
        rad_grid=np.empty_like(self.out_vals,dtype=np.float32)
        lat_grid=np.empty_like(self.out_vals,dtype=np.float32)
        lon_grid=np.empty_like(self.out_vals,dtype=np.float32)
        rows,cols=self.out_vals.shape
        
        for row in range(rows):
            for col in range(cols):
                rad_list=self.out_vals[row,col]
                if len(rad_list)==0:
                    rad_grid[row,col]=np.nan
                    lat_grid[row,col]=np.nan
                    lon_grid[row,col]=np.nan
                else:
                    rad_vals=np.take(self.rads,rad_list)
                    lat_vals=np.take(self.lats,rad_list)
                    lon_vals=np.take(self.lons,rad_list)
                    rad_grid[row,col]=np.sum(rad_vals)
                    lat_grid[row,col]=np.mean(lat_vals)                    
                    lon_grid[row,col]=np.mean(lon_vals)                    

        return np.asarray(lat_grid),np.asarray(lon_grid),np.asarray(rad_grid)