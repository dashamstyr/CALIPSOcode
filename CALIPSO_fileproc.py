# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:49:42 2013

@author: dashamstyr
"""
import os
import CALIPSO_tools as caltools

os.chdir('D:\CALIPSO DATA\April 2013')

#caltools.h5convert_all(verbose=True)

calfiles = caltools.get_files('Select HDF5 files to process', filetype = ('.h5','*.h5'))
    
lats = [30,55]
lons = []
alts = [0,20000]
    
for f in calfiles:
    c1 = caltools.Calipso(h5file = f, raw = True)
    
    c1.window_select(latrange = lats,lonrange = lons,altrange = alts, inplace = True)
    
    c1_masked = c1.feature_mask('aerosol')
    
    c1_dust = caltools.filterandflatten(c1_masked,features = ['Dust','Polluted Dust'], combine = True)
    
    savename = f+'_dust.h5'
    
    try:
        os.chdir('Processed')
    except WindowsError:
        os.mkdir('Processed')
        os.chdir('Processed')
    
    c1_dust.save(savename)
    os.chdir('..')