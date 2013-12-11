# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:49:42 2013

@author: dashamstyr
"""
import os
import CALIPSO_tools as caltools

os.chdir('K:\CALIPSO DATA')

#caltools.h5convert_all(verbose=True)

calfiles = caltools.get_files('Select HDF5 files to process', filetype = ('.h5','*.h5'))
    
lats = [0,70]
lons = []
alts = [0,20000]
    
for f in calfiles:
    c1_masked = caltools.Calipso(h5file = f, raw = False)
    
#    c1.window_select(latrange = lats,lonrange = lons,altrange = alts, inplace = True)
#    
#    c1_masked = c1.feature_mask('aerosol')
    
    c1_smoke = caltools.filterandflatten(c1_masked,features = ['Clean Cont.','Polluted Cont.'], combine = True)
    
    filename = '.'.join(os.path.basename(f).split('.')[:-1])
    
#    maskname = filename+'_aerosol.h5'
#    dustname = filename+'_dust.h5'    
    smokename = filename+'_continental.h5'
    
    savedir = 'K:\\CALIPSO DATA\\Processed-Full\\Continental\\'
#    c1_masked.save(maskname)
#    c1_dust.save(dustname)
    c1_smoke.save(savedir+smokename)