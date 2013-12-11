# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:42:41 2013

@author: Paul Cottle
"""

def set_dir(titlestring):
    from Tkinter import Tk
    import tkFileDialog
     
    # Make a top-level instance and hide since it is ugly and big.
    root = Tk()
    root.withdraw()
    
    # Make it almost invisible - no decorations, 0 size, top left corner.
    root.overrideredirect(True)
    root.geometry('0x0+0+0')
#    
    # Show window again and lift it to top so it can get focus,
    # otherwise dialogs will end up behind the terminal.
    root.deiconify()
    root.attributes("-topmost",1)
    root.focus_force()
    
    file_path = tkFileDialog.askdirectory(parent=root,title=titlestring)
     
    if file_path != "":
       return str(file_path)
     
    else:
       print "you didn't open anything!"
    
    # Get rid of the top-level instance once to make it actually invisible.
    root.destroy() 
     

       
    
     
def get_files(titlestring,filetype = ('.hdf','*.hdf')):
    from Tkinter import Tk
    import tkFileDialog
    import re
     
     
    # Make a top-level instance and hide since it is ugly and big.
    root = Tk()
    root.withdraw()
    
    # Make it almost invisible - no decorations, 0 size, top left corner.
    root.overrideredirect(True)
    root.geometry('0x0+0+0')
#    
    # Show window again and lift it to top so it can get focus,
    # otherwise dialogs will end up behind the terminal.
    root.deiconify()
    root.attributes("-topmost",1)
    root.focus_force()

    filenames = []
     
    filenames = tkFileDialog.askopenfilename(title=titlestring, filetypes=[filetype],multiple='True')
    
    #do nothing if already a python list
    if filenames == "": 
        print "You didn't open anything!"  
        return
        
    if isinstance(filenames,list): return filenames

    #http://docs.python.org/library/re.html
    #the re should match: {text and white space in brackets} AND anynonwhitespacetokens
    #*? is a non-greedy match for any character sequence
    #\S is non white space

    #split filenames string up into a proper python list
    result = re.findall("{.*?}|\S+",filenames)

    #remove any {} characters from the start and end of the file names
    result = [ re.sub("^{|}$","",i) for i in result ]     
    return result

    
    root.destroy()

def h5convert(h4file, verbose = False):
    import subprocess
    
    cmd_arg = 'h4toh5convert '+h4file    
    
    out = subprocess.Popen(cmd_arg, shell=True)
    
    if verbose:
        print out

def h5convert_all(verbose = False):
    import os
    import subprocess
    import glob
    
    olddir = os.getcwd()
    newdir = set_dir('Select Folder to Convert')
    
    os.chdir(newdir)
    
    h4files = glob.glob('*.hdf')
    
    for f in h4files:
        cmd_arg = 'h4toh5convert '+f
        out = subprocess.Popen(cmd_arg, shell=True)
        
        if verbose:
            print out
    
    os.chdir(olddir)
            
def vfm_row2block(rowvals):
    """
        
        input: rowvals - a given row from a CALIPSO VFM data set
        output: datablock - a block of CALIPSO VFM elements covering 30.1km x 5km 
        
        Function to take a 5515 element row of VFM data and convert it to an array
        of 15 profiles, each one 545 elements in length covering the altitude range
        from -0.5 - 30.1km.  Vertical resolution varies with altitude as per the 
        following table (see Hunt et.al. 2009):
        
        Range(km)   Vertical Resolution(m)  Horizontal Resolution(m)    Elements per profile
        -0.5 - 8.2      30                          333                     290
        8.2 - 20.2      60                          1000*                   200
        20.2 - 30.1     180                         1667*                   55
        
        * - higher altitude elements are oversampled to 333m to keep number of columns 
        consistent

        Based on vfm_row2block.m code from  LARC website (http://www-calipso.larc.nasa.gov/resources/calipso_users_guide/data_summaries/vfm/index.php)

    """
    import numpy as np
    
    datablock = np.empty((15,55+200+290),dtype = np.uint16)
    offset = 0
    step = 55
    indA = 0
    indB = 55
    for i in range(0,3):
        iLow = offset+step*i
        iHi = iLow+step
        n = i*5
        for k in range(0,5):
            datablock[n+k,indA:indB] = rowvals[iLow:iHi]
    
    offset = 165
    step = 200
    indA = 55
    indB = 55+200
    for i in range(0,5):
        iLow = offset+step*i
        iHi = iLow+step
        n = i*3
        for k in range(0,3):
            datablock[n+k,indA:indB] = rowvals[iLow:iHi]
    
    offset = 1165
    step = 290
    indA = 55+200
    indB = 55+200+290
    for i in range(0,15):
        iLow = offset+step*i
        iHi = iLow+step
        datablock[i,indA:indB] = rowvals[iLow:iHi]
    
    return datablock
    
    
def vfm_type(rowvals,type_str):
    """
        input:  rowvals :   a 1-d row of uint16 values from the VFM data set (any length)
                type_str:   a string matching one of the cases seen below that deteremines 
                            the feature being determined (e.g. 'type', 'qa','phase', 'aerosol', etc...)
        
        output: row_typed:  a 1-d row containing uint16 numbers that identify the value in 
                            category defined by "type_str"
                classdict: a dict of strings describing the possible types such that
                            classdict['FieldDesc'] returs the property being inquired 
                            about and classdict['ByteTxt'][row_typed[n]] returns the string describing the 
                            nth element of row_typed
                
        Takes a 1-d row of CALIPSO VFM data and extracts bits that determine 
        the user-defined aspect of the data as per the input type_str
        
        Based on vfm_type.m code from  LARC website (http://www-calipso.larc.nasa.gov/resources/calipso_users_guide/data_summaries/vfm/index.php)
        
    """
    import numpy as np
    
    mask3 = np.uint16(7)
    mask2 = np.uint16(3)
    mask1 = np.uint16(1)
    
    case = type_str.lower()
    
    if case in ('type','all'):
        row_typed = [x & mask3 for x in rowvals]
        classdict = {'FieldDesc':'Feature Type','ByteTxt':['Invalid', \
        'Clear Air','Cloud','Aerosol','Strat Feature','Surface','Subsurface', \
        'No Signal']}
    elif case in ('qa'):
        row_typed = [x & mask3 for x in rowvals]
        not_clear_air = [x==0 or x==2 or x==3 or x==4 for x in row_typed]
        temp = [x >> 3 for x in rowvals] 
        row_typed = [x & mask2 for x in temp]
        row_typed += np.uint16(not_clear_air)
        classdict = {'FieldDescription':'Feature Type QA','ByteTxt':['Clear Air',
        'None','Low','Medium','High']}
    elif case in ('phase'):
        temp = [x >> 5 for x in rowvals]
        row_typed = [x & mask2 for x in temp]
        classdict = {'FieldDescription':'Ice/Water Phase','ByteTxt':['Unknown',
        'Ice','Water','HorizOrient']}
    elif case in ('phaseqa'):
        temp = [x >> 7 for x in rowvals]
        row_typed = [x & mask2 for x in temp]
        classdict = {'FieldDescription':'Ice/Water Phase QA','ByteTxt':['None',
        'Low','Medium','High']}
    elif case in ('cloud'):
        temp = [x >> 9 for x in rowvals]
        sub_type = [x & mask3 for x in temp]
        temp_type = [x & mask3 for x in rowvals]
        is_cloud = [x == 2 for x in temp_type]
        row_typed = sub_type*np.uint16(is_cloud)
        classdict = {'FieldDescription':'Cloud Sub-Type','ByteTxt':['NA',
        'Low, overcast, thin','Low, overcast, thick','Trans. StratoCu','Low Broken',
        'Altocumulus','Altostratus','Cirrus (transparent)','Deep Convection']}
    elif case in ('aerosol'):
        temp = [x >> 9 for x in rowvals]
        sub_type = [x & mask3 for x in temp]
        temp_type = [x & mask3 for x in rowvals]
        is_aero = [x == 3 for x in temp_type]
        row_typed = sub_type*np.uint16(is_aero)
        classdict = {'FieldDescription':'Aerosol Sub-Type','ByteTxt':['Not Determined',
        'Clean Marine','Dust','Polluted Cont.','Clean Cont.','Polluted Dust',
        'Smoke','Other']}
    elif case in ('psc'):
        temp = [x >> 9 for x in rowvals]
        sub_type = [x & mask3 for x in temp]
        temp_type = [x & mask3 for x in rowvals]
        is_psc = [x == 4 for x in temp_type]
        row_typed = sub_type*np.uint16(is_psc)
        classdict = {'FieldDescription':'PSC Sub-Type','ByteTxt':['Not Determined',
        'Non-Depol. Large P.','Depol. Large P.','Non-Depol Small P.',
        'Depol. Small P.','spare','spare','Other']}
    elif case in ('typeqa'):
        temp = [x >> 12 for x in rowvals]
        row_typed = [x & mask1 for x in temp]
        classdict = {'FieldDescription':'Sub-Type QA','ByteTxt':['None','Low',
        'Medium','High']}
    elif case in ('averaging'):
        temp = [x >> 13 for x in rowvals]
        row_typed = [x & mask3 for x in temp]
        classdict = {'FieldDescription':'Averaging Required','ByteTxt':['NA',
        '1/3 km','1 km','5 km','20 km','80 km']}
    else:
        print('WARNING: Unknown type specifier!')
        row_typed = 0
        classdict = {'FieldDescription':'empty','ByteTxt':['empty']}
    
    return row_typed,classdict

def to360(lon_in):
    if lon_in >= 0:
        lon_out = lon_in
    else:
        lon_out = 360+lon_in
    return lon_out

def to180(lon_in):
    if lon_in <= 180:
        lon_out = lon_in
    else:
        lon_out = lon_in-360
    return lon_out

class Calipso:
    
    def __init__(self,metadata = [], VFM = [], maskdata = [], h5file=[], raw = True):        
        import pandas as pan
        import tables
        import numpy as np
        import datetime as dt
        import pytz
        from itertools import izip,tee
        
        if h5file:
            print "Opening "+h5file
            
            if raw:
                calfile = tables.openFile(h5file,'r')
                rawlats = calfile.root.Latitude.read()[:,0]
                rawlons = calfile.root.Longitude.read()[:,0]        
                rawtimes = calfile.root.Profile_UTC_Time.read()[:,0]
                rawdata = calfile.root.Feature_Classification_Flags.read()
                
                metadata = calfile.root.metadata   
                metakeys = metadata.colnames
                metadict = {}                
                for k in metakeys:
                    metadict[k] = [m[k] for m in metadata.iterrows()]
                
                calfile.close() 
        
                def timetag_conv(start,end):
                                    
                    d = int(start % 100)
                    m = int((start-d) % 10000)//100
                    y = 2000 + int(start-m-d)//10000
                    
                    starttime = dt.datetime(y, m, d, tzinfo=pytz.utc) + dt.timedelta(start % 1)
                    
                    d = int(end % 100)
                    m = int((end-d) % 10000)//100
                    y = 2000 + int(end-m-d)//10000
                    
                    endtime = dt.datetime(y, m, d, tzinfo=pytz.utc) + dt.timedelta(end % 1)
                    
                    timestep = (endtime-starttime)
                    
                    time = []
                    for n in range(0,15):
                        time.append(starttime+n*timestep/15)
                    
                    return time
                
                def latlon_conv(start,end):
                    
                    step = (end-start)/15.0
                    
                    outvals = []
                    for n in range(0,15):
                        outvals.append(start+n*step)
                    
                    return outvals
                
                def pairwise(iterable):
                    a, b = tee(iterable)
                    next(b, None)
                    return izip(a, b)
    
                times = np.ravel([timetag_conv(t1,t2) for t1,t2 in pairwise(rawtimes)])
                lats = np.ravel([latlon_conv(l1,l2) for l1,l2 in pairwise(rawlats)])
                lons = np.ravel([latlon_conv(l1,l2) for l1,l2 in pairwise(rawlons)])
                
                [r,c] = np.shape(rawdata)
                
                blocksize = 15
                data = np.empty(((r-1)*blocksize,545), dtype = np.uint16)
                i = 0
                for n in range(len(rawdata)-1):
                    block = vfm_row2block(rawdata[n,:])
                    data[i:i+blocksize,:] = block
                    i = i+blocksize
                
                altitudes = metadict['Lidar_Data_Altitudes'][0]*1000  #altitudes of VFM data points converted from km to m
                
                altitudes = altitudes[(altitudes>=-500)&(altitudes<=30100)]  # altitudes limited to range 30.1km--0.5km
                
                ix = zip(*[lats,lons,times])
                
                df_ix = pan.MultiIndex.from_tuples(ix,names = ['Lats','Lons','Times'])
                
                df = pan.DataFrame(data=data, index=df_ix, columns = altitudes)   
                
                df.sort(axis=1, inplace=True)
                
                self.metadata = pan.DataFrame.from_dict(metadict, orient = 'index')
                self.VFM = df
                self.maskdata = []
            else:
                self.metadata = pan.read_hdf(h5file,'metadata')
                self.VFM = pan.read_hdf(h5file,'VFM')
                self.maskdata = pan.read_hdf(h5file,'maskdata')
            print "DONE!"             
        else:
            if VFM:
                print "Creating new Calipso class object from dataframe"
                self.VFM = VFM #slot for VFM data frame
                self.metadata = metadata  #slot for metadata dict
                self.maskdata = maskdata  #slot for dictionary describing applied feature mask
            else:
                print "Creating empty Calipso classs object"
                self.VFM = pan.DataFrame()
                self.metadata = pan.DataFrame()
                self.maskdata = pan.DataFrame()
            print "DONE!"
    
    def save(self,filename):
        """ Saves a Calipso class object in HDF5 format """
        
        import pandas as pan
        
        store = pan.HDFStore(filename)
        
        store['metadata'] = self.metadata
        store['VFM'] = self.VFM
        store['maskdata'] = self.maskdata
        
        store.close()
        
        
    def window_select(self,latrange=[],lonrange=[],timerange=[], altrange=[], inplace = True):
        """
        Select a subset of a calipso dataframe based on some combination of lat,
        lon, time, and altitude
        """
        
        print "Selecting Data Window ..."
        from copy import deepcopy
        import pandas as pan
        
        if inplace:
            dfout = self.VFM
        else:
            dfout = deepcopy(self.VFM)
        
        if latrange:
            dfout = dfout[(dfout.index.get_level_values('Lats')>=latrange[0]) \
            & (dfout.index.get_level_values('Lats')<=latrange[1])]
        
        if lonrange:
            #convert from -180/+180 to 0/360
            lon360 = [to360(l) for l in lonrange]
            lons = dfout.index.get_level_values('Lons').values           
            lons = [to360(l) for l in lons]
            lats = dfout.index.get_level_values('Lats').values
            times = dfout.index.get_level_values('Times').values
            
            ix = zip(*[lats,lons,times])
                
            df_ix = pan.MultiIndex.from_tuples(ix,names = ['Lats','Lons','Times'])
            dfout.index = df_ix
            
            #filter for longitude window              
            dfout = dfout[(dfout.index.get_level_values('Lons')>=lon360[0]) \
            & (dfout.index.get_level_values('Lons')<=lon360[1])]
            
            #convert back to -180/+180            
            lons = dfout.index.get_level_values('Lons').values           
            lons = [to180(l) for l in lons]
            lats = dfout.index.get_level_values('Lats').values
            times = dfout.index.get_level_values('Times').values
            
            ix = zip(*[lats,lons,times])
                
            df_ix = pan.MultiIndex.from_tuples(ix,names = ['Lats','Lons','Times'])
            dfout.index = df_ix
        
        if timerange:
            dfout = dfout[(dfout.index.get_level_values('Times')>=timerange[0]) \
            & (dfout.index.get_level_values('Times')<=timerange[1])]
        
        if altrange:
            dfout = dfout.loc[:,(dfout.columns>=altrange[0])&(dfout.columns<=altrange[1])]
        
        if not inplace:
            calout = Calipso(metadata = self.metadata, VFM = dfout, maskdata = self.maskdata)
#            calout.metadata = deepcopy(self.metadata)
#            calout.VFM = dfout
            print "DONE!"
            return calout
        else:
            self.VFM = dfout
            print "DONE!"
        
        
        
        
    
    def feature_mask(self, maskname, inplace = False):
        """
            inputs: maskname = string representing the type of feature to query
                    inplace = Boolean variable determining whether the Calipso class
                                object will be copied or modified
            outputs: if inplace == False, returns calfile: A new Calipso class
                    object with mask applied
                    if inplace == True, nothing returned
                    
            applies vfm_type function to each row in a Calipso.VFM dataframe, converting
            the 16-bit integer there to a single integer representing the data type under
            the category defined in masktype.  Mask types and interpretations found in
            vfm_type function definition.
        """
        print "Masking for %s Features ..." %maskname
        
        from copy import deepcopy
        import pandas as pan
        
        if self.maskdata:
            print "ERROR:  This object has already been masked for Type:" +self.masktype['FieldDesc']
        else:            
            if inplace:
                dfout = self.VFM
            else:
                dfout = deepcopy(self.VFM)
                
            for i in dfout.index:
#                print "Masking for Index: "+str(i)
                dfout.loc[i],maskdata = vfm_type(dfout.loc[i],maskname)
            
            if inplace:
                self.VFM = dfout
                self.maskdata = pan.DataFrame.from_dict(maskdata, orient = 'index')
                print "DONE!"
            else:
                calout = Calipso()
                calout.metadata = deepcopy(self.metadata)
                calout.VFM = dfout
                calout.maskdata = pan.DataFrame.from_dict(maskdata, orient = 'index')
                print "DONE!"
                return calout
        
        

def filterandflatten(caldat_in, features, combine = False, domask = False, maskname = []):
    """
        inputs:     caldat_in - a Calipso class object to be masked
                    features - a list of strings defining the feature subtypes to be scanned for
                    combine - a boolean to determine whether to separate features in to columns
                    domask - a boolean to determine whether to apply a feature type mask
                    maskname - a string describing the category of fature type 
                    mask to perform, if domask = True
        
        outputs:    caldat_out - a Calipso class object contining the following:
                        dfout - a dataframe, same length as df, but only one column per feature
                        element, each representing the number of 30m x 333m blocks containing the
                        desired feature at that location
                        maskdata - a dataframe containing strings defining the feature type mask
        
    """
    print "Counting Instances of "+', '.join(features)
    
    import pandas as pan
    import numpy as np
    
    if domask:
        caldat_tmp = caldat_in.feature_mask(maskname,inplace = False)
        df = caldat_tmp.VFM
        maskdata = caldat_tmp.maskdata
    else:
        df = caldat_in.VFM
        maskdata = caldat_in.maskdata
    
    featurenums = []
    featurenames = []
    maskdataout = {'FieldDescription':maskdata['FieldDescription'], 'ByteTxt':[]}
    
    for f in features:
        try:
            temp = maskdata.loc['ByteTxt'][0].index(f)
            featurenums.append(temp)
            featurenames.append(f)
            maskdataout['ByteTxt'].append(featurenames)
        except ValueError:
            print "ERROR: The feature %s does not exist in this mask" %f
            print "Available features are: "+ ', '.join(maskdata['ByteTxt'])
    
    if combine:
        #output column is named ofater first feature in list with '+' added
        colname = [featurenames[0]+'+']
        numrows = len(df.index)
        numcols = 1
        dfout = pan.DataFrame(np.zeros((numrows,numcols), dtype = np.uint16), index = df.index, columns = colname)
    else:
        numrows = len(df.index)
        numcols = len(features)
        dfout = pan.DataFrame(np.zeros((numrows,numcols), dtype = np.uint16), index = df.index, columns = featurenames)  
    
    #filter through df and count number of instances for each feature
    
    for fname,fnum in zip(featurenames,featurenums):
        for i in df.index:
            temp = df.loc[i]
            if combine:
                dfout.loc[i] += (temp == fnum).sum()
            else:
                dfout.loc[i,fname] = (temp == fnum).sum()
    
    print "DONE!"
    caldat_out = Calipso()
    caldat_out.metadata = caldat_in.metadata
    caldat_out.VFM = dfout
    caldat_out.maskdata = maskdataout
    
    
    return caldat_out
            
if __name__ == "__main__":
    import os
    from matplotlib import pyplot as plt
    
    olddir = os.getcwd()
    
    os.chdir('C:\Users\dashamstyr\Dropbox\Lidar Files\CALIPSO Data Reader')
    
#    c1 = Calipso(h5file = 'CAL_LID_L2_VFM-ValStage1-V3-30.2013-04-22T00-37-50ZD.h5', raw = True)
#    
#    lats = [30,40]
#    lons = []
#    alts = [0,20000]
#    
#    c1.window_select(latrange = lats,lonrange = lons,altrange = alts, inplace = True)
#    
#    c1_masked = c1.feature_mask('aerosol')
#    
#    c1_dust = filterandflatten(c1_masked,features = ['Dust','Polluted Dust'], combine = True)
#
#    c1_dust.save('test.h5') 
    
    c1_dust = Calipso(h5file = 'test.h5',raw = False)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(c1_dust.VFM.index.get_level_values('Lats').values,c1_dust.VFM.values)
    plt.show()
    

    
    print "Done"
    