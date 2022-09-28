import numpy as np
from load_input import *
import PISM
from bed_inversion import *
import os
import shutil
from oggm.core import massbalance
import pandas as pd

glaciers_Sweden = get_RIDs_Sweden()
RIDs_Sweden = glaciers_Sweden.RGIId
period = '2010-2015'

dhdts = []
dems = []
lats = []

for RID in RIDs_Sweden:
    dem = load_dem_path(RID)
    mask_in = load_mask_path(RID)
    dhdt = load_dhdt_path(RID, period = period)
    dem = crop_border_xarr(dem)
    mask_in = crop_border_xarr(mask_in)
    dhdt = crop_to_xarr(dhdt, mask_in)
    if not (np.round(mask_in.x.data,6) == np.round(dhdt.x.data, 6)).any():
        print('Error: x coordinates not the same')
        break
    if not (np.round(mask_in.y.data,6) == np.round(dhdt.y.data, 6)).any():
        print('Error: y coordinates not the same')
        break
    dhdts.extend(list(dhdt.data[mask_in.data==1]))
    dem_masked = dem.data[mask_in.data==1]
    dem_correct = [x < 100 for x in dem_masked]
    dem_masked[dem_correct] = np.nan
    CenLat = int(glaciers_Sweden.where(glaciers_Sweden.RGIId == RID).dropna().CenLat.values[0].split(',')[0])
    lats.extend([CenLat] * len(dem_masked))
    dems.extend(list(((dem_masked - np.min(dem_masked))/(np.max(dem_masked) - np.min(dem_masked)))))

dhdts = np.array(dhdts)
dems = np.array(dems)
dhdts[dhdts<-200] = np.nan


'''
    try:
        dhdts.extend(list(dhdt_c.data[mask_in_c.data==1]))
        dems.extend(list(dem_c.data[mask_in_c.data==1]))
    except IndexError:
        try:
            dhdts.extend(list(dhdt_c.data[0][:,:-1][mask_in_c.data[0]==1]))
            dems.extend(list(dem_c.data[0][mask_in_c.data[0]==1]))
        except IndexError:
            try:
                dhdts.extend(list(dhdt_c.data[0][:,1:][mask_in_c.data[0]==1]))
                dems.extend(list(dem_c.data[0][mask_in_c.data[0]==1]))
            except IndexError:
            #    print(dhdt_c.data[0][:,1:].shape)
            #    print(mask_in_c.data[0].shape)
                i+=1
                continue
    #dhdts.extend(list(dhdt_c.where(mask_in_c == 1).data.flatten()))
    #dems.extend(list(dem_c.where(mask_in_c == 1).data.flatten()))
    #dhdts.append(list(dhdt.where(mask_in == 1).data.flatten()))
    #dems.append(list(dem.where(mask_in == 1).data.flatten()))

#dh = np.array(dhdts).flatten()
#de = np.array(dems).flatten()
'''
