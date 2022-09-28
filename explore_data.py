import numpy as np
from load_input import *
import PISM
from bed_inversion import *
import os
import shutil
from oggm.core import massbalance
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

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
    ### concatenate arrays and select based on that ###
    dem_c = xr.concat([dem, mask_in], 'band')
    dem_c = dem_c.where(dem_c[1] == 1)
    dhdt_c = xr.concat([mask_in, dhdt], 'band')
    dhdt_c = dhdt_c.where(dhdt_c[1] == 1)

    dhdts.extend(list(dhdt.data[mask_in.data==1]))
    dem_masked = dem.data[mask_in.data==1]
    dem_correct = [x < 100 for x in dem_masked]
    dem_masked[dem_correct] = np.nan
    CenLat = int(glaciers_Sweden.where(glaciers_Sweden.RGIId == RID).dropna().CenLat.values[0].split(',')[0])
    CenLat = float(dem.y.mean().values)
    lats.extend([CenLat] * len(dem_masked))
    dems.extend(list(((dem_masked - np.min(dem_masked))/(np.max(dem_masked) - np.min(dem_masked)))))

dhdts = np.array(dhdts)
dems = np.array(dems)
lats = np.array(lats)
nan_dhdts = dhdts>-200
dhdts = dhdts[nan_dhdts]
dems = dems[nan_dhdts]
lats = lats[nan_dhdts]

model = LinearRegression().fit(dems.reshape((-1,1)), dhdts)
dems_m = sm.add_constant(dems)
lats_m = sm.add_constant(lats)
res = sm.OLS(dhdts, dems_m).fit()
res1 = sm.OLS(dhdts, lats_m).fit()
#res = mod.fit()
