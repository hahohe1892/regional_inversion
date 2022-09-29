import numpy as np
from load_input import *
from bed_inversion import *
import os
import shutil
from oggm.core import massbalance
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import xarray as xr

glaciers_Sweden = get_RIDs_Sweden()
RIDs_Sweden = glaciers_Sweden.RGIId
period = '2010-2015'

rids = []
dhdts = []
dems = []
areas = []
dhdts_small = []
dems_small = []
dhdts_large = []

dems_large = []
dhdt_fields_small = []
dhdt_fields_large = []
dem_fields_small = []
dem_fields_large = []
lats = []

for RID in RIDs_Sweden:
    dem = load_dem_path(RID)
    gdir = load_dem_gdir(RID)[0]
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
    '''
    dem_c = xr.concat([dem, mask_in], 'band')
    dem_c = dem_c.where(dem_c[1] == 1)
    dhdt_c = xr.concat([mask_in, dhdt], 'band')
    dhdt_c = dhdt_c.where(dhdt_c[1] == 1)
    '''

    dhdt_masked = list(dhdt.data[mask_in.data==1])
    dem_masked = dem.data[mask_in.data==1]
    dem_correct = [x < 100 for x in dem_masked]
    dem_masked[dem_correct] = np.nan
    dem_scaled = list(((dem_masked - np.min(dem_masked))/(np.max(dem_masked) - np.min(dem_masked))))
    a = gdir.rgi_area_m2
    if a > 1000**2:
        dhdts_large.extend(dhdt_masked)
        dems_large.extend(dem_scaled)
        dem_fields_large.append(dem)
        dhdt_fields_large.append(dhdt)
    else:
        dhdts_small.extend(dhdt_masked)
        dems_small.extend(dem_scaled)
        dem_fields_small.append(dem)
        dhdt_fields_small.append(dhdt)
    dhdts.extend(dhdt_masked)
    dems.extend(dem_scaled)
    rids.extend([RID] * len(dem_scaled))
    areas.extend([a] * len(dem_scaled))
    CenLat = int(glaciers_Sweden.where(glaciers_Sweden.RGIId == RID).dropna().CenLat.values[0].split(',')[0])
    CenLat = float(dem.y.mean().values)
    lats.extend([CenLat] * len(dem_masked))

dhdts = np.array(dhdts)
dems = np.array(dems)
rids = np.array(rids)
areas = np.array(areas)
dhdts_small = np.array(dhdts_small)
dhdts_large = np.array(dhdts_large)
dems_small = np.array(dems_small)
dems_large = np.array(dems_large)
lats = np.array(lats)
nan_dhdts = dhdts>-200
nan_dhdts_small = dhdts_small>-200
nan_dhdts_large = dhdts_large>-200
dhdts = dhdts[nan_dhdts]
dhdts_small = dhdts_small[nan_dhdts_small]
dhdts_large = dhdts_large[nan_dhdts_large]
dems_small = dems_small[nan_dhdts_small]
dems_large = dems_large[nan_dhdts_large]
dems = dems[nan_dhdts]
rids = rids[nan_dhdts]
areas = areas[nan_dhdts]
#lats = lats[nan_dhdts]

summary_array = np.array((rids, dems, dhdts, areas))

#model = LinearRegression().fit(dems.reshape((-1,1)), dhdts)
dems_small_m = sm.add_constant(dems_small)
dems_large_m = sm.add_constant(dems_large)
#lats_m = sm.add_constant(lats)
res_small = sm.OLS(dhdts_small, dems_small_m).fit()
res_large = sm.OLS(dhdts_large, dems_large_m).fit()
#res = mod.fit()

fig, ax = plt.subplots(1,2)
ax[0].scatter(dems_small, dhdts_small, alpha = .01)
ax[0].scatter(dems_small, res_small.fittedvalues)
ax[1].scatter(dems_large, dhdts_large, alpha = .01)
ax[1].scatter(dems_large, res_large.fittedvalues)
ax[0].set_ylim([-5,5])
ax[1].set_ylim([-5,5])
plt.show()


ks =[0]
misfit = 100
learning_rate = .2
while abs(misfit) > .005:
    k = ks[-1]
    slope_fit = k + res_small.params[1] * np.array(dem_scaled)
    median_extrapolated = np.median(slope_fit)
    misfit = np.median(dhdt_masked) - median_extrapolated
    k += misfit * learning_rate
    ks.append(k)

dhdt_fit = k + res_small.params[1] * np.array(dem_scaled)
dhdt_fit_field = (k + res_small.params[1] * ((dem - np.min(dem))/(np.max(dem) - np.min(dem))))
plt.scatter(dem_scaled, dhdt_masked)
plt.scatter(dem_scaled, dhdt_fit)
plt.show()
