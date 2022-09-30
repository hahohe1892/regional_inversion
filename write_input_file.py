import numpy as np
from load_input import *
import PISM
from bed_inversion import *
import os
import shutil
from oggm.core import massbalance
import statsmodels.api as sm

def write_input_file(RID):
    working_dir = '/home/thomas/regional_inversion/output/' + RID
    input_file = working_dir + '/input.nc'
    input_dir = '/home/thomas/regional_inversion/input_data/'

    # default DEM (COPDEM) is from 2010 - 2015
    period = '2010-2015'
    dhdt_dir = input_dir + 'dhdt_' + period + '/per_glacier/RGI60-08/RGI60-08.0' + RID[10] + '/'+ RID 


    if not os.path.exists(input_dir + 'dhdt_2000-2020/per_glacier/RGI60-08/RGI60-08.0' + RID[10] + '/'+ RID):
        shutil.copyfile(input_dir + 'dhdt_2000-2020/per_glacier/RGI60-08/RGI60-08.0' + RID[10] + '/'+ RID + '/gridded_data.nc', input_file)

    if not os.path.isdir(working_dir):
        os.mkdir(working_dir)

    dem = load_dem_path(RID)
    mask_in = load_mask_path(RID)
    dhdt = load_dhdt_path(RID, period = period)

    dem = crop_border_xarr(dem)
    mask_in = crop_border_xarr(mask_in)
    dhdt = crop_to_xarr(dhdt, mask_in)
    
    topg = np.copy(dem)-1

    smb = np.ones_like(dem[0])
    heights = dem.data.flatten()
    gdir = load_dem_gdir(RID)[0]
    mbmod = massbalance.MultipleFlowlineMassBalance(gdir)
    mbmod1 = mbmod.flowline_mb_models[0]
    mb_years = []
    for year in range(2010, 2015+1):
        mb_years.append(mbmod1.get_annual_mb(heights, year=year) * secpera * 900)
    mb = [heights, np.mean(np.array(mb_years), axis = 0)]

    for i in range(dem.data[0].shape[0]):
        for j in range(dem.data[0].shape[1]):
            smb[i,j] = (mb[1][mb[0] == dem.data[0][i,j]])[0]

    # calculate dhdt based on regional fitting of elevation trend
    A = gdir.rgi_area_m2
    if A > 1000*2:
        output = 'large'
        dems_large, dems_small, dhdts_large, dhdts_small = partition_dhdt(output=output)
        dhdt_fit, dhdt_fit_field, dem_masked = get_dhdt(RID, dems_large, dhdts_large)
    else:
        output = 'small'
        dems_large, dems_small, dhdts_large, dhdts_small = partition_dhdt(output=output)
        dhdt_fit, dhdt_fit_field, dem_masked = get_dhdt(RID, dems_small, dhdts_small)

    # modify either smb or dhdt so that they balance
    k = 0
    learning_rate = 0.2
    smb_misfit = np.mean(smb[mask_in.data[0]==1]/900) - np.mean(dhdt_fit_field.data[0][mask_in.data[0]==1])
    while abs(smb_misfit)>0.01:
        smb_misfit = np.mean(smb[mask_in.data[0]==1]/900) - np.mean(dhdt_fit_field.data[0][mask_in.data[0]==1]) - k
        k += smb_misfit * learning_rate

    smb -= k * 900

    x = dem.x
    y = np.flip(dem.y)
    create_input_nc(input_file, x, y, dem, topg, mask_in, dhdt_fit_field, smb, ice_surface_temp=273)


def partition_dhdt(output = 'all'):
    sum_arr = np.load('/home/thomas/regional_inversion/all_dhdt_dem.npy')
    RIDs = np.unique(sum_arr[0])
    dhdts_small = []
    dems_small = []
    dhdts_large = []
    dems_large = []

    for rid in RIDs:
        loc = sum_arr[0] == rid
        A = sum_arr[-1, loc][0].astype(float)
        if A > 1000**2:
            if output == 'small':
                continue
            else:
                dhdts_large.extend(sum_arr[2, loc])
                dems_large.extend(sum_arr[1, loc])
        else:
            if output == 'large':
                continue
            else:
                dhdts_small.extend(sum_arr[2, loc])
                dems_small.extend(sum_arr[1, loc])
    dhdts_small = np.array(dhdts_small)
    dhdts_large = np.array(dhdts_large)
    dems_small = np.array(dems_small)
    dems_large = np.array(dems_large)
    dhdts_large = dhdts_large.astype(float)
    dhdts_small = dhdts_small.astype(float)
    dems_large = dems_large.astype(float)
    dems_small = dems_small.astype(float)

    return dems_large, dems_small, dhdts_large, dhdts_small


def get_dhdt(RID, dems, dhdts):
    sum_arr = np.load('/home/thomas/regional_inversion/all_dhdt_dem.npy')
    dems_m = sm.add_constant(dems)
    res = sm.OLS(dhdts, dems_m).fit()
    loc = sum_arr[0] == RID
    dem_scaled = sum_arr[1, loc].astype(float)
    dhdt = sum_arr[2, loc].astype(float)
    dem = load_dem_path(RID)
    dem = crop_border_xarr(dem)

    ks =[0]
    misfit = 100
    learning_rate = .2
    while abs(misfit) > .005:
        k = ks[-1]
        slope_fit = k + res.params[1] * dem_scaled
        median_extrapolated = np.median(slope_fit)
        misfit = np.median(dhdt) - median_extrapolated
        k += misfit * learning_rate
        ks.append(k)

    dhdt_fit = k + res.params[1] * dem_scaled
    dhdt_fit_field = (k + res.params[1] * ((dem - np.min(dem))/(np.max(dem) - np.min(dem))))

    return dhdt_fit, dhdt_fit_field, dem_scaled


glaciers_Sweden = get_RIDs_Sweden()
RIDs_Sweden = glaciers_Sweden.RGIId

for RID in RIDs_Sweden:
    write_input_file(RID)
