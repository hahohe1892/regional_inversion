import numpy as np
from load_input import *
import PISM
from bed_inversion import *
import os
import shutil
from oggm.core import massbalance
import statsmodels.api as sm

def write_input_file(RID, new_mask = False):
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
    if new_mask:
        mask_in = load_georeferenced_mask(RID)
        mask_smb = load_mask_path(RID)
    else:
        mask_in = load_mask_path(RID)
    dhdt = load_dhdt_path(RID, period = period)

    dem = crop_border_xarr(dem)
    mask_in = crop_to_xarr(mask_in, dem)
    mask_in.data[0][:3,:] = 0
    mask_in.data[0][:,:3] = 0
    mask_in.data[0][:,-3:] = 0
    mask_in.data[0][-3:,:] = 0

    mask_smb = crop_to_xarr(mask_smb, dem)

    dhdt = crop_to_xarr(dhdt, dem)

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
    smb_misfit = np.mean(smb[mask_smb.data[0]==1]/900) - np.mean(dhdt_fit_field.data[0][mask_smb.data[0]==1])
    while abs(smb_misfit)>0.01:
        smb_misfit = np.mean(smb[mask_smb.data[0]==1]/900) - np.mean(dhdt_fit_field.data[0][mask_smb.data[0]==1]) - k
        k += smb_misfit * learning_rate

    smb -= k * 900

    apparent_mb = smb - dhdt_fit_field * 900
    #apparent_mb.data[0][mask_in.data[0] == 0] = 0
    # smooth input DEM
    k = np.ones((3,3))
    dem.data[0] = ndimage.convolve(dem.data[0], k)/9
    topg = np.copy(dem)-1

    x = dem.x
    y = np.flip(dem.y)
    create_input_nc(input_file, x, y, dem, topg, mask_in, dhdt_fit_field, smb, apparent_mb, ice_surface_temp=273)


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
    write_input_file(RID, new_mask = True)

    
def correct_mask(RID):
    dem = load_dem_path(RID)
    mask = load_mask_path(RID)
    #dem = crop_border_xarr(dem)
    #mask = crop_border_xarr(mask)
    x = dem.x
    y = dem.y
    slope = calc_slope(dem.data[0], dem.rio.resolution()[0])
    slope_sums = []
    Is = []
    Js = []
    m = np.copy(mask[0].data)
    for j in range(-10, 10+1):
        for i in range(-10, 10+1):
            mask_new = np.copy(m)
            if i > 0:
                mask_new[:,:-i] = mask_new[:,i:]
            if i < 0:
                mask_new[:,-i:] = mask_new[:,:i]
            if j > 0:
                mask_new[:-j,:] = mask_new[j:,:]
            if j < 0:
                mask_new[-j:,:] = mask_new[:j,:]
            if i == 0 and j == 0:
                mask_new = np.copy(m)

            slope_sums.append(np.sum(slope/(dem.data[0]**.5) * mask_new))
            Is.append(i)
            Js.append(j)
    slope_sums = np.array(slope_sums)
    min = np.min(slope_sums)
    index = np.argwhere(slope_sums == min).squeeze()
    print(Is[index])
    print(Js[index])
    i = Is[index]
    j = Js[index]
    mask_new = np.copy(m)
    if i > 0:
        mask_new[:,:-i] = mask_new[:,i:]
    if i < 0:
        mask_new[:,-i:] = mask_new[:,:i]
    if j > 0:
        mask_new[:-j,:] = mask_new[j:,:]
    if j < 0:
        mask_new[-j:,:] = mask_new[:j,:]
    if i == 0 and j == 0:
        mask_new = np.copy(m)

    mask[0].data -= (m - mask_new)
    path = '/home/thomas/regional_inversion/input_data/dhdt_2000-2020/per_glacier/RGI60-08/RGI60-08.0' + RID[10] +'/' + RID +'/gridded_data.nc'
    new_file = '/home/thomas/regional_inversion/input_data/dhdt_2000-2020/per_glacier/RGI60-08/RGI60-08.0' + RID[10] +'/' + RID +'/gridded_data_new.nc'
    if not os.path.exists(new_file):
        shutil.copyfile(path, new_file)
    nc = NC(new_file, 'r+')
    if 'mask_new' not in nc.variables.keys():
        nc.createVariable('mask_new', int, ('x', 'y'))
    nc.close()
    nc = NC(new_file, 'r+')
    nc['mask_new'][:,:] = mask_new
    nc.close()
    
    mask.rio.to_raster(path)


#glaciers_Sweden = get_RIDs_Sweden()
#RIDs_Sweden = glaciers_Sweden.RGIId

#for RID in RIDs_Sweden:
#    correct_mask(RID)
