import numpy as np
from load_input import *
import PISM
from bed_inversion import *
import os
import shutil
from oggm.core import massbalance
import statsmodels.api as sm
import xarray as xr
from copy import deepcopy
import rioxarray as rioxr


def write_input_file_Sweden_Norway(RID, period = '2000-2020'):
    
    input_dir = '/home/thomas/regional_inversion/input_data/'
    RGI_region = RID.split('-')[1].split('.')[0]
    per_glacier_dir = 'per_glacier/RGI60-' + RGI_region + '/RGI60-' + RGI_region + '.0' + RID[10] + '/'+ RID
    glaciers_Sweden = get_RIDs_Sweden()
    RIDs_Sweden = glaciers_Sweden.RGIId

    dem_Norway = rioxr.open_rasterio(os.path.join(input_dir, 'DEM_Norway', per_glacier_dir, 'dem.tif'))
    dem_Sweden = rioxr.open_rasterio(os.path.join(input_dir, 'DEM_Sweden', per_glacier_dir, 'dem.tif'))
    # choose the DEM which contains no nans
    if (dem_Sweden.data == dem_Sweden._FillValue).any():
        if not (dem_Norway.data == dem_Norway._FillValue).any():
            print('Choosing Norwegian DEM')
            dem = deepcopy(dem_Norway)
    elif (dem_Norway.data == dem_Norway._FillValue).any():
        if not (dem_Sweden.data == dem_Sweden._FillValue).any():
            dem = deepcopy(dem_Sweden)
            print('Choosing Swedish DEM')
    elif (dem_Norway.data == dem_Norway._FillValue).any() and (dem_Norway.data == dem_Norway._FillValue).any():
        raise ValueError('No suitable DEM found, cannot proceed')

    dhdt = rioxr.open_rasterio(os.path.join(input_dir, 'dhdt_' + period, per_glacier_dir, 'dem.tif'))

    if RID in RIDs_Sweden.tolist():
        mask = load_georeferenced_mask(RID)
    else:
        mask = rioxr.open_rasterio(os.path.join(input_dir, 'masks', per_glacier_dir, RID + '_mask.tif'))

    consensus_thk = rioxr.open_rasterio(os.path.join(input_dir, 'consensus_thk', 'RGI60-' + RGI_region, RID + '_thickness.tif'))
    vel_Millan = rioxr.open_rasterio(os.path.join(input_dir, 'vel_Millan', per_glacier_dir, 'dem.tif'))
    # mass balance; VERY PRELIMINARY
    

    
def write_input_file(RID, period = '2010-2015', new_mask = False, output_resolution = None, fit_dhdt_regionally = True, modify_dhdt_or_smb = 'smb'):

    # default DEM (COPDEM) is from 2010 - 2015

    working_dir = '/home/thomas/regional_inversion/output/' + RID
    input_file = working_dir + '/input.nc'
    input_dir = '/home/thomas/regional_inversion/input_data/'

    RGI_region = RID.split('-')[1].split('.')[0]
    #dhdt_dir = input_dir + 'dhdt_' + period + '/per_glacier/RGI60-' + RGI_region + '/RGI60-' + RGI_region + '.0' + RID[10] + '/'+ RID 

    if not os.path.exists(input_dir + 'dhdt_2000-2020/per_glacier/RGI60-' + RGI_region + '/RGI60-' + RGI_region + '.0' + RID[10] + '/'+ RID):
        shutil.copyfile(input_dir + 'dhdt_2000-2020/per_glacier/RGI60-' + RGI_region + '/RGI60-' + RGI_region + '.0' + RID[10] + '/'+ RID + '/gridded_data.nc', input_file)

    if not os.path.isdir(working_dir):
        os.mkdir(working_dir)

    dem = load_dem_path(RID)
    if new_mask == True:
        mask_in = load_georeferenced_mask(RID)
        # decide buffer around glacier; here based on centering mask and taking this as standard extent
        mask_in = crop_to_new_mask(mask_in, mask_in, 10)
        mask_in = mask_in.rio.reproject(mask_in.rio.crs, resolution = 100)
        mask_in.data[mask_in.data == mask_in._FillValue] = 0
        dem = dem.rio.reproject_match(mask_in)
    else:
        mask_in = load_mask_path(RID)
        mask_in.data[0][mask_in.data[0] == mask_in.rio.nodata] = 0
        # decide buffer around glacier; below is standard procedure to crop as is done with OGGM
        dem = crop_border_xarr(dem)
        mask_in = mask_in.rio.reproject_match(dem)
        mask_in.data[0][:2,:] = 0
        mask_in.data[0][:,:2] = 0
        mask_in.data[0][-2:,:] = 0
        mask_in.data[0][:,-2:] = 0

    dhdt = load_dhdt_path(RID, period = period)


    #thk_oggm_in = load_thk_path(RID)
    thk_oggm = load_consensus_thk(RID)
    thk_oggm = thk_oggm.rio.reproject_match(dem)
    thk_oggm.data[thk_oggm.data < 0] = 0
    thk_oggm.data[thk_oggm.data > 1e5] = 0
    thk_oggm.data[thk_oggm.data == thk_oggm._FillValue] = 0
    #thk_oggm = np.zeros_like(dem.data[0])
    #thk_oggm = thk_oggm_in
    thk_oggm.data[0][mask_in.data[0] == 0] = 0


    #dhdt = crop_to_xarr(dhdt, dem)
    dhdt = dhdt.rio.reproject_match(dem)

    smb = deepcopy(dem)#np.ones_like(dem[0])
    smb.data[0] = np.ones_like(dem[0])
    heights = dem.data.flatten()
    gdir = load_dem_gdir(RID)[0]
    #mbmod = massbalance.MultipleFlowlineMassBalance(gdir)
    mbmod = massbalance.PastMassBalance(gdir, check_calib_params = False)
    #mbmod1 = mbmod.flowline_mb_models[0]
    mb_years = []
    mb_start_yr = int(period.split('-')[0])
    mb_end_yr = int(period.split('-')[1])
    mb_end_yr = np.minimum(mb_end_yr, 2019-1) #oggm smb calculation goes only until 2019, and -1 because of next line
    for year in range(mb_start_yr, mb_end_yr+1):
        mb_years.append(mbmod.get_annual_mb(heights, year=year) * secpera * 900)
    mb = [heights, np.mean(np.array(mb_years), axis = 0)]

    for i in range(dem.data[0].shape[0]):
        for j in range(dem.data[0].shape[1]):
            smb.data[0][i,j] = (mb[1][mb[0] == dem.data[0][i,j]])[0]

    if fit_dhdt_regionally is True:
        # calculate dhdt based on regional fitting of elevation trend
        A = gdir.rgi_area_m2
        if A > 1000*2:
            output = 'large'
            dems_large, dems_small, dhdts_large, dhdts_small = partition_dhdt(output=output)
            dhdt_fit, dhdt_fit_field, dem_masked = get_dhdt(RID, dem, dems_large, dhdts_large)
        else:
            output = 'small'
            dems_large, dems_small, dhdts_large, dhdts_small = partition_dhdt(output=output)
            dhdt_fit, dhdt_fit_field, dem_masked = get_dhdt(RID, dem, dems_small, dhdts_small)

    else:
        dhdt_fit_field = deepcopy(dhdt)

    # modify either smb or dhdt so that they balance
    k = 0
    learning_rate = 0.2
    smb_misfit = np.mean(smb.data[0][mask_in.data[0]==1]/900) - np.mean(dhdt_fit_field.data[0][mask_in.data[0]==1])
    while abs(smb_misfit)>0.01:
        smb_misfit = np.mean(smb.data[0][mask_in.data[0]==1]/900) - np.mean(dhdt_fit_field.data[0][mask_in.data[0]==1]) - k
        k += smb_misfit * learning_rate

    if modify_dhdt_or_smb == 'smb':
        smb.data[0] -= k * 900
    else:
        dhdt_fit_field.data[0] += k 
    
    apparent_mb = smb.data[0] - dhdt_fit_field * 900
    #apparent_mb.data[0][mask_in.data[0] == 0] = 0
    # smooth input DEM
    #k = np.ones((3,3))
    #dem.data[0] = ndimage.convolve(dem.data[0], k)/9
    topg = dem.data[0] - thk_oggm

    #slope = np.rad2deg(np.arctan(calc_slope(dem.data[0], dem.rio.resolution()[0])))
    #slope_mask = slope < 35
    #mask_in.data[0] *= slope_mask

    x = dem.x
    y = np.flip(dem.y)

    dem.name = 'usurf'
    dem = dem.squeeze()
    topg.name = 'topg'
    topg = topg.squeeze()
    mask_in.name = 'mask'
    mask_in = mask_in.squeeze()
    mask_in.astype('int')
    dhdt_fit_field.name = 'dhdt'
    dhdt_fit_field = dhdt_fit_field.squeeze()
    smb.name = 'climatic_mass_balance'
    smb.attrs['units'] = 'kg m-2 year-1'
    smb = smb.squeeze()
    apparent_mb.name = 'apparent_mass_balance'
    apparent_mb = apparent_mb.squeeze()
    ice_surface_temperature = deepcopy(dem)
    ice_surface_temperature.data = np.ones_like(dem.data)*273
    ice_surface_temperature.name = 'ice_surface_temp'
    thk = dem - topg
    thk.name = 'thk'

    all_xr = xr.merge([thk, dem, topg, mask_in, dhdt_fit_field, smb, apparent_mb, ice_surface_temperature])
    if output_resolution is not None:
        all_xr = all_xr.rio.reproject(all_xr.rio.crs, resolution = output_resolution, nodata = 3333)
    all_xr = all_xr.reindex(y=list(reversed(all_xr.y)))
    all_xr.to_netcdf(input_file)
    #create_input_nc(input_file, x, y, dem, topg, mask_in, dhdt_fit_field, smb, apparent_mb, ice_surface_temp=273) 


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


def get_dhdt(RID, dem, dems, dhdts):
    sum_arr = np.load('/home/thomas/regional_inversion/all_dhdt_dem.npy')
    dems_m = sm.add_constant(dems)
    res = sm.OLS(dhdts, dems_m).fit()
    loc = sum_arr[0] == RID
    dem_scaled = sum_arr[1, loc].astype(float)
    dhdt = sum_arr[2, loc].astype(float)

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


glaciers_Sweden = get_RIDs_Sweden()
RIDs_Sweden = glaciers_Sweden.RGIId

#for RID in RIDs_Sweden:
#    write_input_file(RID, new_mask = True, period = '2000-2020')
