import numpy as np
from load_input import *
import PISM
from bed_inversion import *
import os
import shutil
from oggm.core import massbalance

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
    dhdt = crop_border_xarr(dhdt, pixels = 30)

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
    create_input_nc(input_file, x, y, dem, topg, mask_in, dhdt, smb, ice_surface_temp=273)

    