import numpy as np
import rioxarray as rioxr
from oggm import cfg, workflow, utils, GlacierDirectory
from funcs import *
import shutil
from load_input import *
import datetime
import time

def nc_out(RID, field, i=-1, file = 'output.nc'):
    dem = load_dem_path(RID)
    dem = crop_border_xarr(dem)
    nc = get_nc_data('/home/thomas/regional_inversion/output/' + RID + '/' + file, field, i)
    dem.data[0] = nc

    return dem


def out_to_tif(RID, field, i=-1, file = 'output.nc'):
    out = nc_out(RID, field, i, file = file)
    path = '/home/thomas/regional_inversion/output/' + RID + '/' + field + '.tif'
    out.rio.to_raster(path)


def all_out_to_Win(field, file = 'output.nc', date = '01/01/01/1970'):
    ''' date should be given as "hh/dd/mm/yyyy'''
    glaciers_Sweden = get_RIDs_Sweden()
    RIDs_Sweden = glaciers_Sweden.RGIId

    for RID in RIDs_Sweden:
        try:
            path = '/home/thomas/regional_inversion/output/' + RID + '/' + file
            date_unix = time.mktime(datetime.datetime.strptime(date, "%H/%d/%m/%Y").timetuple())
            file_time = os.path.getmtime(path)
            if file_time > date_unix:
                out_to_tif(RID, field, file = file, i = ':')
                shutil.copy('/home/thomas/regional_inversion/output/' + RID + '/'+ field + '.tif', '/mnt/c/Users/thofr531/Documents/Global/Scandinavia/outputs/' + RID + '_' + field + '.tif')
        except FileNotFoundError:
            print('field {} does not exist for glacier {}'.format(field, RID))
            continue

def dem_to_Win(date = '01/01/1970'):
    ''' date should be given as "dd/mm/yyyy'''
    glaciers_Sweden = get_RIDs_Sweden()
    RIDs_Sweden = glaciers_Sweden.RGIId

    for RID in RIDs_Sweden:
        try:
            path = '/home/thomas/regional_inversion/input_data/DEMs/per_glacier/RGI60-08/RGI60-08.0' + RID[10] + '/'+RID + '/dem.tif'
            date_unix = time.mktime(datetime.datetime.strptime(date, "%d/%m/%Y").timetuple())
            file_time = os.path.getmtime(path)
            if file_time > date_unix:
                shutil.copy(path, '/mnt/c/Users/thofr531/Documents/Global/Scandinavia/DEMs/' + RID + '_dem.tif')
        except FileNotFoundError:
            print('dem for glacier {} not found'.format(RID))


def get_all_output(field, in_or_out = 'out'):
    glaciers_Sweden = get_RIDs_Sweden()
    RIDs_Sweden = glaciers_Sweden.RGIId

    all_out = []
    for RID in RIDs_Sweden:
        if in_or_out == 'out':
            data = nc_out(RID, field)
        elif in_or_out == 'in':
            data = nc_out(RID, field, file = 'input.nc', i = ':')
        else:
            raise ValueError('neither input nor output recognized as data source')
        all_out.extend(data.data[0].flatten())
    return np.array(all_out)


def raw_mask_out_to_Win(field = 'mask', file = 'gridded_data.nc'):
    ''' date should be given as "dd/mm/yyyy'''
    glaciers_Sweden = get_RIDs_Sweden()
    RIDs_Sweden = glaciers_Sweden.RGIId

    for RID in RIDs_Sweden:
        path_dem = '/home/thomas/regional_inversion/input_data/DEMs/per_glacier/RGI60-08/RGI60-08.0' + RID[10] + '/'+RID+ '/dem.tif'
        path = '/home/thomas/regional_inversion/input_data/DEMs/per_glacier/RGI60-08/RGI60-08.0' + RID[10] + '/'+RID + '/gridded_data.nc'
        data = get_nc_data(path, 'glacier_' + field, ':')
        dem = rioxr.open_rasterio(path_dem)
        dem.data[0, :, :] = np.copy(data)
        path_out = '/home/thomas/regional_inversion/input_data/outlines/raw_masks/mask_' + RID + '.tif'
        dem.rio.to_raster(path_out)
        shutil.copy(path_out, '/mnt/c/Users/thofr531/Documents/Global/Scandinavia/08_rgi60_Scandinavia/raw_masks/mask_' + RID + '.tif')


