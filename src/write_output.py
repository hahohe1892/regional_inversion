import numpy as np
import rioxarray as rioxr
from oggm import cfg, workflow, utils, GlacierDirectory
from funcs import *
import shutil
from load_input import *
import datetime
import time
import subprocess

def nc_out(RID, field, i=-1, file = 'output.nc', file_not_standard_dims = False, flip = False):
    dem = load_dem_path(RID)
    dem = crop_border_xarr(dem)
    if file_not_standard_dims is True:
        nc = rioxr.open_rasterio('/home/thomas/regional_inversion/output/' + RID + '/' + file)[field][i]
        if flip is True:
            nc.data = np.flip(nc.data, axis = 0)
        nc = nc.rio.write_crs(dem.rio.crs)
        return nc
    else:
        nc = get_nc_data('/home/thomas/regional_inversion/output/' + RID + '/' + file, field, i)
        dem.data[0] = nc

        return dem


def out_to_tif(RID, field, i=-1, file = 'output.nc', threshold = np.nan, file_not_standard_dims = False, flip = False):
    out = nc_out(RID, field, i, file = file, file_not_standard_dims = file_not_standard_dims, flip = flip)
    path = '/home/thomas/regional_inversion/output/' + RID + '/' + field + '.tif'
    if not np.isnan(threshold):
        out.data[0][out.data[0]<threshold] = out.attrs['_FillValue']
    out.rio.to_raster(path)


def all_out_to_Win(field, i = ':', file = 'output.nc', date = '01/01/01/1970', threshold=np.nan, file_not_standard_dims = False, flip = False, only_Sweden = False):
    ''' date should be given as "hh/dd/mm/yyyy'''
    if only_Sweden is True:
        glaciers_Sweden = get_RIDs_Sweden()
        RIDs = glaciers_Sweden.RGIId
    else:
        RGI_region = '08'
        fr = utils.get_rgi_region_file(RGI_region, version='62')
        gdf = gpd.read_file(fr)
        RIDs = gdf.RGIId.to_list()[4:]

    for RID in RIDs:
        try:
            path = '/home/thomas/regional_inversion/output/' + RID + '/' + file
            date_unix = time.mktime(datetime.datetime.strptime(date, "%H/%d/%m/%Y").timetuple())
            file_time = os.path.getmtime(path)
            if file_time > date_unix:
                out_to_tif(RID, field, file = file, i = i, threshold = threshold, file_not_standard_dims = file_not_standard_dims, flip = flip)
                shutil.copy('/home/thomas/regional_inversion/output/' + RID + '/'+ field + '.tif', '/mnt/c/Users/thofr531/Documents/Global/Scandinavia/outputs/' + RID + '_' + field + '.tif')
                print(RID + ' done')
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


def retrieve_from_HPC(glaciers, file = 'ex.nc', date = '01/01/01/1970', new_name = 'ex.nc'):
    for glacier in glaciers:
        remote_path = '/mimer/NOBACKUP/groups/snic2022-22-55/regional_inversion/output/{}/{}'.format(glacier, file)
        mosaic_ref_path = '/mimer/NOBACKUP/groups/snic2022-22-55/regional_inversion/output/{}/mosaic_reference.txt'.format(glacier)
        date_unix = time.mktime(datetime.datetime.strptime(date, "%H/%d/%m/%Y").timetuple())
        # retrieve mosaic reference for all glaciers, regardless of when they were modelled last
        subprocess.call(['scp', 'alvis2:' + mosaic_ref_path, '/home/thomas/regional_inversion/output/{}/'.format(glacier)])
        try:
            file_time = int(subprocess.check_output(['ssh', 'alvis2', 'stat -c %Y', remote_path]).strip())
        except(subprocess.CalledProcessError):
            continue
        if file_time > date_unix:
            print(glacier)
            if new_name == 'ex.nc':
                subprocess.call(['scp', 'alvis2:' + remote_path, '/home/thomas/regional_inversion/output/{}/'.format(glacier)])
            else:
                subprocess.call(['scp', 'alvis2:' + remote_path, '/home/thomas/regional_inversion/output/{}/{}'.format(glacier, new_name)])


def get_mosaic_reference(RID):
    mosaic_reference = '/home/thomas/regional_inversion/output/{}/mosaic_reference.txt'.format(RID)
    file_content = []
    with open(mosaic_reference, 'r') as fp:
            for item in fp:
                file_content.append(item)
    return file_content[-1]


def copy_all_ex_to_Win(date = '01/01/01/1970'):
    RGI_region = '08'
    fr = utils.get_rgi_region_file(RGI_region, version='62')
    gdf = gpd.read_file(fr)
    RIDs = gdf.RGIId.to_list()[4:]
    
    for RID in RIDs:
        try:
            path = '/home/thomas/regional_inversion/output/' + RID + '/ex.nc'
            date_unix = time.mktime(datetime.datetime.strptime(date, "%H/%d/%m/%Y").timetuple())
            file_time = os.path.getmtime(path)
            if file_time > date_unix:
                shutil.copy('/home/thomas/regional_inversion/output/' + RID + '/ex.nc', '/mnt/c/Users/thofr531/Documents/Global/Scandinavia/outputs/' + RID + '_ex.nc')
                print(RID + ' done')
        except FileNotFoundError:
            print('ex.nc does not exist for glacier {}'.format(RID))
            continue

def rename_to_version(glaciers, old_name, new_name):
    for glacier in glaciers:
        print('renaming ' + glacier + '...')
        path = '/home/thomas/regional_inversion/output/{}/'.format(glacier)
        if os.path.exists(path + old_name):
            subprocess.call(['mv', path + old_name, path + new_name])
            print('...done')
        else:
            print('{} not found; skipping'.format(path + old_name))
            continue


def ex_on_Win_to_output(glaciers, version):
    for glacier in glaciers:
        print('moving ' + glacier + '...')
        Win_path = '/mnt/c/Users/thofr531/Documents/Global/Scandinavia/outputs/{}/ex/{}_ex.nc'.format(version, glacier)
        path = '/home/thomas/regional_inversion/output/{}/'.format(glacier)
        if os.path.exists(Win_path):
            subprocess.call(['cp', Win_path, path + 'ex_{}.nc'.format(version)])
            print('...done')
        else:
            print('{} not found; skipping'.format(path + old_name))
            continue
    
