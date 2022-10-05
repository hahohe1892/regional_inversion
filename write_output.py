import numpy as np
import rioxarray as rioxr
from oggm import cfg, workflow, utils, GlacierDirectory
from funcs import *
import shutil

def nc_out(RID, field, i=-1):
    dem = load_dem_path(RID)
    dem = crop_border_xarr(dem)
    nc = get_nc_data('/home/thomas/regional_inversion/output/' + RID + '/output.nc', field, i)
    dem.data[0] = nc

    return dem


def out_to_tif(RID, field, i=-1):
    out = nc_out(RID, field, i)
    path = '/home/thomas/regional_inversion/output/' + RID + '/' + field + '.tif'
    out.rio.to_raster(path)


def all_out_to_Win(field):
    glaciers_Sweden = get_RIDs_Sweden()
    RIDs_Sweden = glaciers_Sweden.RGIId

    for RID in RIDs_Sweden:
        try:
            out_to_tif(RID, field)
            shutil.move('/home/thomas/regional_inversion/output/' + RID + '/'+ field + '.tif', '/mnt/c/Users/thofr531/Documents/Global/Scandinavia/outputs/' + RID + '_' + field + '.tif')
        except FileNotFoundError:
            print('field {} does not exist for glacier {}'.format(field, RID))
            continue
