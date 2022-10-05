import numpy as np
import rioxarray as rioxr
from oggm import cfg, workflow, utils, GlacierDirectory
from funcs import *


def nc_out(RID, field, i=-1):
    dem = load_dem_path(RID)
    dem = crop_border_xarr(dem)
    nc = get_nc_data('/home/thomas/regional_inversion/output/' + RID + '/extra.nc', field, i)
    dem.data[0] = nc

    return dem


def out_to_tiff(RID, field, i=-1):
    out = nc_out(RID, field, i)
    path = '/home/thomas/regional_inversion/output/' + RID + '/' + field + '.tif'
    out.rio.to_raster(path)


