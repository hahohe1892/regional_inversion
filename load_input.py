import numpy as np
import rioxarray as rioxr
from oggm import cfg, workflow, utils, GlacierDirectory
from shapely.geometry import box
import geopandas as gpd
from netCDF4 import Dataset as NC
import pandas as pd
from funcs import *

#RID = 'RGI60-08.00010'
glacier_dir = '/home/thomas/regional_inversion/input_data/'
#period = '2000-2020'


def load_dhdt_path(RID, period):
    path = glacier_dir + 'dhdt_{}/per_glacier/RGI60-08/RGI60-08.0'.format(period) + RID[10] + '/'+RID+ '/dem.tif'
    dhdt = rioxr.open_rasterio(path)

    return dhdt


def load_dem_path(RID):
    path = glacier_dir + 'DEMs/per_glacier/RGI60-08/RGI60-08.0' + RID[10] + '/'+RID + '/dem.tif'
    dem = rioxr.open_rasterio(path)

    return dem


def load_mask_path(RID):
    path = glacier_dir + 'dhdt_2000-2020/per_glacier/RGI60-08/RGI60-08.0' + RID[10] + '/'+RID + '/gridded_data.nc'
    path_tif = glacier_dir + 'DEMs/per_glacier/RGI60-08/RGI60-08.0' + RID[10] + '/'+RID+ '/dem.tif'
    with utils.ncDataset(path) as nc:
        mask = nc.variables['glacier_mask'][:]
    tif = rioxr.open_rasterio(path_tif)
    tif.data[0, :, :] = np.copy(mask)

    return tif

def in_field(RID, field):
    path = '/home/thomas/regional_inversion/output/' + RID + '/input.nc'
    data = get_nc_data(path, field, ':')

    return data


def crop_border_xarr(xarr, pixels=150):
    res = xarr.rio.resolution()[0]
    x_min = float(xarr.x.min()) + pixels * res
    x_max = float(xarr.x.max()) - pixels * res
    y_min = float(xarr.y.min()) + pixels * res
    y_max = float(xarr.y.max()) - pixels * res
    geodf = gpd.GeoDataFrame(
        geometry=[
            box(x_min, y_min, x_max, y_max)],
        crs=xarr.rio.crs)
    clipped = xarr.rio.clip(geodf.geometry)

    return clipped


def crop_to_xarr(xarr_target, xarr_source):
    geodf = gpd.GeoDataFrame(
        geometry=[
            box(xarr_source.x.min(), xarr_source.y.min(), xarr_source.x.max(), xarr_source.y.max())],
        crs=xarr_source.rio.crs)
    clipped = xarr_target.rio.clip(geodf.geometry, all_touched = True)

    return clipped


def crop_border_arr(arr, pixels=150):
    return arr[pixels:-pixels, pixels:-pixels]


def load_dhdt_gdir(RID, period):
    cfg.initialize(logging_level='WARNING')
    cfg.PATHS['working_dir'] = '~/regional_inversion/input_data/dhdt_' + period
    gdir = workflow.init_glacier_directories(RID)

    return gdir


def load_dem_gdir(RID):
    cfg.initialize(logging_level='WARNING')
    cfg.PATHS['working_dir'] = '~/regional_inversion/input_data/DEMs'
    gdir = workflow.init_glacier_directories(RID)

    return gdir


def create_nc(vars, WRIT_FILE):
    ncfile = NC(WRIT_FILE, 'w', format='NETCDF3_CLASSIC')
    xdim = ncfile.createDimension('x', int(vars['x'][-1].shape[0]))
    ydim = ncfile.createDimension('y', int(vars['y'][-1].shape[0]))

    for name in list(vars.keys()):
        [_, _, _, fill_value, data] = vars[name]
        if name in ['x', 'y']:
            var = ncfile.createVariable(name, 'f4', (name,))
        else:
            var = ncfile.createVariable(name, 'f4', ('y', 'x'), fill_value=fill_value)
        for each in zip(['units', 'long_name', 'standard_name'], vars[name]):
            if each[1]:
                setattr(var, each[0], each[1])
        var[:] = data

    # finish up
    ncfile.close()
    print("NetCDF file " + WRIT_FILE + " created")


def create_input_nc(file, x, y, dem, topg, mask, dhdt, smb, apparent_mb, ice_surface_temp=273):
    vars = {'y':    ['m',
                     'y-coordinate in Cartesian system',
                     'projection_y_coordinate',
                     None,
                     y],
            'x':    ['m',
                     'x-coordinate in Cartesian system',
                     'projection_x_coordinate',
                     None,
                     x],
            'thk':  ['m',
                     'floating ice shelf thickness',
                     'land_ice_thickness',
                     None,
                     dem - topg],
            'topg': ['m',
                     'bedrock surface elevation',
                     'bedrock_altitude',
                     None,
                     topg],
            'usurf': ['m',
                      'landscape surface',
                      'surf',
                      None,
                      dem],
            'ice_surface_temp': ['K',
                                 'annual mean air temperature at ice surface',
                                 'surface_temperature',
                                 None,
                                 ice_surface_temp],
            'climatic_mass_balance': ['kg m-2 year-1',
                                      'mean annual net ice equivalent accumulation rate',
                                      'land_ice_surface_specific_mass_balance_flux',
                                      None,
                                      smb],
            'precip': ['kg m-2 year-1',
                                      'not actually precip, but mass balance minus dhdt',
                                      'apparent_mb',
                                      None,
                                      apparent_mb],
            'dhdt': ['m/yr',
                     "rate of surface elevation change",
                     'dhdt',
                     None,
                     dhdt],
            'mask': ['',
                     'ice extent mask',
                     'mask',
                     None,
                     mask],
            }
    create_nc(vars, file)


def get_RIDs_Sweden(file = glacier_dir + 'Glaciers_Sweden.txt'):
    return pd.read_table(file, delimiter = ';')


def get_gdir_info(RID):
    cfg.initialize(logging_level='WARNING')
    cfg.PATHS['working_dir'] = '~/regional_inversion/input_data/DEMs'    
    return GlacierDirectory(glacier_dir + 'DEMs/per_glacier/RGI60-08/RGI60-08.0' + RID[10] + '/'+ RID)
