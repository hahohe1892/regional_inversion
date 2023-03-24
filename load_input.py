import numpy as np
import rioxarray as rioxr
from oggm import cfg, workflow, utils, GlacierDirectory
from shapely.geometry import box
import geopandas as gpd
from netCDF4 import Dataset as NC
import pandas as pd
from funcs import *
from rasterio import merge

#RID = 'RGI60-08.00005'
glacier_dir = '/home/thomas/regional_inversion/input_data/'
#period = '2000-2020'


def load_dhdt_path(RID, period):
    RGI_region = RID.split('-')[1].split('.')[0]
    path = glacier_dir + 'dhdt_{}/per_glacier/RGI60-'.format(period) + RGI_region + '/RGI60-' + RGI_region + '.0' + RID[10] + '/'+RID+ '/dem.tif'
    dhdt = rioxr.open_rasterio(path)

    return dhdt


def load_dem_path(RID):
    RGI_region = RID.split('-')[1].split('.')[0]
    path = glacier_dir + 'DEMs/per_glacier/RGI60-' + RGI_region + '/RGI60-' + RGI_region + '.0'  + RID[10] + '/'+RID + '/dem.tif'
    dem = rioxr.open_rasterio(path)

    return dem


def load_mask_path(RID, mask_new = False):
    RGI_region = RID.split('-')[1].split('.')[0]
    path_tif = glacier_dir + 'DEMs/per_glacier/RGI60-' + RGI_region + '/RGI60-' + RGI_region + '.0'  + RID[10] + '/'+RID + '/dem.tif'
    path = glacier_dir + 'dhdt_2000-2020/per_glacier/RGI60-' + RGI_region + '/RGI60-' + RGI_region + '.0' + RID[10] + '/'+RID + '/gridded_data_new.nc'
    if not os.path.isfile(path):
        path = glacier_dir + 'DEMs/per_glacier/RGI60-' + RGI_region + '/RGI60-' + RGI_region + '.0'  + RID[10] + '/'+RID + '/gridded_data.nc'
    with utils.ncDataset(path) as nc:
        if mask_new is True:
            mask = nc.variables['mask_new'][:]
            tif = rioxr.open_rasterio(path_tif)
            tif.data[0, :, :] = (mask).T
        else:
            mask = nc.variables['glacier_mask'][:]
            tif = rioxr.open_rasterio(path_tif)
            tif.data[0, :, :] = mask

    return tif


def load_consensus_thk(RID):
    RGI_region = RID.split('-')[1].split('.')[0]
    path = glacier_dir + 'consensus_thk/RGI60-' + RGI_region + '/' + RID + '_thickness.tif'
    return rioxr.open_rasterio(path)

    
def load_georeferenced_mask(RID):
    path = glacier_dir + 'outlines/georeferenced_masks/mask_{}_new.tif'.format(RID)
    mask = rioxr.open_rasterio(path)

    return mask


def load_thk_path(RID):
    RGI_region = RID.split('-')[1].split('.')[0]
    path_tif = glacier_dir + 'DEMs/per_glacier/RGI60-' + RGI_region + '/RGI60-' + RGI_region + '.0'  + RID[10] + '/'+RID + '/dem.tif'
    path = glacier_dir + 'dhdt_2000-2020/per_glacier/RGI60-' + RGI_region + '/RGI60-' + RGI_region + '.0' + RID[10] + '/'+RID + '/gridded_data_new.nc'
    if not os.path.isfile(path):
        path = glacier_dir + 'DEMs/per_glacier/RGI60-' + RGI_region + '/RGI60-' + RGI_region + '.0'  + RID[10] + '/'+RID + '/gridded_data.nc'
    
    #path_tif = glacier_dir + 'DEMs/per_glacier/RGI60-08/RGI60-08.0' + RID[10] + '/'+RID+ '/dem.tif'
    #path = glacier_dir + 'DEMs/per_glacier/RGI60-08/RGI60-08.0' + RID[10] + '/'+RID + '/gridded_data.nc'
    with utils.ncDataset(path) as nc:
        thk = nc.variables['consensus_ice_thickness'][:]
    tif = rioxr.open_rasterio(path_tif)
    tif.data[0, :, :] = np.copy(thk)

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


def crop_to_new_mask(xarr, mask, pixels):
    x_grid, y_grid = np.meshgrid(xarr.x, xarr.y)
    res = xarr.rio.resolution()[0]
    x_min = np.min(x_grid[mask.data[0]==1]) - pixels * res
    x_max = np.max(x_grid[mask.data[0]==1]) + pixels * res
    y_min = np.min(y_grid[mask.data[0]==1]) - pixels * res
    y_max = np.max(y_grid[mask.data[0]==1]) + pixels * res
    geodf = gpd.GeoDataFrame(
        geometry=[
            box(x_min, y_min, x_max, y_max)],
        crs=xarr.rio.crs)
    clipped = xarr.rio.clip(geodf.geometry)

    return clipped

        
def crop_to_xarr(xarr_target, xarr_source, from_disk=False):
    geodf = gpd.GeoDataFrame(
        geometry=[
            box(xarr_source.x.min(), xarr_source.y.min(), xarr_source.x.max(), xarr_source.y.max())],
        crs=xarr_source.rio.crs)
    clipped = xarr_target.rio.clip(geodf.geometry, all_touched = True, from_disk=from_disk)

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
            'apparent_mass_balance': ['kg m-2 year-1',
                                      'mass balance minus dhdt',
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


def read_pkl(file):
    return pd.read_pickle(file, 'gzip')


def print_all():
    RIDs = get_RIDs_Sweden()
    RIDs_Sweden = RIDs.RGIId
    for RID in RIDs_Sweden:
        gdir = load_dem_gdir(RID)
        print(gdir[0].form)


def get_mb_Rounce(RID, last_n_years = 20, standardize = False):
    RID_id = RID.split('-')[1]
    if RID_id[0] == '0':
        RID_id = RID_id[1:]
    mb_xr = rioxr.open_rasterio(os.path.join(glacier_dir, 'mass_balance', RID_id + '_ERA5_MCMC_ba1_50sets_1979_2019_binned.nc'))
    elevation_bins = np.flip(mb_xr[0].bin_surface_h_initial[0][0])
    mass_balance = mb_xr[1].bin_massbalclim_annual
    mass_balance_last_n_years = mass_balance[0][:,-last_n_years:]
    mass_balance_mean = mass_balance_last_n_years.mean(axis = 1)
    if standardize is True:
        mass_balance_mean = mass_balance_mean.assign_coords(y = (elevation_bins.data-np.min(elevation_bins.data))/(np.max(elevation_bins.data) - np.min(elevation_bins.data)))
    else:
        mass_balance_mean = mass_balance_mean.assign_coords(y = elevation_bins.data)
    return mass_balance_mean


def nearby_glacier(RID, RGI, buffer_width):
    '''
    supply a geodataframe with RGI outlines such as obtained from:
    fr = utils.get_rgi_region_file(RGI_region, version='62')
    gdf = gpd.read_file(fr)
    gdf_crs = gdf.to_crs(some_crs)
    '''

    def neighboring_glacier(RID, RGI, buffer_width):
        glacier_buffer = RGI[RGI.RGIId == RID].buffer(buffer_width)
        intersection = RGI.intersection(glacier_buffer.iloc[0])
        intersection_indices = np.nonzero(~intersection.is_empty.to_numpy())
        intersection_RIDs = RGI.iloc[intersection_indices].RGIId
        return intersection_RIDs
    area_RIDs = neighboring_glacier(RID, RGI, buffer_width).to_list()
    old_len = 0
    while len(area_RIDs) > old_len:
        old_len = len(area_RIDs)
        print(old_len)
        new_area_RIDs = []
        for area_RID in area_RIDs:
            new_area_RIDs.extend(neighboring_glacier(area_RID, RGI, 200))
        area_RIDs.extend(new_area_RIDs)
        area_RIDs = np.unique(area_RIDs).tolist()

    return area_RIDs


def obtain_area_mosaic(RID):
    working_dir = '/home/thomas/regional_inversion/output/' + RID
    input_file = working_dir + '/input.nc'
    Fill_Value = 9999.0

    RGI_region = RID.split('-')[1].split('.')[0]
    input_igm = rioxr.open_rasterio(input_file)
    input_igm = input_igm.squeeze()
    input_igm = input_igm.where(input_igm.mask != 0)
    input_igm.attrs['_FillValue'] = Fill_Value
    for var in input_igm.data_vars:
        input_igm.data_vars[var].rio.write_nodata(Fill_Value, inplace=True)
    input_igm = input_igm.fillna(Fill_Value)
    fr = utils.get_rgi_region_file(RGI_region, version='62')
    gdf = gpd.read_file(fr)
    gdf_crs = gdf.to_crs(input_igm.rio.crs)
    area_RIDs = nearby_glacier(RID, gdf_crs, 200)
    mosaic_list = [input_igm]
    for glacier in area_RIDs[1:]:
        glacier_dir = '/home/thomas/regional_inversion/output/' + glacier
        input_file = glacier_dir + '/input.nc'
        input = rioxr.open_rasterio(input_file)
        input = input.squeeze()
        input.attrs['_FillValue'] = Fill_Value
        input = input.where(input.mask != 0)
        for var in input.data_vars:
            input.data_vars[var].rio.write_nodata(Fill_Value, inplace=True)
        input = input.fillna(Fill_Value)
        input = input.rio.reproject(input_igm.rio.crs)
        mosaic_list.append(input)
    mosaic = rioxr.merge.merge_datasets(mosaic_list)#, nodata = 9999.0)#, method = 'max')
    mosaic = mosaic.where(mosaic != Fill_Value)
    return mosaic
