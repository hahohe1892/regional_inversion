from oggm import utils, workflow, cfg, exceptions
from oggm.core import gis
import geopandas as gpd
import os
from oggm.shop import bedtopo
from load_input import *
import geopy.distance
import xarray as xr
import rasterio

#periods = ['2000-2005', '2005-2010', '2010-2015', '2015-2020']

def get_DEM_dir(RID, DEM_file = None, DEM_Dir_out = '/home/thomas/regional_inversion/input_data/DEMs', custom_dx = None, obtain_consensus_thk_with_oggm = True):

    '''
    if custom_dx is set but no custom DEM_file is specified, custom_dx will be ignored (which is due to the fact
    that oggm then downloads the dem file from a sever in whatever resolution it is stored there)
    '''
    if DEM_file is not None:
        DEM_file = '/home/thomas/regional_inversion/input_data/' + DEM_file
        if not os.path.isfile(DEM_file):
            raise FileNotFoundError('no file found from which to extract DEM values; check DEM_file')

    cfg.initialize(logging_level='CRITICAL')
    cfg.PATHS['working_dir'] = DEM_Dir_out
    if custom_dx is not None:
        cfg.PARAMS['grid_dx_method'] = 'fixed'
        cfg.PARAMS['fixed_dx'] = custom_dx
    if os.path.isfile(cfg.PATHS['dem_file']):
        cfg.PATHS['dem_file'] = 'bla'
    RGI_region = RID.split('-')[1].split('.')[0]
    fr = utils.get_rgi_region_file(RGI_region, version='62')
    gdf = gpd.read_file(fr)
    if DEM_file is not None:
        cfg.PATHS['dem_file'] = DEM_file
    gdirs = workflow.init_glacier_directories(gdf[gdf.RGIId == RID], from_prepro_level = 3, prepro_border=40)

    if DEM_file is not None:
        print('obtained DEM from {} and placed it in subdirectory of {}'.format(DEM_file, DEM_Dir_out))
        gis.define_glacier_region(gdirs[0], source = 'USER')
    else:
        print('obtained DEM from oggm bedtopo (likely COPDEM) and placed it in subdirectory of {}'.format(DEM_Dir_out))
    if obtain_consensus_thk_with_oggm is True:
        workflow.execute_entity_task(bedtopo.add_consensus_thickness, gdirs)

def get_dhdt_dir(RID, dhdt_file, dhdt_period = '2000-2020',  dhdt_Dir_out = '/home/thomas/regional_inversion/input_data/dhdt_', custom_dx = None):

    '''
    dhdt is provided as m/yr. For conversion to m.w.eq. a factor of 0.85 needs to be applied as in Hugonnet et al. (2021)
    '''
    dhdt_Dir_out = dhdt_Dir_out + dhdt_period
    dhdt_file = '/home/thomas/regional_inversion/input_data/dhdt_all/dhdt/' + dhdt_file
    if not os.path.isfile(dhdt_file):
        raise FileNotFoundError('no file found from which to extract dhdt values; check dhdt_file')
    RGI_region = RID.split('-')[1].split('.')[0]
    cfg.initialize(logging_level='CRITICAL')
    cfg.PATHS['working_dir'] = dhdt_Dir_out
    if custom_dx is not None:
        cfg.PARAMS['grid_dx_method'] = 'fixed'
        cfg.PARAMS['fixed_dx'] = custom_dx
    fr = utils.get_rgi_region_file(RGI_region, version='62')
    gdf = gpd.read_file(fr)
    cfg.PATHS['dem_file'] = dhdt_file
    gdirs = workflow.init_glacier_directories(gdf[gdf.RGIId == RID], from_prepro_level = 2, prepro_border=160)
    gis.define_glacier_region(gdirs[0], source = 'USER')
    print('obtained dhdt for period {} and placed it in subdirectory of {}'.format(dhdt_period, dhdt_Dir_out))


def get_vel_dir(RID, custom_dx = None):
    vel_Dir_out = '/home/thomas/regional_inversion/input_data/vel_Millan'
    RGI_region = RID.split('-')[1].split('.')[0]

    vel_file = os.path.join('/home/thomas/regional_inversion/input_data/vel_Millan', 'RGI-' + str(int(RGI_region)), 'vel_mosaic_Millan.tif')

    if not os.path.isfile(vel_file):
        raise FileNotFoundError('no file found from which to extract vel values; check vel_file')

    cfg.initialize(logging_level='WARNING')
    cfg.PATHS['working_dir'] = vel_Dir_out
    if custom_dx is not None:
        cfg.PARAMS['grid_dx_method'] = 'fixed'
        cfg.PARAMS['fixed_dx'] = custom_dx
    fr = utils.get_rgi_region_file(RGI_region, version='62')
    gdf = gpd.read_file(fr)

    cfg.PATHS['dem_file'] = vel_file
    gdirs = workflow.init_glacier_directories(gdf[gdf.RGIId == RID], from_prepro_level = 2, prepro_border=160, reset = True, force = True)
    gis.define_glacier_region(gdirs[0], source = 'USER')
    print('obtained velocity and placed it in subdirectory of {}'.format(vel_Dir_out))


#glaciers_Sweden = get_RIDs_Sweden()
#RIDs_Sweden = glaciers_Sweden.RGIId

#poly = gdf.geometry.loc[4]

def mask_from_polygon(poly, gdf, out_path = '/home/thomas/regional_inversion/test.tif', projected = True):
    '''
    creates a 30 m resolution mask based on a polygone outline for a glacier with a specified buffer around it. 
    '''
    if projected is False:
        dy = geopy.distance.distance((poly.bounds[3], poly.bounds[0]), (poly.bounds[1], poly.bounds[0])).m
        dx = geopy.distance.distance((poly.bounds[1], poly.bounds[0]),  (poly.bounds[1], poly.bounds[2])).m
        margin_deg = 0.1 #how much space around polygon should there be (in deg)?
        dx_margin = geopy.distance.distance((poly.bounds[1], poly.bounds[0]-margin_deg),  (poly.bounds[1], poly.bounds[0])).m

    else:
        dy = poly.bounds[3] - poly.bounds[1]
        dx = poly.bounds[2] - poly.bounds[0]
        dx_margin = 1000

    nx = int((dx + 2*dx_margin)/30)
    ny = int((dy + 2*dx_margin)/30)

    if projected is False:
        lats = np.linspace(poly.bounds[1]-margin_deg,poly.bounds[3]+margin_deg,ny)
        lons = np.linspace(poly.bounds[0]-margin_deg, poly.bounds[2]+margin_deg,nx)
    else:
        lats = np.linspace(poly.bounds[1]-dx_margin,poly.bounds[3]+dx_margin,ny)
        lons = np.linspace(poly.bounds[0]-dx_margin, poly.bounds[2]+dx_margin, nx)


    # create empty xr dataset
    ds = xr.Dataset({
        'mask': xr.DataArray(
            data = np.ones((ny,nx), dtype = 'int'),
            dims = ['latitude','longitude'],
            coords = {'latitude': lats, 'longitude': lons},
            attrs = {'long_name': 'ice_mask', 'units': 'bool'}
            )})
    ds = ds.rio.write_crs(gdf.crs)
    ds.rio.to_raster(out_path, compress='LZW')

    with rasterio.open(out_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, [poly], crop=False)
        out_meta = src.meta

    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(out_image)

    print('obtained mask based on RGI inventory and placed it in {}'.format(out_path))


def fill_nans_mosaic(mosaic, RGI_region = '08'):
    '''
    This may be the most accurate way of interpolating nan values of dhdt.
    However, it crashes the RAM if run for the whole RGI region.
    Instead, simply interpolating (e.g. in write_input_file_Sweden_Norway)
    is probably also fine.
    '''
    fr = utils.get_rgi_region_file(RGI_region, version='62')
    gdf = gpd.read_file(fr)
    gdf = gdf.to_crs(mosaic.rio.crs) # project RGI inventory to prevent distortion
    #gdf = gdf.buffer(500)
    gdf = gdf[gdf.RGIId == RID].buffer(500)
    mask = deepcopy(mosaic)
    mask.data[0] = np.ones_like(mask.data[0])
    mask = mask.rio.clip(gdf.geometry, drop = True)
    mask.data[0][mask.data[0] == mask.rio.nodata] = 0
    mosaic = mosaic.rio.clip(gdf.geometry, drop = True)
    mosaic.data[0] = mosaic.data[0]*mask.data[0]
    mosaic = mosaic.rio.interpolate_na()
    
    
def get_complete_input(RGI_region):
    fr = utils.get_rgi_region_file(RGI_region, version='62')
    gdf = gpd.read_file(fr)
    gdf = gdf.to_crs(gdf.crs.from_epsg('32632')) # project RGI inventory to prevent distortion
    for RID in gdf.RGIId:
        get_DEM_dir(RID, DEM_file = 'DEM_Sweden/DEM_mosaic_Sweden.tif', custom_dx = 100, obtain_consensus_thk_with_oggm = False, DEM_Dir_out = '/home/thomas/regional_inversion/input_data/DEM_Sweden')
        get_dhdt_dir(RID, dhdt_file = 'mosaic_2000-2020.tif', custom_dx = 100)
        get_DEM_dir(RID, DEM_file = 'DEM_Norway/DEM_mosaic_Norway.tif', custom_dx = 100, obtain_consensus_thk_with_oggm = False, DEM_Dir_out = '/home/thomas/regional_inversion/input_data/DEM_Norway')
        get_vel_dir(RID, custom_dx = 100)

        mask_path = '/home/thomas/regional_inversion/input_data/masks/per_glacier/RGI60-' + RGI_region + '/RGI60-' + RGI_region + '.0' + RID[10] + '/'+RID
        if not os.path.isdir(mask_path):
            os.makedirs(mask_path)
        mask_from_polygon(gdf[gdf.RGIId == RID].geometry.iloc[0], gdf, out_path = os.path.join(mask_path, RID + '_mask.tif'))
