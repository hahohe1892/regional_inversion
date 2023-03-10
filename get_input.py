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
    gdirs = workflow.init_glacier_directories(gdf[gdf.RGIId == RID], from_prepro_level = 3, prepro_border=160)

    if DEM_file is not None:
        print('obtained DEM from {} and placed it in subdirectory of {}'.format(DEM_file, DEM_Dir_out))
        gis.define_glacier_region(gdirs[0], source = 'USER')
    else:
        print('obtained DEM from oggm bedtopo (likely COPDEM) and placed it in subdirectory of {}'.format(DEM_Dir_out))
    if obtain_consensus_thk_with_oggm is True:
        workflow.execute_entity_task(bedtopo.add_consensus_thickness, gdirs)

def get_dhdt_dir(RID, dhdt_file, dhdt_period = '2000-2020',  dhdt_Dir_out = '/home/thomas/regional_inversion/input_data/dhdt_', custom_dx = None):
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

def mask_from_polygon(poly, out_path = '/home/thomas/regional_inversion/test.tif'):
    dy = geopy.distance.distance((poly.bounds[3], poly.bounds[0]), (poly.bounds[1], poly.bounds[0])).m
    dx = geopy.distance.distance((poly.bounds[1], poly.bounds[0]),  (poly.bounds[1], poly.bounds[2])).m
    margin_deg = 0.1 #how much space around polygon should there be (in deg)?
    dx_margin = geopy.distance.distance((poly.bounds[1], poly.bounds[0]-margin_deg),  (poly.bounds[1], poly.bounds[0])).m

    nx = int((dx + 2*dx_margin)/30)
    ny = int((dy + 2*dx_margin)/30)

    lats = np.linspace(poly.bounds[1]-margin_deg,poly.bounds[3]+margin_deg,ny)
    lons = np.linspace(poly.bounds[0]-margin_deg, poly.bounds[2]+margin_deg,nx)

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


def get_complete_input():
    fr = utils.get_rgi_region_file(RGI_region, version='62')
    gdf = gpd.read_file(fr)
    for RID in gdf.RGIId:
        get_DEM_dir(RID, DEM_file = 'DEM_Sweden/DEM_mosaic_Sweden.tif', custom_dx = 100, obtain_consensus_thk_with_oggm = False, DEM_Dir_out = '/home/thomas/regional_inversion/input_data/DEM_Sweden')
        get_dhdt_dir(RID, dhdt_file = 'mosaic_2000-2020.tif', custom_dx = 100)
        get_DEM_dir(RID, DEM_file = 'DEM_Norway/DEM_mosaic_Norway.tif', custom_dx = 100, obtain_consensus_thk_with_oggm = False, DEM_Dir_out = '/home/thomas/regional_inversion/input_data/DEM_Norway')
        get_vel_dir(RID, custom_dx = 100)

        mask_path = '/home/thomas/regional_inversion/input_data/masks/per_glacier/RGI60-' + RGI_region + '/RGI60-' + RGI_region + '.0' + RID[10] + '/'+RID
        if not os.path.isdir(mask_path):
            os.makedirs(mask_path)
        mask_from_polygon(gdf[gdf.RGIId == RID].geometry.iloc[0], out_path = os.path.join(mask_path, RID + '_mask.tif'))

'''
DEMs = utils.DEM_SOURCES

for DEM in [DEMs[0]]:
    #DEM_dir = '/home/thomas/regional_inversion/input_data/DEM_' + DEM
    DEM_dir = '/home/thomas/regional_inversion/input_data/outlines'
    cfg.initialize(logging_level='WARNING')
    if not os.path.isdir(DEM_dir):
        os.mkdir(DEM_dir)
    cfg.PATHS['working_dir'] = DEM_dir

    fr = utils.get_rgi_region_file('08', version='62')  # Scandinavia
    gdf = gpd.read_file(fr)
    gdirs = workflow.init_glacier_directories(gdf, from_prepro_level = 1, prepro_border=160)

    for gdir in gdirs:
        #cfg.PATHS['dem_file'] = '/home/thomas/regional_inversion/input_data/dhdt_all/dhdt/mosaic_{}.tif'.format(period)
        cfg.PATHS['dem_file'] = '/home/thomas/regional_inversion/input_data/outlines/RGI_outlines_raster_georeferenced_noData.tif'
        try:
            gis.define_glacier_region(gdir, source = 'USER')
            #workflow.execute_entity_task(bedtopo.add_consensus_thickness, gdir)
        except exceptions.InvalidDEMError:
p            print('no DEM found here')
p            continue


# add conensus ice thickness to DEM gridded data
fr = utils.get_rgi_region_file('08', version='62')  # Scandinavia
gdf = gpd.read_file(fr)
RIDs = gdf['RGIId']
for RID in RIDs.loc[10:]:
    gdir = load_dem_gdir(RID)
    workflow.execute_entity_task(bedtopo.add_consensus_thickness, gdir)
'''
