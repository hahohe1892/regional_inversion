from oggm import utils, workflow, cfg, exceptions
from oggm.core import gis
import geopandas as gpd
import os
from oggm.shop import bedtopo
from load_input import *
#periods = ['2000-2005', '2005-2010', '2010-2015', '2015-2020']
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
            print('no DEM found here')
            continue

'''
# add conensus ice thickness to DEM gridded data
fr = utils.get_rgi_region_file('08', version='62')  # Scandinavia
gdf = gpd.read_file(fr)
RIDs = gdf['RGIId']
for RID in RIDs.loc[10:]:
    gdir = load_dem_gdir(RID)
    workflow.execute_entity_task(bedtopo.add_consensus_thickness, gdir)
'''

def get_DEM_dir(RID,  DEM_Dir = '/home/thomas/regional_inversion/input_data/DEMs'):
    cfg.initialize(logging_level='WARNING')
    cfg.PATHS['working_dir'] = DEM_Dir
    if os.path.isfile(cfg.PATHS['dem_file']):
        cfg.PATHS['dem_file'] = 'bla'
    RGI_region = RID.split('-')[1].split('.')[0]
    fr = utils.get_rgi_region_file(RGI_region, version='62')  # Scandinavia
    gdf = gpd.read_file(fr)
    gdirs = workflow.init_glacier_directories(gdf[gdf.RGIId == RID], from_prepro_level = 3, prepro_border=160)
    workflow.execute_entity_task(bedtopo.add_consensus_thickness, gdirs)


def get_dhdt_dir(RID, dhdt_file, dhdt_period = '2000-2020',  dhdt_Dir = '/home/thomas/regional_inversion/input_data/dhdt_'):
    dhdt_Dir = dhdt_Dir + dhdt_period
    dhdt_file = '/home/thomas/regional_inversion/input_data/dhdt_all/dhdt/' + dhdt_file
    if not os.path.isfile(dhdt_file):
        raise FileNotFoundError('no file found from which to extract dhdt values; check dhdt_file')
    RGI_region = RID.split('-')[1].split('.')[0]
    cfg.initialize(logging_level='WARNING')
    cfg.PATHS['working_dir'] = dhdt_Dir
    fr = utils.get_rgi_region_file(RGI_region, version='62')  # Scandinavia
    gdf = gpd.read_file(fr)
    cfg.PATHS['dem_file'] = dhdt_file
    gdirs = workflow.init_glacier_directories(gdf[gdf.RGIId == RID], from_prepro_level = 2, prepro_border=160)
    gis.define_glacier_region(gdirs[0], source = 'USER')
