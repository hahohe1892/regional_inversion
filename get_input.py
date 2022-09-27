from oggm import utils, workflow, cfg
from oggm.core import gis
import geopandas as gpd
import os
#periods = ['2000-2005', '2005-2010', '2010-2015', '2015-2020']
DEMs = utils.DEM_SOURCES

for DEM in DEMs:
    DEM_dir = '/home/thomas/regional_inversion/input_data/DEM_' + DEM
    cfg.initialize(logging_level='WARNING')
    if not os.path.isdir(DEM_dir):
        os.mkdir(DEM_dir)
    cfg.PATHS['working_dir'] = DEM_dir

    fr = utils.get_rgi_region_file('08', version='62')  # Scandinavia
    gdf = gpd.read_file(fr)
    gdirs = workflow.init_glacier_directories(gdf, from_prepro_level = 3, prepro_border=160)

    for gdir in gdirs:
        #cfg.PATHS['dem_file'] = '/home/thomas/regional_inversion/input_data/dhdt_all/dhdt/mosaic_{}.tif'.format(period)
        gis.define_glacier_region(gdir, source = DEM)
