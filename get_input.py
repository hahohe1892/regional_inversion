from oggm import utils, workflow, cfg
from oggm.core import gis
import geopandas as gpd

cfg.initialize(logging_level='WARNING')
cfg.PATHS['working_dir'] = '~/regional_inversion/input_data/dhdt'#utils.gettempdir(dirname='OGGM_Scandinavia')

fr = utils.get_rgi_region_file('08', version='62')  # Scandinavia
gdf = gpd.read_file(fr)
gdirs = workflow.init_glacier_directories(gdf, from_prepro_level = 3, prepro_border=160)

for gdir in gdirs:
    cfg.PATHS['dem_file'] = '/home/thomas/regional_inversion/input_data/dhdt/dhdt/mosaic.tif'
    gis.define_glacier_region(gdir, source = 'USER')
