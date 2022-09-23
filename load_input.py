import rioxarray as rioxr
from oggm import cfg, workflow, tasks, utils
from shapely.geometry import box
import geopandas as gpd

RID = 'RGI60-08.00010'


def load_dhdt_path(RID):
    path = '/home/thomas/regional_inversion/input_data/dhdt/per_glacier/RGI60-08/RGI60-08.0' + RID[10] + '/'+RID+ '/dem.tif'
    dhdt = rioxr.open_rasterio(path)

    return dhdt


def load_dem_path(RID):
    path = '/home/thomas/regional_inversion/input_data/DEMs/per_glacier/RGI60-08/RGI60-08.0' + RID[10] + '/'+RID + '/dem.tif'
    dem = rioxr.open_rasterio(path)

    return dem


def load_mask_path(RID):
    path = '/home/thomas/regional_inversion/input_data/dhdt/per_glacier/RGI60-08/RGI60-08.0' + RID[10] + '/'+RID + '/gridded_data.nc'
    with utils.ncDataset(path) as nc:
        mask = nc.variables['glacier_mask'][:]

    return mask


def crop_border_rio(xarr):
    res = xarr.rio.resolution()[0]
    x_min = float(xarr.x.min()) + 150 * res
    x_max = float(xarr.x.max()) - 150 * res
    y_min = float(xarr.y.min()) + 150 * res
    y_max = float(xarr.y.max()) - 150 * res
    geodf = gpd.GeoDataFrame(
        geometry=[
            box(x_min, y_min, x_max, y_max)],
        crs=xarr.rio.crs)
    clipped = xarr.rio.clip(geodf.geometry)

    return clipped


def crop_border_arr(arr):
    return arr[150:-150, 150:-150]


def load_dhdt_gdir(RID):
    cfg.initialize(logging_level='WARNING')
    cfg.PATHS['working_dir'] = '~/regional_inversion/input_data/dhdt'
    gdir = workflow.init_glacier_directories(RID)

    return gdir


def load_dem_gdir(RID):
    cfg.initialize(logging_level='WARNING')
    cfg.PATHS['working_dir'] = '~/regional_inversion/input_data/DEMs'
    gdir = workflow.init_glacier_directories(RID)

    return gdir
