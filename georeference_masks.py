import rioxarray as rioxr
from load_input import *
from scipy.ndimage import label
import numpy as np
import rasterio

def get_georeferenced_mosaic(path = '/home/thomas/regional_inversion/input_data/outlines/georeferenced_mask_mosaic.tif'):
    geo_mosaic = rioxr.open_rasterio(path)
    return geo_mosaic


def find_overlap(mosaic_c, glacier_mask):
    overlap = (mosaic_c + glacier_mask) == 2
    return overlap * 1


def get_new_mask(mosaic_c, overlap, glacier_mask):
    regions = label(mosaic_c)[0].squeeze()
    region_size = []
    overlap_size = []
    for i in range(np.max(regions)+1):
        region_n = regions == i
        overlap_size.append((region_n * overlap).sum().data)
        region_size.append(region_n.sum())
    region_size = np.array(region_size)
    # assuming that the mask has only been shifted by a little, we take that
    # region which has the largest overlap with the old mask as the new mask
    largest_overlapping_region = np.argwhere(overlap_size == np.max(overlap_size))[0][0]
    if np.max(overlap_size) == 0:
        print('no overlap found; going by mask size')
        size_difference = region_size - glacier_mask.sum().data
        largest_overlapping_region = np.argwhere(abs(size_difference) == np.min(abs(size_difference)))[0][0]
    new_mask = deepcopy(mosaic_c)
    new_mask.data[0] = regions == largest_overlapping_region
    # for a small glacier where the mask is shifted a lot, it could be that
    # the largest overlap is with the ice-free area as represented in the old mask
    # we check by comparing sizes between old and new mask
    # and print a warning if there is a clear mismatch
    if abs(new_mask.sum()  - glacier_mask.sum()) > 0.2 * glacier_mask.sum():
        print('WARNING: new mask likely representes either connected glaciers or a neighboring glacier! Double-check!')
    return new_mask * 1


def produce_new_mask(RID, mosaic):
    glacier_mask = load_mask_path(RID)
    mosaic_c = mosaic.rio.reproject_match(glacier_mask)
    overlap = find_overlap(mosaic_c, glacier_mask)
    new_mask = get_new_mask(mosaic_c, overlap, glacier_mask)
    new_mask.rio.to_raster('/home/thomas/regional_inversion/input_data/outlines/georeferenced_masks/mask_{}_new.tif'.format(RID))


def mask_from_shapefile(RID, all_shapes):
    glacier = all_shapes[all_shapes.RGIId == RID]
    mask_old = load_mask_path(RID)
    glacier_reprojected = glacier.to_crs(mask_old.rio.crs)
    src = rasterio.open('/home/thomas/regional_inversion/input_data/outlines/georeferenced_masks/mask_{}_new.tif'.format(RID)) 
    mask_new_data = rasterio.mask.mask(src, glacier_reprojected.geometry, crop = False)
    mask_new = deepcopy(mask_old)
    mask_new.data[0] = mask_new_data[0][0]
    return mask_new



# originally, I tried to use a georeferenced raster that contained all glaciers. I tried to extract individual
# glaciers from that raster. Although this worked in general, it clustered glaciers together if their
# gridded masks were connected.
# The new approach as shown below uses a georeferenced shapefile (produced with the 'spatial adjustment'
# arcgis tool) of outlines and turns these into raster
# masks for each glacier. Works well, although the quality of the georeferenced shapefile could still be improved
if __name__ == '__main__':
    RIDs = get_RIDs_Sweden()
    RIDs_Sweden = RIDs.RGIId
    all_shapes = gpd.read_file("/home/thomas/regional_inversion/input_data/outlines/08_rgi60_Scandinavia_georeferenced.shp")
    #mosaic = get_georeferenced_mosaic()
    for RID in RIDs_Sweden:
        new_mask = mask_from_shapefile(RID, all_shapes)
        new_mask.rio.to_raster('/home/thomas/regional_inversion/input_data/outlines/georeferenced_masks/mask_{}_new.tif'.format(RID))
        #produce_new_mask(RID, mosaic)
        print('glacier {} done'.format(RID))
