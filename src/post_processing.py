from pathlib import Path
import numpy as np
import rioxarray as rioxr
from funcs import *
from oggm import utils
import geopandas as gpd

home_dir = Path('/home/thomas')
Fill_Value = 9999.0
RID = 'RGI60-08.00434' # Tunsbergdalsbreen
RIDs_with_obs = ['RGI60-08.00434', 'RGI60-08.01657', 'RGI60-08.01779', 'RGI60-08.02666', 'RGI60-08.01258', 'RGI60-08.02382', 'RGI60-08.00966', 'RGI60-08.00987', 'RGI60-08.00312', 'RGI60-08.02972', 'RGI60-08.01103', 'RGI60-08.00435', 'RGI60-08.00213']
fr = utils.get_rgi_region_file('08', version='62')
gdf = gpd.read_file(fr)
all_glaciers = gdf.RGIId.to_list()[4:]

RID = 'RGI60-08.00139'
RID = 'RGI60-08.00314'
#RID = 'RGI60-08.00078'
for i,RID in enumerate(all_glaciers):
    working_dir = home_dir / 'regional_inversion/output/' / RID
    output_file = 'ex_v7.0.nc'
    post_processed_file = 'ex_v7.0_pp_{}.tif'
    if not os.path.exists(working_dir / output_file):
        continue
    print('processing {}...'.format(RID))
    input_file = working_dir / 'igm_input.nc'
    input_igm = rioxr.open_rasterio(input_file)
    ncfile = rioxr.open_rasterio(working_dir / output_file, mode = 'r')
    ncfile = ncfile.rio.write_crs(input_igm.rio.crs)
    ncfile.attrs['_FillValue'] = Fill_Value
    last_iteration = deepcopy(ncfile.isel({'time': -2}))
    last_iteration = last_iteration.rio.write_crs(ncfile.rio.crs)
    ncfile.close()
    thk_orig = deepcopy(last_iteration.thk.data)

    # establish buffer
    buffer_outline = internal_buffer(1, input_igm.mask.data[0])
    buffer_outline2 = internal_buffer(2, input_igm.mask.data[0])
    buffer = internal_buffer(1, last_iteration.thk.data > 10)
    buffer = np.maximum(buffer - buffer_outline2, 0)
    buffer2 = np.maximum(internal_buffer(1, buffer == 0) - buffer_outline2, 0)
    buffer_adjacent_icefree = np.maximum(buffer2 - buffer, 0)
    buffer_norm1 = buffer_adjacent_icefree + buffer
    last_iteration.thk.data[buffer == 1] = 0
    
    for p in range(5):
        last_iteration.thk.data[np.logical_and(input_igm.mask.data[0] == 1, last_iteration.thk.data < 10)] = last_iteration.thk.attrs['_FillValue']
        if p % 2 == 0:
            last_iteration['thk'] = last_iteration.thk.rio.interpolate_na(method = 'nearest')
        else:
            last_iteration['thk'] = last_iteration.thk.rio.interpolate_na(method = 'linear')

    #normalize thickness field relative to a max thickness of 500 m (i.e. all thicknesses >=500 m get the max smoothing)
    norm = normalize(last_iteration.thk.data) * np.minimum((np.nanmax(last_iteration.thk.data)/500), 1)
    norm[abs(thk_orig - last_iteration.thk.data)>1] = 1 * input_igm.mask.data[0][abs(thk_orig - last_iteration.thk.data)>1]
    norm[buffer_norm1 == 1] = 1
    norm_close = close_internal_holes((norm == 1) * 1)
    norm[norm_close == 1] = 1
    last_iteration.thk.data = gauss_filter(last_iteration.thk.data, 2, 4) * norm + last_iteration.thk.data * (-norm+ 1)
    last_iteration.thk.data[buffer_outline == 1] = last_iteration.thk.attrs['_FillValue']
    last_iteration['thk'] = last_iteration.thk.rio.interpolate_na(method = 'linear')
    
    last_iteration.thk.rio.to_raster(working_dir / post_processed_file.format('thk'))
    
    last_iteration['topg'] = last_iteration.usurf - last_iteration.thk
    last_iteration.topg.rio.to_raster(working_dir / post_processed_file.format('topg'))
