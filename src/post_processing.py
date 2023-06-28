from pathlib import Path
import numpy as np
import rioxarray as rioxr
from funcs import *

home_dir = Path('/home/thomas')
Fill_Value = 9999.0
RID = 'RGI60-08.00434' # Tunsbergdalsbreen
RIDs_with_obs = ['RGI60-08.00434', 'RGI60-08.01657', 'RGI60-08.01779', 'RGI60-08.02666', 'RGI60-08.01258', 'RGI60-08.02382', 'RGI60-08.00966', 'RGI60-08.00987', 'RGI60-08.00312', 'RGI60-08.02972', 'RGI60-08.01103', 'RGI60-08.00435', 'RGI60-08.00213']
for i,RID in enumerate(RIDs_with_obs):
    working_dir = home_dir / 'regional_inversion/output/' / RID
    output_file = 'ex_v6.6.nc'
    post_processed_file = 'ex_v6.6_pp_{}.tif'
    if not os.path.exists(working_dir / output_file):
        continue
    input_file = working_dir / 'igm_input.nc'
    input_igm = rioxr.open_rasterio(input_file)
    ncfile = rioxr.open_rasterio(working_dir / output_file, mode = 'r')
    ncfile = ncfile.rio.write_crs(input_igm.rio.crs)
    ncfile.attrs['_FillValue'] = Fill_Value
    last_iteration = deepcopy(ncfile.isel({'time': -1}))
    last_iteration = last_iteration.rio.write_crs(ncfile.rio.crs)
    #last_iteration['time'] = ncfile.time[[0,-1]]
    #last_iteration['time'][-1] = ncfile.time[-1].data + 1
    ncfile.close()
    #last_iteration.close()

    last_iteration.thk.data = gauss_filter(last_iteration.thk.data, 1, 3)
    for p in range(5):
        last_iteration.thk.data[np.logical_and(input_igm.mask.data[0] == 1, last_iteration.thk.data < 5)] = last_iteration.thk.attrs['_FillValue']
        if p % 2 == 0:
            last_iteration['thk'] = last_iteration.thk.rio.interpolate_na(method = 'nearest')
        else:
            last_iteration['thk'] = last_iteration.thk.rio.interpolate_na(method = 'linear')
            
    last_iteration.thk.rio.to_raster(working_dir / post_processed_file.format('thk'))
    
    last_iteration['topg'] = last_iteration.usurf - last_iteration.thk
    last_iteration.topg.rio.to_raster(working_dir / post_processed_file.format('topg'))
