from pathlib import Path
import numpy as np
import xarray as xr
from funcs import *

home_dir = Path('/home/thomas')
RID = 'RGI60-08.00434' # Tunsbergdalsbreen
working_dir = home_dir / 'regional_inversion/output/' / RID
output_file = 'ex2.nc'
post_processed_file = 'ex2_pp.nc'

ncfile = xr.open_dataset(working_dir / output_file, mode = 'r')
last_iteration = ncfile.isel({'time': -1})
last_iteration['time'] = ncfile.time[-1].data + 1
ncfile.close()
last_iteration.close()

last_thk = ncfile.thk[-1].data
last_thk = gauss_filter(last_thk, 1, 3)

last_iteration.thk.data = last_thk
nc_done = xr.concat([ncfile, last_iteration], dim = 'time')
nc_done.to_netcdf(working_dir / post_processed_file)
