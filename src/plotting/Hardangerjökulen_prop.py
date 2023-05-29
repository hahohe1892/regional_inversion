import tensorflow as tf
import math
from igm.igm import Igm
from rasterio.enums import Resampling
import rasterio
import numpy as np
from load_input import *
import os
from write_output import *
from funcs import normalize
from geocube.api.core import make_geocube
import sys
from pathlib import Path
from scipy.stats import gaussian_kde

RID = 'RGI60-08.01779' # Hardangerjökulen
home_dir = Path('/home/thomas')
Fill_Value = 9999.0

working_dir = home_dir / 'regional_inversion/output/' / RID
input_file = working_dir / 'igm_input.nc'
input_igm = rioxr.open_rasterio(input_file)
for var in input_igm.data_vars:
    input_igm.data_vars[var].rio.write_nodata(Fill_Value, inplace=True)

out_to_tif(RID, 'topg', i = -1, file = 'ex.nc', file_not_standard_dims = True)
out_to_tif(RID, 'thk', i = -1, file = 'ex.nc', file_not_standard_dims = True)
out_to_tif(RID, 'velsurf_mag', i = -1, file = 'ex.nc', file_not_standard_dims = True)
out_to_tif(RID, 'usurf', i = 0, file = 'igm_input.nc', file_not_standard_dims = True)
out_to_tif(RID, 'mask', i = 0, file = 'igm_input.nc', file_not_standard_dims = True)
out_to_tif(RID, 'apparent_mass_balance', i = 0, file = 'igm_input.nc', file_not_standard_dims = True)
out_thk = rasterio.open(working_dir / 'thk.tif')
out_topg = rasterio.open(working_dir / 'topg.tif')
out_velsurf = rasterio.open(working_dir / 'velsurf_mag.tif')
in_usurf = rasterio.open(working_dir / 'usurf.tif')
in_mask = rasterio.open(working_dir / 'mask.tif')
in_smb = rasterio.open(working_dir / 'apparent_mass_balance.tif')
thk_consensus = rasterio.open('/mnt/c/Users/thofr531/Documents/Global/Scandinavia/consensus_thk/RGI60-08/all_thk_consensus.tif')
glathida_NO = pd.read_csv('/mnt/c/Users/thofr531/Documents/Global/glathida-3.1.0/data/glathida_NO.csv')
glathida_NO = gpd.GeoDataFrame(glathida_NO, geometry=gpd.points_from_xy(glathida_NO.POINT_LON, glathida_NO.POINT_LAT),crs = 'epsg: 4326')
glathida_NO = glathida_NO.to_crs(thk_consensus.crs)
coords = [(x,y) for x, y in zip(glathida_NO.geometry.x, glathida_NO.geometry.y)]
glathida_NO['THICK_CONS'] = [x[0] for x in thk_consensus.sample(coords)]
glathida_NO = glathida_NO.to_crs(out_thk.crs)
glathida_NO['TOPG'] = glathida_NO.ELEVATION - glathida_NO.THICKNESS
coords = [(x,y) for x, y in zip(glathida_NO.geometry.x, glathida_NO.geometry.y)]
glathida_NO['THICK_MOD'] = [x[0] for x in out_thk.sample(coords)]
glathida_NO['TOPG_MOD'] = [x[0] for x in out_topg.sample(coords)]
glathida_NO['USURF_TODAY'] = [x[0] for x in in_usurf.sample(coords)]
glathida_NO['VELSURF_MOD'] = [x[0] for x in out_velsurf.sample(coords)]
glathida_NO['THICK_OBS_CORR'] = (glathida_NO.USURF_TODAY) - glathida_NO.TOPG
glathida_NO = glathida_NO.where(glathida_NO.THICK_MOD < 1e5)
glathida_NO = glathida_NO.where(glathida_NO.THICK_MOD > -1e5)
glathida_NO['Difference'] = glathida_NO.THICK_OBS_CORR - glathida_NO.THICK_MOD
glathida_NO['Percent_difference'] = glathida_NO.Difference/glathida_NO.THICK_OBS_CORR
glathida_NO = glathida_NO.dropna()
gla_ras = make_geocube(glathida_NO, resolution = (100,100), output_crs = out_thk.crs)
gla_ras = gla_ras.rio.reproject_match(input_igm)
gla_ras = gla_ras.where(in_mask.read()[0] == 1, np.nan)
not_nan = ~np.isnan(gla_ras.THICK_OBS_CORR.data.flatten())
xy = np.vstack([gla_ras.THICK_OBS_CORR.data.flatten()[not_nan], gla_ras.THICK_MOD.data.flatten()[not_nan]])
z = gaussian_kde(xy)(xy)

fs = 17
fig, ax = plt.subplots(figsize = (7,6))
ax.scatter(gla_ras.THICK_OBS_CORR.data.flatten()[not_nan], gla_ras.THICK_MOD.data.flatten()[not_nan], c = z)
ax.plot(range(int(gla_ras.THICKNESS.max())), range(int(gla_ras.THICKNESS.max())), '--', c='r', label = '1:1 line')
ax.set(adjustable='box', aspect='equal')
ax.set_xlabel('Observed ice thickness (m)', fontsize = fs)
ax.set_ylabel('Modelled ice thickness (m)', fontsize = fs)
ax.set_xticks([0,100,200,300,400])
ax.set_yticks([0,100,200,300,400])
ax.tick_params(axis='both', which='major', labelsize=fs*.8)
ax.set_axisbelow(True)
ax.grid()
ax.legend(fontsize = fs)
#plt.savefig('/mnt/c/Users/thofr531/Documents/Global/Scandinavia/outputs/plots/Hardangerjökulen_correlation_proposal.png', dpi = 300)
plt.show()
