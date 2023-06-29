import rasterio
import numpy as np
from load_input import *
from write_output import *
import os
from sklearn.preprocessing import normalize
from load_input import obtain_area_mosaic
from funcs import normalize
from sklearn.linear_model import LinearRegression
from geocube.api.core import make_geocube
import sys
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import math
from igm.igm import Igm

home_dir = Path('/home/thomas')
output_file = 'ex_v6.6.nc'
output_file_pp = 'ex_v6.6_pp_{}.tif'
Fill_Value = 9999.0
iteration = -1
RIDs_with_obs = ['RGI60-08.00434', 'RGI60-08.01657', 'RGI60-08.01779', 'RGI60-08.02666', 'RGI60-08.01258', 'RGI60-08.02382', 'RGI60-08.00966', 'RGI60-08.00987', 'RGI60-08.00312', 'RGI60-08.02972', 'RGI60-08.01103', 'RGI60-08.00435', 'RGI60-08.00213']
icecaps = ['RGI60-08.00434', 'RGI60-08.01657', 'RGI60-08.01779', 'RGI60-08.00435']
glaciers = [i for i in RIDs_with_obs if i not in icecaps]


for i,RID in enumerate(RIDs_with_obs):
    #RID = get_mosaic_reference(RID_obs)
    working_dir = home_dir / 'regional_inversion/output/' / RID
    input_file = working_dir / 'igm_input.nc'
    input_igm = rioxr.open_rasterio(input_file)
    for var in input_igm.data_vars:
        input_igm.data_vars[var].rio.write_nodata(Fill_Value, inplace=True)
    if not os.path.exists(working_dir / output_file):
        continue
    slope_x = input_igm.usurf.differentiate('x')
    slope_y = input_igm.usurf.differentiate('y')
    slope = np.sqrt(slope_x**2 + slope_y**2) * input_igm.mask
    sin_slope = np.sin(np.arctan(slope))
    #taub = sin_slope * thk * 9.8 * 910 * .9
    slope.rio.to_raster(working_dir / 'slope.tif')
    if RID == 'RGI60-08.00213':
        if output_file_pp is not None:
            out_thk = rasterio.open(working_dir / output_file_pp.format('thk')).read()
        else:
            out_thk = rioxr.open_rasterio(working_dir / output_file).thk[iteration].data
        in_slope = rioxr.open_rasterio(working_dir / 'slope.tif')
        thk_consensus = rioxr.open_rasterio('/mnt/c/Users/thofr531/Documents/Global/Scandinavia/consensus_thk/RGI60-08/RGI60-08.00213_thickness.tif')
        thk_Millan = rioxr.open_rasterio('/mnt/c/Users/thofr531/Documents/Global/Scandinavia/thk_Millan/THICKNESS_RGI-8.2_2021July09.tif')
        radar_bed = rioxr.open_rasterio('/home/thomas/regional_inversion/Storglaciären_radar_bed.tif')
        radar_bed = radar_bed.rio.reproject_match(input_igm)
        thk_consensus = thk_consensus.rio.reproject_match(input_igm)
        thk_Millan = thk_Millan.rio.reproject_match(input_igm)
        radar_bed.data[radar_bed.data == radar_bed.attrs['_FillValue']] = np.nan
        radar_thk = input_igm.usurf - radar_bed.data[0]
        radar_thk = np.maximum(0, radar_thk)
        Storglaciären = pd.DataFrame({'THICKNESS': radar_thk.data[0].flatten(),
                                      'THICK_MOD': out_thk.flatten(),
                                      'THICK_CONS': thk_consensus.data.flatten(),
                                      'THICK_Millan': thk_Millan.data.flatten(),
                                      'SLOPE': in_slope.data[0].flatten()})
        Storglaciären = Storglaciären.dropna(subset = ['THICKNESS'])
        Storglaciären['glacier'] = RID
        all_NO = pd.concat([all_NO, Storglaciären])
    else:
        if output_file_pp is not None:
            out_thk = rasterio.open(working_dir / output_file_pp.format('thk'))
            out_topg = rasterio.open(working_dir / output_file_pp.format('topg'))
        else:
            out_to_tif(RID, 'topg', i = iteration, file = output_file, file_not_standard_dims = True)
            out_to_tif(RID, 'thk', i = iteration, file = output_file, file_not_standard_dims = True)
            out_thk = rasterio.open(working_dir / 'thk.tif')
            out_topg = rasterio.open(working_dir / 'topg.tif')

        out_to_tif(RID, 'velsurf_mag', i = iteration, file = output_file, file_not_standard_dims = True)
        out_to_tif(RID, 'usurf', i = 0, file = output_file, file_not_standard_dims = True)
        #out_to_tif(RID, 'mask', i = 0, file = 'igm_input.nc', file_not_standard_dims = True)
        out_to_tif(RID, 'smb', i = 0, file = output_file, file_not_standard_dims = True)
        out_velsurf = rasterio.open(working_dir / 'velsurf_mag.tif')
        in_usurf = rasterio.open(working_dir / 'usurf.tif')
        #in_mask = rasterio.open(working_dir / 'mask.tif')
        in_smb = rasterio.open(working_dir / 'apparent_mass_balance.tif')
        in_slope = rasterio.open(working_dir / 'slope.tif')
        thk_consensus = rasterio.open('/mnt/c/Users/thofr531/Documents/Global/Scandinavia/consensus_thk/RGI60-08/all_thk_consensus.tif')
        thk_Millan = rasterio.open('/mnt/c/Users/thofr531/Documents/Global/Scandinavia/thk_Millan/all_thk_Millan.tif')
        glathida_NO = pd.read_csv('/mnt/c/Users/thofr531/Documents/Global/glathida-3.1.0/data/glathida_NO.csv')
        glathida_NO = gpd.GeoDataFrame(glathida_NO, geometry=gpd.points_from_xy(glathida_NO.POINT_LON, glathida_NO.POINT_LAT),crs = 'epsg: 4326')
        glathida_NO = glathida_NO.to_crs(thk_consensus.crs)
        coords = [(x,y) for x, y in zip(glathida_NO.geometry.x, glathida_NO.geometry.y)]
        glathida_NO['THICK_CONS'] = [x[0] for x in thk_consensus.sample(coords)]
        glathida_NO = glathida_NO.to_crs(thk_Millan.crs)
        coords = [(x,y) for x, y in zip(glathida_NO.geometry.x, glathida_NO.geometry.y)]
        glathida_NO['THICK_Millan'] = [x[0] for x in thk_Millan.sample(coords)]
        glathida_NO.THICK_Millan = glathida_NO.THICK_Millan.where(glathida_NO.THICK_Millan >= 0, np.nan)
        glathida_NO = glathida_NO.to_crs(out_thk.crs)
        glathida_NO['TOPG'] = glathida_NO.ELEVATION - glathida_NO.THICKNESS
        coords = [(x,y) for x, y in zip(glathida_NO.geometry.x, glathida_NO.geometry.y)]
        glathida_NO['THICK_MOD'] = [x[0] for x in out_thk.sample(coords)]
        glathida_NO['TOPG_MOD'] = [x[0] for x in out_topg.sample(coords)]
        glathida_NO['USURF_TODAY'] = [x[0] for x in in_usurf.sample(coords)]
        glathida_NO['VELSURF_MOD'] = [x[0] for x in out_velsurf.sample(coords)]
        glathida_NO['SLOPE'] = [x[0] for x in in_slope.sample(coords)]
        glathida_NO['THICK_OBS_CORR'] = (glathida_NO.USURF_TODAY) - glathida_NO.TOPG
        glathida_NO.THICK_OBS_CORR = glathida_NO.THICK_OBS_CORR.where(glathida_NO.THICK_OBS_CORR >=0, np.nan)
        glathida_NO = glathida_NO.where(glathida_NO.THICK_MOD < 1e5)
        glathida_NO = glathida_NO.where(glathida_NO.THICK_MOD > -1e5)
        glathida_NO['Difference'] = glathida_NO.THICK_OBS_CORR - glathida_NO.THICK_MOD
        glathida_NO['Percent_difference'] = glathida_NO.Difference/glathida_NO.THICK_OBS_CORR
        glathida_NO = glathida_NO.dropna(subset = ['THICK_MOD'])
        glathida_NO['glacier'] = RID
        if i == 0:
            all_NO = glathida_NO.copy(deep = True)
        else:
            all_NO = pd.concat([all_NO, glathida_NO])


reference_thickness = all_NO.THICKNESS
rmse_mod = math.sqrt(np.square(np.subtract(reference_thickness,all_NO.THICK_MOD)).mean())
rmse_cons = math.sqrt(np.square(np.subtract(reference_thickness,all_NO.THICK_CONS)).mean())
rmse_Millan = math.sqrt(np.square(np.subtract(reference_thickness,all_NO.THICK_Millan)).mean())
MAE_mod = np.mean(abs(reference_thickness-all_NO.THICK_MOD))
MAE_cons = np.mean(abs(reference_thickness-all_NO.THICK_CONS))
MAE_Millan = np.mean(abs(reference_thickness-all_NO.THICK_Millan))
slope_mod = np.polyfit(reference_thickness, all_NO.THICK_MOD, 1)[0]
slope_cons = np.polyfit(reference_thickness, all_NO.THICK_CONS, 1)[0]
slope_Millan = np.polyfit(reference_thickness[~np.isnan(all_NO.THICK_Millan)], all_NO.THICK_Millan[~np.isnan(all_NO.THICK_Millan)], 1)[0]
R_mod = all_NO.corr(numeric_only = True).THICKNESS['THICK_MOD']
R_cons = all_NO.corr(numeric_only = True).THICKNESS['THICK_CONS']
R_Millan = all_NO.corr(numeric_only = True).THICKNESS['THICK_Millan']
var_obs = np.var(reference_thickness)
var_mod = np.var(all_NO.THICK_MOD)
var_cons = np.var(all_NO.THICK_CONS)
var_Millan = np.var(all_NO.THICK_Millan)
dev_mod = np.subtract(reference_thickness,all_NO.THICK_MOD).mean()
dev_cons = np.subtract(reference_thickness,all_NO.THICK_CONS).mean()
dev_Millan = np.subtract(reference_thickness,all_NO.THICK_Millan).mean()

stats_table = pd.DataFrame({'r':[np.nan, R_mod, R_cons, R_Millan],
                            'slope': [np.nan, slope_mod, slope_cons, slope_Millan],
                            'RMSE': [np.nan, rmse_mod, rmse_cons, rmse_Millan],
                            'mean abs deviation': [np.nan, MAE_mod, MAE_cons, MAE_Millan],
                            'variance': [var_obs, var_mod, var_cons, var_Millan],
                            'mean deviation': [np.nan, dev_mod, dev_cons, dev_Millan]},
                           index = ['Observations', 'This study', 'Consensus estimate', 'Millan'])

print(stats_table)


pd_MOD = calc_point_density(reference_thickness, all_NO.THICK_MOD)
pd_CONS = calc_point_density(reference_thickness, all_NO.THICK_CONS)
cmap = plt.get_cmap('Set1')
colors = cmap(np.linspace(0, 1, len(all_NO.glacier)))
fs = 17
fig, ax = plt.subplots(1,3, figsize = (18,6))
ax[0].scatter(reference_thickness, all_NO.THICK_Millan, alpha = .1, label = 'Millan (2022)')
ax[1].scatter(reference_thickness, all_NO.THICK_CONS, alpha = .1, label = 'Consensus', c = pd_CONS)
ax[2].scatter(reference_thickness, all_NO.THICK_MOD, alpha = .1, label = 'This study', c = pd_MOD)
for axes in ax:
    axes.plot(range(700), range(700), '--', c='r')
    axes.set(adjustable='box', aspect='equal')
    axes.set_xlabel('Observed ice thickness (m)', fontsize = fs)
    axes.set_ylabel('Modelled ice thickness (m)', fontsize = fs)
    axes.set_xticks(range(0, 800, 100))
    axes.set_yticks(range(0, 800, 100))
    axes.tick_params(axis='both', which='major', labelsize=fs*.8)
    axes.set_axisbelow(True)
    axes.grid()
    axes.legend(fontsize = fs/2)
plt.show()

thk_mosaic_mod = rioxr.open_rasterio('/mnt/c/Users/thofr531/Documents/Global/Scandinavia/outputs/v5.0/thk/all_thk_v5.0.tif')
thk_mosaic_mod.values[thk_mosaic_mod.values == thk_mosaic_mod._FillValue] = 0
#thk_mosaic_cons = rioxr.open_rasterio('/mnt/c/Users/thofr531/Documents/Global/Scandinavia/consensus_thk/RGI60-08/all_thk_consensus.tif')
#thk_mosaic_cons.values[thk_mosaic_cons.values == thk_mosaic_cons._FillValue] = 0
#thk_mosaic_Millan = rioxr.open_rasterio('/mnt/c/Users/thofr531/Documents/Global/Scandinavia/thk_Millan/all_thk_Millan.tif')
#thk_mosaic_Millan.values[thk_mosaic_Millan.values == thk_mosaic_Millan._FillValue] = 0

V_mod = thk_mosaic_mod.sum() * 100 * 100 / 1e9
#V_cons = thk_mosaic_cons.sum() * 100 * 100 / 1e9
#V_Millan = thk_mosaic_Millan.sum() * 100 * 100 / 1e9


T_pd = pd.read_csv('/mnt/c/Users/thofr531/Documents/Global/glathida-3.1.0/data/T.csv')#, usecols = ['LAT', 'LON', 'MEAN_THICKNESS', 'POLITICAL_UNIT', 'MEAN_THICKNESS_UNCERTAINTY', 'MAXIMUM_THICKNESS', 'MAX_THICKNESS_UNCERTAINTY', 'GLACIER_NAME'])
T = gpd.GeoDataFrame(T_pd, geometry=gpd.points_from_xy(T_pd.LON, T_pd.LAT),crs = 'epsg: 4326')
T  = T[T.POLITICAL_UNIT == 'SE']
T['RGI_ID'] = ['RGI60-08.00251', 'RGI60-08.00188', 'RGI60-08.00213', 'RGI60-08.00146', 'RGI60-08.00121']

for RID_obs in T.RGI_ID.to_list():
    mask = rioxr.open_rasterio(home_dir / 'regional_inversion/output/' / RID_obs / 'input.nc').mask.squeeze()
    RID = get_mosaic_reference(RID_obs)    
    area_thk = rioxr.open_rasterio(home_dir / 'regional_inversion/output/' / RID / output_file).thk.squeeze()
    area_thk_crs = rioxr.open_rasterio(home_dir / 'regional_inversion/output/' / RID / 'input_igm.nc').rio.crs
    area_thk = area_thk.rio.write_crs(area_thk_crs)[-1]
    area_thk = area_thk.rio.reproject_match(mask)
    area_thk.values[area_thk.values == area_thk._FillValue] = 0
    V = np.sum(area_thk * mask) # omitting multiplication with resolution squared here
    mean_thk = np.round(V / len(mask.data[mask.data == 1]), 1)
    max_thk = np.round((area_thk * mask).max(), 1)
    print('{}\nObserved mean thickness: {}, Modelled mean thickness: {}'.format(T[T.RGI_ID == RID_obs].GLACIER_NAME.values[0], T[T.RGI_ID == RID_obs].MEAN_THICKNESS.values[0], mean_thk.values))
    print('Observed max thickness: {}, Modelled max thickness: {}'.format(T[T.RGI_ID == RID_obs].MAXIMUM_THICKNESS.values[0], max_thk.values))


by_glacier = all_NO.set_index([all_NO['glacier'], all_NO.groupby('glacier').cumcount()]).drop('glacier', 1).unstack(0)
by_glacier.columns = [f'{y}_{x}' for x,y in by_glacier.columns]

r_per_glacier = pd.DataFrame({*all_NO.glacier.unique()})
rs_MOD = []
rs_CONS = []
rs_Millan = []
for glacier in all_NO.glacier.unique():
    cors = by_glacier.corr(numeric_only = True)[glacier + '_THICKNESS']
    rs_MOD.append(cors.loc[glacier + '_THICK_MOD'])
    rs_CONS.append(cors.loc[glacier + '_THICK_CONS'])
    rs_Millan.append(cors.loc[glacier + '_THICK_Millan'])
r_per_glacier['r_MOD'] = rs_MOD
r_per_glacier['r_CONS'] = rs_CONS
r_per_glacier['r_Millan'] = rs_Millan
    
