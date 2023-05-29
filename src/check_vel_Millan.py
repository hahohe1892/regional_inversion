import numpy as np
import rioxarray as rioxr
import matplotlib.pyplot as plt
import rasterio
from scipy.optimize import curve_fit
from funcs import *
import pwlf
from statsmodels.stats import weightstats

V_8_1 = rioxr.open_rasterio('/mnt/c/Users/thofr531/Documents/Global/Scandinavia/vel_Millan/RGI-8/V_RGI-8.1_2021July01.tif').squeeze()
VX_8_1 = rioxr.open_rasterio('/mnt/c/Users/thofr531/Documents/Global/Scandinavia/vel_Millan/RGI-8/VX_RGI-8.1_2021July01.tif').squeeze()
VY_8_1 = rioxr.open_rasterio('/mnt/c/Users/thofr531/Documents/Global/Scandinavia/vel_Millan/RGI-8/VY_RGI-8.1_2021July01.tif').squeeze()
STD_X_8_1 = rioxr.open_rasterio('/mnt/c/Users/thofr531/Documents/Global/Scandinavia/vel_Millan/RGI-8/STDX_RGI-8.1_2021July01.tif').squeeze()
STD_Y_8_1 = rioxr.open_rasterio('/mnt/c/Users/thofr531/Documents/Global/Scandinavia/vel_Millan/RGI-8/STDY_RGI-8.1_2021July01.tif').squeeze()

STD_sum = abs(STD_X_8_1) + abs(STD_Y_8_1) 
error = (STD_sum)/V_8_1
error_2 = (STD_sum)/(V_8_1 + 5)

vel_mod_crs = rasterio.open('/mnt/c/Users/thofr531/Documents/Global/Scandinavia/outputs/v4.0/velsurf_mag/all_velsurf_mag_v4.0.tif').crs
vel_mod = rioxr.open_rasterio('/mnt/c/Users/thofr531/Documents/Global/Scandinavia/outputs/v4.0/velsurf_mag/all_velsurf_mag_v4.0.tif')
#vel_mod = rioxr.open_rasterio('/home/thomas/regional_inversion/output/RGI60-08.01779/ex.nc').velsurf_mag.squeeze()
#vel_mod = vel_mod.rio.write_crs(vel_mod_crs)[-1]
#mask = rioxr.open_rasterio('/home/thomas/regional_inversion/output/RGI60-08.01779/igm_input.nc').mask.squeeze()

error_crop = error_2.rio.reproject_match(vel_mod)
V_obs_crop = V_8_1.rio.reproject_match(vel_mod)
STD_X_crop = STD_X_8_1.rio.reproject_match(vel_mod)
STD_Y_crop = STD_Y_8_1.rio.reproject_match(vel_mod)
STD_sum_crop = STD_sum.rio.reproject_match(vel_mod)

err_tol = 1e10

vel_mod_tol = vel_mod.where(error_crop < err_tol)
V_obs_tol = V_obs_crop.where(error_crop < err_tol)
STD_sum_tol = STD_sum_crop.where(error_crop < err_tol)

nan_inds = ~(np.isnan(vel_mod_tol) + np.isnan(V_obs_tol) + np.isnan(STD_sum_tol))
zero_inds = vel_mod_tol > 0
out_inds = np.nonzero((nan_inds * zero_inds).data.flatten())

def linearFunc(x,slope):
    y = slope * x
    return y

def quadFunc(x, exponent, factor):
    y = (x) ** exponent
    return y

a_fit,cov=curve_fit(linearFunc, vel_mod_tol.data.flatten()[out_inds], V_obs_tol.data.flatten()[out_inds], sigma=STD_sum_tol.data.flatten()[out_inds], absolute_sigma=True)
pw = pwlf.PiecewiseLinFit(vel_mod_tol.data.flatten()[out_inds], V_obs_tol.data.flatten()[out_inds], weights = 1/STD_sum_tol.data.flatten()[out_inds])
pw_fit = pw.fit(2)

y_fit = linearFunc(np.sort(vel_mod_tol.data.flatten()[out_inds]), a_fit[0])
y_pw_fit = pw.predict(np.sort(vel_mod_tol.data.flatten()[out_inds]))

plt.scatter(vel_mod_tol.data.flatten()[out_inds], V_obs_tol.data.flatten()[out_inds])
plt.plot(np.sort(vel_mod_tol.data.flatten()[out_inds]), y_fit, c = 'g')
plt.plot(np.sort(vel_mod_tol.data.flatten()[out_inds]), y_pw_fit, c = 'orange')
plt.plot(range(100), range(100), '--', c='r')
plt.show()

plt.scatter(V_obs_tol.data.flatten()[out_inds], vel_mod_tol.data.flatten()[out_inds])
plt.plot(y_fit, np.sort(vel_mod_tol.data.flatten()[out_inds]), c='g')
plt.plot(range(100), range(100), '--', c='r')
plt.show()



mean_obs = np.nanmean(V_obs_crop.data[vel_mod.data > 1])
mean_mod = np.nanmean(vel_mod.data[vel_mod.data > 1])
std_obs = np.sqrt(STD_X_crop**2 + STD_Y_crop**2)

np.sqrt(np.sum(std_obs**2))

plt.hist((V_obs_crop.data[vel_mod.data>1]).flatten(), 20)
plt.hist((vel_mod.data[vel_mod.data>1]).flatten(), 20, alpha = .3)
plt.show()

t_test = weightstats.ttest_ind(V_obs_crop.data.flatten()[out_inds], vel_mod.data.flatten()[out_inds], weights = (1/std_obs.data.flatten()[out_inds], None), usevar = 'unequal')


V_8_1_open = rasterio.open('/mnt/c/Users/thofr531/Documents/Global/Scandinavia/vel_Millan/RGI-8/V_RGI-8.1_2021July01.tif')
vel_mod_open = rasterio.open('/home/thomas/regional_inversion/output/RGI60-08.01779/velsurf_mag.tif')
vel_Norway = gpd.read_file('/mnt/c/Users/thofr531/Documents/Global/Scandinavia/vel_Norway/Glacier_velocity_Norway_2015_2018_v2_1.shp')
vel_Norway = vel_Norway.to_crs(V_8_1_open.crs)
coords = [(x,y) for x, y in zip(vel_Norway.geometry.x, vel_Norway.geometry.y)]
vel_Norway['vel_Millan'] = [x[0] for x in V_8_1_open.sample(coords)]
vel_Norway['vel_365'] = vel_Norway.velocity_m * 365
vel_Norway = vel_Norway.to_crs(vel_mod_open.crs)
coords = [(x,y) for x, y in zip(vel_Norway.geometry.x, vel_Norway.geometry.y)]
vel_Norway['vel_mod'] = [x[0] for x in vel_mod_open.sample(coords)]
vel_Norway.vel_mod = vel_Norway.vel_mod.where(vel_Norway.vel_mod<1e4)
