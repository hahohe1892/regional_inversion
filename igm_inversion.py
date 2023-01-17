import tensorflow as tf
import math
from igm.igm import Igm
from rasterio.enums import Resampling
import rasterio
import numpy as np
from load_input import *
import os
from write_output import *


RID = 'RGI60-08.00213' # Storglaciären
working_dir = '/home/thomas/regional_inversion/output/' + RID
input_file = working_dir + '/input.nc'

dem = load_dem_path(RID)
dem = crop_border_xarr(dem)



input_pism = rioxr.open_rasterio(input_file)
input_pism = input_pism.rio.write_crs(dem.rio.crs)
input_igm = input_pism.rio.reproject(input_pism.rio.crs, resolution = (100,100), nodata=0)
input_igm.ice_surface_temp.data = np.ones_like(input_igm.ice_surface_temp)*273
input_igm = input_igm.squeeze()
input_igm = input_igm.reindex(y=list(reversed(input_igm.y)))
input_igm.to_netcdf(working_dir + '/input_igm.nc')

S_ref = input_igm.usurf.data
S_ref = gauss_filter(S_ref, .6, 2)
B_init = input_igm.topg.data
dh_ref = input_igm.dhdt.data
smb = input_igm.climatic_mass_balance.data
a_smb = input_igm.apparent_mass_balance.data


dt = 1
pmax = 500
beta = 1
B_rec_all = []
misfit_all = []
glacier = Igm()
glacier.config.tstart                 = 0
glacier.config.tend                   = dt
glacier.config.tsave                  = dt#dt * pmax * 1e5
glacier.config.cfl                    = 0.3
glacier.config.init_slidingco         = 12
glacier.config.init_arrhenius         = 78
glacier.config.iceflow_model_lib_path = 'f15_cfsflow_GJ_22_a'
glacier.config.type_mass_balance      = 'sinus'
glacier.config.geology_file = working_dir + '/input_igm.nc'
glacier.config.iceflow_model_lib_path = '/home/thomas/regional_inversion/igm/f15_cfsflow_GJ_22_a/100'
glacier.initialize()
with tf.device(glacier.device_name):
    glacier.load_ncdf_data(glacier.config.geology_file)
    glacier.initialize_fields()

    glacier.smb.assign(smb/900)
    glacier.topg.assign(B_init)
    glacier.usurf.assign(S_ref)
    glacier.thk.assign((glacier.usurf - glacier.topg)*glacier.mask)#np.zeros_like(glacier.topg))
    p = 0
    while p < pmax:
        while glacier.t < glacier.config.tend:
            #glacier.update_smb()
            glacier.update_iceflow()
            glacier.update_t_dt()
            glacier.update_thk()
            glacier.print_info()
            #glacier.update_ncdf_ex()
            #glacier.update_ncdf_ts()
            #glacier.update_plot()

        dhdt = (glacier.usurf - input_igm.usurf.data)/dt
        misfit = (dhdt - dh_ref)
        glacier.usurf.assign(S_ref)
        glacier.topg.assign(np.minimum(glacier.usurf, glacier.topg - beta * misfit))
        glacier.thk.assign((glacier.usurf - glacier.topg)*glacier.mask)
        p+=1
        glacier.config.tend = p*dt+dt
        glacier.config.tstart = p*dt
        del glacier.already_called_update_t_dt
        B_rec_all.append(glacier.thk.numpy())
        misfit_all.append(misfit)
glacier.print_all_comp_info()
