import tensorflow as tf
import math
from igm.igm import Igm
from rasterio.enums import Resampling
import rasterio
import numpy as np
from load_input import *
import os
from write_output import *


#RID = 'RGI60-08.00213' # Storglaciären
RID = 'RGI60-08.00188' # Rabots
#RID = 'RGI60-08.00005' # Salajekna
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
#S_ref = gauss_filter(S_ref, .6, 2)
B_init = input_igm.topg.data
dh_ref = input_igm.dhdt.data
smb = input_igm.climatic_mass_balance.data
a_smb = input_igm.apparent_mass_balance.data
mask = input_igm.mask.data

dt = 1
pmax = 1000
beta = 1
theta = 0
bw = 0
p_save = 10

# establish buffer
mask_iter = mask == 1
mask_bw = (~mask_iter)*1
buffer = np.zeros_like(mask)
for i in range(bw):
    boundary_mask = mask_bw==0
    k = np.ones((3,3),dtype=int)
    boundary = nd.binary_dilation(boundary_mask==0, k) & boundary_mask
    mask_bw[boundary] = 1
buffer = ((mask_bw + mask_iter*1)-1)

B_rec_all = []
misfit_all = []
glacier = Igm()
glacier.config.tstart                 = 0
glacier.config.tend                   = dt
glacier.config.tsave                  = dt * 10
glacier.config.cfl                    = 0.3
glacier.config.init_slidingco         = 0
glacier.config.init_arrhenius         = 78
glacier.config.working_dir = working_dir
glacier.config.vars_to_save.append('velbase_mag')
glacier.config.geology_file = working_dir + '/input_igm.nc'
glacier.config.iceflow_model_lib_path = '/home/thomas/regional_inversion/igm/f15_cfsflow_GJ_22_a/100'
glacier.initialize()
with tf.device(glacier.device_name):
    glacier.load_ncdf_data(glacier.config.geology_file)
    glacier.initialize_fields()

    glacier.icemask.assign(mask)
    glacier.smb.assign(a_smb/900 * glacier.icemask)
    glacier.topg.assign(B_init)
    glacier.usurf.assign(S_ref)
    glacier.thk.assign((glacier.usurf - glacier.topg)*glacier.icemask)#np.zeros_like(glacier.topg))
    p = 0
    while p < pmax:
        S_old = glacier.usurf.numpy()
        while glacier.t < glacier.config.tend:

            glacier.update_iceflow()
            glacier.update_t_dt()
            glacier.update_thk()
            glacier.print_info()
            if p % p_save == 0:
                glacier.update_ncdf_ex()
                glacier.update_ncdf_ts()

        #glacier.icemask.assign(mask)

        dhdt = (glacier.usurf - S_old)/dt
        misfit = (dhdt)# - dh_ref)

        # update surface
        S_ref[mask == 1] = S_ref[mask == 1] + theta * beta * misfit.numpy()[mask == 1]
        #S_ref = gauss_filter(S_ref, .3, 2)
        glacier.usurf.assign(S_ref)

        # update bed and thickness
        new_bed = glacier.topg.numpy() - beta * misfit.numpy()
        new_bed[mask == 0] = B_init[mask == 0]
        new_bed[np.where(buffer == 1)] = np.nan
        new_bed = inpaint_nans(new_bed)
        glacier.topg.assign(np.minimum(glacier.usurf, new_bed))
        glacier.thk.assign((glacier.usurf - glacier.topg)*glacier.icemask)

        #save data
        B_rec_all.append(glacier.thk.numpy())
        misfit_all.append(misfit)

        # prepare next iteration
        p+=1
        glacier.config.tend = p*dt+dt
        glacier.config.tstart = p*dt
        del glacier.already_called_update_t_dt
        
glacier.print_all_comp_info()
misfit_vs_iter = [np.mean(abs(x[mask == 1])) for x in misfit_all]
