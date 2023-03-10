import tensorflow as tf
import math
from igm.igm import Igm
from rasterio.enums import Resampling
import rasterio
import numpy as np
from load_input import *
import os
from write_output import *
from sklearn.preprocessing import normalize


#RID = 'RGI60-08.00213' # Storglaciären
RID = 'RGI60-08.00188' # Rabots
#RID = 'RGI60-08.00005' # Salajekna
#RID = 'RGI60-08.00146' # Paartejekna
#RID = 'RGI60-08.00121' # Mikkajekna
#RID = 'RGI60-07.01475' # Droenbreen
#RID = 'RGI60-07.00389' # Bergmesterbreen

glaciers_Sweden = get_RIDs_Sweden()
RIDs_Sweden = glaciers_Sweden.RGIId
RIDs_Sweden = [RID]
for RID in RIDs_Sweden:
    working_dir = '/home/thomas/regional_inversion/output/' + RID
    input_file = working_dir + '/input.nc'
    resolution = 100

    dem = load_dem_path(RID)
    dem = crop_border_xarr(dem)

    input_pism = rioxr.open_rasterio(input_file)
    input_pism = input_pism.rio.write_crs(dem.rio.crs)
    input_pism.mask.data = input_pism.mask.astype('int').data
    input_igm = input_pism.rio.reproject(input_pism.rio.crs, resolution = (resolution,resolution), resampling = Resampling.bilinear)
    input_igm.ice_surface_temp.data = np.ones_like(input_igm.ice_surface_temp)*273
    input_igm = input_igm.squeeze()
    #input_igm = input_igm.reindex(y=list(reversed(input_igm.y)))
    for var in input_igm.data_vars:
        input_igm.data_vars[var].data = input_igm.data_vars[var].rio.interpolate_na()

    input_igm.to_netcdf(working_dir + '/input_igm.nc')

    dummy_var = input_igm.usurf

    S_ref = deepcopy(input_igm.usurf.data)
    B_init = input_igm.topg.data
    dh_ref = input_igm.dhdt.data
    smb = input_igm.climatic_mass_balance.data
    a_smb = input_igm.apparent_mass_balance.data
    if RID == 'RGI60-07.01475':
        a_smb = deepcopy(smb) * 0.7
        a_smb += 195

    if RID == 'RGI60-07.00389':
        a_smb = deepcopy(smb) * 0.7
        a_smb += 163#232 


    mask = input_igm.mask.data
    #S_ref[mask==0] = np.nan
    S_ref = gauss_filter(S_ref,1, 3)
    #S_ref[mask==0] = input_igm.usurf.data[mask==0]
    #S_ref[mask == 0] += 100
    B_init[mask == 0] = S_ref[mask == 0]
    B_init[mask == 1] = S_ref[mask == 1]

    dt = 1
    pmax = 1000
    beta = 1
    theta = 0.3
    bw = 0
    p_save = 10
    p_mb = -500 #iterations before end when mass balance is recalculated

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
    #buffer *= (a_smb<0)

    B_rec_all = []
    misfit_all = []
    glacier = Igm()
    glacier.config.tstart = 0
    glacier.config.tend = dt
    glacier.config.tsave = dt * 10
    glacier.config.cfl = 0.3
    glacier.config.init_slidingco = 6
    glacier.config.init_arrhenius = 78 #78
    glacier.config.working_dir = working_dir
    glacier.config.vars_to_save.extend(['velbase_mag', 'uvelsurf', 'vvelsurf'])
    glacier.config.geology_file = working_dir + '/input_igm.nc'
    glacier.config.iceflow_model_lib_path = '/home/thomas/regional_inversion/igm/f15_cfsflow_GJ_22_a/{}'.format(resolution)
    glacier.initialize()
    with tf.device(glacier.device_name):
        glacier.load_ncdf_data(glacier.config.geology_file)
        glacier.initialize_fields()

        glacier.icemask.assign(mask)
        glacier.smb.assign((0.0 + a_smb/900) * glacier.icemask)
        glacier.topg.assign(B_init)
        glacier.usurf.assign(S_ref)
        glacier.thk.assign((glacier.usurf - glacier.topg)*glacier.icemask)#np.zeros_like(glacier.topg))
        p = 0
        left_sum = []
        s_refresh = 50000
        while p < pmax:
            if p > 0 and p%s_refresh == 0:
                glacier.usurf.assign(S_ref)
                theta *= 3
            S_old = glacier.usurf.numpy()
            while glacier.t < glacier.config.tend:

                glacier.update_iceflow()
                glacier.update_t_dt()
                glacier.update_thk()
                glacier.print_info()
                if p % p_save == 0:
                    glacier.update_ncdf_ex()
                    glacier.update_ncdf_ts()

            dhdt = (glacier.usurf - S_old)/dt

            misfit = (dhdt) #+ (S_ref - S_old) # - dh_ref)
            left_sum.append(np.sum(dhdt*(-mask+1)))
            # update surface
            S_old[mask == 1] = S_old[mask == 1] + theta * beta * misfit.numpy()[mask == 1] #*(normalize(calc_slope(glacier.thk.numpy()*mask, resolution).reshape(-1,1), axis = 0, norm = 'max').reshape(S_ref.shape))[mask==1]
            #(1-normalize(glacier.thk.numpy().reshape(-1,1), axis = 0, norm = 'max').reshape(S_ref.shape))[mask==1]
            glacier.usurf.assign(S_old)

            # update bed and thickness
            new_bed = glacier.topg.numpy() - beta * misfit.numpy()
            new_bed[mask == 0] = B_init[mask == 0]
            bed_before_buffer = deepcopy(new_bed)
            new_bed[np.where(buffer == 1)] = dummy_var.attrs['_FillValue']
            #new_thk = glacier.usurf.numpy() - (glacier.topg.numpy() - beta * misfit.numpy())
            #new_thk[mask == 0] = 0
            #new_thk[np.where(buffer == 1)] = dummy_var.attrs['_FillValue']

            dummy_var.data = new_bed
            dummy_var = dummy_var.rio.interpolate_na(method = 'cubic')

            #new_thk = dummy_var.data
            new_bed = dummy_var.data
            left_sum[-1] += np.sum(new_bed - bed_before_buffer)
            left_sum[-1] += np.sum(np.maximum(new_bed - glacier.usurf, 0))
            new_bed = np.minimum(glacier.usurf, new_bed)

            if p == (pmax - p_mb):
                left_sum_mean = np.mean(left_sum[-100:])
                #abl_area_size = len(np.nonzero(glacier.smb<0)[0])
                abl_area_size = len(np.nonzero(np.logical_and(glacier.thk<10, mask == 1))[0])
                left_sum_per_area = left_sum_mean / abl_area_size
                smb_new = a_smb/900
                #smb_new[glacier.smb<0] += left_sum_per_area
                smb_new[np.logical_and(glacier.thk<10, mask == 1)] += left_sum_per_area
                glacier.smb.assign(smb_new)
                #new_bed[np.logical_and(glacier.thk<10, mask == 1)] -= left_sum_per_area

            if p>0 and p%100000==0:
                new_bed = gauss_filter(new_bed, 1, 3)
                new_bed[mask==0] = B_init[mask == 0]
            glacier.topg.assign(new_bed)
            glacier.thk.assign(np.maximum(0, (glacier.usurf - glacier.topg)))#*glacier.icemask)
            # save data
            B_rec_all.append(glacier.thk.numpy())
            misfit_all.append(misfit)

            # prepare next iteration
            p+=1
            glacier.config.tend = p*dt+dt
            glacier.config.tstart = p*dt
            del glacier.already_called_update_t_dt

    glacier.print_all_comp_info()
    misfit_vs_iter = [np.mean(abs(x[mask == 1])) for x in misfit_all]

radar_bed = rioxr.open_rasterio('/home/thomas/regional_inversion/Storglaciären_radar_bed.tif')
radar_bed = radar_bed.rio.reproject_match(input_igm)
radar_bed.data[radar_bed.data == radar_bed.attrs['_FillValue']] = np.nan
radar_thk = S_ref - radar_bed.data[0]
