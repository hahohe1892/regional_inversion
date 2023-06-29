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
from load_input import obtain_area_mosaic
from funcs import normalize
from sklearn.linear_model import LinearRegression
from geocube.api.core import make_geocube
import sys
from pathlib import Path
from tensorflow.python.client import device_lib

#RID = 'RGI60-08.00213' # Storglaciären
#RID = 'RGI60-08.00188' # Rabots
#RID = 'RGI60-08.00005' # Salajekna
#RID = 'RGI60-08.00146' # Paartejekna
#RID = 'RGI60-08.00121' # Mikkajekna
#RID = 'RGI60-07.01475' # Droenbreen
#RID = 'RGI60-07.00389' # Bergmesterbreen
RID = 'RGI60-08.00434' # Tunsbergdalsbreen
#RID = 'RGI60-08.01657' # Svartisen
#RID = 'RGI60-08.01779' # Hardangerjökulen
#RID = 'RGI60-08.02666' # Aalfotbreen
#RID = 'RGI60-08.01258' # Langfjordjökulen
#RID = 'RGI60-08.02382' # Rundvassbreen
#RID = 'RGI60-08.00966' # Vestre Memurubreen
#RID = 'RGI60-08.00987' # Graasubreen
#RID = 'RGI60-08.00312' # Storbreen
#RID = 'RGI60-08.02972' # Botnabrea
#RID = 'RGI60-08.01103' # Boeyabreen
#RID = 'RGI60-08.00435' # Folgefonna

RIDs_with_obs = ['RGI60-08.00434', 'RGI60-08.01657', 'RGI60-08.01779', 'RGI60-08.02666', 'RGI60-08.01258', 'RGI60-08.02382', 'RGI60-08.00966', 'RGI60-08.00987', 'RGI60-08.00312', 'RGI60-08.02972', 'RGI60-08.01103', 'RGI60-08.00435', 'RGI60-08.00213']

if os.getcwd() == '/home/thomas/regional_inversion/src':
    home_dir = Path('/home/thomas')
    model_dir = home_dir / 'regional_inversion/src/igm'
else:
    home_dir = Path('/mimer/NOBACKUP/groups/snic2022-22-55/')
    model_dir = Path('{}/regional_inversion/src/igm/'.format(os.environ['HOME']))

override_inputs = True
check_global_already_modelled = True # if this is false, global checking is activated
override_global = True
if check_global_already_modelled is True: # in that case, we check if this glacier should be modelled in check_file
    check_file = home_dir / 'regional_inversion/already_checked.txt'
    if override_global is True:
        open(check_file, 'w').close()
    else:
        open(check_file, 'a').close()
fr = utils.get_rgi_region_file('08', version='62')
gdf = gpd.read_file(fr)
Fill_Value = 9999.0
already_checked = []
#for RID in gdf.RGIId.to_list()[4:]:
for RID in [RID]:
#for RID in RIDs_with_obs:
    # check if this glacier has been modelled either in this session, or ever
    if check_global_already_modelled is True:
        already_checked = []
        with open(check_file, 'r') as fp:
            for item in fp:
                already_checked.append(item[:-1])
    if RID in already_checked:
        continue
    
    working_dir = home_dir / 'regional_inversion/output/' / RID
    input_file = working_dir / 'igm_input.nc'
    resolution = 100
    area_RIDs = []
    if not os.path.exists(input_file) or override_inputs is True:

        # obtain all glaciers connected in one glacier complex
        input_igm, internal_boundaries, area_RIDs = obtain_area_mosaic(RID, discard_list = already_checked)
        write_path_to_mosaic(RID, area_RIDs)

        input_igm.usurf_oggm.data = input_igm.usurf_oggm.where(input_igm.usurf_oggm != -9999, 9999).data
        input_igm.usurf_oggm.data = input_igm.usurf_oggm.where(~np.isnan(input_igm.usurf_oggm), 9999).data

        input_igm = input_igm.fillna(0)
        if not internal_boundaries is None:
            input_igm = input_igm.where(internal_boundaries == 0, Fill_Value)
        input_igm.mask.values = input_igm.mask.fillna(1).values
        input_igm = input_igm.rio.interpolate_na()
        input_igm.topg.data = input_igm.usurf.data - input_igm.thk.data
        input_igm = input_igm.rio.write_crs(input_igm.rio.crs)
        input_igm.to_netcdf(input_file)

        already_checked.extend(area_RIDs)
        if check_global_already_modelled is True:
            with open(check_file, 'w') as fp:
                for item in already_checked:
                    fp.write("{}\n".format(item))
            
    else:
        input_igm = rioxr.open_rasterio(input_file)
        for var in input_igm.data_vars:
            input_igm.data_vars[var].rio.write_nodata(Fill_Value, inplace=True)
    input_igm = input_igm.squeeze()

    # obtain input fields from input_igm
    S_ref = deepcopy(input_igm.usurf.data)
    S_old = tf.Variable(S_ref)
    B_init = input_igm.topg.data
    dh_ref = input_igm.dhdt.data
    vel_ref = input_igm['velocity by Millan']
    vel_ref = vel_ref.where(abs(vel_ref.data) < 1e3, Fill_Value)
    vel_ref = vel_ref.rio.interpolate_na().data
    smb = input_igm.climatic_mass_balance.data
    a_smb = input_igm.apparent_mass_balance.data
    mask = tf.Variable(input_igm.mask.data, dtype = tf.int8)
    basin = input_igm.mask_count
    basin.data[basin.data == 0] = Fill_Value
    basin = tf.Variable(basin.rio.interpolate_na().data)

    # smooth apparent mass balance to even out basin boundaries
    a_smb[mask == 0] = np.nan
    a_smb = gauss_filter(a_smb,1, 3)
    a_smb[mask == 0] = 0

    # below is possibility to experiment with some settings
    S_ref[mask == 0] += 0
    B_init[mask == 0] = S_ref[mask == 0]
    B_init[mask == 1] = gauss_filter(B_init, 2, 4)[mask == 1] #S_ref[mask == 1]
    
    # smooth velocity field
    vel_smooth = gauss_filter(vel_ref, 2, 4)

    # set A and c based on velocity pattern
    A_tilde = np.zeros_like(mask) + 30
    if (vel_smooth == 0).all(): # velocity not available for all glaciers
        A = np.ones_like(A_tilde) * 60
        c = np.zeros_like(A_tilde)
    else:
        q25 = np.quantile(vel_smooth[mask == 1], .05)
        q75 = np.quantile(vel_smooth[mask == 1], .95)
        A_tilde[np.logical_and(mask == 1, vel_smooth <= q25)] = 30
        A_tilde[np.logical_and(mask == 1, vel_smooth >= q75)] = 84
        A_tilde[np.logical_and(mask == 1, np.logical_and(vel_smooth > q25, vel_smooth < q75))] = normalize(vel_smooth[np.logical_and(mask == 1, np.logical_and(vel_smooth > q25, vel_smooth < q75))]) * 54 + 30
        A = np.where(A_tilde <= 78, A_tilde, 78)
        c = np.where(A_tilde <= 78, 0, A_tilde - 78)

    # set inversion parameters (note: no buffer used currently)
    dt = .2
    pmax = 6000
    beta_0 = 0.5
    theta = 0.8
    p_save = 200 # number of iterations when output is saved
    p_mb = 1500  # iterations before end when mass balance is recalculated
    s_refresh = 250 # number of iterations when surface is reset

    # if Jostedalsbreen is simulated, change inversion parameters
    if 'RGI60-08.00434' in [area_RIDs, RID]:
        pmax = 10000
        p_mb = 3000
        s_refresh = 600
        beta_0 = .5
    
    # prepare vector where last 50 beds will be saved in (used for bed averaging)
    B_rec_all = tf.Variable(tf.zeros(shape=(50, mask.shape[0], mask.shape[1])))

    # initialize and fill igm glacier class
    glacier = Igm()
    glacier.config.tstart = 0
    glacier.config.tend = dt
    glacier.config.tsave = 1
    glacier.config.cfl = 0.3
    glacier.config.init_slidingco = 6
    glacier.config.init_arrhenius = 55
    glacier.config.working_dir = working_dir
    glacier.config.vars_to_save.extend(['velbase_mag', 'uvelsurf', 'vvelsurf', 'dhdt'])
    glacier.config.verbosity = 0
    glacier.config.geology_file = working_dir / 'igm_input.nc'
    glacier.config.iceflow_model_lib_path = model_dir / 'f15_cfsflow_GJ_22_a/{}'.format(resolution)
    glacier.config.usegpu = True
    glacier.initialize()
    with tf.device(glacier.device_name):
        glacier.load_ncdf_data(glacier.config.geology_file)
        glacier.initialize_fields()

        # assign input fields to glacier class
        glacier.icemask.assign(tf.cast(mask, tf.float32))
        glacier.smb.assign((a_smb) * glacier.icemask)
        glacier.topg.assign(B_init)
        glacier.usurf.assign(S_ref)
        glacier.thk.assign((glacier.usurf - glacier.topg)*glacier.icemask)
        glacier.dhdt.assign(S_ref * 0)
        glacier.var_info['dhdt'] = ['surface elevation change', 'm/yr']

        # set variables that change with iterations
        p = 0
        beta_update = tf.Variable(0.0)
        update_surface = tf.Variable(0.0)

        # prepare vectors where info on ice leaving domain will be saved
        left_sum = tf.Variable(tf.zeros(shape=pmax))
        basin_left = tf.Variable(tf.zeros(shape=(100, mask.shape[0], mask.shape[1])))
        
        while p < pmax:

            # every s_refresh iterations, reset surface by adding smoothed difference between
            # original and present surface to present surface
            if p > 0 and p % s_refresh == 0:
                S_diff = S_ref - glacier.usurf.numpy()
                S_diff = gauss_filter(S_diff, 2,4)*tf.cast(mask, tf.float32)
                glacier.usurf.assign(glacier.usurf + S_diff)
                
                glacier.topg.assign(tf.reduce_mean(B_rec_all[-20:], axis = 0))
                update_surface.assign(5)
                glacier.slopsurfx, glacier.slopsurfy = glacier.compute_gradient_tf(glacier.usurf, glacier.dx, glacier.dx)
                glacier.thk.assign(glacier.usurf - glacier.topg)
                
            update_surface.assign_sub(1)

            # apply beta ramp-up
            beta_update.assign_add(1)
            beta = ((-10*beta_0)/(beta_update+10)) + beta_0

            # set S_old: needed to calculate dhdt
            S_old.assign(glacier.usurf)

            # run forward simulation with IGM
            while glacier.t < glacier.config.tend:
                glacier.update_iceflow()
                glacier.update_t_dt()
                glacier.update_thk()
                glacier.print_info()

                # save output every p_save iterations
                if p % p_save == 0:
                    if p != pmax:
                        glacier.update_ncdf_ex()
                        glacier.update_ncdf_ts()

            # calculate dhdt
            dhdt = (glacier.usurf - S_old)/dt
            glacier.dhdt.assign(dhdt)
            # dhdt is equal to misfit when using apparent mass balance
            misfit = tf.math.minimum(tf.math.maximum(dhdt, -2), 2)

            # check how much ice has left mask
            left_sum[p].assign(tf.math.reduce_sum(tf.where(mask == 0, dhdt, 0)))

            # only add spatially distributed left ice to vector if
            # the mass balance update happens within the next 100 iterations
            if ((p >= (pmax - p_mb - 100)) & (p < (pmax - p_mb))):
                basin_left[p - (pmax - p_mb - 100)].assign(tf.where(mask == 0, dhdt, 0))

            # mask misfit
            misfit_masked = misfit * tf.cast(mask, tf.float32)
            
            # update surface
            S_new = S_old + theta * beta * misfit_masked
            glacier.usurf.assign(S_new)

            # update bed
            new_bed = tf.math.minimum(glacier.usurf, glacier.topg - beta * misfit_masked)

            # do bed averaging
            if p>10 and p%40 == 0 and update_surface < 0:
                if p < 50:
                    new_bed = tf.math.minimum(tf.reduce_mean(B_rec_all[p-4:p], axis = 0), S_new)
                else:
                    new_bed = tf.math.minimum(tf.reduce_mean(B_rec_all[-4:], axis = 0), S_new)

            if p>10 and p%50 == 0 and update_surface < 0:
                new_bed = tf.math.minimum(tf.reduce_mean(B_rec_all[-50:], axis = 0), S_new)

            # assign new ice thickness and bed
            glacier.topg.assign(new_bed)
            glacier.thk.assign(tf.math.maximum(0, (glacier.usurf - glacier.topg)))

            #set minimum thickness
            #glacier.thk.assign(tf.where(glacier.thk<10, 10 * tf.cast(mask, tf.float32), glacier.thk))
            #glacier.topg.assign(glacier.usurf - glacier.thk)
            #print(np.min(glacier.thk[mask == 1]))
            # update mass balance to account for ice leaving mask
            if p == (pmax - p_mb):
                left_sum_mean = tf.math.reduce_mean(left_sum[-100:])
                basin_left_mean = tf.math.reduce_mean(basin_left[-100:], axis = 0)
                smb_new = tf.Variable(a_smb)

                # ice leaving mask is calculated seperately for each glacier basin
                for b in tf.unique(tf.reshape(basin, -1))[0]:
                    aoi = tf.math.logical_and(basin == b, mask == 1)
                    abl_area_size = tf.math.count_nonzero(aoi)
                    b_left_mean = tf.math.reduce_sum(basin_left_mean[basin == b])
                    b_sum_per_area = b_left_mean / tf.cast(abl_area_size, tf.float32)
                    smb_new = tf.where(aoi, smb_new + b_sum_per_area, smb_new)
                beta_update = tf.Variable(0.0)
                    
                glacier.smb.assign(smb_new)

            # save data
            if p >= 50:
                B_rec_all[:-1].assign(B_rec_all[1:])
                B_rec_all[-1].assign(glacier.topg)
            else:
                B_rec_all[p].assign(glacier.topg)

            # prepare next iteration
            p += 1
            glacier.config.tend = p*dt+dt
            glacier.config.tstart = p*dt
            del glacier.already_called_update_t_dt
            glacier.slopsurfx, glacier.slopsurfy = glacier.compute_gradient_tf(glacier.usurf, glacier.dx, glacier.dx)
            #min_slope = 0.02
            #slope = glacier.getmag(glacier.slopsurfx, glacier.slopsurfy)
            #slope_factor = tf.math.minimum(min_slope/slope, 1e3)
            #glacier.slopsurfx = tf.where(slope < min_slope, glacier.slopsurfx * slope_factor, glacier.slopsurfx)
            #glacier.slopsurfy = tf.where(slope < min_slope, glacier.slopsurfy * slope_factor, glacier.slopsurfy)
                
    # establish buffer
    bw = 1
    mask_iter = mask == 1
    mask_bw = tf.cast(~mask_iter, tf.int8)
    buffer = np.zeros_like(mask)
    for i in range(bw):
        boundary_mask = mask_bw==0
        k = np.ones((3,3),dtype=int)
        boundary = nd.binary_dilation(tf.cast(boundary_mask, tf.int8)==0, k) & boundary_mask
        mask_bw = tf.where(boundary, 1, mask_bw)
    buffer = ((mask_bw + tf.cast(mask_iter, tf.int8))-1)

    # interpolate around ice margin
    dummy_var = deepcopy(input_igm.usurf)
    dummy_var.data = glacier.thk.numpy()
    dummy_var.data[np.where(buffer == 1)] = dummy_var.attrs['_FillValue']
    dummy_var = dummy_var.rio.interpolate_na(method = 'linear')
    glacier.thk.assign(dummy_var.data)
    glacier.topg.assign(glacier.usurf - glacier.thk)
    glacier.update_ncdf_ex()
    glacier.print_all_comp_info()
