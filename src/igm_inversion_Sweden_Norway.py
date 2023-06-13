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
#RID = 'RGI60-08.00434' # Tunsbergdalsbreen
#RID = 'RGI60-08.01657' # Svartisen
#RID = 'RGI60-08.01779' # Hardangerjökulen
#RID = 'RGI60-08.02666' # Aalfotbreen
RID = 'RGI60-08.01258' # Langfjordjökulen
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

override_inputs = False
check_in_session = True # if this is false, global checking is activated
override_global = False
if check_in_session is False: # in that case, we check if this glacier should be modelled in check_file
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
#for RID in RIDs_with_obs[1:]:
    # check if this glacier has been modelled either in this session, or ever
    if check_in_session is False:
        already_checked = []
        with open(check_file, 'r') as fp:
            for item in fp:
                already_checked.append(item[:-1])
    if RID in already_checked:
        continue
    
    working_dir = home_dir / 'regional_inversion/output/' / RID
    input_file = working_dir / 'igm_input.nc'
    resolution = 100
    if not os.path.exists(input_file) or override_inputs is True:
        input_igm, internal_boundaries, area_RIDs = obtain_area_mosaic(RID, max_n_glaciers = 30, discard_list = already_checked)
        write_path_to_mosaic(RID, area_RIDs)
        if 'usurf_oggm' in input_igm.keys():
            input_igm.usurf_oggm.data = input_igm.usurf_oggm.where(input_igm.usurf_oggm != -9999, 9999).data
            input_igm.usurf_oggm.data = input_igm.usurf_oggm.where(~np.isnan(input_igm.usurf_oggm), 9999).data
        #input_igm.usurf_oggm.attrs['_FillValue'] = -9999.0
        input_igm = input_igm.fillna(0)
        if not internal_boundaries is None:
            input_igm = input_igm.where(internal_boundaries == 0, Fill_Value)
        input_igm.mask.values = input_igm.mask.fillna(1).values
        input_igm = input_igm.rio.interpolate_na()
        input_igm.topg.data = input_igm.usurf.data - input_igm.thk.data
        input_igm = input_igm.rio.write_crs(input_igm.rio.crs)
        input_igm.to_netcdf(input_file)

        already_checked.extend(area_RIDs)
        if check_in_session is False:
            with open(check_file, 'w') as fp:
                for item in already_checked:
                    fp.write("{}\n".format(item))
            
    else:
        input_igm = rioxr.open_rasterio(input_file)
        for var in input_igm.data_vars:
            input_igm.data_vars[var].rio.write_nodata(Fill_Value, inplace=True)
    input_igm = input_igm.squeeze()
    
    dummy_var = deepcopy(input_igm.usurf)

    S_ref = deepcopy(input_igm.usurf.data)
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
    
    #S_ref[mask == 0] = np.nan
    #S_ref = gauss_filter(S_ref,1, 3)
    #S_ref[mask == 0] = input_igm.usurf.data[mask == 0]
    a_smb[mask == 0] = np.nan
    a_smb = gauss_filter(a_smb,1, 3)
    a_smb[mask == 0] = 0

    S_ref[mask == 0] += 0
    S_old = tf.Variable(S_ref)
    B_init[mask == 0] = S_ref[mask == 0]
    B_init[mask == 1] = S_ref[mask == 1]#-300
    slope = calc_slope(S_ref, 100)
    slope_norm = np.zeros_like(slope)
    slope_norm[mask == 1] = normalize(-slope[mask == 1]**(1/2))

    A_tilde = np.zeros_like(mask) + 30
    vel_smooth = gauss_filter(vel_ref, 2, 4)
    
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

    dt = 1
    pmax = 5000
    beta_0 = 2.0
    theta = 0.3
    bw = 1
    p_save = 500
    p_mb = 1100  # iterations before end when mass balance is recalculated
    s_refresh = 550

    if RID in 'RGI60-08.00434' in area_RIDs:
        pmax = 12000
        p_mb = 4400
        s_refresh = 1200
        beta_0 = 1
    
    # establish buffer
    mask_iter = mask == 1
    #mask_bw = (~mask_iter)*1
    mask_bw = tf.cast(~mask_iter, tf.int8)
    buffer = np.zeros_like(mask)
    for i in range(bw):
        boundary_mask = mask_bw==0
        k = np.ones((3,3),dtype=int)
        boundary = nd.binary_dilation(tf.cast(boundary_mask, tf.int8)==0, k) & boundary_mask
        #mask_bw[boundary] = 1
        mask_bw = tf.where(boundary, 1, mask_bw)
    buffer = ((mask_bw + tf.cast(mask_iter, tf.int8))-1)

    B_rec_all = tf.Variable(tf.zeros(shape=(50, mask.shape[0], mask.shape[1])))
    #S_rec_all = []
    #misfit_all = []
    glacier = Igm()
    glacier.config.tstart = 0
    glacier.config.tend = dt
    glacier.config.tsave = 1
    glacier.config.cfl = 0.3
    glacier.config.init_slidingco = c
    glacier.config.init_arrhenius = A #np.ones_like(S_ref)*50
    glacier.config.working_dir = working_dir
    glacier.config.vars_to_save.extend(['velbase_mag', 'uvelsurf', 'vvelsurf'])
    glacier.config.verbosity = 0
    glacier.config.geology_file = working_dir / 'igm_input.nc'
    #glacier.config.iceflow_model_lib_path = home_dir / 'regional_inversion/igm/f17_pismbp_GJ_22_a/{}'.format(resolution)
    glacier.config.iceflow_model_lib_path = model_dir / 'f15_cfsflow_GJ_22_a/{}'.format(resolution)
    glacier.config.usegpu = True
    glacier.initialize()
    with tf.device(glacier.device_name):
        glacier.load_ncdf_data(glacier.config.geology_file)
        glacier.initialize_fields()

        glacier.icemask.assign(tf.cast(mask, tf.float32))
        glacier.smb.assign((a_smb) * glacier.icemask)
        glacier.topg.assign(B_init)
        glacier.usurf.assign(S_ref)
        glacier.thk.assign((glacier.usurf - glacier.topg)*glacier.icemask)
        p = 0
        #left_sum = []
        left_sum = tf.Variable(tf.zeros(shape=pmax))
        #basin_left = []
        basin_left = tf.Variable(tf.zeros(shape=(100, mask.shape[0], mask.shape[1])))
        beta_update = tf.Variable(0.0)
        update_surface = tf.Variable(0.0)
        while p < pmax:
            if p > 0 and p % s_refresh == 0:
                #glacier.usurf.assign(S_ref)
                S_diff = S_ref - glacier.usurf.numpy()
                S_diff = gauss_filter(S_diff, 2,4)*tf.cast(mask, tf.float32)
                #S_diff = tf.where(buffer == 1, 0, S_diff)
                glacier.usurf.assign(glacier.usurf + S_diff)
                glacier.topg.assign(tf.reduce_mean(B_rec_all[-20:], axis = 0))
                update_surface.assign(5)
                #beta_update = 0

            update_surface.assign_sub(1)
            beta_update.assign_add(1)
            beta = ((-10*beta_0)/(beta_update+10)) + beta_0
            #beta = ((300*beta_0)/(beta_update+300))
            S_old.assign(glacier.usurf)
            while glacier.t < glacier.config.tend:
                glacier.update_iceflow()
                glacier.update_t_dt()
                glacier.update_thk()
                glacier.print_info()
                if p % p_save == 0:
                    if p != pmax:
                        glacier.update_ncdf_ex()
                        glacier.update_ncdf_ts()

            dhdt = (glacier.usurf - S_old)/dt
            misfit = tf.math.minimum(tf.math.maximum(dhdt, -10), 10)
            #misfit[mask == 0] = np.nan
            #misfit = gauss_filter(misfit, 1, 3)
            #misfit[mask == 0] = 0
            left_sum[p].assign(tf.math.reduce_sum(tf.where(mask == 0, dhdt, 0)))#dhdt*(-mask+1)))
            if ((p >= (pmax - p_mb - 100)) & (p < (pmax - p_mb))):
                basin_left[p - (pmax - p_mb - 100)].assign(tf.where(mask == 0, dhdt, 0))#dhdt*(-mask+1))
            bed_before_buffer = glacier.topg - beta * misfit
            misfit_masked = misfit * tf.cast(mask, tf.float32)
            # update surface
            S_new = S_old + theta * beta * misfit_masked#  * (normalize(glacier.thk.numpy()**2))
            #S_new = tf.where(buffer == 1, S_ref-50, S_new)
            glacier.usurf.assign(S_new)

            # update bed and thickness
            new_bed = glacier.topg - beta * misfit_masked
            #new_bed[np.where(buffer == 1)] = 9999.0
            #new_thk = glacier.usurf.numpy() - (glacier.topg.numpy() - beta * misfit.numpy())
            #new_thk[mask == 0] = 0
            #new_thk[np.where(buffer == 1)] = dummy_var.attrs['_FillValue']

            #dummy_var.data = new_bed

            ## the below is causing issues, not sure why. Needs fixing if buffer should be used!
            #dummy_var = dummy_var.rio.interpolate_na(method = 'linear')

            #new_thk = dummy_var.data
            #new_bed = dummy_var.data
            #left_sum[-1] += np.sum(new_bed - bed_before_buffer)
            #left_sum[-1] += np.sum(np.maximum(new_bed - glacier.usurf, 0))
            #new_bed = tf.where(buffer == 1, shift(new_bed.numpy(), glacier.uvelsurf, glacier.vvelsurf, tf.cast(mask, tf.float32),  1), new_bed)
            new_bed = tf.math.minimum(glacier.usurf, new_bed)
            if p>10 and p%4 == 0 and update_surface < 0:
                if p < 50:
                    new_bed = tf.math.minimum(tf.reduce_mean(B_rec_all[p-4:p], axis = 0), S_new)
                else:
                    new_bed = tf.math.minimum(tf.reduce_mean(B_rec_all[-4:], axis = 0), S_new)
                #glacier.usurf.assign(np.mean(S_rec_all[-4:], axis = 0))
            if p>10 and p%50 == 0 and update_surface < 0:
                new_bed = tf.math.minimum(tf.reduce_mean(B_rec_all[-50:], axis = 0), S_new)
                #glacier.usurf.assign(np.mean(S_rec_all[-50:], axis = 0))
                
            if p == (pmax - p_mb):
            #if p == 600:
                left_sum_mean = tf.math.reduce_mean(left_sum[-100:])
                basin_left_mean = tf.math.reduce_mean(basin_left[-100:], axis = 0)
                smb_new = tf.Variable(a_smb)
                for b in tf.unique(tf.reshape(basin, -1))[0]:
                    aoi = tf.math.logical_and(basin == b, mask == 1)
                    abl_area_size = tf.math.count_nonzero(aoi)
                    b_left_mean = tf.math.reduce_sum(basin_left_mean[basin == b])
                    b_sum_per_area = b_left_mean / tf.cast(abl_area_size, tf.float32)
                    smb_new = tf.where(aoi, smb_new + b_sum_per_area, smb_new)
                beta_update = tf.Variable(0.0)
                    
                #abl_area_size = len(np.nonzero(a_smb<0)[0])
                #abl_area_size = len(np.nonzero(np.logical_and(glacier.thk<10, mask == 1))[0])
                #abl_area_size = len(np.nonzero(mask == 1)[0])
                #left_sum_per_area = left_sum_mean / abl_area_size
                #smb_new = deepcopy(a_smb)
                #smb_new[mask == 1] += left_sum_per_area
                #smb_new[np.logical_and(glacier.thk<10, mask == 1)] += left_sum_per_area
                glacier.smb.assign(smb_new)
                #new_bed[np.logical_and(glacier.thk<10, mask == 1)] -= left_sum_per_area
                #trend = LinearRegression().fit((S_ref[mask == 1]).reshape(-1,1), (glacier.usurf.numpy()-S_ref)[mask == 1].reshape(-1,1)).predict(S_ref[mask == 1].reshape(-1,1)).reshape(1,-1)[0]
                #S_new[mask == 1] = S_new[mask == 1] - trend
                #glacier.topg.assign(S_ref - glacier.thk.numpy())
                #glacier.usurf.assign(S_ref)

            glacier.topg.assign(new_bed)
            glacier.thk.assign(tf.math.maximum(0, (glacier.usurf - glacier.topg)))#*glacier.icemask)

            # save data
            if p >= 50:
                B_rec_all[:-1].assign(B_rec_all[1:])
                B_rec_all[-1].assign(glacier.topg)
            else:
                B_rec_all[p].assign(glacier.topg)
            #S_rec_all.append(glacier.usurf.numpy())
            #misfit_all.append(misfit)

            # prepare next iteration
            p += 1
            glacier.config.tend = p*dt+dt
            glacier.config.tstart = p*dt
            del glacier.already_called_update_t_dt
                
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

    dummy_var.data = glacier.thk.numpy()
    dummy_var.data[np.where(buffer == 1)] = dummy_var.attrs['_FillValue']
    dummy_var = dummy_var.rio.interpolate_na(method = 'linear')
    glacier.thk.assign(dummy_var.data)
    glacier.topg.assign(glacier.usurf - glacier.thk)
    glacier.update_ncdf_ex()
    glacier.print_all_comp_info()
    #misfit_vs_iter = [np.mean(abs(x[mask == 1])) for x in misfit_all]

    #raise(ValueError)
'''
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('/home/thomas/regional_inversion/RIDs_with_obs.pdf')
for RID in RIDs_with_obs[:-1]:
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
    #glathida_pd = pd.read_csv('/mnt/c/Users/thofr531/Documents/Global/glathida-3.1.0/data/TTT.csv', usecols = ['POINT_LAT', 'POINT_LON', 'THICKNESS', 'POLITICAL_UNIT', 'ELEVATION'])
    #glathida = gpd.GeoDataFrame(glathida_pd, geometry=gpd.points_from_xy(glathida_pd.POINT_LON, glathida_pd.POINT_LAT),crs = 'epsg: 4326')

    #glathida_NO = glathida[glathida.POLITICAL_UNIT == 'NO']
    #glathida_NO.to_csv('/mnt/c/Users/thofr531/Documents/Global/glathida-3.1.0/data/glathida_NO.csv')
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
    fig, ax = plt.subplots(2,3)
    fig.suptitle('OBS: {}, MOD: {}, CONS: {}'.format(np.round(np.var(gla_ras.THICK_OBS_CORR), 0), np.round(np.var(gla_ras.THICK_MOD), 0), np.round(np.var(gla_ras.THICK_CONS), 0)))
    ax[0,0].imshow(gla_ras.THICK_OBS_CORR, vmin = 0, vmax = np.maximum(gla_ras.THICK_OBS_CORR.max(), np.max(out_thk.read())))
    ax[0,1].imshow(gla_ras.THICK_MOD, vmin = 0, vmax = np.maximum(gla_ras.THICK_OBS_CORR.max(), np.max(out_thk.read())))
    glathida_NO.plot(glathida_NO.THICK_OBS_CORR, ax = ax[0,2])
    ax[1,0].imshow(out_thk.read()[0], vmin = 0, vmax = np.maximum(gla_ras.THICK_OBS_CORR.max(), np.max(out_thk.read())))
    ax[1,1].scatter(gla_ras.THICK_OBS_CORR, gla_ras.THICK_MOD, alpha  = .3)#, c = basin)
    #ax_ins = ax[1,1].inset_axes()
    #ax_ins.imshow(basin)
    ax[1,1].plot(range(int(gla_ras.THICKNESS.max())), range(int(gla_ras.THICKNESS.max())), '--', c='r')
    ax[1,1].set(adjustable='box', aspect='equal')
    ax[1,1].annotate('R = {}'.format(glathida_NO.corr().THICK_OBS_CORR['THICK_MOD'].round(4)),xy=(0.05, 0.75), xycoords='axes fraction')
    ax[1,1].annotate('R = {}'.format(glathida_NO.corr().THICK_OBS_CORR['THICK_CONS'].round(4)),xy=(0.05, 0.65), xycoords='axes fraction')
    ax[1,2].imshow(in_smb.read()[0], vmin = -3, vmax = 3, cmap = 'RdBu')
    #plt.show()
    pp.savefig()
    plt.close()
pp.close()

'''
