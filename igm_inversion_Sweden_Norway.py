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

#RID = 'RGI60-08.00213' # Storglaciären
#RID = 'RGI60-08.00188' # Rabots
#RID = 'RGI60-08.00005' # Salajekna
#RID = 'RGI60-08.00146' # Paartejekna
#RID = 'RGI60-08.00121' # Mikkajekna
#RID = 'RGI60-07.01475' # Droenbreen
#RID = 'RGI60-07.00389' # Bergmesterbreen
#RID = 'RGI60-08.00434' # Tunsbergdalsbreen
RID = 'RGI60-08.01657' # Svartisen
#RID = 'RGI60-08.01779' # Hardangerjökulen

override_inputs = True
check_in_session = False # if this is false, global checking is activated
override_global = True
if check_in_session is False: # in that case, we check if this glacier should be modelled in check_file
    check_file = '/home/thomas/regional_inversion/already_checked.txt'
    if override_global is True:
        open(check_file, 'w').close()
    else:
        open(check_file, 'a').close()
fr = utils.get_rgi_region_file('08', version='62')
gdf = gpd.read_file(fr)
Fill_Value = 9999.0
already_checked = []
#for RID in gdf.RGIId.to_list():
for RID in [RID]:
    # check if this glacier has been modelled either in this session, or ever
    if check_in_session is False:
        already_checked = []
        with open(check_file, 'r') as fp:
            for item in fp:
                already_checked.append(item[:-1])
    if RID in already_checked:
        continue
    
    working_dir = '/home/thomas/regional_inversion/output/' + RID
    input_file = working_dir + '/igm_input.nc'
    resolution = 100

    if not os.path.exists(input_file) or override_inputs is True:
        input_igm, internal_boundaries, area_RIDs = obtain_area_mosaic(RID)
        write_path_to_mosaic(RID, area_RIDs)
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
    smb = input_igm.climatic_mass_balance.data
    a_smb = input_igm.apparent_mass_balance.data
    mask = input_igm.mask.data
    
    #S_ref[mask == 0] = np.nan
    #S_ref = gauss_filter(S_ref,1, 3)
    #S_ref[mask == 0] = input_igm.usurf.data[mask == 0]
    a_smb[mask == 0] = np.nan
    a_smb = gauss_filter(a_smb,1, 3)
    a_smb[mask == 0] = 0

    S_ref[mask == 0] += 0
    B_init[mask == 0] = S_ref[mask == 0]
    B_init[mask == 1] = S_ref[mask == 1]#-300
    slope = calc_slope(S_ref, 100)
    slope_norm = np.zeros_like(slope)
    slope_norm[mask == 1] = normalize(-slope[mask == 1]**(1/2))

    dt = 1
    pmax = 2500
    beta = 1
    theta = 0.3
    bw = 0
    p_save = 10
    p_mb = -800  # iterations before end when mass balance is recalculated
    s_refresh = 400
    
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
    glacier.config.tstart = 0
    glacier.config.tend = dt
    glacier.config.tsave = dt * 10
    glacier.config.cfl = 0.3
    glacier.config.init_slidingco = 0
    arr_norm = np.zeros_like(S_ref)
    arr_norm[mask == 1] = normalize(-S_ref[mask == 1]) * 60
    glacier.config.init_arrhenius = np.ones_like(S_ref)*78# + arr_norm
    glacier.config.working_dir = working_dir
    glacier.config.vars_to_save.extend(['velbase_mag', 'uvelsurf', 'vvelsurf'])
    glacier.config.verbosity = 0
    glacier.config.geology_file = working_dir + '/igm_input.nc'
    glacier.config.iceflow_model_lib_path = '/home/thomas/regional_inversion/igm/f15_cfsflow_GJ_22_a/{}'.format(resolution)
    glacier.initialize()
    with tf.device(glacier.device_name):
        glacier.load_ncdf_data(glacier.config.geology_file)
        glacier.initialize_fields()

        glacier.icemask.assign(mask)
        glacier.smb.assign((a_smb) * glacier.icemask)
        glacier.topg.assign(B_init)
        glacier.usurf.assign(S_ref)
        glacier.thk.assign((glacier.usurf - glacier.topg)*glacier.icemask)
        p = 0
        left_sum = []
        beta_update = 0
        while p < pmax:
            if p > 600 and p % s_refresh == 0:
                #glacier.usurf.assign(S_ref)
                S_diff = S_ref - glacier.usurf.numpy()
                S_diff = gauss_filter(S_diff, 2,4)*mask
                glacier.usurf.assign(glacier.usurf.numpy() + S_diff)
                theta+=.1
            #beta = ((-10*beta_0)/(beta_update+10)) +beta_0
            #beta = ((300*beta_0)/(beta_update+300))
            #beta_update += 1
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
            misfit = dhdt.numpy()
            #misfit[mask == 0] = np.nan
            #misfit = gauss_filter(misfit, 1, 3)
            #misfit[mask == 0] = 0
            left_sum.append(np.sum(dhdt*(-mask+1)))
            bed_before_buffer = glacier.topg.numpy() - beta * misfit
            misfit_masked = misfit * mask
            # update surface
            S_new = S_old + theta * beta * misfit_masked#  * (normalize(glacier.thk.numpy()**2))
            glacier.usurf.assign(S_new)

            # update bed and thickness
            new_bed = glacier.topg.numpy() - beta * misfit_masked
            new_bed[np.where(buffer == 1)] = 9999.0
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
            new_bed = np.minimum(glacier.usurf, new_bed)

            if p == (pmax - p_mb):
                left_sum_mean = np.mean(left_sum[-100:])
                #abl_area_size = len(np.nonzero(glacier.smb<0)[0])
                abl_area_size = len(np.nonzero(np.logical_and(glacier.thk<10, mask == 1))[0])
                #abl_area_size = len(np.nonzero(mask == 1)[0])
                left_sum_per_area = left_sum_mean / abl_area_size
                smb_new = deepcopy(a_smb)
                #smb_new[mask == 1] += left_sum_per_area
                smb_new[np.logical_and(glacier.thk<10, mask == 1)] += left_sum_per_area
                glacier.smb.assign(smb_new)
                #new_bed[np.logical_and(glacier.thk<10, mask == 1)] -= left_sum_per_area
                #trend = LinearRegression().fit((S_ref[mask == 1]).reshape(-1,1), (glacier.usurf.numpy()-S_ref)[mask == 1].reshape(-1,1)).predict(S_ref[mask == 1].reshape(-1,1)).reshape(1,-1)[0]
                #S_new[mask == 1] = S_new[mask == 1] - trend
                #glacier.topg.assign(S_ref - glacier.thk.numpy())
                #glacier.usurf.assign(S_ref)

            glacier.topg.assign(new_bed)
            glacier.thk.assign(np.maximum(0, (glacier.usurf - glacier.topg)))#*glacier.icemask)
            # save data
            #B_rec_all.append(glacier.thk.numpy())
            #misfit_all.append(misfit)

            # prepare next iteration
            p += 1
            glacier.config.tend = p*dt+dt
            glacier.config.tstart = p*dt
            del glacier.already_called_update_t_dt

    glacier.print_all_comp_info()
    #misfit_vs_iter = [np.mean(abs(x[mask == 1])) for x in misfit_all]


'''
out_to_tif(RID, 'topg', i = -1, file = 'ex.nc', file_not_standard_dims = True)
out_to_tif(RID, 'thk', i = -1, file = 'ex.nc', file_not_standard_dims = True)
out_to_tif(RID, 'usurf', i = 0, file = 'ex.nc', file_not_standard_dims = True)
out_thk = rasterio.open(working_dir + '/thk.tif')
out_topg = rasterio.open(working_dir + '/topg.tif')
in_usurf = rasterio.open(working_dir + '/usurf.tif')
#thk_consensus = rasterio.open('/mnt/c/Users/thofr531/Documents/Global/Scandinavia/consensus_thk/RGI60-08/all_thk_consensus.tif')
glathida_pd = pd.read_csv('/mnt/c/Users/thofr531/Documents/Global/glathida-3.1.0/data/TTT.csv', usecols = ['POINT_LAT', 'POINT_LON', 'THICKNESS', 'POLITICAL_UNIT', 'ELEVATION'])
glathida = gpd.GeoDataFrame(glathida_pd, geometry=gpd.points_from_xy(glathida_pd.POINT_LON, glathida_pd.POINT_LAT),crs = 'epsg: 4326')

glathida_NO = glathida[glathida.POLITICAL_UNIT == 'NO']
glathida_NO = glathida_NO.to_crs(out_thk.crs)
glathida_NO['TOPG'] = glathida_NO.ELEVATION - glathida_NO.THICKNESS
coords = [(x,y) for x, y in zip(glathida_NO.geometry.x, glathida_NO.geometry.y)]
glathida_NO['THICK_MOD'] = [x[0] for x in out_thk.sample(coords)]
#glathida_NO['THICK_CONS'] = [x[0] for x in thk_consensus.sample(coords)]
glathida_NO['TOPG_MOD'] = [x[0] for x in out_topg.sample(coords)]
glathida_NO['USURF_TODAY'] = [x[0] for x in in_usurf.sample(coords)]
glathida_NO['THICK_OBS_CORR'] = (glathida_NO.USURF_TODAY) - glathida_NO.TOPG
glathida_NO = glathida_NO.where(glathida_NO.THICK_MOD < 1e5)
glathida_NO = glathida_NO.where(glathida_NO.THICK_MOD > -1e5)
glathida_NO['Difference'] = glathida_NO.THICK_OBS_CORR - glathida_NO.THICK_MOD
glathida_NO['Percent_difference'] = glathida_NO.Difference/glathida_NO.THICK_OBS_CORR
'''
