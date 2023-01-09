import numpy as np
from load_input import *
import PISM
from bed_inversion import *
import os
import shutil
from oggm.core import massbalance
import subprocess
from mpi4py import MPI
from write_output import *

comm = MPI.COMM_WORLD
size = comm.Get_size() # new: gives number of ranks in comm
rank = comm.Get_rank()

#RID = 'RGI60-08.00010'
#RID = 'RGI60-08.00213' # Storglaciären
#RID = 'RGI60-08.00006'
#RID = 'RGI60-08.00085'
glaciers_Sweden = get_RIDs_Sweden()
RIDs_Sweden = glaciers_Sweden.RGIId
sample_glaciers = ['RGI60-08.00005', 'RGI60-08.00146', 'RGI60-08.00233', 'RGI60-08.00223', 'RGI60-08.00021']
#for RID in RIDs_Sweden:
for i in range(1):
    try:
        #RID = 'RGI60-08.00251'
        RID = 'RGI60-08.00213' # Storglaciären
        #RID = 'RGI60-08.00188' # Rabots
        #RID = 'RGI60-08.00202'
        #RID = 'RGI60-08.00227'
        working_dir = '/home/thomas/regional_inversion/output/' + RID
        input_file = working_dir + '/input.nc'

        # default DEM (COPDEM) is from 2010 - 2015
        period = '2010-2015'

        dem = load_dem_path(RID)
        dem = crop_border_xarr(dem)

        x = dem.x
        y = np.flip(dem.y)

        options = {
            "-Mz": 50,
            "-Lz": 1500,
            "-Mx": x.size,
            "-Lx": int(x.max() - x.min())/1000,
            "-My": y.size,
            "-Ly": int(y.max() - y.min())/1000,
            "-Lbz": 1,
            "-allow_extrapolation": True,
            "-surface": "given",
            #"-surface.given.file": input_file,
            "-i": input_file,
            "-bootstrap": "",
            "-energy": "none",
            "-sia_flow_law": "isothermal_glen",
            "-ssa_flow_law": "isothermal_glen",
            "-stress_balance": "sia",
            "-yield_stress": "constant",
            #"-pseudo_plastic": "",
            "-pseudo_plastic_q": 0.2,
            "-pseudo_plastic_uthreshold": 3.1556926e7,
            "-geometry.update.use_basal_melt_rate": "no",
            "-stress_balance.ssa.compute_surface_gradient_inward": "yes",
            "-stress_balance.sia.surface_gradient_method": "haseloff",
            "-flow_law.isothermal_Glen.ice_softness": 1.733e3*np.exp(-13.9e4/(8.3*272)), # 3.9565534675428266e-24,
            "-constants.ice.density": 900.,
            "-constants.sea_water.density": 1000.,
            "-bootstrapping.defaults.geothermal_flux": 0.0,
            "-stress_balance.ssa.Glen_exponent": 3.,
            "-constants.standard_gravity": 9.8,
            "-stress_balance.sia.bed_smoother.range": 0.0,
            #"-basal_resistance.beta_lateral_margin": 0,
            "-o": working_dir + "/output.nc",
            "-output.timeseries.times": 1,
            "-output.timeseries.filename": working_dir + "/timeseries.nc",
            "-output.extra.times": .5,
            "-output.extra.file": working_dir + "/extra.nc",
            "-output.extra.vars": "diffusivity,thk,topg,usurf,velsurf_mag,mask,taub_mag,taud_mag,velbar_mag,flux_mag,velbase_mag,climatic_mass_balance,rank,uvel,vvel",
            "-sea_level.constant.value": -10000,
            "-time_stepping.assume_bed_elevation_changed": "true"
            }

        # switch between using apparent mass balance (the default, use_apparent_mb = True)
        # and actual mass balance (use_apparent_mb = False)
        # use carfully to not mess up what is written in input file

        use_apparent_mb = False
        if rank == 0:
            if 'original_climatic_mass_balance' in NC(input_file).variables.keys(): # checks input file
                raise ValueError('check whether apparent mass balance and actual mass balance are in the right spot in the input file; something is probably messed up here')

        if use_apparent_mb is True:
            if rank == 0:
                cmd = ['ncrename', '-h', '-O', '-v', 'climatic_mass_balance,original_climatic_mass_balance', input_file]
                subprocess.run(cmd)
                cmd = ['ncatted', '-a', 'standard_name,original_climatic_mass_balance,o,c,original_mass_flux',input_file]
                subprocess.run(cmd)
                cmd = ['ncrename', '-h', '-O', '-v', 'apparent_mass_balance,climatic_mass_balance', input_file]
                subprocess.run(cmd)
                cmd = ['ncatted', '-a', 'standard_name,climatic_mass_balance,o,c,land_ice_surface_specific_mass_balance_flux', input_file]
                subprocess.run(cmd)

        comm.Barrier() #make sure that process on rank 0 is done before proceeding; perhaps not needed though

        #X, Y = np.meshgrid(x,y) 
        #nc_input = NC(input_file, 'r+')
        #mask_infile = nc_input['mask'][:,:]
        #cmb = nc_input['climatic_mass_balance'][:,:]
        #new_mb = np.zeros_like(Y)
        #new_mb[mask_infile == 1] = cmb[mask_infile==1]#(((Y - np.min(Y))-2000)/10)[5:-5,5:-5]
        #nc_input['climatic_mass_balance'][:,:] = new_mb
        #nc_input.close()

        pism = create_pism(input_file = input_file, options = options, grid_from_options = False)

        #nc_input = NC(input_file, 'r+')
        #nc_input['climatic_mass_balance'][:,:] = cmb
        #nc_input.close()


        if use_apparent_mb is True:
            if rank == 0:
                cmd = ['ncrename', '-h', '-O', '-v', 'climatic_mass_balance,apparent_mass_balance', input_file]
                subprocess.run(cmd)
                cmd = ['ncatted', '-a', 'standard_name,apparent_mass_balance,o,c,apparent_mb', input_file]
                subprocess.run(cmd)
                cmd = ['ncrename', '-h', '-O', '-v', 'original_climatic_mass_balance,climatic_mass_balance', input_file]
                subprocess.run(cmd)
                cmd = ['ncatted', '-a', 'standard_name,climatic_mass_balance,o,c,land_ice_surface_specific_mass_balance_flux',input_file]
                subprocess.run(cmd)

        dh_ref = read_variable(pism.grid(), input_file, 'dhdt', 'm year-1')
        mask = read_variable(pism.grid(), input_file, 'mask', '')
        mask[mask>=0.5] = 1
        mask[mask<0.5] = 0
        S_rec = read_variable(pism.grid(), input_file, 'usurf', 'm')
        B_rec = read_variable(pism.grid(), input_file, 'topg', 'm')
        if use_apparent_mb is True:
            smb = read_variable(pism.grid(), input_file, 'apparent_mass_balance', 'kg m-2 year-1')
        else:
            smb = read_variable(pism.grid(), input_file, 'climatic_mass_balance', 'kg m-2 year-1')
        #smb = np.zeros_like(mask)
        tauc = np.ones_like(mask)*5e5

        if use_apparent_mb is True:
            dh_ref *= 0
        # set inversion paramters
        dt = .1
        beta = .25
        theta = 0.075
        bw = 1
        pmax = 1000
        p_friction = 1000
        max_steps_PISM = 25
        res = dem.rio.resolution()[0]
        A = 1.733e3*np.exp(-13.9e4/(8.3*272)) # 3.9565534675428266e-24

        B_init = np.copy(B_rec)
        S_ref = np.copy(S_rec)
        B_rec_all = []
        S_rec_all = []
        misfit_all = []

        ### mask out steep sections and where slope change is large ###
        data = np.copy(mask)
        buffer_mask = np.copy(mask)
        mask_b = create_buffer(data, buffer_mask,3)
        mask = np.maximum(mask, mask_b)

        slope = np.rad2deg(np.arctan(calc_slope(S_rec, res)))
        mask[slope > 30] = 0
        slope_change = calc_slope(calc_slope(calc_slope(S_rec, res), res), res)
        slope_change_new = np.gradient(calc_slope(calc_slope(S_rec, res), res))
        slope_change_new = np.sqrt(slope_change_new[0]**2+ slope_change_new[1]**2)*np.sign(np.maximum(slope_change_new[0], slope_change_new[1]))

        slope_change_new *= (slope_change_new<0)
        ### establish buffer to correct mask ###
        bw_m = 5
        mask_iter = mask == 1
        mask_bw = (~mask_iter)*1
        criterion = np.zeros_like(mask)
        for i in range(bw_m):
            boundary_mask = mask_bw==0
            k = np.ones((3,3),dtype=int)
            boundary = nd.binary_dilation(boundary_mask==0, k) & boundary_mask
            mask_bw[boundary] = 1
        criterion[3:-3,3:-3] = ((mask_bw + mask_iter*1)-1)[3:-3,3:-3]

        aoi = np.logical_and(slope_change**4>10e-20, criterion == 1)
        aoi_new = np.logical_and(slope_change_new**2>1e-7, criterion == 1)
        mask -=aoi

        ### establish buffer lift up sides ###
        bw_s = 20
        mask_s = mask == 1
        mask_bws = (~mask_s)*1
        criterion_s = np.zeros_like(mask)
        for i in range(bw_s):
            boundary_mask_s = mask_bws==0
            k = np.ones((3,3),dtype=int)
            boundary_s = nd.binary_dilation(boundary_mask_s==0, k) & boundary_mask_s
            mask_bws[boundary_s] += bw_s - i
        criterion_s[3:-3,3:-3] = ((mask_bws + mask_s*1)-1)[3:-3,3:-3]
        #dh_ref = -mask_bws/28
        #dh_mod = ((-mask_bws**2+1))*dh_ref/15-11.35
        #dh_mod = ((-mask_bws**1.5+1))*dh_ref/15-3.42
        #dh_mod = ((-mask_bws+1))*dh_ref/15-1.23

        mask_bws[mask==0] = 20
        #S_rec = ndimage.convolve(S_rec, np.ones((3,3)))/(3**2)
        for i in range(3):
            mask_bws = ndimage.convolve(mask_bws, np.ones((3,3)))/9
        #S_rec += mask_bws - np.min(mask_bws)
        #S_rec[mask==0] += 20
    
        ### derive initial bed from perfect plasticity ###
        dH = (np.max(S_rec[mask==1]) - np.min(S_rec[mask==1]))/1000
        tau = 0.005+1.598*dH-0.435*dH**2  #Haeberli and Hoelzle
        slope[slope<1.5] = 1.5
        sin_slope = np.sin(np.deg2rad(slope))
        H=((tau)*1e5)/(sin_slope*9.8*910)
        for i in range(5):
            H = ndimage.convolve(H, np.ones((3,3)))/9
        #B_rec = S_rec - H
        #S_rec[mask==0] = S_ref[mask==0]+50
        B_rec[mask==0] = S_rec[mask==0]
        B_rec = np.minimum(S_rec, B_rec)
        B_rec[mask==1] = np.minimum(B_rec, B_rec - 50)[mask==1]

        '''
        ### start with tilted plane as surface ###
        mask *= 0
        mask[15:-15,15:-15] = 1
        X, Y = np.meshgrid(x,y) 
        S_rec[2:-2,2:-2] = ((Y - np.min(Y)) * .25 + 1000)
        S_rec[mask==0] +=50
        B_rec[mask==1] = S_rec[mask==1] - 30
        # do the inversion
        B_rec = np.minimum(S_rec, B_rec)
        B_rec[mask==0] = S_rec[mask==0]-1e-5

        h_old = S_rec - B_rec
        #h_old = h_old * mask

        # run PISM forward for dt years
        #(h_rec, mask_iter, u_rec, v_rec, tauc_rec, h_old) = run_pism(pism, dt, B_rec, h_old, tauc)
        '''
        break
        for p in range(pmax):
            B_rec, S_rec, tauc_rec, misfit, taud = iteration(pism,
                                                       B_rec, S_rec, tauc, mask, dh_ref, np.zeros_like(dem), smb,
                                                       dt=dt,
                                                       beta=beta,
                                                       theta=theta,
                                                       bw=bw,
                                                       update_friction='no',
                                                       res=res,
                                                       A=A,
                                                       max_steps_PISM=max_steps_PISM,
                                                       treat_ocean_boundary='no',
                                                       correct_diffusivity='no',
                                                       contact_zone=np.zeros_like(dem),
                                                       ocean_mask=np.zeros_like(dem))

            B_rec_all.append(np.copy(B_rec))
            S_rec_all.append(np.copy(S_rec))
            misfit_all.append(misfit)
            tauc = taud*.8

        pism.save_results()
        misfit_vs_iter = [np.mean(abs(x[mask == 1])) for x in misfit_all]
    except KeyError:
        continue


#B = np.copy(B_rec)
#for i in range(2):
#    B = ndimage.convolve(B, np.ones((3,3)))/9
