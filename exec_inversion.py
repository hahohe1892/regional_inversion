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
for RID in sample_glaciers[-2:]:
    try:
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
            "-stress_balance.ssa.compute_surface_gradient_inward": "no",
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
            "-output.extra.vars": "diffusivity,thk,topg,usurf,velsurf_mag,mask,taub_mag,taud_mag,velbar_mag,flux_mag,velbase_mag,climatic_mass_balance,rank",
            "-sea_level.constant.value": -10000,
            "-time_stepping.assume_bed_elevation_changed": "true"
            }

        # switch between using apparent mass balance (the default, use_apparent_mb = True)
        # and actual mass balance (use_apparent_mb = False)
        # use carfully to not mess up what is written in input file

        use_apparent_mb = True
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

        pism = create_pism(input_file = input_file, options = options, grid_from_options = False)

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
        smb = read_variable(pism.grid(), input_file, 'apparent_mass_balance', 'kg m-2 year-1')
        tauc = np.ones_like(mask)*1e10

        if use_apparent_mb is True:
            dh_ref *= 0
        # set inversion paramters
        dt = .1
        beta = .25
        theta = 0.4
        bw = 0
        pmax = 3000
        p_friction = 1000
        max_steps_PISM = 25
        res = dem.rio.resolution()[0]
        A = 1.733e3*np.exp(-13.9e4/(8.3*272)) # 3.9565534675428266e-24

        B_init = np.copy(B_rec)
        S_ref = np.copy(S_rec)
        B_rec_all = []
        S_rec_all = []
        misfit_all = []

        mask[smb<=0] = 1
        
        #smb[np.logical_and(mask == 0, smb<0)] = 0
        #S_rec[mask == 0] += 500
        #B_rec[mask==0] = S_rec[mask==0]
        #S = np.copy(S_rec)
        #S_rec[2:-2,2:-2] = nc_out(RID, 'usurf', file='output_v0.2.nc')
        #topg = np.copy(B_rec)
        #B_rec[2:-2,2:-2] = nc_out(RID, 'topg', file='output_v0.2.nc')
        #S_rec[mask==1] = S[mask==1] - np.mean(S[mask == 1] - S_ref[mask==1])
        #B_rec = topg + (S - S_ref)

        '''
        mask_iter = mask == 1
        mask_bw = (~mask_iter)*1
        criterion = np.zeros_like(mask_iter)
        for i in range(bw):
            boundary_mask = mask_bw==0
            k = np.ones((3,3),dtype=int)
            boundary = nd.binary_dilation(boundary_mask==0, k) & boundary_mask
            mask_bw[boundary] = 1
        criterion[3:-3,3:-3] = ((mask_bw + mask_iter*1)-1)[3:-3,3:-3]
        criterion[criterion!=1] = 0

        S_rec[criterion==1] = np.nan
        S_rec = inpaint_nans(S_rec)
        B_rec = np.minimum(B_rec, S_rec)
        '''
        # do the inversion
        for p in range(pmax):
            B_rec, S_rec, tauc_rec, misfit = iteration(pism,
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

        pism.save_results()
        break
    except ValueError:
        continue
