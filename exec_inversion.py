import numpy as np
from load_input import *
import PISM
from bed_inversion import *
import os
import shutil
from oggm.core import massbalance
import subprocess

#RID = 'RGI60-08.00010'
#RID = 'RGI60-08.00213' # Storglaciären
#RID = 'RGI60-08.00006'
RID = 'RGI60-08.00005'
glaciers_Sweden = get_RIDs_Sweden()
RIDs_Sweden = glaciers_Sweden.RGIId

for RID in RIDs_Sweden:
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
            "-stress_balance": "ssa+sia",
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
            "-output.extra.vars": "diffusivity,thk,topg,usurf,velsurf_mag,mask,taub_mag,taud_mag,velbar_mag,flux_mag,velbase_mag",
            "-sea_level.constant.value": -10000,
            "-time_stepping.assume_bed_elevation_changed": "true"
            }

        #smb = get_nc_data(input_file, 'climatic_mass_balance', ':')
        #dh_ref = get_nc_data(input_file, 'dhdt', ':')
        #input = NC(input_file, 'r+')
        #input['climatic_mass_balance'][:,:] = smb - dh_ref*900
        #input.close()
        '''
        try:
            cmd = ['ncrename', '-h', '-O', '-v', '.climatic_mass_balance,original_climatic_mass_balance', input_file]
            subprocess.run(cmd)
        except:
            print('nothing to be done')
        try:
            cmd = ['ncrename', '-h', '-O', '-v', '.precip,climatic_mass_balance', input_file]
            subprocess.run(cmd)
        except:
            print('nothing to be done')
        '''    
        pism = create_pism(input_file = input_file, options = options, grid_from_options = False)

        dh_ref = read_variable(pism.grid(), input_file, 'dhdt', 'm year-1')
        mask = read_variable(pism.grid(), input_file, 'mask', '')
        mask[mask>=0.5] = 1
        mask[mask<0.5] = 0
        S_rec = read_variable(pism.grid(), input_file, 'usurf', 'm')
        B_rec = read_variable(pism.grid(), input_file, 'topg', 'm')
        tauc = np.ones_like(mask)*1e10


        dh_ref *= 0
        # set inversion paramters
        dt = .1
        beta = 1
        theta = 0.1
        bw = 1
        pmax = 3000
        p_friction = 1000
        max_steps_PISM = 25
        res = dem.rio.resolution()[0]
        A = 1.733e3*np.exp(-13.9e4/(8.3*272)) # 3.9565534675428266e-24

        B_init = np.copy(B_rec)
        S_ref = np.copy(S_rec)
        B_rec_all = []
        misfit_all = []

        #k = np.ones((2,2))
        #S_rec = ndimage.convolve(S_rec, k)/9
        #S_rec = ndimage.median_filter(S_rec, 9)
        #B_rec[B_rec>S_rec] = S_rec[B_rec>S_rec]
        #mask = create_buffer(mask, np.copy(mask), 5)

        # do the inversion
        for p in range(pmax):
            B_rec, S_rec, tauc_rec, misfit = iteration(pism,
                                                       B_rec, S_rec, tauc, mask, dh_ref, np.zeros_like(dem),
                                                       dt=dt,
                                                       beta=beta,
                                                       theta=theta,
                                                       bw=bw,
                                                       update_friction='no',
                                                       res=res,
                                                       A=A,
                                                       max_steps_PISM=max_steps_PISM,
                                                       treat_ocean_boundary='no',
                                                       correct_diffusivity='yes',
                                                       contact_zone=np.zeros_like(dem),
                                                       ocean_mask=np.zeros_like(dem))
            #taud = get_nc_data(working_dir + '/extra.nc', 'taud_mag', -1)
            #tauc[2:-2,2:-2] = 0.9 * taud
            B_rec_all.append(np.copy(B_rec))
            misfit_all.append(misfit)

        pism.save_results()

    except:
        continue
