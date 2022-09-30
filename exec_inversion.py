import numpy as np
from load_input import *
import PISM
from bed_inversion import *
import os
import shutil
from oggm.core import massbalance

#RID = 'RGI60-08.00010'
#RID = 'RGI60-08.00213' # Storglaciären
#RID = 'RGI60-08.00006'
RID = 'RGI60-08.00005'
glaciers_Sweden = get_RIDs_Sweden()
RIDs_Sweden = glaciers_Sweden.RGIId


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
    "-surface.given.file": input_file,
    "-i": input_file,
    "-bootstrap": "",
    "-energy": "none",
    "-sia_flow_law": "isothermal_glen",
    "-ssa_flow_law": "isothermal_glen",
    "-stress_balance": "sia",
    "-yield_stress": "constant",
    "-pseudo_plastic": "",
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
    "-output.extra.vars": "diffusivity,thk,topg,usurf,velsurf_mag,mask,taub_mag,taud_mag,velbar_mag,flux_mag",
    "-sea_level.constant.value": -10000,
    "-time_stepping.assume_bed_elevation_changed": "true"
    }

pism = create_pism(input_file = input_file, options = options, grid_from_options = False)

dh_ref = read_variable(pism.grid(), input_file, 'dhdt', 'm year-1')
mask = read_variable(pism.grid(), input_file, 'mask', '')
mask[mask>=0.5] = 1
mask[mask<0.5] = 0
S_rec = read_variable(pism.grid(), input_file, 'usurf', 'm')
B_rec = read_variable(pism.grid(), input_file, 'topg', 'm')
smb = get_nc_data(input_file, 'climatic_mass_balance', ':')

# set inversion paramters
dt = .1
beta = 1
theta = 0.05
bw = 1
pmax = 1000
p_friction = 1000
max_steps_PISM = 25
res = dem.rio.resolution()
A = 1.733e3*np.exp(-13.9e4/(8.3*272)) # 3.9565534675428266e-24

B_init = np.copy(B_rec)
S_ref = np.copy(S_rec)
B_rec_all = []
misfit_all = []

#k = np.ones((3,3))
#B_rec = ndimage.convolve(B_rec, k)/9
#S_rec[B_rec>S_rec] = B_rec[B_rec>S_rec]
#mask = create_buffer(mask, np.copy(mask), 5)

# do the inversion
for p in range(pmax):
    B_rec, S_rec, tauc_rec, misfit = iteration(pism,
                                               B_rec, S_rec, np.zeros_like(dem), mask, dh_ref, np.zeros_like(dem),
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

    B_rec_all.append(np.copy(B_rec))
    misfit_all.append(misfit)

pism.save_results()
