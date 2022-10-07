import numpy as np
from load_input import *
import PISM
from bed_inversion import *
import os
import shutil
from oggm.core import massbalance
import subprocess
from mpi4py import MPI
#from write_output import *

comm = MPI.COMM_WORLD
size = comm.Get_size() # new: gives number of ranks in comm
rank = comm.Get_rank()
#RID = 'RGI60-08.00010'
RID = 'RGI60-08.00213' # Storglaciären
#RID = 'RGI60-08.00006'
#RID = 'RGI60-08.00005'
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
    "-Mz": 5,
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
    "-stress_balance": "blatter",
    "-stress_balance.blatter.flow_law": "isothermal_glen",
    "-stress_balance.blatter.coarsening_factor": 2,
    "-stress_balance­.ice_free_thickness_standard": 1,
    "-yield_stress": "constant",
    #"-pseudo_plastic": "",
    "-pseudo_plastic_q": 0.2,
    "-pseudo_plastic_uthreshold": 3.1556926e7,
    "-geometry.update.use_basal_melt_rate": "no",
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
    "-output.extra.times": .1,
    "-output.extra.file": working_dir + "/extra.nc",
    "-output.extra.vars": "thk,topg,usurf,velsurf_mag,mask,taub_mag,taud_mag,velbar_mag,flux_mag,velbase_mag,tauc",
    "-sea_level.constant.value": -10000,
    "-time_stepping.assume_bed_elevation_changed": "true"
    }
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
smb = get_nc_data(input_file, 'climatic_mass_balance', ':')
tauc = np.ones_like(mask) * 1e20

if use_apparent_mb is True:
    dh_ref *= 0
# set inversion paramters
dt = 50
beta = .5
theta = 0.05
bw = 0
pmax = 1
p_friction = 1000
max_steps_PISM = 25
res = dem.rio.resolution()
A = 1.733e3*np.exp(-13.9e4/(8.3*272)) # 3.9565534675428266e-24

B_init = np.copy(B_rec)
S_ref = np.copy(S_rec)
B_rec_all = []
misfit_all = []

#k = np.ones((3,3))
#S_rec = ndimage.convolve(S_rec, k)/9
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
                                               correct_diffusivity='no',
                                               contact_zone=np.zeros_like(dem),
                                               ocean_mask=np.zeros_like(dem))

    B_rec_all.append(np.copy(B_rec))
    misfit_all.append(misfit)
pism.save_results()
