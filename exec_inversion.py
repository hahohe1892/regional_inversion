import numpy as np
from load_input import *
import PISM
from bed_inversion import *
import os
import shutil

RID = 'RGI60-08.00010'
working_dir = '/home/thomas/regional_inversion/output/' + RID
input_dir = '/home/thomas/regional_inversion/input_data/dhdt/per_glacier/RGI60-08/RGI60-08.0' + RID[10] + '/'+ RID 

if not os.path.exists(working_dir + '/input.nc'):
    shutil.copyfile(input_dir + '/gridded_data.nc', working_dir + '/input.nc')

if not os.path.isdir(working_dir):
    os.mkdir(working_dir)


dem = load_dem_path(RID)
mask = load_mask_path(RID)
dhdt = load_dhdt_path(RID)

dem = crop_border_xarr(dem)
mask = crop_border_xarr(mask)
dhdt = crop_border_xarr(dhdt, pixels = 30)

topg = np.copy(dem)
smb = np.ones_like(dem)

x = dem.x
y = dem.y

create_input_nc(working_dir + '/input.nc', x, y, dem, topg, mask, dhdt, smb, ice_surface_temp=273)

options = {
    "-Mz": 50,
    "-Lz": 1500,
    "-Mx": dem.sizes['x'],
    "-Lx": int(dem.x.max() - dem.x.min()),
    "-My": dem.sizes['y'],
    "-Ly": int(dem.y.max() - dem.y.min()),
    "-Lbz": 1,
    "-allow_extrapolation": True,
    "-surface": "given",
    "-surface.given.file": working_dir + "/gridded_data.nc",
    "-i": working_dir + "/gridded_data.nc",
    "-bootstrap": "",
    "-energy": "none",
    "-sia_flow_law": "isothermal_glen",
    "-ssa_flow_law": "isothermal_glen",
    "-stress_balance": "ssa+sia",
    "-yield_stress": "constant",
    "-pseudo_plastic": "",
    "-pseudo_plastic_q": 0.2,
    "-pseudo_plastic_uthreshold": 3.1556926e7,
    "-geometry.update.use_basal_melt_rate": "no",
    "-stress_balance.ssa.compute_surface_gradient_inward": "no",
    "-flow_law.isothermal_Glen.ice_softness": 3.9565534675428266e-24,
    "-constants.ice.density": 900.,
    "-constants.sea_water.density": 1000.,
    "-bootstrapping.defaults.geothermal_flux": 0.0,
    "-stress_balance.ssa.Glen_exponent": 3.,
    "-constants.standard_gravity": 9.8,
    "-stress_balance.sia.bed_smoother.range": 0.0,
    #"-basal_resistance.beta_lateral_margin": 0,
    "-o": working_dir + "output.nc",
    "-output.timeseries.times": 1,
    "-output.timeseries.filename": working_dir + "/timeseries.nc",
    "-output.extra.times": .5,
    "-output.extra.file": working_dir + "/extra.nc",
    "-output.extra.vars": "diffusivity,thk,topg,usurf,velsurf_mag,tauc,mask,taub_mag,taud_mag,velbar_mag,velbase_mag",
    "-sea_level.constant.value": -10000,
    "-time_stepping.assume_bed_elevation_changed": "true"
    }

pism = create_pism(input_file = working_dir + '/gridded_data.nc', options = options, grid_from_options = True)

dh_ref = read_variable(pism.grid(), "Kronebreen_input.nc", 'dhdt', 'm year-1')
mask = read_variable(pism.grid(), "Kronebreen_input.nc", 'mask', '')
