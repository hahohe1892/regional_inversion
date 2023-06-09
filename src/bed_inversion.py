#!/usr/bin/env python3
import PISM
import numpy as np
from scipy import ndimage
from netCDF4 import Dataset as NC
from funcs import *

ctx = PISM.Context()
secpera = 365*24*60*60

def read_variable(grid, filename, variable_name, units=None):
    """Read a 2D array `variable_name` from a NetCDF file `filename` and
    return its 'local' part (the part owned by a particular MPI rank),
    including ghosts.

    Uses the last time record found in `filename` and converts into
    units specified by `units`.

    """
    # stencil width has to match the one used by PISM
    stencil_width = 2

    # allocate storage
    array = PISM.IceModelVec2S(grid, variable_name,
                               PISM.WITH_GHOSTS, stencil_width)

    if units is not None:
        # array.regrid() will automatically convert to these units
        array.metadata().set_string("units", units)

    # read from `filename` using bilinear interpolation (if necessary)
    #
    # uses the last time record found in the file
    #
    # PISM.CRITICAL means "stop with an error message if not found"
    #
    # the default value is not used
    array.regrid(filename, PISM.CRITICAL, default_value=0)

    # return a copy to make sure that the caller owns it (array
    # allocated here will go out of scope and may get de-allocated)
    return np.array(array.local_part(), copy=True)

def create_pism(input_file, options, grid_from_options = False):
    """Allocate and initialize PISM, using bed topography 'topg' in the
       `input_file` to set the horizontal grid.

    `options` is a dictionary containing command-line options.
    """

    # set command line options
    opt = PISM.PETSc.Options()
    for key, value in options.items():
        opt.setValue(key, value)

    PISM.set_config_from_options(ctx.unit_system, ctx.config)

    if grid_from_options:
        grid_parameters = PISM.GridParameters()
        grid_parameters.horizontal_size_from_options()
        grid_parameters.horizontal_extent_from_options(ctx.unit_system)
    else:
        # get horizontal grid parameters from the input file
        grid_parameters = PISM.GridParameters(ctx.ctx, input_file, "topg", PISM.CELL_CENTER)

    # get vertical grid parameters from command line options
    grid_parameters.vertical_grid_from_options(ctx.config)
    grid_parameters.ownership_ranges_from_options(ctx.ctx.size())

    # allocate the computational grid
    grid = PISM.IceGrid(ctx.ctx, grid_parameters)

    model = PISM.IceModel(grid, ctx.ctx)
    model.init()

    return model

def run_pism(pism, dt_years, bed_elevation, ice_thickness, yield_stress):
    """Run PISM for `dt_years` years using provided `bed_elevation` and `ice_thickness`.

    Here `bed_elevation` and `ice_thickness` are NumPy arrays
    containing the local (sub-domain) parts of these fields,
    *including* the two grid cell wide border or ghosts.

    returns a tuple containing updated (ice_thickness, mask, u_surface, v_surface)

    """

    # Copy ice thickness and bed elevation into PISM's storage
    #
    # Note that this abuses "const" access to various parts of PISM.

    # bed deformation models are in charge of bed elevation
    pism.bed_deformation_model().bed_elevation().local_part()[:] = np.copy(bed_elevation)
    pism.bed_deformation_model().bed_elevation().update_ghosts()

    # we also need to update this copy of bed elevation (for consistency)
    pism.geometry().bed_elevation.local_part()[:] = np.copy(bed_elevation)
    pism.geometry().bed_elevation.update_ghosts()
    
    # pism.geometry() stores ice thickness
    pism.geometry().ice_thickness.local_part()[:] = np.copy(ice_thickness)
    pism.geometry().ice_thickness.update_ghosts()
    h_old = np.copy(pism.geometry().ice_thickness.local_part()[:])
    
    H_min = ctx.config.get_number("geometry.ice_free_thickness_standard")
    pism.geometry().ensure_consistency(H_min)

    # set basal yield stress
    if hasattr(pism.basal_yield_stress_model(), 'basal_material_yield_stress'):
        pism.basal_yield_stress_model().basal_material_yield_stress().local_part()[:] = yield_stress
        pism.basal_yield_stress_model().basal_material_yield_stress().update_ghosts()

    dt = PISM.util.convert(dt_years, "year", "second")
    pism.run_to(ctx.time.current() + dt)

    # get new ice thickness from PISM:
    H    = pism.geometry().ice_thickness.local_part()
    H    = np.array(H, copy=True)

    # get new cell type mask from PISM:
    mask = pism.geometry().cell_type.local_part()
    mask = np.array(mask, copy=True)

    # get basal yield stress from PISM:
    if hasattr(pism.basal_yield_stress_model(), 'basal_material_yield_stress'):
        tauc = pism.basal_yield_stress_model().basal_material_yield_stress().local_part()
        tauc = np.array(tauc, copy=True)
    else:
        tauc = np.ones_like(mask)

    # compute surface velocity:
    stress_balance_model = pism.stress_balance()
    diags                = stress_balance_model.diagnostics()
    velsurf              = diags["velsurf"].compute().local_part()
    taud = diags["taud_mag"].compute().local_part()

    # extract U and V components and paste them into arrays with
    # "ghosts" to ensure that all arrays have the same shape:

    # stencil width:
    w = 2
    u_surface = np.zeros_like(H)
    u_surface[w:-w, w:-w] = velsurf[:, :, 0]

    v_surface = np.zeros_like(H)
    v_surface[w:-w, w:-w] = velsurf[:, :, 1]

    taud = np.zeros_like(H)
    taud[w:-w, w:-w] = diags["taud_mag"].compute().local_part()
    
    return (H, mask, u_surface, v_surface, tauc, h_old, taud)


def iteration(model, bed, usurf, yield_stress, mask, dh_ref, vel_ref, smb, dt, beta, theta, bw, update_friction, res, A, correct_diffusivity ='no', max_steps_PISM = 50, treat_ocean_boundary='no', contact_zone = None, ocean_mask = None):
        
    h_old = usurf - bed
    #h_old = h_old * mask

    # run PISM forward for dt years
    (h_rec, mask_iter, u_rec, v_rec, tauc_rec, h_old, taud) = run_pism(model, dt, bed, h_old, yield_stress)

    # set velocities to 0 outside mask
    u_rec = u_rec * mask * secpera
    v_rec = v_rec * mask * secpera

    # calculate modelled dh/dt
    dh_rec = (h_rec - h_old)/dt
    
    # calculate dh/dt misfit and shift it
    misfit = dh_rec - dh_ref
    #try based on surface misfit
    #misfit = (bed + h_rec) - usurf

    #misfit = np.maximum(np.minimum(misfit, 5), -5)
    #misfit = shift(misfit, u_rec, v_rec, mask, .3)

    # apply bed and surface corrections
    B_rec = bed - beta * misfit
    #S_rec = usurf - h_old + h_rec
    S_rec = usurf + beta * theta * misfit

    #B_rec[mask == 0] = bed[mask == 0] + beta * (misfit[mask == 0] - smb[mask == 0] / 900)
    #S_rec[mask == 0] = bed[mask == 0] + beta * (misfit[mask == 0] - smb[mask == 0] / 900)
    
    # interpolate around ice margin
    if bw > 0:
        # consider that in some cases, mass is only added to glacier through misfit
        #mask_iter = (S_rec*mask) > (B_rec * mask)
        mask_iter = h_rec >= 10
        mask_bw = (~mask_iter)*1
        criterion = np.zeros_like(mask_iter)
        for i in range(bw):
            boundary_mask = mask_bw==0
            k = np.ones((3,3),dtype=int)
            boundary = nd.binary_dilation(boundary_mask==0, k) & boundary_mask
            mask_bw[boundary] = 1
        criterion[3:-3,3:-3] = ((mask_bw + mask_iter*1)-1)[3:-3,3:-3]
        criterion[criterion!=1] = 0

        S_rec[criterion==1] = usurf[criterion==1]
        h_inpaint = S_rec - B_rec
        h_inpaint[criterion==1] = np.nan
        h_inpaint = inpaint_nans(h_inpaint)
        B_rec = S_rec - h_inpaint
        #S_rec[criterion == 1] = usurf[criterion == 1] + beta * theta*5 * misfit[criterion == 1]
        #S_rec[criterion==1] = np.nan
        #S_rec = inpaint_nans(S_rec)
        #B_rec[criterion==1]=S_rec[criterion==1]
        #S_rec[criterion==1]=usurf[criterion==1]

    # correct bed in locations where a large diffusivity would cause pism to take many internal time steps
    if correct_diffusivity == 'yes':
        if treat_ocean_boundary == 'yes':
            B_rec[contact_zone!=1] = correct_high_diffusivity(S_rec, B_rec, dt, max_steps_PISM, res, A, return_mask = False)[contact_zone!=1]
        else:
            B_rec = correct_high_diffusivity(S_rec, B_rec, dt, max_steps_PISM, res, A, return_mask = False)

    if treat_ocean_boundary == 'yes':
        B_rec[contact_zone==1] = shift(B_rec, u_rec, v_rec, mask,  1)[contact_zone==1]
        B_rec[ocean_mask==1] = shift(B_rec, u_rec, v_rec,  mask, 2)[ocean_mask==1]

    # mask out 
    B_rec[mask==0] = bed[mask==0]
    S_rec[mask==0] = usurf[mask==0]
    B_rec = np.minimum(B_rec, S_rec)

    if update_friction == 'yes':   
        vel_rec = np.sqrt(u_rec**2+v_rec**2)
        vel_rec[vel_rec>1000000]=np.nan
        vel_mismatch = np.maximum(np.minimum((vel_rec - vel_ref)/vel_ref, 0.5), -0.5)
        vel_mismatch[mask==0]=np.nan
        vel_mismatch =  gauss_filter(vel_mismatch, 2,4)
        vel_mismatch[np.isnan(vel_mismatch)]=0
        tauc_rec += vel_mismatch * tauc_rec

    return B_rec, S_rec, tauc_rec, misfit, taud

def iteration_friction_first(model, bed, usurf, yield_stress, mask, dh_ref, vel_ref, dt, beta, bw, update_friction, res, A, correct_diffusivity ='no', max_steps_PISM = 50, treat_ocean_boundary='no', contact_zone = None, ocean_mask = None):
        
    h_old = usurf - bed
    
    # run PISM forward for dt years
    (h_rec, mask_iter, u_rec, v_rec, tauc_rec) = run_pism(model, dt, bed, h_old, yield_stress)
    B_rec = np.copy(bed)
    S_rec = np.copy(usurf)
    misfit = np.zeros_like(B_rec)
    
    vel_rec = np.sqrt(u_rec**2+v_rec**2)*secpera
    vel_mismatch = np.maximum(np.minimum((vel_rec - vel_ref)/vel_ref, 0.5), -0.5)
    vel_mismatch[mask==0]=np.nan
    vel_mismatch =  gauss_filter(vel_mismatch, .6,2)
    vel_mismatch[np.isnan(vel_mismatch)]=0
    tauc_rec += vel_mismatch * tauc_rec

    if update_friction == 'yes':   
        # calculate modelled dh/dt
        dh_rec = (h_rec - h_old)/dt

        # calculate dh/dt misfit and shift it
        misfit = dh_rec - dh_ref
        misfit = shift(misfit, u_rec, v_rec, mask, .3)
        # apply bed and surface corrections
        B_rec = bed - beta * misfit
        S_rec = usurf + beta * 0.025 * misfit
        # interpolate around ice margin
        if bw > 0:
            k = np.ones((bw, bw))
            buffer = ndimage.convolve(mask_iter, k)/(bw)**2 
            criterion = np.logical_and(np.logical_and(buffer > 0, buffer != 2), mask == 1)
            B_rec[criterion]=0
            S_rec[criterion]=usurf[criterion]

    # mask out
    B_rec = np.minimum(B_rec, S_rec)
    B_rec[mask==0] = bed[mask==0]
    S_rec[mask==0] = usurf[mask==0]
            
    return B_rec, S_rec, tauc_rec, misfit

if __name__ == "__main__":
    input_file = PISM.OptionString("-i", "input file name")

    if not input_file.is_set():
        raise RuntimeError("-i is required")

    options = {
        "-Mz": 21,
        "-Lz": 4000,
        "-z_spacing": "equal",
        "-surface" : "given",
        "-surface_given_file": input_file.value(),
        "-i": input_file.value(),
        "-bootstrap": "",
        "-energy": "none",
        "-sia_flow_law": "isothermal_glen",
        "-ssa_flow_law": "isothermal_glen",
        "-stress_balance": "ssa+sia",
        "-yield_stress": "constant"
    }

    pism = create_pism("input.nc", options)

    H   = np.array(pism.geometry().ice_thickness.local_part(), copy=True)
    bed = np.array(pism.geometry().bed_elevation.local_part(), copy=True)

    tauc = np.array(pism.basal_yield_stress_model().basal_material_yield_stress().local_part(),
                    copy=True)

    tauc[10:-10,:] *= 0.95

    bed[10:-10,:] -= 1

    dt_years = 1
    (H, mask, u_surface, v_surface, tauc) = run_pism(pism, dt_years, bed, H, tauc)

    u_surface[mask == 0] = np.nan
    u_surface[mask == 4] = np.nan
    v_surface[mask == 0] = np.nan
    v_surface[mask == 4] = np.nan

    import pylab as plt

    for each in (H, mask, u_surface, v_surface, tauc):
        plt.figure()
        plt.imshow(each)
        plt.colorbar()
    plt.show()


