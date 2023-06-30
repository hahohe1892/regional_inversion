import numpy as np
import rioxarray as rioxr
from oggm import cfg, workflow, utils, GlacierDirectory
from shapely.geometry import box
import geopandas as gpd
from netCDF4 import Dataset as NC
import pandas as pd
from funcs import *
from rioxarray import merge
import xarray as xr
import scipy
from shapely.ops import unary_union
from get_input import *
from sklearn.linear_model import LinearRegression
import pwlf
from pathlib import Path

if os.getcwd() == '/home/thomas/regional_inversion/src':
    glacier_dir = '/home/thomas/regional_inversion/input_data/'
    home_dir = '/home/thomas/'
else:
    glacier_dir = '/mimer/NOBACKUP/groups/snic2022-22-55/regional_inversion/input_data/'
    home_dir = '/mimer/NOBACKUP/groups/snic2022-22-55/'

def load_dhdt_path(RID, period):
    RGI_region = RID.split('-')[1].split('.')[0]
    path = glacier_dir + 'dhdt_{}/per_glacier/RGI60-'.format(period) + RGI_region + '/RGI60-' + RGI_region + '.0' + RID[10] + '/'+RID+ '/dem.tif'
    dhdt = rioxr.open_rasterio(path)

    return dhdt


def load_dem_path(RID):
    RGI_region = RID.split('-')[1].split('.')[0]
    path = glacier_dir + 'DEMs/per_glacier/RGI60-' + RGI_region + '/RGI60-' + RGI_region + '.0'  + RID[10] + '/'+RID + '/dem.tif'
    dem = rioxr.open_rasterio(path)

    return dem


def load_mask_path(RID, mask_new = False):
    RGI_region = RID.split('-')[1].split('.')[0]
    path_tif = glacier_dir + 'DEMs/per_glacier/RGI60-' + RGI_region + '/RGI60-' + RGI_region + '.0'  + RID[10] + '/'+RID + '/dem.tif'
    path = glacier_dir + 'dhdt_2000-2020/per_glacier/RGI60-' + RGI_region + '/RGI60-' + RGI_region + '.0' + RID[10] + '/'+RID + '/gridded_data_new.nc'
    if not os.path.isfile(path):
        path = glacier_dir + 'DEMs/per_glacier/RGI60-' + RGI_region + '/RGI60-' + RGI_region + '.0'  + RID[10] + '/'+RID + '/gridded_data.nc'
    with utils.ncDataset(path) as nc:
        if mask_new is True:
            mask = nc.variables['mask_new'][:]
            tif = rioxr.open_rasterio(path_tif)
            tif.data[0, :, :] = (mask).T
        else:
            mask = nc.variables['glacier_mask'][:]
            tif = rioxr.open_rasterio(path_tif)
            tif.data[0, :, :] = mask

    return tif


def load_consensus_thk(RID):
    RGI_region = RID.split('-')[1].split('.')[0]
    path = glacier_dir + 'consensus_thk/RGI60-' + RGI_region + '/' + RID + '_thickness.tif'
    return rioxr.open_rasterio(path)

    
def load_georeferenced_mask(RID):
    path = glacier_dir + 'outlines/georeferenced_masks/mask_{}_new.tif'.format(RID)
    mask = rioxr.open_rasterio(path)

    return mask


def load_thk_path(RID):
    RGI_region = RID.split('-')[1].split('.')[0]
    path_tif = glacier_dir + 'DEMs/per_glacier/RGI60-' + RGI_region + '/RGI60-' + RGI_region + '.0'  + RID[10] + '/'+RID + '/dem.tif'
    path = glacier_dir + 'dhdt_2000-2020/per_glacier/RGI60-' + RGI_region + '/RGI60-' + RGI_region + '.0' + RID[10] + '/'+RID + '/gridded_data_new.nc'
    if not os.path.isfile(path):
        path = glacier_dir + 'DEMs/per_glacier/RGI60-' + RGI_region + '/RGI60-' + RGI_region + '.0'  + RID[10] + '/'+RID + '/gridded_data.nc'
    
    #path_tif = glacier_dir + 'DEMs/per_glacier/RGI60-08/RGI60-08.0' + RID[10] + '/'+RID+ '/dem.tif'
    #path = glacier_dir + 'DEMs/per_glacier/RGI60-08/RGI60-08.0' + RID[10] + '/'+RID + '/gridded_data.nc'
    with utils.ncDataset(path) as nc:
        thk = nc.variables['consensus_ice_thickness'][:]
    tif = rioxr.open_rasterio(path_tif)
    tif.data[0, :, :] = np.copy(thk)

    return tif


def in_field(RID, field):
    path = home_dir + '/regional_inversion/output/' + RID + '/input.nc'
    data = get_nc_data(path, field, ':')

    return data


def crop_border_xarr(xarr, pixels=150):
    res = xarr.rio.resolution()[0]
    x_min = float(xarr.x.min()) + pixels * res
    x_max = float(xarr.x.max()) - pixels * res
    y_min = float(xarr.y.min()) + pixels * res
    y_max = float(xarr.y.max()) - pixels * res
    geodf = gpd.GeoDataFrame(
        geometry=[
            box(x_min, y_min, x_max, y_max)],
        crs=xarr.rio.crs)
    clipped = xarr.rio.clip(geodf.geometry)

    return clipped


def crop_to_new_mask(xarr, mask, pixels):
    x_grid, y_grid = np.meshgrid(xarr.x, xarr.y)
    res = xarr.rio.resolution()[0]
    x_min = np.min(x_grid[mask.data[0]==1]) - pixels * res
    x_max = np.max(x_grid[mask.data[0]==1]) + pixels * res
    y_min = np.min(y_grid[mask.data[0]==1]) - pixels * res
    y_max = np.max(y_grid[mask.data[0]==1]) + pixels * res
    geodf = gpd.GeoDataFrame(
        geometry=[
            box(x_min, y_min, x_max, y_max)],
        crs=xarr.rio.crs)
    clipped = xarr.rio.clip(geodf.geometry)

    return clipped

        
def crop_to_xarr(xarr_target, xarr_source, from_disk=False):
    geodf = gpd.GeoDataFrame(
        geometry=[
            box(xarr_source.x.min(), xarr_source.y.min(), xarr_source.x.max(), xarr_source.y.max())],
        crs=xarr_source.rio.crs)
    clipped = xarr_target.rio.clip(geodf.geometry, all_touched = True, from_disk=from_disk)

    return clipped


def crop_border_arr(arr, pixels=150):
    return arr[pixels:-pixels, pixels:-pixels]


def load_dhdt_gdir(RID, period):
    cfg.initialize(logging_level='WARNING')
    cfg.PATHS['working_dir'] = '~/regional_inversion/input_data/dhdt_' + period
    gdir = workflow.init_glacier_directories(RID)

    return gdir


def load_dem_gdir(RID):
    cfg.initialize(logging_level='WARNING')
    cfg.PATHS['working_dir'] = '~/regional_inversion/input_data/DEMs'
    gdir = workflow.init_glacier_directories(RID)

    return gdir


def create_nc(vars, WRIT_FILE):
    ncfile = NC(WRIT_FILE, 'w', format='NETCDF3_CLASSIC')
    xdim = ncfile.createDimension('x', int(vars['x'][-1].shape[0]))
    ydim = ncfile.createDimension('y', int(vars['y'][-1].shape[0]))

    for name in list(vars.keys()):
        [_, _, _, fill_value, data] = vars[name]
        if name in ['x', 'y']:
            var = ncfile.createVariable(name, 'f4', (name,))
        else:
            var = ncfile.createVariable(name, 'f4', ('y', 'x'), fill_value=fill_value)
        for each in zip(['units', 'long_name', 'standard_name'], vars[name]):
            if each[1]:
                setattr(var, each[0], each[1])
        var[:] = data

    # finish up
    ncfile.close()
    print("NetCDF file " + WRIT_FILE + " created")


def create_input_nc(file, x, y, dem, topg, mask, dhdt, smb, apparent_mb, ice_surface_temp=273):
    vars = {'y':    ['m',
                     'y-coordinate in Cartesian system',
                     'projection_y_coordinate',
                     None,
                     y],
            'x':    ['m',
                     'x-coordinate in Cartesian system',
                     'projection_x_coordinate',
                     None,
                     x],
            'thk':  ['m',
                     'floating ice shelf thickness',
                     'land_ice_thickness',
                     None,
                     dem - topg],
            'topg': ['m',
                     'bedrock surface elevation',
                     'bedrock_altitude',
                     None,
                     topg],
            'usurf': ['m',
                      'landscape surface',
                      'surf',
                      None,
                      dem],
            'ice_surface_temp': ['K',
                                 'annual mean air temperature at ice surface',
                                 'surface_temperature',
                                 None,
                                 ice_surface_temp],
            'climatic_mass_balance': ['kg m-2 year-1',
                                      'mean annual net ice equivalent accumulation rate',
                                      'land_ice_surface_specific_mass_balance_flux',
                                      None,
                                      smb],
            'apparent_mass_balance': ['kg m-2 year-1',
                                      'mass balance minus dhdt',
                                      'apparent_mb',
                                      None,
                                      apparent_mb],
            'dhdt': ['m/yr',
                     "rate of surface elevation change",
                     'dhdt',
                     None,
                     dhdt],
            'mask': ['',
                     'ice extent mask',
                     'mask',
                     None,
                     mask],
            }
    create_nc(vars, file)


def get_RIDs_Sweden(file=glacier_dir + 'Glaciers_Sweden.txt'):
    return pd.read_table(file, delimiter=';')


def get_gdir_info(RID):
    cfg.initialize(logging_level='WARNING')
    cfg.PATHS['working_dir'] = '~/regional_inversion/input_data/DEMs'
    return GlacierDirectory(glacier_dir + 'DEMs/per_glacier/RGI60-08/RGI60-08.0' + RID[10] + '/'+ RID)


def read_pkl(file):
    return pd.read_pickle(file, 'gzip')


def print_all():
    RIDs = get_RIDs_Sweden()
    RIDs_Sweden = RIDs.RGIId
    for RID in RIDs_Sweden:
        gdir = load_dem_gdir(RID)
        print(gdir[0].form)


def nearby_glacier(RID, RGI, buffer_width):
    '''
    supply a geodataframe with RGI outlines such as obtained from:
    fr = utils.get_rgi_region_file(RGI_region, version='62')
    gdf = gpd.read_file(fr)
    gdf_crs = gdf.to_crs(some_crs)
    '''

    def neighboring_glacier(RID, RGI, buffer_width):
        glacier_buffer = RGI[RGI.RGIId == RID].buffer(buffer_width)
        RGI_minus_one = deepcopy(RGI)
        RGI_minus_one[RGI_minus_one.RGIId == RID] = None
        intersection = RGI_minus_one.intersection(glacier_buffer.iloc[0])
        intersection_indices = np.nonzero(~intersection.is_empty.to_numpy())
        intersection_RIDs = RGI.iloc[intersection_indices].RGIId
        intersection_polys = intersection.iloc[intersection_indices]
        bounds = RGI.iloc[intersection_indices].geometry.bounds
        return intersection_RIDs, intersection_polys, bounds
    area_RIDs, intersection_polys, bounds = neighboring_glacier(RID, RGI, buffer_width)
    area_RIDs = area_RIDs.to_list()
    old_len = 0
    checked_RIDs = []
    new_RIDs = deepcopy(area_RIDs)
    intersection_polys_union = deepcopy(intersection_polys)
    while len(new_RIDs) > 0:
        old_len = len(new_RIDs)
        print(old_len)
        new_area_RIDs = []
        new_area_polys = gpd.geoseries.GeoSeries()
        new_area_bounds = pd.DataFrame()
        for area_RID in new_RIDs:
            neighbor_RIDs, neighbor_polys, neighbor_bounds = neighboring_glacier(area_RID, RGI, 200)
            new_area_RIDs.extend(neighbor_RIDs)
            intersection_polys_union = intersection_polys_union.append(neighbor_polys)
            new_area_polys = new_area_polys.append(neighbor_polys)
            #bounds = pd.concat([bounds, neighbor_bounds])
            new_area_bounds = pd.concat([new_area_bounds, neighbor_bounds])
            checked_RIDs.append(area_RID)
        new_RIDs = [x for x in new_area_RIDs if x not in checked_RIDs]
        new_polys = [np.array(new_area_polys)[np.where(np.array(new_area_RIDs) == x)] for x in new_area_RIDs if x not in checked_RIDs]
        new_bounds = [np.array(new_area_bounds)[np.where(np.array(new_area_RIDs) == x)[0][0]] for x in new_area_RIDs if x not in checked_RIDs]
        new_RIDs = np.unique(new_RIDs).tolist()
        new_polys = [new_polys[np.where(np.array(new_RIDs) == x)[0][0]] for x in np.unique(new_RIDs)]
        new_bounds = [new_bounds[np.where(np.array(new_RIDs) == x)[0][0]] for x in np.unique(new_RIDs)]
        if (len(new_polys) != len(new_RIDs)) or (len(new_polys) != len(new_bounds)):
            break
        area_RIDs.extend(new_RIDs)
        for new_poly in new_polys:
            intersection_polys = intersection_polys.append(gpd.geoseries.GeoSeries(new_poly[0]), ignore_index = True)
        bounds = pd.concat([bounds, pd.DataFrame(new_bounds, columns = ['minx', 'miny', 'maxx', 'maxy'])], ignore_index = True)
    return area_RIDs, intersection_polys, bounds, intersection_polys_union


def obtain_area_mosaic(RID, buffer_width = 200, max_n_glaciers = 1000, discard_list = []):
    '''
    searches for nearby (within 200 m buffer) glaciers based on RGI polygons,
    extracts their RID, bounds and the overlap between the buffer and the nearby glaciers.
    Creates a big raster from bounds using crs from input glacier and reproject_matches
    all nearby glaciers to this raster. Then mosaics all that into one big raster.
    This procedure can leave holes at glacier boundaries due to rasterization.
    Identifies these holes based on overlap.
    '''
    working_dir = home_dir + '/regional_inversion/output/' + RID
    input_file = working_dir + '/input.nc'
    Fill_Value = 9999.0

    RGI_region = RID.split('-')[1].split('.')[0]
    input_igm = rioxr.open_rasterio(input_file)
    input_igm.attrs['_FillValue'] = Fill_Value
    input_igm = input_igm.assign(mask_count = input_igm.mask)
    for var in input_igm.data_vars:
        input_igm.data_vars[var].rio.write_nodata(Fill_Value, inplace=True)
        if var != 'usurf':
            input_igm.data_vars[var].values = input_igm.data_vars[var].where(input_igm.mask == 1)
    input_igm = input_igm.fillna(Fill_Value)
    fr = utils.get_rgi_region_file(RGI_region, version='62')
    gdf = gpd.read_file(fr)
    gdf = gdf.to_crs(input_igm.rio.crs)#gdf.crs.from_epsg('32632'))
    area_RIDs, area_polys, bounds, area_polys_union = nearby_glacier(RID, gdf, buffer_width)
    area_polys = area_polys.reset_index()
    bounds = bounds.reset_index()
    already_done_inds = [np.where(np.array(area_RIDs) == x)[0][0] for x in area_RIDs if x in discard_list]
    already_done_inds = sorted(already_done_inds, reverse = True)
    for index in already_done_inds:
        area_RIDs.pop(index)
        area_polys = area_polys.drop(index)
        bounds = bounds.drop(index)
    area_RIDs = area_RIDs[:max_n_glaciers]
    area_polys = area_polys[:max_n_glaciers]
    bounds = bounds[:max_n_glaciers]
    #min_x, min_y = bounds.min()['minx']-1000, bounds.min()['miny']-1000 # add 1000 m to bounds as extra padding
    #max_x, max_y = bounds.max()['maxx']+1000, bounds.max()['maxy']+1000
    min_x, min_y = gdf[gdf.RGIId.isin(area_RIDs)].geometry.total_bounds[:2] - 1000 # add 1000 m to bounds as extra padding
    max_x, max_y = gdf[gdf.RGIId.isin(area_RIDs)].geometry.total_bounds[2:] + 1000
    input_igm = input_igm.rio.pad_box(min_x, min_y, max_x, max_y)
    mosaic_list = [input_igm]
    i = 2
    for glacier in area_RIDs:
        if glacier == RID:
            continue
        glacier_dir = home_dir + '/regional_inversion/output/' + glacier
        input_file = glacier_dir + '/input.nc'
        input = rioxr.open_rasterio(input_file)
        input.attrs['_FillValue'] = Fill_Value
        input = input.assign(mask_count = input.mask * i)
        i+=1
        for var in input.data_vars:
            input.data_vars[var].rio.write_nodata(Fill_Value, inplace=True)
            if var not in ['usurf', 'usurf_oggm']:
                input.data_vars[var].values = input.data_vars[var].where(input.mask == 1)
        input = input.fillna(Fill_Value)
        # using reproject as below somehow didn't work (creates displacement between orignal glacier and mosaic), not sure why
        #input = input.rio.reproject(input_igm.rio.crs)
        input = input.rio.reproject_match(input_igm)
        mosaic_list.append(input)
    mosaic = merge.merge_datasets(mosaic_list, crs = input_igm.rio.crs)
    mosaic = mosaic.where(mosaic != Fill_Value)
    # to find gaps between glaciers, create new mask mosaic based on polygons
    # and take difference to mosaic mask created above
    area_polys_union = area_polys_union[area_polys_union != None]
    if len(area_polys_union) == 0:
        internal_boundaries = None
    else:
        union = gpd.GeoSeries(unary_union(area_polys_union))
        out_path = os.path.join(working_dir, 'intersection_mask.tif')
        raster_union_out = mask_from_polygon(union[0], area_polys_union, out_path = out_path)
        raster_union = rioxr.open_rasterio(out_path)
        raster_union = raster_union.rio.write_nodata(0)
        raster_union = raster_union.rio.reproject_match(mosaic, all_touched = True)
        internal_boundaries = np.maximum(raster_union - mosaic.mask.fillna(0), 0)
    return mosaic, internal_boundaries, area_RIDs


def get_mb_gradient_Rounce(RID, use_generic_dem_heights=True):
    RID_id = RID.split('-')[1]
    if RID_id[0] == '0':
        RID_id = RID_id[1:]
    mb_xr = xr.open_dataset(os.path.join(glacier_dir, 'mass_balance', RID_id + '_ERA5_MCMC_ba1_50sets_1979_2019_binned.nc'))
    elevation_bins = mb_xr.bin_surface_h_initial.squeeze()
    mass_balance = mb_xr.bin_massbalclim_annual
    mass_balance = mass_balance.where(mass_balance.year >= 2000, drop=True)
    mass_balance_mean = mass_balance.mean(axis = 2).squeeze()
    if use_generic_dem_heights is True:
        mass_balance_mean = mass_balance_mean.assign_coords({'bin':  (elevation_bins.data-np.min(elevation_bins.data))/(np.max(elevation_bins.data) - np.min(elevation_bins.data))})
    else:
        mass_balance_mean = mass_balance_mean.assign_coords({'bin': elevation_bins.data})
    return mass_balance_mean


def get_mb_Rounce(RID, dem, mask, use_generic_dem_heights=True):
    mb_gradient = get_mb_gradient_Rounce(RID, use_generic_dem_heights=use_generic_dem_heights)
    if use_generic_dem_heights is True:
        heights = (dem.data[0][mask.data[0] == 1]-np.min(dem.data[0][mask.data[0] == 1]))/(np.max(dem.data[0][mask.data[0] == 1])-np.min(dem.data[0][mask.data[0] == 1]))
    else:
        heights = dem.data[0][mask.data[0] == 1]

    mb_interp = mb_gradient.interp(bin=heights)#, kwargs={"fill_value": "extrapolate"})
    mb = dem*mask
    mb.data[0][mask.data[0] == 1] = mb_interp
    return mb, heights


def resolve_mb_dhdt(RID, dhdt, dem, mask, use_generic_dem_heights=True, bin_heights=False, modify_dhdt_or_smb='smb', cutoff_elevation=None):

    mb, heights = get_mb_Rounce(RID, dem, mask, use_generic_dem_heights=use_generic_dem_heights)
    # need to standardize elevation range regardless of previous
    # standardization choice to have a change to fit to Huss
    if use_generic_dem_heights is False:
        heights = np.array(standardize(heights))

    glacier_area = len(mask.data[0] == 1) * 100 * 100

    glacier_elevation_range = np.max(dem.data[0][mask.data[0] == 1])-np.min(dem.data[0][mask.data[0] == 1])

    if glacier_elevation_range > 300:
        #n_bins = int(np.maximum(len(np.nonzero(mask.data[0] == 1)[0]) / 100, 10))
        n_bins = int(glacier_elevation_range / 30)
        bins = np.linspace(np.min(heights), np.max(heights), n_bins)
        bin_inds = np.digitize(heights, bins)
        heights_binned = np.array([heights[bin_inds == i].mean() for i in range(1, len(bins))])
        dhdt_binned = np.array([dhdt.data[0][mask.data[0] == 1][bin_inds == i].mean() for i in range(1, len(bins))])
        if bin_heights is True:
            fitting_input = [heights_binned,  dhdt_binned]
        else:
            fitting_input = [heights, dhdt.data[0][mask.data[0] == 1]]

        def least_squares_dhdt(x, a, b, c, gamma):
            y = (x + a)**gamma + b*(x + a) + c
            return y#.astype('float64')

        def find_cutoff_percent():#heights, dhdt_masked, n_bins):
            bins = np.linspace(np.min(heights), np.max(heights), n_bins * 3)
            bin_inds = np.digitize(heights, bins)
            heights_binned = np.array([heights[bin_inds == i].mean() for i in range(1, len(bins))])
            dhdt_binned = np.array([dhdt.data[0][mask.data[0] == 1][bin_inds == i].mean() for i in range(1, len(bins))])
            min_ind = np.where(dhdt_binned == np.nanmin(dhdt_binned))[0]
            cutoff_percent = heights_binned[min_ind]
            return cutoff_percent

        # remove the lowest x m as dh/dt may be to high because of too-large mask
        if cutoff_elevation is None:
            cutoff_percent = find_cutoff_percent()[0]
            print('cutoff elevation: {}'.format(cutoff_percent*glacier_elevation_range))
        else:
            cutoff_percent = cutoff_elevation/glacier_elevation_range

        # fitting input needs to have at least 4 data points to fit 4 params;
        # if not, assign mean to dhdt
        if (1-cutoff_percent) * glacier_elevation_range < 150:
            dhdt_fit_field = dem*mask
            dhdt_fit_field.data[0][mask.data[0] == 1] = np.mean(dhdt.data[0][mask.data[0] == 1])
        else:
            # values for the params as given by Huss; currently not used
            if glacier_area < 5 * 1e6:
                p0 = np.array([0., 0.6, 0.09, 2.0]).astype('float64')
                bounds = (np.array([-.5, -5, -.5, -3], dtype = 'float64'), np.array([.5, 5, .5, 6], dtype = 'float64'))
            elif glacier_area > 20 * 1e6:
                p0 = [-0.02, 0.12, 0, 6.0]
                bounds = []
            else:
                p0 = [-0.05, 0.19, 0.01, 4.0]
                bounds = []

            # try to fit. Since params can't always be found, try the following order:
            # 1) with cutoff height
            # 2) with cutoff height and not the choice of bin_heights
            # 3) without cutoff height
            # 4) without cutoff height and not the choice of bin_heights

            def fit_Huss(fitting_input, cutoff, cutoff_percent):
                # change dhdt to follow Huss
                fit_in = [[], []]
                fit_in[1] = -(fitting_input[1]-1)
                fit_in[0] = -(fitting_input[0]-1)
                fit_in_min = np.min(fit_in[1])
                fit_in_max = np.max(fit_in[1])
                fit_in[0] = normalize(fit_in[0])
                fit_in[1] = normalize(fit_in[1])
                if cutoff is True:
                    fit_in_cutoff = [[], []]
                    fit_in_cutoff[0] = fit_in[0][fit_in[0] <= (1-cutoff_percent)]
                    fit_in_cutoff[1] = fit_in[1][fit_in[0] <= (1-cutoff_percent)]
                    params = scipy.optimize.curve_fit(least_squares_dhdt, xdata = fit_in_cutoff[0].astype('float64'), ydata = fit_in_cutoff[1].astype('float64'))[0]#, p0=p0.astype('float64'), bounds=bounds)[0]#, p0 = p0, bounds = bounds)[0]
                else:
                    params = scipy.optimize.curve_fit(least_squares_dhdt, xdata = fit_in[0].astype('float64'), ydata = fit_in[1].astype('float64'))[0]#, p0=p0, bounds=bounds)[0]
                return params, fit_in_max, fit_in_min

            cutoff = True
            try:
                print('trying to fit with preferred options...')
                params, f_max, f_min = fit_Huss(fitting_input, cutoff, cutoff_percent)
            except(RuntimeError):
                print('option 1 for fitting failed, trying option 2...')
                try:
                    if bin_heights is True:
                        fitting_input = [heights, dhdt.data[0][mask.data[0] == 1]]
                    else:
                        fitting_input = [heights_binned, dhdt_binned]
                    params, f_max, f_min = fit_Huss(fitting_input, cutoff, cutoff_percent)
                except(RuntimeError):
                    print('option 2 for fitting failed, trying option 3...')
                    try:
                        if bin_heights is True:
                            fitting_input = [heights_binned, dhdt_binned]
                        else:
                            fitting_input = [heights, dhdt.data[0][mask.data[0] == 1]]
                        params, f_max, f_min = fit_Huss(fitting_input, not cutoff, cutoff_percent)
                    except(RuntimeError):
                        print('option 3 for fitting failed, trying option 4...')
                        if bin_heights is True:
                            fitting_input = [heights, dhdt.data[0][mask.data[0] == 1]]
                        else:
                            fitting_input = [heights_binned, dhdt_binned]
                        params, f_max, f_min = fit_Huss(fitting_input, not cutoff, cutoff_percent)
            print('...fitting successfull')
            dhdt_fit = least_squares_dhdt(-(heights-1), params[0], params[1], params[2], params[3])
            # scale dhdt back to original range
            dhdt_fit = (dhdt_fit * (f_max - f_min) + f_min) * -1 + 1
            dhdt_fit[np.isnan(dhdt_fit)] = np.nanmin(dhdt_fit)
            dhdt_fit_field = dem*mask
            dhdt_fit_field.data[0][mask.data[0] == 1] = dhdt_fit
    else:
        dhdt_fit_field = dem*mask
        dhdt_fit_field.data[0][mask.data[0] == 1] = np.mean(dhdt.data[0][mask.data[0] == 1])

    # modify either smb or dhdt so that they balance
    k = 0
    learning_rate = 0.2
    mb_misfit = np.mean(mb.data[0][mask.data[0]==1]) - np.mean(dhdt_fit_field.data[0][mask.data[0]==1])
    print('There is a mismatch of {} m w eq. between mass balance and dhdt which is being resolved now...'.format(round(mb_misfit/1, 2)))
    while abs(mb_misfit) > 0.01:
        mb_misfit = np.mean(mb.data[0][mask.data[0]==1]) - np.mean(dhdt_fit_field.data[0][mask.data[0]==1]) - k 
        k += mb_misfit * learning_rate
    print('...a bias of {} had to be applied to make mass balance and dhdt match'.format(round(k, 2)))
    #if modify_dhdt_or_smb == 'smb':
    #    mb -= k * mask
    #else:
    #    dhdt_fit_field += k

    return mb, dhdt_fit_field



def resolve_mb_dhdt_smoothing(RID, dhdt, dem, mask, use_generic_dem_heights=True, modify_dhdt_or_smb='smb'):

    mb, heights = get_mb_Rounce(RID, dem, mask, use_generic_dem_heights=use_generic_dem_heights)
    dhdt.data[0][mask.data[0] == 0] = np.nan
    dhdt.data[0] = gauss_filter(dhdt.data[0], 1, 3)
    dhdt.data[0][mask.data[0] == 0] = 0
    mb_misfit = np.mean(mb.data[0][mask.data[0]==1]) - np.mean(dhdt.data[0][mask.data[0]==1])
    print('There was a mismatch of {} m w eq. between mass balance and dhdt which is resolved now...'.format(round(mb_misfit/1, 2)))
    if modify_dhdt_or_smb == 'smb':
        mb -= mb_misfit
    else:
        dhdt += mb_misfit

    return mb, dhdt


def resolve_mb_dhdt_piecewise_linear(RID, dhdt, dem, mask, use_generic_dem_heights=True, modify_dhdt_or_smb='smb'):
    mb, heights = get_mb_Rounce(RID, dem, mask, use_generic_dem_heights=use_generic_dem_heights)
    if np.isnan(heights[0]).all(): # for very small glaciers (1 pixel), there sometimes is no mass balance
        apparent_mb_fit = np.zeros_like(dem.data[0])[mask.data[0] == 1]
    else:
        mb_misfit = np.mean(mb.data[0][mask.data[0]==1]) - np.mean(dhdt.data[0][mask.data[0]==1])
        print('There was a mismatch of {} m w eq. between mass balance and dhdt which is resolved now...'.format(round(mb_misfit/1, 2)))
        if modify_dhdt_or_smb == 'smb':
            mb -= mb_misfit
        else:
            dhdt += mb_misfit

        apparent_mb = (mb - dhdt)*mask
        my_pwlf = pwlf.PiecewiseLinFit(dem.data[0][mask.data[0] == 1], apparent_mb.data[0][mask.data[0] == 1])
        res = my_pwlf.fit(2)
        apparent_mb_fit = np.zeros_like(apparent_mb.data[0])
        apparent_mb_fit[mask.data[0] == 1] = my_pwlf.predict(dem.data[0][mask.data[0] == 1])

        # The fit above produces a knick-point wherever it matches the data best,
        # but we want the knick point at the ELA.
        # Therefore, take the previous fit, determine where the ELA is according to it,
        # and then calculate a new fit where the knick-point is forced to be at the ELA.
        # Do this iteratively to ensure convergence.

        for i in range(5):
            ELA_ind = np.where(abs(apparent_mb_fit[mask.data[0] == 1])==np.min(abs(apparent_mb_fit[mask.data[0] == 1])))
            ELA = dem.data[0][mask.data[0] == 1][ELA_ind[0][0]]
            res = my_pwlf.fit_with_breaks(np.array([min(dem.data[0][mask.data[0] == 1]), ELA, max(dem.data[0][mask.data[0] == 1])]))
            apparent_mb_fit[mask.data[0] == 1] = my_pwlf.predict(dem.data[0][mask.data[0] == 1])
        if (my_pwlf.slopes < 0).any():
            X = dem.data[0][mask.data[0] == 1]
            Y = apparent_mb.data[0][mask.data[0] == 1]
            apparent_mb_fit = np.zeros_like(apparent_mb.data[0])
            apparent_mb_fit[mask.data[0] == 1] = LinearRegression().fit(X.reshape(-1,1), Y.reshape(-1,1)).predict(X.reshape(-1,1)).reshape(1,-1)[0]

    return mb, dhdt, apparent_mb_fit
        
def write_path_to_mosaic(RID, area_RIDs):
    for area_RID in area_RIDs:
        working_dir = os.path.join(home_dir + '/regional_inversion/output/', area_RID)
        with open(os.path.join(working_dir, 'mosaic_reference.txt'), 'w') as fp:
            fp.write('This glacier is part of a larger glaciated area,\nand the results therefore can be found under the following RGI ID:\n{}'.format(RID))


def calc_h_perfect_plasticity(tau_b, usurf, mask, f = 1):
    if tau_b is None:
        dH = (usurf.where(mask == 1).max() - usurf.where(mask == 1).min())/1000
        if dH < 1.6:
            tau_b = 1.5e5
        else:
            tau_b = (0.005 + 1.598 * dH - 0.435 * dH**2) * 1e5
        tau_b = np.maximum(tau_b, 0.005 * 1e6)
    slope_x = usurf.differentiate('x')
    slope_y = usurf.differentiate('y')
    slope = np.sqrt(slope_x**2 + slope_y**2) * mask
    sin_slope = np.sin(np.arctan(slope))
    sin_slope = np.maximum(sin_slope, 0.04)
    sin_slope.data[0] = gauss_filter(sin_slope.data[0], 1, 3)
    rho = 910
    g = 9.8
    h = tau_b / (rho * g * sin_slope * f)
    return h
            
'''
cutoffs = [0,50,100,200,300, None]
colormap = plt.cm.viridis
colors = [colormap(i) for i in np.linspace(0, 1,len(cutoffs))]
plt.scatter(dem.data[0][mask.data[0] == 1], dhdt.data[0][mask.data[0] == 1])
for q,i in enumerate(cutoffs):
    mb, dhdt_fit_field2 = resolve_mb_dhdt(RID, dhdt, dem, mask, use_generic_dem_heights=True, bin_heights = False, cutoff_elevation = i)
    plt.scatter(dem.data[0][mask.data[0] == 1], dhdt_fit_field2.data[0][mask.data[0] == 1], color = colors[q])

plt.show()
'''
