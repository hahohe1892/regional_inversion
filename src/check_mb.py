import numpy as np
import rioxarray as rioxr
import matplotlib.pyplot as plt
import rasterio
from scipy.optimize import curve_fit
from funcs import *
import pwlf
from statsmodels.stats import weightstats
import glob
from load_input import *
from matplotlib.backends.backend_pdf import PdfPages

mb_files = glob.glob('/mnt/c/Users/thofr531/Documents/Global/Scandinavia/mass_balance_Norway/*massbalanceData*')

mb_NVE = pd.read_csv(mb_files[0], sep = ';', header = 0, encoding= 'unicode_escape')
for file in mb_files[1:]:
    mb_NVE = pd.concat([mb_NVE, pd.read_csv(file, sep = ';', header = 0, encoding= 'unicode_escape')])


ID_links = pd.read_csv('/mnt/c/Users/thofr531/Documents/Global/WGMS/WGMS-FoG-2021-05-AA-GLACIER-ID-LUT.csv', header = 0, encoding= 'unicode_escape')
mb_bins = pd.read_csv('/mnt/c/Users/thofr531/Documents/Global/WGMS/WGMS-FoG-2021-05-EE-MASS-BALANCE.csv', header = 0, encoding= 'unicode_escape')
mb_bins = mb_bins[np.logical_or(mb_bins.POLITICAL_UNIT == 'NO', mb_bins.POLITICAL_UNIT == 'SE')]
mb_bins['RGI_ID'] = np.nan
for ID in mb_bins.WGMS_ID.unique():
    RGI_ID = ID_links.RGI_ID[ID_links.WGMS_ID == ID].iloc[0]
    mb_bins.RGI_ID = mb_bins.RGI_ID.where(mb_bins.WGMS_ID != ID, RGI_ID)


pp = PdfPages('/home/thomas/regional_inversion/mb_check_figures.pdf')
for RID in mb_bins.RGI_ID.unique():
    if isinstance(RID, float):
        continue
    if RID[3] != '6':
        RID = RID.split(RID[3])[0] + '6' + RID.split(RID[3])[1]
    mb_glacier = mb_bins.loc[mb_bins.RGI_ID == RID]
    mb_glacier = mb_glacier.loc[mb_glacier.YEAR >= 2000]
    mb_bin_mean = (mb_glacier.LOWER_BOUND + mb_glacier.UPPER_BOUND)/2

    fig, ax = plt.subplots(1,2)
    ax[0].scatter(mb_bin_mean[mb_bin_mean!=9999.0], mb_glacier.ANNUAL_BALANCE[mb_bin_mean!=9999.0] / 910, c = mb_glacier.YEAR[mb_bin_mean!=9999.0])
    mb_Rounce = get_mb_gradient_Rounce(RID, use_generic_dem_heights = False)
    mb_Rounce.plot(ax = ax[0])


    period = '2000-2020'
    use_generic_dem_heights = True
    modify_dhdt_or_smb = 'smb'
    Fill_Value = 9999.0
    glaciers_Sweden = get_RIDs_Sweden()
    RIDs_Sweden = glaciers_Sweden.RGIId
    input_dir = '/home/thomas/regional_inversion/input_data/'
    output_dir = '/home/thomas/regional_inversion/output/' + RID
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    input_file = output_dir + '/input.nc'
    RGI_region = RID.split('-')[1].split('.')[0]
    per_glacier_dir = 'per_glacier/RGI60-' + RGI_region + '/RGI60-' + RGI_region + '.0' + RID[10] + '/'+ RID
    dem_Norway = rioxr.open_rasterio(os.path.join(input_dir, 'DEM_Norway', per_glacier_dir, 'dem.tif'))
    dem_Sweden = rioxr.open_rasterio(os.path.join(input_dir, 'DEM_Sweden', per_glacier_dir, 'dem.tif'))
    # choose the DEM which contains no nans
    if not (dem_Sweden.data == dem_Sweden._FillValue).any():
            dem = deepcopy(dem_Sweden)
            print('Choosing Swedish DEM')
    elif not (dem_Norway.data == dem_Norway._FillValue).any():
            print('Choosing Norwegian DEM')
            dem = deepcopy(dem_Norway)
    elif (dem_Norway.data == dem_Norway._FillValue).any() and (dem_Sweden.data == dem_Sweden._FillValue).any():
        raise ValueError('No suitable DEM found, cannot proceed')

    dhdt = rioxr.open_rasterio(os.path.join(input_dir, 'dhdt_' + period, per_glacier_dir, 'dem.tif')) * 0.85 #convert from m/yr to m.w.eq.
    dhdt = dhdt.rio.write_nodata(Fill_Value)
    dhdt.data[0][abs(dhdt.data[0])>1e3] = dhdt.rio.nodata
    dhdt = dhdt.rio.interpolate_na()

    dem = dem.rio.reproject_match(dhdt)

    if RID in RIDs_Sweden.tolist():
        mask = load_georeferenced_mask(RID)
        mask = mask.rio.set_attrs({'nodata': 0})
        mask = mask.rio.reproject_match(dhdt)
    else:
        mask = rioxr.open_rasterio(os.path.join(input_dir, 'masks', per_glacier_dir, RID + '_mask.tif'))
        mask = mask.rio.set_attrs({'nodata': 0})
        mask = mask.rio.reproject_match(dhdt)
    consensus_thk = rioxr.open_rasterio(os.path.join(input_dir, 'consensus_thk', 'RGI60-' + RGI_region, RID + '_thickness.tif'))
    consensus_thk = consensus_thk.rio.set_attrs({'nodata': Fill_Value})
    consensus_thk = consensus_thk.rio.reproject_match(dhdt)
    consensus_thk.data[0][consensus_thk.data[0] == Fill_Value] = 0
    consensus_thk.data[0][mask.data[0] == 0] = 0
    vel_Millan = rioxr.open_rasterio(os.path.join(input_dir, 'vel_Millan', per_glacier_dir, 'dem.tif'))

    #mb, dhdt = resolve_mb_dhdt_smoothing(RID, dhdt, dem, mask, use_generic_dem_heights = use_generic_dem_heights, modify_dhdt_or_smb = modify_dhdt_or_smb)
    #apparent_mb = (mb - dhdt)*mask
    print(RID)
    app_mb = deepcopy(dhdt)

    mb_res, dhdt_res, app_mb.data[0] = resolve_mb_dhdt_piecewise_linear(RID, dhdt, dem, mask, use_generic_dem_heights = use_generic_dem_heights, modify_dhdt_or_smb = modify_dhdt_or_smb)
    ax[0].scatter(dem.data[0][mask.data[0] == 1], mb_res.data[0][mask.data[0] == 1], color = 'orange')
    ax[0].scatter(dem.data[0][mask.data[0] == 1], dhdt_res.data[0][mask.data[0] == 1], color = 'purple')
    ax[0].scatter(dem.data[0][mask.data[0] == 1], app_mb.data[0][mask.data[0] == 1], color = 'black')
    ax[0].annotate(np.sum(app_mb.data[0][mask.data[0] == 1]), xy=(0.05, 0.95), xycoords='axes fraction')
    ax[1].pcolor(mask.data[0])
    ax[1].annotate(RID, xy=(0.05, 0.85), xycoords='axes fraction', c='w')
    ax[1].annotate(mb_glacier.NAME.iloc[0], xy=(0.05, 0.75), xycoords='axes fraction', c='w')
    pp.savefig()
    plt.close()
pp.close()
#    plt.show()
