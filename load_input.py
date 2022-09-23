import numpy as np
import rioxarray as rioxr

RID = 'RGI60-08.00001'
def load_dhdt(RID):
    path = '/home/thomas/regional_inversion/input_data/dhdt/RGI60-08/RGI60-08.0'+RID[10] + '/'+RID+ '/dem.tif'
    dhdt = rioxr.open_rasterio(path)

    return dhdt

def load_dem(RID):
    path = '/home/thomas/regional_inversion/input_data/DEMs/RGI60-08/RGI60-08.0'+RID[10] + '/'+RID+ '/dem.tif'
    dem = rioxr.open_rasterio(path)

    return dem
