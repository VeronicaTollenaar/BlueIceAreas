# merge daily composites of MODIS into a multi-day composite
# import packages
import xarray as xr
import pandas as pd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt

import os
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

# define function to compute the median of a number of passes
def compute_nanmedian(band_number, # band number to process
                    directory, # directory of daily composites
                    window_size, # window to reduce amount of RAM
                    ):
    # list filenames
    filenames = os.listdir(f'{directory}')
    filenames = [s for s in filenames if s.startswith(f'MOD09')]
    window_size = window_size
    # open one file to obtain profile values
    data_day1 = rasterio.open(f'{directory}/{filenames[0]}')
    profile = data_day1.profile
    profile.update(count=1)
    profile.update(nodata=0)
    
    # define window columns and rows, widths and heights
    window_cols = np.arange(0,data_day1.shape[1],window_size)
    window_rows = np.arange(0,data_day1.shape[0],window_size)
    window_widths = np.ones((len(window_cols)))*window_size
    window_heights = np.ones((len(window_rows)))*window_size
    window_widths[-1] = data_day1.shape[1] - (len(window_cols)-1)*window_size
    window_heights[-1] = data_day1.shape[0] - (len(window_rows)-1)*window_size
    
    print(f'opening file')
    # open file to which to write values
    with rasterio.open(
            f'{directory}/nanmedian_B{band_number}.tif', 'w', **profile) as dst:
        # loop over windows
        for ix_col, window_col in enumerate(window_cols):
            for ix_row, window_row in enumerate(window_rows):
                # predefine empty array to save values of different days
                data_band = np.zeros((len(filenames),int(window_heights[ix_row]),int(window_widths[ix_col]))) 
                # define window sizes
                window_ = Window(window_col,window_row,window_widths[ix_col],window_heights[ix_row])
                for idx, filename in enumerate(filenames):
                    # open data
                    data_day = rasterio.open(f'{directory}/{filename}')
                    # read in data for selected window and band
                    data_day_band = data_day.read(band_number, window=window_)  
                    # assign nans to missing values (nodata=0)
                    data_day_band_nan = np.where(data_day_band==0,np.nan,data_day_band)
                    # store data of file in array
                    data_band[idx,:,:] = data_day_band_nan
                    # delete intermediate files from memory
                    del(data_day_band, data_day_band_nan)
                # compute nanmedian
                nanmedian = np.nanmedian(data_band,axis=0)
                # append values to tif file
                dst.write(nanmedian.astype(np.int16), window=window_, indexes=1)
            print(f'part {ix_col+1}/{len(window_cols)} is done')

directory = '../../../../../../home/DISK1/files_vero/daily_passes'

string_filenames = ''
# compute nanmedian
for i in range(1,8):
    print(f'starting band {i}')
    compute_nanmedian(i,directory,window_size=3500)
    # append filename to string of filenames to merge bands later
    string_filenames = string_filenames + f'{directory}/nanmedian_B{i}.tif '

# merge all bands to one tif
os.system(f'gdalbuildvrt -separate -resolution "highest" {directory}/merged_bands_composite080910.vrt {string_filenames}')
os.system(f'gdal_translate {directory}/merged_bands_composite080910.vrt {directory}/merged_bands_composite080910.tif')

# reproject final tif
proj_str = "+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181"
# gdalwarp uses nearest neighbor interpolation --> check quality
os.system(f'gdalwarp -s_srs "{proj_str}" -t_srs "EPSG:3031" -srcnodata 0 -multi {directory}/merged_bands_composite080910.tif {directory}/merged_bands_composite3031080910.tif')

