# python wrapper to download and reproject MODIS data automatically
# import packages
import os
from pyproj import Proj, Transformer
import subprocess
import pandas as pd
import numpy as np
import xarray as xr
from python_wrapper_functions import reproject_tiles
from python_wrapper_functions import merge_adjectent_tiles
from python_wrapper_functions import merge_bands
from python_wrapper_functions import merge_passes
from QA_tomask import QA_binary_mask
from QA_tomask import stateQA_binary_mask
from download_data import download_files
from datetime import date
from urllib.error import HTTPError
import urllib.request
import urllib.request, json
import pandas as pd
import os
from rasterio.windows import Window
from multiprocessing import Pool

# function to download data in parallel
def download_parallel(i):
    dayn = str(2010001 + i)
    print(dayn)
    # download data
    target_dir = '../../../../../../home/DISK1/files_vero/MOD_data/'
    LAADS_query = '../../../../../../home/DISK1/files_vero/MOD_data/LAADS_query.2010001to2010016.csv'
    download_files(target_dir, LAADS_query, dayn)
    # set directories
    data_path = target_dir
    output_path = f'../../../../../../home/DISK1/files_vero/files_{dayn}/'
    # make directory to store intermediate files
    os.system(f'mkdir {output_path}')

# run download in parallel
p = Pool(processes=16)
p.map(download_parallel, np.arange(0,16,1))
p.close()


def main():
    # (re)download data (basically just checks if all files are downloaded)
    for i in range(1):
        dayn = str(2008023 + i)
        print(dayn)
        # # download data
        target_dir = '../../../../../../home/DISK1/files_vero/MOD_data/'
        LAADS_query = '../../../../../../home/DISK1/files_vero/MOD_data/LAADS_query.2008017to2008032.csv'
        download_files(target_dir, LAADS_query, dayn)

        # set directories
        data_path = target_dir 
        output_path = f'../../../../../../home/DISK1/files_vero/files_{dayn}/'
        # make directory to store intermediate files
        os.system(f'mkdir {output_path}')
    
        # process data
        # band 1
        band = '"250m Surface Reflectance Band 1"'
        output_name_band = 'B1'
        count_corrupted, count_toocloudy, corrupted_files, cloudy_files, failed_files = reproject_tiles(data_path, dayn, output_path, band, output_name_band, 0.25)
        assert failed_files == 0
        merge_adjectent_tiles(data_path, dayn, output_path, output_name_band, count_corrupted, count_toocloudy, corrupted_files, cloudy_files, -200)

        # band 2
        band = '"250m Surface Reflectance Band 2"'
        output_name_band = 'B2'
        count_corrupted, count_toocloudy, corrupted_files, cloudy_files, failed_files = reproject_tiles(data_path, dayn, output_path, band, output_name_band, 0.25)
        assert failed_files == 0
        merge_adjectent_tiles(data_path, dayn, output_path, output_name_band, count_corrupted, count_toocloudy, corrupted_files, cloudy_files, -200)
        
        # band 3
        band = '"500m Surface Reflectance Band 3"'
        output_name_band = 'B3'
        count_corrupted, count_toocloudy, corrupted_files, cloudy_files, failed_files = reproject_tiles(data_path, dayn, output_path, band, output_name_band, 0.5)
        assert failed_files == 0
        merge_adjectent_tiles(data_path, dayn, output_path, output_name_band, count_corrupted, count_toocloudy, corrupted_files, cloudy_files, -200)

        # band 4
        band = '"500m Surface Reflectance Band 4"'
        output_name_band = 'B4'
        count_corrupted, count_toocloudy, corrupted_files, cloudy_files, failed_files = reproject_tiles(data_path, dayn, output_path, band, output_name_band, 0.5)
        assert failed_files == 0
        merge_adjectent_tiles(data_path, dayn, output_path, output_name_band, count_corrupted, count_toocloudy, corrupted_files, cloudy_files, -200)
        
        # band 5
        band = '"500m Surface Reflectance Band 5"'
        output_name_band = 'B5'
        count_corrupted, count_toocloudy, corrupted_files, cloudy_files, failed_files = reproject_tiles(data_path, dayn, output_path, band, output_name_band, 0.5)
        assert failed_files == 0
        merge_adjectent_tiles(data_path, dayn, output_path, output_name_band, count_corrupted, count_toocloudy, corrupted_files, cloudy_files, -200)

        # band 6
        band = '"500m Surface Reflectance Band 6"'
        output_name_band = 'B6'
        count_corrupted, count_toocloudy, corrupted_files, cloudy_files, failed_files = reproject_tiles(data_path, dayn, output_path, band, output_name_band, 0.5)
        assert failed_files == 0
        merge_adjectent_tiles(data_path, dayn, output_path, output_name_band, count_corrupted, count_toocloudy, corrupted_files, cloudy_files, -200)

        # band 7
        band = '"500m Surface Reflectance Band 7"'
        output_name_band = 'B7'
        count_corrupted, count_toocloudy, corrupted_files, cloudy_files, failed_files = reproject_tiles(data_path, dayn, output_path, band, output_name_band, 0.5)
        assert failed_files == 0
        merge_adjectent_tiles(data_path, dayn, output_path, output_name_band, count_corrupted, count_toocloudy, corrupted_files, cloudy_files, -200)

        # band QA
        band = '"500m Reflectance Band Quality"'
        output_name_band = 'QA'
        count_corrupted, count_toocloudy, corrupted_files, cloudy_files, failed_files = reproject_tiles(data_path, dayn, output_path, band, output_name_band, 0.5)
        assert failed_files == 0
        merge_adjectent_tiles(data_path, dayn, output_path, output_name_band, count_corrupted, count_toocloudy, corrupted_files, cloudy_files, 2147483600)
        
        # band stateQA
        band = '"1km Reflectance Data State QA"'
        output_name_band = 'stateQA'
        count_corrupted, count_toocloudy, corrupted_files, cloudy_files, failed_files = reproject_tiles(data_path, dayn, output_path, band, output_name_band, 1)
        assert failed_files == 0
        merge_adjectent_tiles(data_path, dayn, output_path, output_name_band, count_corrupted, count_toocloudy, corrupted_files, cloudy_files, 32750)
        
        # create binary masks of QA values and stateQA values
        QA_binary_mask(output_path)
        stateQA_binary_mask(output_path)
        
        # merge bands
        merge_bands(output_path,['B1','B2','B3','B4','B5','B6','B7','QAbinary','stateQAbinary'])

        # merge passes
        merge_passes(output_path,
                    18000)

    # remove directory with intermediate files
    # os.system(f'rm -r {output_path}')


if __name__ == "__main__":
    main()