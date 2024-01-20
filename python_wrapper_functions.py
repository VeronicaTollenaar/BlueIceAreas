# define different functions to reproject and process MODIS data
# import packages
import os
from pyproj import Proj, Transformer
import subprocess
import pandas as pd
import numpy as np
import xarray as xr
import rasterio
from rasterio.windows import Window
#%%
# function to reproject L2 MODIS data
# loop over all files in data_path
def reproject_tiles(data_path, # path of data
                    daynumber, # number of the day
                    output_path, # output path to save data
                    band, # band to process (e.g., '"250m Surface Reflectance Band 1"')
                    output_name_band, # extension to filename to save band (e.g., 'B1')
                    pixsz, # pixel size to process data in km
                    ):
    # list filenames
    list_filenames_all = os.listdir(data_path)
    list_filenames = [s for s in list_filenames_all if s.startswith('MOD09')]
    list_filenames = [s for s in list_filenames if (f'{daynumber}' == s.split('.')[1][1:])] # check if this line works well!
    print(len(list_filenames))
    # predefine parameters
    n_tries = 0
    count_corrupted = 0
    count_toocloudy = 0
    corrupted_files = []
    cloudy_files = []
    # calculate values for -box parameter needed for NASA reprojection software (depends on output resolution)
    resolution = pixsz #km
    km_globalproj = 25485 # if resolution is 1 km
    n_pix_globalproj = km_globalproj//resolution
    # define rectangular box around south pole
    ext_left = 2750 # km
    ext_up = 2350 # km
    ext_right = 2900 # km
    ext_down = 2250 # km
    pix_ul_col = int(n_pix_globalproj//2 - ext_left//resolution)
    pix_ul_row = int(n_pix_globalproj//2 - ext_up//resolution)
    width = int(ext_left//resolution + ext_right//resolution)
    height = int(ext_up//resolution + ext_down//resolution)
    # loop over filenames
    while (len(list_filenames) > 0) and (n_tries < 100):
        for file in list_filenames:
            print(f'processing {file}')
            # step 0. check whether to process or not
            # define input file name
            input_filename = file #e.g., 'MOD09.A2022064.2125.006.2022066021950.hdf'
            # define band
            band = band #e.g., '"1km Surface Reflectance Band 1"' "500m Reflectance Band Quality"
            # define output file name
            output_filename = f'{input_filename[:-4]}_{output_name_band}.hdf'
            # check if output filename already exists
            if os.path.exists(f'{output_path}{output_filename[:-4]}.tif') == False:
                try:
                    # read in metadata of file
                    cmd_preproc = f'gdalinfo {data_path}{input_filename}'
                    info_preproc=subprocess.Popen(cmd_preproc, shell=True, stdout=subprocess.PIPE, )
                    info_preproc_str=str(info_preproc.communicate()[0])
                    # check whether internal cloud mask <90
                    info_preproc_splitted =info_preproc_str.split("\\n")
                    intcloudmask_text = 'Internal cloud mask'
                    intcloudmask_str = [s for s in info_preproc_splitted if intcloudmask_text in s][0]
                    intcloudmask = float(intcloudmask_str.split(":\\t")[-1])
                    # check whether cloudy pixels <90
                    cloudpix_text = 'Cloudy'
                    cloudpix_str = [s for s in info_preproc_splitted if cloudpix_text in s][0]
                    cloudpix = float(cloudpix_str.split(":\\t")[-1])
                    # check if atmospheric correction is attempted
                    atmospherecorr_text = 'ATMOSPHERICCORRECTIONATTEMPTED'
                    atmospherecorr_str = [s for s in info_preproc_splitted if atmospherecorr_text in s][0]
                    atmospherecorr = atmospherecorr_str[-3:]
                except:
                    print(f"problem with file {input_filename}, removed from list filenames")
                    list_filenames.remove(file)
                    count_corrupted = count_corrupted + 1
                    corrupted_files.append(input_filename)
                    break
                if (intcloudmask > 90) | (cloudpix > 90) | (atmospherecorr != 'yes'):
                    print(f'too cloudy, data will not be processed and file {input_filename} is removed')
                    print('internal cloudmask', intcloudmask)
                    print('cloudpix', cloudpix)
                    print('atmospherecorr', atmospherecorr)
                    list_filenames.remove(file)
                    count_toocloudy = count_toocloudy + 1
                    cloudy_files.append(input_filename)
                else:
                    # step 1. use NASA software (projbrowse_pkg) to reproject 5-min swath data to stereographic projection
                    # reproject data !!! Sometimes this causes errors (the geolocation file cannot be found for some reason), the filename will be saved and tried later
                    # NB using os.system is not the safest method
                    os.system(f'projbrowse -proj=14 -MOD03 -depth -center=0,-90 -pixsz={pixsz} -box={pix_ul_col},{pix_ul_row},{width},{height} -sds={band} {data_path}{input_filename} -of={output_path}{output_filename} > terminal_output.txt')
                    # CHECK IF REPROJECTION IS DONE CORRECTLY!!!
                    with open('terminal_output.txt', 'r') as file_:
                        terminal_output = file_.read().split("\n")
                    geo_file = [s.split(':')[1].split('/')[-1] for s in terminal_output if 'Found geolocation file' in s]
                    print(geo_file)
                    if len(geo_file) != 1:
                        print(f"reprojection didn't work, trying again later")
                        continue
                    else:
                        geo_file_time = geo_file[0].split('.')[2]
                        print('geo file time:', geo_file_time)
                        file_time = file.split('.')[2]
                        print('file time:', file_time)
                        if geo_file_time == file_time:
                            # catch errors if file is not within bounds
                            outofbounds = [s for s in terminal_output if 'Warning: none of the input pixels was projected successfully.' in s]
                            if len(outofbounds)>0:
                                print(f"file {input_filename} is not in the AOI, removed from list filenames")
                                list_filenames.remove(input_filename)
                                count_corrupted = count_corrupted + 1
                                corrupted_files.append(input_filename)
                                continue
                            try:
                                # step 2. assign coordinates and reference system to reprojected data
                                # define projection of tile boundaries
                                inProj = Proj('EPSG:4326')
                                # define projection of reprojected data
                                proj_str = "+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181"
                                outProj = Proj(proj_str)

                                # extract tile boundaries from data
                                cmd = f'gdalinfo {output_path}{output_filename}'
                                info=subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, )
                                info_str=str(info.communicate()[0])
                                info_str_splitted =info_str.split("\\n")
                                
                                LR_lat_text = 'Lower Right Corner Bounding Latitude (deg.)'
                                LR_lat_str = [s for s in info_str_splitted if LR_lat_text in s][0]
                                LR_lat = float(LR_lat_str.split("=")[-1])

                                LR_lon_text = 'Lower Right Corner Bounding Longitude (deg.)'
                                LR_lon_str = [s for s in info_str_splitted if LR_lon_text in s][0]
                                LR_lon = float(LR_lon_str.split("=")[-1])

                                UL_lat_text = 'Upper Left Corner Bounding Latitude (deg.)'
                                UL_lat_str = [s for s in info_str_splitted if UL_lat_text in s][0]
                                UL_lat = float(UL_lat_str.split("=")[-1])

                                UL_lon_text = 'Upper Left Corner Bounding Longitude (deg.)'
                                UL_lon_str = [s for s in info_str_splitted if UL_lon_text in s][0]
                                UL_lon = float(UL_lon_str.split("=")[-1])

                                transformer = Transformer.from_proj(inProj, outProj)
                                x_UL, y_UL = transformer.transform(UL_lat, UL_lon)
                                x_LR, y_LR = transformer.transform(LR_lat, LR_lon)

                                os.system(f'gdal_translate -a_srs "{proj_str}" -a_ullr {x_UL} {y_UL} {x_LR} {y_LR} {output_path}{output_filename} {output_path}{output_filename[:-4]}.tif')

                                # remove intermediate files
                                os.system(f'rm {output_path}{output_filename}')
                                # remove processed file from list filenames
                                list_filenames.remove(file)
                            except:
                                print("reprojection didn't work, trying again later")
                        if geo_file_time != file_time:
                            print("Geolocation file not correctly selected, trying again later")
            else:
                print("file already exists")
                list_filenames.remove(file)
            
        n_tries = n_tries + 1

    # print number of files to check if all are processed correctly
    print(f"files too cloudy: {count_toocloudy}")
    print(f"files corrupted: {count_corrupted}")
    failed_files = len(list_filenames)
    print(f"files failed to process: {failed_files}")
    return(count_corrupted, count_toocloudy, corrupted_files, cloudy_files, failed_files)

# function to merge adjecent tiles (i.e., one satellite pass over Antarctica)
def merge_adjectent_tiles(data_path, # path of data files
                          dayn, # number of the day
                          output_path, # path to save the data
                          output_name_band, # band number
                          count_corrupted, # number of corrupted files (output of prev function)
                          count_toocloudy, # number of cloudy files (output of prev function)
                          corrupted_files, # list of corrupted files
                          cloudy_files, # list of cloudy files
                          fill_value, # assign fill value for no data
                          ):
    # merge adjacent tiles --> ca. 15 passes per day at the polar region
    # group filenames of adjacent tiles
    # create new list of all filenames
    list_filenames_all2 = os.listdir(data_path)
    list_filenames2 = [s for s in list_filenames_all2 if s.startswith('MOD09')]
    list_filenames2 = [s for s in list_filenames2 if s.split('.')[1][1:]==f'{dayn}']
    # remove corrupted and cloudy filenames
    if count_corrupted > 0:
        for i in range(count_corrupted):
            list_filenames2.remove(corrupted_files[i])
    if count_toocloudy > 0:
        for j in range(count_toocloudy):
            list_filenames2.remove(cloudy_files[j])
    # create a dataframe with filenames
    group_filenames = pd.DataFrame({"list_filenames":list_filenames2})
    # extract acquisition time from filenames
    group_filenames['acquisition_time'] = [float(s.split('.')[2]) for s in group_filenames.list_filenames]
    # sort dataframe by acquisition time
    group_filenames_sorted = group_filenames.sort_values(by=['acquisition_time'],ignore_index=True)
    group_n = 1
    group_filenames_sorted['group_n'] = np.zeros(len(group_filenames_sorted))
    # loop through dataframe to group values
    for row in range(len(group_filenames_sorted)-1):
        time_diff = group_filenames_sorted['acquisition_time'].iloc[row + 1] - group_filenames_sorted['acquisition_time'].iloc[row]
        # time difference equals 5 or 45 (in case xx55 and xx00)
        group_filenames_sorted.iloc[row,2] = group_n
        if (time_diff == 5) or (time_diff == 45):
            group_n = group_n
        else:
            group_n = group_n + 1
    group_filenames_sorted.iloc[-1,2] = group_n

    # use gdalbuildvrt with -srcnodata "-200" to merge tiles
    # create strings of files to merge
    # add column to dataframe with associated output filenames
    group_filenames_sorted['files_to_merge'] = [f'{output_path}{s[:-4]}_{output_name_band}.tif' for s in group_filenames_sorted['list_filenames']]
    # group dataframe by the group_n
    grouped = group_filenames_sorted.groupby(['group_n'])['files_to_merge'].transform(lambda x: ' '.join(x))
    grouped = grouped.drop_duplicates()
    grouped_df = pd.DataFrame(grouped)
    # create strings of merged file names
    # extract group number
    grouped_df['group_n'] = group_filenames_sorted.iloc[grouped_df.index]['group_n'].astype(int).astype(str).str.zfill(2)
    # extract day of the year
    grouped_df['new_filename_p1'] = group_filenames_sorted.iloc[grouped_df.index]['list_filenames']
    grouped_df['new_filename_p1'] = [s[:15] for s in grouped_df['new_filename_p1']]
    # combine day of the year and group number ("pass")
    series_list = [grouped_df[c] for c in ['new_filename_p1','group_n']]
    grouped_df['new_filename'] = series_list[0].str.cat(series_list[1:],sep='pass')
    # append .vrt to string
    grouped_df['new_filename'] = [f'{output_path}{s}_{output_name_band}.vrt' for s in grouped_df['new_filename']]
    # loop over all rows in dataframe
    for k in range(len(grouped_df)):
        os.system(f'gdalbuildvrt -srcnodata "{fill_value}" {grouped_df.iloc[k]["new_filename"]} {grouped_df.iloc[k]["files_to_merge"]}')
        os.system(f'gdal_translate {grouped_df.iloc[k]["new_filename"]} {grouped_df.iloc[k]["new_filename"][:-4]}.tif')
        os.system(f'rm {grouped_df.iloc[k]["new_filename"]}')
        # remove files
        os.system(f'rm {grouped_df.iloc[k]["files_to_merge"]}')

# function to merge bands
# select different bands to combine into tiff file
def merge_bands(output_path, # path to save data
                list_of_bands, # list of bands to merge
                ):
    # list all files in output directory
    all_files_outputdir = os.listdir(output_path)
    # select all files that need to be merged (if the word 'pass' is in the filename)
    files_to_merge = [s for s in all_files_outputdir if 'pass' in s]
    files_to_merge = [s for s in files_to_merge if '_' in s]
    # save as dataframe
    files_to_merge_df = pd.DataFrame({"filenames":files_to_merge})
    # extract the pass number from the filenames
    files_to_merge_df["pass_n"] = [s.split('.')[2].split('_')[0] for s in files_to_merge_df["filenames"]]
    # extract the band from the filenames
    files_to_merge_df["band"] = [s.split('_')[1].split('.')[0] for s in files_to_merge_df["filenames"]]
    # create empty dataframe
    files_to_merge_bands = files_to_merge_df[0:0]
    # select bands listed in list_of_bands
    for band_ in list_of_bands:
        selected_band = files_to_merge_df[files_to_merge_df['band']==band_]
        files_to_merge_bands = pd.concat([files_to_merge_bands,selected_band])
    files_to_merge_sorted = files_to_merge_bands.sort_values(by=["band"],ignore_index=True)
    # add output directory to filename
    files_to_merge_sorted['files_to_merge'] = [f'{output_path}{s}' for s in files_to_merge_sorted['filenames']]
    # group the filenames by the pass numbers and create input text for gdalbuildvrt
    files_to_merge_grouped = files_to_merge_sorted.groupby(['pass_n'])['files_to_merge'].transform(lambda x: ' '.join(x))
    files_to_merge_grouped = files_to_merge_grouped.drop_duplicates()
    files_to_merge_grouped_df = pd.DataFrame(files_to_merge_grouped)
    # create new filename based on pass_n
    files_to_merge_grouped_df['pass_n'] = files_to_merge_sorted.iloc[files_to_merge_grouped_df.index]['pass_n']
    files_to_merge_grouped_df['new_filename_p1'] = files_to_merge_sorted.iloc[files_to_merge_grouped_df.index]['filenames']
    files_to_merge_grouped_df['new_filename_p1'] = [s.split('_')[0] for s in files_to_merge_grouped_df['new_filename_p1']]
    files_to_merge_grouped_df['new_filename'] = [f'{output_path}{s}.vrt' for s in files_to_merge_grouped_df['new_filename_p1']]
    # loop over filenames to merge
    for l in range(len(files_to_merge_grouped_df)):
        os.system(f'gdalbuildvrt -separate -resolution "highest" {files_to_merge_grouped_df.iloc[l]["new_filename"]} {files_to_merge_grouped_df.iloc[l]["files_to_merge"]}')
        os.system(f'gdal_translate {files_to_merge_grouped_df.iloc[l]["new_filename"]} {files_to_merge_grouped_df.iloc[l]["new_filename"][:-4]}.tif')

# create binary Quality masks (in separate file QA_tomask.py)

# function to merge passes
def merge_passes(output_path, # define output path
                window_size, # window is used to reduce the amount of RAM needed
                ):
    window_size = window_size

    # list all files (merged tifs of each pass)
    all_files = os.listdir(f'{output_path}')
    filenames_selected = [s for s in all_files if 'pass' in s]
    filenames_list = [s for s in filenames_selected if '_' not in s]
    filenames_list = [s for s in filenames_list if '.aux' not in s]
    filenames_list = [s for s in filenames_list if '.tif' in s]

    # read in one dataset to retrieve the shape
    dataset_example = rasterio.open(f'{output_path}{filenames_list[0]}')
    dataset_shape = dataset_example.read().shape

    # read in profile to save new .tif file later
    profile = dataset_example.profile
    profile.update(count=1)
    # delete variables from memory
    del(dataset_example)
    
    # define window columns and rows, widths and heights
    window_cols = np.arange(0,dataset_shape[2],window_size)
    window_rows = np.arange(0,dataset_shape[1],window_size)
    window_widths = np.ones((len(window_cols)))*window_size
    window_heights = np.ones((len(window_rows)))*window_size
    window_widths[-1] = dataset_shape[2] - (len(window_cols)-1)*window_size
    window_heights[-1] = dataset_shape[1] - (len(window_rows)-1)*window_size
    n_windows = len(window_widths)*len(window_heights)

    # predefine empty sting to store all names of individual files that are combined later
    string_files = ''
    # not most efficient, but because of memory loop over bands
    # loop over all bands
    for band_ in range(dataset_shape[0]-2):
        print(f'preparing files band {band_+1}')
        # define first part of filename
        filename_tosave = f"{filenames_list[0].split('.')[0]}.{filenames_list[0].split('.')[1]}"
        # open file
        with rasterio.open(f'{output_path}{filename_tosave}_allpasses_b{int(band_+1)}.tif', 'w', **profile) as dst:
            # loop over all windows
            for ix_col, window_col in enumerate(window_cols):
                for ix_row, window_row in enumerate(window_rows):
                    
                    # create empty array to fill with values per pass, per band
                    # NB dtype has to be np.float32, because we need to save np.nan in the array to calculate nanmean later
                    masked_bands_tofill = np.zeros((len(filenames_list),int(window_heights[ix_row]),int(window_widths[ix_col])),dtype=np.float32) 
                    # define window sizes
                    window_ = Window(window_col,window_row,window_widths[ix_col],window_heights[ix_row])
                    # loop over all files
                    for fileno,filename in enumerate(filenames_list):
                        # read in data
                        dataset = rasterio.open(f'{output_path}{filename}')
                        QAband1 = dataset.read(int(dataset_shape[0]), window=window_)
                        QAband2 = dataset.read(int(dataset_shape[0]-1), window=window_)
                        banddata = dataset.read(int(band_+1), window=window_)
                        # combine two quality bands into one
                        QAband = QAband1*QAband2
                        # mask values using the last two bands of dataset (QA and stateQA)
                        masked_band_step1 = np.where((QAband==1), banddata, np.nan)
                        # mask reflectance values smaller or equal to zero
                        masked_band_step2 = np.where(masked_band_step1>0, masked_band_step1, np.nan) 
                        # store values in array
                        masked_bands_tofill[fileno,:,:] = masked_band_step2 #band_
                        # delete variables from memory
                        del(dataset,QAband1,QAband2,banddata,QAband,masked_band_step1,masked_band_step2)
                    # compute median of all files ignoring nan values
                    print(f'merging files band {int(band_+1)}, part {(ix_col)*len(window_rows)+ix_row+1}/{n_windows}')
                    merged_files = np.nanmedian(masked_bands_tofill[:,:,:],axis=0) #USED TO BE NANMEAN
                    # save as tif file
                    print(f'saving tif ... part {(ix_col)*len(window_rows)+ix_row+1}/{n_windows}')
                    dst.write_band(1,(merged_files[:,:]).astype(rasterio.int16),window=window_)
                    # delete variables from memory
                    del(merged_files, masked_bands_tofill)

        # append filename of intermediate file to string of filenames
        string_files = f'{string_files}{output_path}{filename_tosave}_allpasses_b{int(band_+1)}.tif '

    # merge  tifs to multi-band tif
    os.system(f'gdalbuildvrt -separate {output_path}{filename_tosave}_allpasses.vrt {string_files}') #-srcnodata "0" 
    os.system(f'gdal_translate {output_path}{filename_tosave}_allpasses.vrt {output_path}{filename_tosave}_allpasses.tif')
    # remove intermediate files
    os.system(f'rm {string_files}')
    os.system(f'rm {output_path}{filename_tosave}_allpasses.vrt')
    # move file to daily_passes directory
    os.system(f'mv {output_path}{filename_tosave}_allpasses.tif ../../../../../../home/DISK1/files_vero/daily_passes/{filename_tosave}_allpasses.tif')
