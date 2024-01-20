# process quality bands of MODIS to mask out e.g., cloudy observations
# import packages
import xarray as xr
import numpy as np
import os
#%%
def QA_binary_mask(QA_files_dir):
    # defining possible values for the QA (see Supporting Information Text S1)
    row1 = ['00','01']
    row2to8 = ['0000','0111','1000','1100']
    row9 = ['1']
    row10 = ['0','1']
    # generate numbers from binary values
    possible_values_QA = []
    for r1 in row1:
        for r2 in row2to8:
            for r3 in row2to8:
                for r4 in row2to8:
                    for r5 in row2to8:
                        for r6 in row2to8:
                            for r7 in row2to8:
                                for r8 in row2to8:
                                    for r9 in row9:
                                        for r10 in row10:
                                            string = r10 + r9 + r8 + r7 + r6 + r5 + r4 + r3 + r2 + r1
                                            number = float(int(string,2))
                                            # select possible values for the QA band
                                            possible_values_QA.append(number)

    # read in files
    files_all = os.listdir(QA_files_dir)
    QA_files_all = [s for s in files_all if '_QA.tif' in s]
    QA_files = [s for s in QA_files_all if 'pass' in s]

    for file in QA_files:
        # read in data
        data = xr.open_rasterio(f'{QA_files_dir}{file}')
        # create binary mask
        QA_passed = np.in1d(data.values, possible_values_QA).reshape(data.values.shape)
        # write to dataarray
        QA_passed_da = xr.DataArray(QA_passed.astype(int).astype(float),
                coords={"band":data.band, 
                "y":data.y, 
                "x":data.x},
                dims=["band","y","x"])
        # write to .nc file
        QA_passed_da.to_netcdf(f'{QA_files_dir}{file[:-4]}binary.nc')
        # convert to .tif
        proj_str = "+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181"
        os.system(f'gdal_translate -a_srs "{proj_str}" -ot Int16 {QA_files_dir}{file[:-4]}binary.nc {QA_files_dir}{file[:-4]}binary.tif')
        # remove .nc file
        os.system(f'rm {QA_files_dir}{file[:-4]}binary.nc')
        os.system(f'rm {QA_files_dir}{file[:-4]}binary.tif.aux.xml')

def stateQA_binary_mask(stateQA_files_dir):
    # defining possible values for the state QA (see Supporting Information Text S1)
    row1 = ['00','10','11']
    row2 = ['0']
    row3 = ['000','001','010','011','100','101','110','111']
    row4 = ['00','01','10','11']
    row5 = ['00','01','10']
    row6 = ['0','1']
    row7 = ['0','1']
    row8 = ['0','1']
    row9 = ['0','1']
    row10 = ['0','1']
    row11 = ['0','1']
    # generate numbers from binary values
    possible_values_stateQA = []
    for r1 in row1:
        for r2 in row2:
            for r3 in row3:
                for r4 in row4:
                    for r5 in row5:
                        for r6 in row6:
                            for r7 in row7:
                                for r8 in row8:
                                    for r9 in row9:
                                        for r10 in row10:
                                            for r11 in row11:
                                                string = r11 + r10 + r9 + r8 + r7 + r6 + r5 + r4 + r3 + r2 + r1
                                                number = float(int(string,2))
                                                # select possible values for the state QA band
                                                possible_values_stateQA.append(number)
    # read in files
    files_all = os.listdir(stateQA_files_dir)
    stateQA_files_all = [s for s in files_all if '_stateQA.tif' in s]
    stateQA_files = [s for s in stateQA_files_all if 'pass' in s]

    for file in stateQA_files:
        # read in data
        data = xr.open_rasterio(f'{stateQA_files_dir}{file}')
        # create binary mask
        stateQA_passed = np.in1d(data.values, possible_values_stateQA).reshape(data.values.shape)
        # write to dataarray
        stateQA_passed_da = xr.DataArray(stateQA_passed.astype(int).astype(float),
                coords={"band":data.band, 
                "y":data.y, 
                "x":data.x},
                dims=["band","y","x"])
        # write to .nc file
        stateQA_passed_da.to_netcdf(f'{stateQA_files_dir}{file[:-4]}binary.nc')
        # convert to .tif
        proj_str = "+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181"
        os.system(f'gdal_translate -a_srs "{proj_str}" -ot Int16 {stateQA_files_dir}{file[:-4]}binary.nc {stateQA_files_dir}{file[:-4]}binary.tif')
        # remove .nc file
        os.system(f'rm {stateQA_files_dir}{file[:-4]}binary.nc')
        os.system(f'rm {stateQA_files_dir}{file[:-4]}binary.tif.aux.xml')
