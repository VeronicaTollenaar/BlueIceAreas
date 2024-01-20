# function to download data from MODIS (based on https://www.moonbooks.org/Articles/How-to-download-a-file-from-NASA-LAADS-DAAC-using-python-/)
# import packages
from datetime import date
from urllib.error import HTTPError

import urllib.request
import urllib.request, json

import pandas as pd
import subprocess
import os

# function to download files
def download_files(target_dir, # directory to save data
                    LAADS_query, # csv with all filenames
                    dayn, # number of the day of which to download files
                    ):
    # set directory
    path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(path)
    # read filenames from list of LAADS (obtained by searching for acquisitions over a certain area during a certain time)
    filenames_all = pd.read_csv(f'{LAADS_query}')
    filenames_all['day'] = [s.split('/')[-1].split('.')[1][1:] for s in filenames_all.iloc[:,1]]
    filenames_all_day = filenames_all[filenames_all['day']==dayn].reset_index(drop=True)
    assert len(filenames_all_day) > 0
    # set directory in which to save data (relative path)
    target_dir = target_dir 
    # check if filenames already exist in directory!
    fileexists = os.listdir(target_dir)
    filenames_all_day['fileexists_check'] = [s.split('/')[-1] for s in filenames_all_day.iloc[:,1]]
    # check if existing files are not corrupted
    check_existingfiles = filenames_all_day[filenames_all_day['fileexists_check'].isin(fileexists)]
    check_existingfiles_list = check_existingfiles.iloc[:,1].values.tolist()
    for existing_file in check_existingfiles_list:
        cmd_preproc = f"gdalinfo {target_dir}{existing_file.split('/')[-1]}"
        info_preproc=subprocess.Popen(cmd_preproc, shell=True, stdout=subprocess.PIPE, )
        info_preproc_str=str(info_preproc.communicate()[0]) 
        if len(info_preproc_str) < 10:
            print('File corrupted, trying to download')
            fileexists.remove(f"{existing_file.split('/')[-1]}")
        else:
            print(f"File {existing_file.split('/')[-1]} already exists")

    # select filenames to download
    filenames = filenames_all_day[~filenames_all_day['fileexists_check'].isin(fileexists)]
    # print estimated time
    print(f"will download {sum(filenames['size'])*1e-9} GB")
    print(f"with 2 MB/s that takes {(sum(filenames['size']*1e-6)/2)/3600} hours")
    # add authorization code (see https://www.moonbooks.org/Articles/How-to-download-a-file-from-NASA-LAADS-DAAC-using-python-/)
    opener = urllib.request.build_opener()
    opener.addheaders = [('Authorization', 'Bearer authorization-code (removed here)')]
    urllib.request.install_opener(opener)

    # list of filenames
    filenames_list = filenames.iloc[:,1].values.tolist()
    # predefine value for n_tries
    n_tries = 0
    # try to download
    while (len(filenames_list)>0) and (n_tries < 40):
        for idx, ladsweb_url_p2 in enumerate(filenames_list):
            print(f'downloading next file - {len(filenames_list)} files remaining ...')
            try:
                ladsweb_url = f'https://ladsweb.modaps.eosdis.nasa.gov{ladsweb_url_p2}'
                target_name = f"{target_dir}{ladsweb_url_p2.split('/')[-1]}"
                urllib.request.urlretrieve(ladsweb_url,target_name)
                # try to open file
                cmd_preproc = f"gdalinfo {target_dir}{ladsweb_url_p2.split('/')[-1]}"
                info_preproc=subprocess.Popen(cmd_preproc, shell=True, stdout=subprocess.PIPE, )
                info_preproc_str=str(info_preproc.communicate()[0])
                if len(info_preproc_str) < 10:
                    print('File corrupted, trying to download again')
                else:
                    # remove downloaded file from list
                    filenames_list.remove(ladsweb_url_p2)
            except Exception as ex: #HTTPError or ContentTooShortError:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(f'{message}, trying again later')
        # stop trying to download after 40 attempts
        n_tries = n_tries + 1
    if len(filenames_list)==0:
        print(f'finished downloading!')    
    else:
        print(f'finished downloading with {len(filenames_list)} errors')
