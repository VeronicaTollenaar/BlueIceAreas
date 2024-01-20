# generate continent-wide predictions
# set working directory
import os
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

# import packages
import torch
from dataset import BIADataset
from model import UNET
from utils import (
    load_checkpoint,
        get_loaders,)
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import xarray as xr
import numpy as np
import affine
#%%
# loop over all trained models
for run in range(0,5,1):
    if run == 0:
        model_run = f'hp_opt_1'
    else:
        model_run = f'final_model_{run}'
    # loop over all rotations
    for rot in [0,1,2,3]:
        # loop over all flips
        for flip in ['flip','noflip']:
            # define directory of prediction tiles
            prediction_tile_directory = f'../pred_tiles'
            # define device
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            # import model
            model = UNET(in_channels=10, out_channels=1).to(DEVICE)
            # import trained parameters
            model_run = model_run
            load_checkpoint(model_run, model)
            # set model to evaluation mode
            model.eval()
            # list all filenames
            all_tiles = [f for f in os.listdir(f'{prediction_tile_directory}') if not f.startswith('.')]
            # obtain predictions for tile
            # import data
            # object to load
            pred_ds = BIADataset(
                image_dir=prediction_tile_directory,
                target_dir=None,
                transform_p=0.0,
                normalize_elevation=True,
            )
            # define dataloader prediction data
            pred_loader = DataLoader(
                pred_ds,
                batch_size=1,
                num_workers=0,
                pin_memory=True,
                shuffle=False,
            )

            # define 2d smoothing window to assign weights to each prediction within tile (Figure S5)
            n_pixels = 512
            resolution = 200 #m
            # 32 = the outer pixels that should not be considered
            # --> 64 pixels in the inner part of the tile should have the weight of 1
            len_spline = (n_pixels - (64+32+32))//2

            # define 1d window
            # Tukey window https://en.wikipedia.org/wiki/List_of_window_functions
            n = torch.linspace(0,len_spline-1,len_spline)
            Nalpha = (len_spline-1)*2
            cosine_wind = 0.5*((1-np.cos((2*torch.pi*n)/(Nalpha))))
            cosine_wind_rev = cosine_wind.flip(0)
            tuk_window_1d = torch.zeros(n_pixels)
            tuk_window_1d[32:32+len_spline] = cosine_wind
            tuk_window_1d[32+len_spline:32+len_spline+64] = 1.
            tuk_window_1d[32+len_spline+64:32+len_spline*2+64] = cosine_wind_rev

            # expand 1d window to 2d
            tuk_window_2d = tuk_window_1d.expand(512,512) * tuk_window_1d.expand(512,512).transpose(0,1)
            tuk_window_2d = tuk_window_2d.to(device=DEVICE)

            # create large empty tensor to store all values
            # define lowerleft and upper right corner of array (these values need to align with the grid of the observations)
            ll_x_main = -2716100
            ll_y_main = -2600100
            ur_x_main = 2836100
            ur_y_main = 2390100
            # calculate number of entries
            n_x = ((ur_x_main - ll_x_main)/resolution) + 1
            n_y = ((ur_y_main - ll_y_main)/resolution) + 1
            # predefine empty tensors
            all_weighted_preds = torch.zeros((int(n_y),int(n_x)))
            all_weights = torch.zeros((int(n_y),int(n_x)))

            # send tensors to GPU
            all_weighted_preds = all_weighted_preds.to(device=DEVICE)
            all_weights = all_weights.to(device=DEVICE)


            # classify data
            for idx, (data_pred, filename) in enumerate(pred_loader):
                # flip data
                if flip=='flip':
                    data_pred = torch.flip(data_pred,dims=[-2,-1])
                # rotate data
                data_pred = torch.rot90(data_pred, k=rot, dims=[-2,-1])
                
                # send data to GPU
                data_pred = data_pred.to(device=DEVICE)
                with torch.no_grad():
                    # apply trained model to tile
                    predictions = torch.sigmoid(model(data_pred))
                    # multiply with 2d spline function to smooth overlapping tiles
                    weighted_preds = predictions*tuk_window_2d
                # rotate data back
                weighted_preds = torch.rot90(weighted_preds,k=-1*rot,dims=[-2,-1])
                # flip data back
                if flip=='flip':
                    weighted_preds = torch.flip(weighted_preds,dims=[-2,-1])

                # extract lowerleft corner from filename
                filename_splitted = filename[0].split('_')
                ll_x = float(filename_splitted[0])
                ll_y = float(filename_splitted[1][:-3])

                # translate lowerleft corner to indices in array
                idx_x = int((ll_x - ll_x_main)/resolution)
                idx_y = int((ll_y - ll_y_main)/resolution)

                # sum weighted preds and weights to Antarctic-wide tensor
                all_weighted_preds[idx_y:idx_y+512,idx_x:idx_x+512] = all_weighted_preds[idx_y:idx_y+512,idx_x:idx_x+512] + weighted_preds.flip(-2)
                all_weights[idx_y:idx_y+512,idx_x:idx_x+512] = all_weights[idx_y:idx_y+512,idx_x:idx_x+512] + tuk_window_2d

            # divide weighted predictions over summed weights
            all_preds = (all_weighted_preds/all_weights)

            # assign coordinates and export as netcdf
            ds_preds = xr.Dataset(data_vars=dict(
                                    all_preds=(["y","x"],all_preds.cpu().numpy().astype('float32')),
                                    weights=(["y","x"],all_weights.cpu().numpy().astype('float32')))
                                )
            # define coordinates
            x_coords_main = np.arange(ll_x_main,ur_x_main+resolution,resolution).astype('int32')
            y_coords_main = np.arange(ll_y_main,ur_y_main+resolution,resolution).astype('int32')
            # assign coordinates
            ds_preds['x'] = x_coords_main
            ds_preds['y'] = y_coords_main


            # define transform (a, b, c, d, e, f)
                # a = width of a pixel
                # b = row rotation (typically zero)
                # c = x-coordinate of the upper-left corner of the upper-left pixel
                # d = column rotation (typically zero)
                # e = height of a pixel (typically negative)
                # f = y-coordinate of the of the upper-left corner of the upper-left pixel
            ds_preds.attrs['transform'] = affine.Affine(resolution, 0.0, ll_x_main-(resolution/2), 0.0, -1*resolution, ur_y_main+(resolution/2))
            ds_preds.attrs['res'] = resolution
            # export data
            ds_preds.to_netcdf(f'../output/preds_{model_run}_rot{rot}_{flip}.nc')

#%%
# plot 2d smoothing window that assigns weights to each prediction within the tile

# predefine tensor with zeros
n_pixels = 512
resolution = 200 #m
# 32 = the outer pixels that should not be considered
# --> 64 pixels in the inner part of the tile should have the weight of 1
len_spline = (n_pixels - (64+32+32))//2

# define 1d window
# Tukey window https://en.wikipedia.org/wiki/List_of_window_functions
n = torch.linspace(0,len_spline-1,len_spline)
Nalpha = (len_spline-1)*2
cosine_wind = 0.5*((1-np.cos((2*torch.pi*n)/(Nalpha))))
cosine_wind_rev = cosine_wind.flip(0)
tuk_window_1d = torch.zeros(n_pixels)
tuk_window_1d[32:32+len_spline] = cosine_wind
tuk_window_1d[32+len_spline:32+len_spline+64] = 1.
tuk_window_1d[32+len_spline+64:32+len_spline*2+64] = cosine_wind_rev

# expand 1d window to 2d
tuk_window_2d = tuk_window_1d.expand(512,512) * tuk_window_1d.expand(512,512).transpose(0,1)

# plot figure (Figure S5)
fig, ax1 = plt.subplots(1,1,figsize=(9/2.54,9/2.54))
plt.imshow(tuk_window_2d,cmap='gray')
cbar = plt.colorbar(label='Weight',shrink=0.82)
contour = plt.contour(tuk_window_2d,levels=[0,0.9999999],colors=['turquoise','firebrick'])
cbar.ax.plot([0, 1], [0, 0], 'turquoise', lw=6)
cbar.ax.plot([0, 1], [0.9999999, 0.9999999], 'firebrick', lw=6)
plt.xlabel('Pixels')
plt.ylabel('Pixels')

# adjust spacing
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0)
plt.margins(0,0)
# save figure
fig.savefig(f'../output/figures/cos_window.png',bbox_inches = 'tight',
            pad_inches = 0.01,dpi=300,facecolor='white')

