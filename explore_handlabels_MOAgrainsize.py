# compare handlabels to MOA grainsize data (Figure S10)
# import packages
import gc
gc.collect()
# import packages
import numpy as np
import xarray as xr
import os
#import pandas as pd

import geopandas
from rasterio import features
import matplotlib.pyplot as plt

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

#%%
# open MOA grainsize data            
MOA_path = r'../data/moa125_2004_grn_v02.0.tif'
MOA_raw = xr.open_rasterio(MOA_path)

# convert DataArray to DataSet
MOA_ds = MOA_raw.drop('band')[0].to_dataset(name='gs')

# reset nodatavalues to nan (to avoid artifacts in interpolation)
MOA_ds = MOA_ds.where(MOA_ds['gs'] != 0)

#%%
# open handlabels
sq_246_path = r'../data/handlabeled_sq246.shx'
sq_265_path = r'../data/handlabeled_sq265.shx'
sq_278_path = r'../data/handlabeled_sq278.shx'
sq_264_path = r'../data/handlabelled_sq264.shx'
sq_409_path = r'../data/handlabelled_sq409.shx'
hl_sq_246 = geopandas.read_file(sq_246_path)
hl_sq_265 = geopandas.read_file(sq_265_path)
hl_sq_278 = geopandas.read_file(sq_278_path)
hl_sq_264 = geopandas.read_file(sq_264_path)
hl_sq_409 = geopandas.read_file(sq_409_path)

# load squares of training/validation/testing
sqs_train_path = r'../data/train_squares.shx'
sqs_train = geopandas.read_file(sqs_train_path).rename(columns={'geometry':'square'}).set_geometry(col='square')
    
sqs_val_path = r'../data/validation_squares.shx'
sqs_val = geopandas.read_file(sqs_val_path).rename(columns={'geometry':'square'}).set_geometry(col='square')
    
sqs_test_path = r'../data/test_squares.shx'
sqs_test = geopandas.read_file(sqs_test_path).rename(columns={'geometry':'square'}).set_geometry(col='square')

# select areas corresponding to handlabels
sq_265 = sqs_train[sqs_train['id_square']=='265']
sq_246 = sqs_val[sqs_val['id_square']=='246']
sq_278 = sqs_val[sqs_val['id_square']=='278']
sq_264 = sqs_test[sqs_test['id_square']=='264']
sq_409 = sqs_test[sqs_test['id_square']=='409']

#%%

# function to extract grainsize values at handlabelled locations and plot histograms
def plot_data(sq_n,hl_sq_n,axis,sq_title,plt_legend=False):
    # first clip data to extent of square area
    ShapeMask = features.geometry_mask(sq_n.geometry,
                                        out_shape=(len(MOA_ds.y), len(MOA_ds.x)),
                                        transform=MOA_raw.transform,
                                        invert=True)
    # assign coordinates and create a data array
    ShapeMask = xr.DataArray(ShapeMask , coords={"y":MOA_ds.y,
                                                "x":MOA_ds.x},
                            dims=("y", "x"))
    # apply Mask to MOA data
    data_in_sq = MOA_ds.where((ShapeMask==True),drop=True)

    # define transform of slected data
    data_in_sq_transform = list(MOA_raw.transform)
    data_in_sq_transform[2] = data_in_sq.x.min().values - 125/2
    data_in_sq_transform[5] = data_in_sq.y.max().values + 125/2
    data_in_sq_transform = tuple(data_in_sq_transform)

    # select labeled BIA and labeled no BIA in sq
    # rasterize polygons: create mask where only pixels whose center is within the polygon will be burned in.
    ShapeMask = features.geometry_mask(hl_sq_n.geometry,
                                        out_shape=(len(data_in_sq.y), len(data_in_sq.x)),
                                        transform=data_in_sq_transform,
                                        invert=True)
    # assign coordinates and create a data array
    ShapeMask = xr.DataArray(ShapeMask , coords={"y":data_in_sq.y,
                                                "x":data_in_sq.x},
                            dims=("y", "x"))

    # extract grain sizes at BIAs and at noBIAs
    gs_at_BIA_nans = data_in_sq.where((ShapeMask==True),drop=True).to_dataframe()
    gs_at_noBIA_nans = data_in_sq.where((ShapeMask==False),drop=True).to_dataframe()
    
    # drop nan values
    gs_at_BIA = gs_at_BIA_nans[~np.isnan(gs_at_BIA_nans['gs'])]
    gs_at_noBIA = gs_at_noBIA_nans[~np.isnan(gs_at_noBIA_nans['gs'])]

    # define bins for histogram
    bins = np.linspace(0,1150,24)

    # plot grain sizes at BIAs
    gs_at_BIA_bins = axis.hist(gs_at_BIA,
            label='blue ice areas', 
            edgecolor='black', 
            linewidth=0.6,
            bins=bins, 
            color='blue',
            alpha=0.7,
            density=True)
    # plot grain sizes at no BIAs
    gs_at_noBIA_bins = axis.hist(gs_at_noBIA,
            label='other areas', 
            edgecolor='black', 
            linewidth=0.6,
            bins=bins, 
            color='grey',
            alpha=0.7,
            density=True)
    # set ylabel and xlabel
    axis.set_ylabel('Normalized bin counts')
    axis.set_xlabel('MOA grain size (μm)')
    # plot legend
    if plt_legend==True:
        axis.legend()

    # calculate threshold of grainsize follow Bayes' rule and the law of total probability
    # probability to have blue ice or not
    p_BIA = len(gs_at_BIA)/(len(gs_at_BIA)+len(gs_at_noBIA))
    p_noBIA = len(gs_at_noBIA)/(len(gs_at_BIA)+len(gs_at_noBIA))
    # probability of a grain size (per bin)
    p_obs = gs_at_BIA_bins[0]*50*p_BIA + gs_at_noBIA_bins[0]*50*p_noBIA
    # probability to have blue ice given the grainsize
    p_BIA_gs = p_BIA * (gs_at_BIA_bins[0]*50/p_obs)
    # probability to have no blue ice given the grainsize
    p_noBIA_gs = p_noBIA * (gs_at_noBIA_bins[0]*50/p_obs)

    # calculate threshold
    center_bins = bins[:-1] + (bins[1]-bins[0])/2
    # check if there is a threshold
    if sum(p_BIA_gs>p_noBIA_gs)>0:
        # calculate the threshold (based on the center of the bin)
        threshold = center_bins[p_BIA_gs>p_noBIA_gs][0] - (bins[1]-bins[0])/2
        axis.annotate(f'threshold = {threshold} μm',xy=(300,gs_at_noBIA_bins[0].max()*0.6))
    else:
        # in case no threshold is found
        threshold = bins[-1]
        axis.annotate(f'no threshold found',xy=(300,gs_at_noBIA_bins[0].max()*0.6))
    # set title and ylimit
    axis.set_title(sq_title)
    axis.set_ylim([0,gs_at_noBIA_bins[0].max()*1.05])
    # estimate how many BIAs are missed through thresholding approach
    missed_BIAs = len(gs_at_BIA[gs_at_BIA['gs']<threshold])/len(gs_at_BIA)
    print(missed_BIAs)
    return(missed_BIAs)

#%%
# set font for plot
font = {'family': 'Arial', # normally Calibri
        'weight' : 'normal',
        'size'   : 10}
plt.rc('font', **font)
# plot histograms
fig, axis = plt.subplots(3,2,figsize=(16/2.54, 24/2.54))
# apply function to different squares
plot_data(sq_246,hl_sq_246,axis[0,0],'Mc Murdo Dry Valleys',plt_legend=True)
plot_data(sq_278,hl_sq_278,axis[0,1],'Sor Rondane Mts (W)',plt_legend=False)
plot_data(sq_409,hl_sq_409,axis[1,0],'Denman/Apfel Glacier',plt_legend=False)
plot_data(sq_265,hl_sq_265,axis[1,1],'Prince Albert Mts',plt_legend=False)
plot_data(sq_264,hl_sq_264,axis[2,0],'Victoria Land (West)',plt_legend=False)
# switch off last axis
axis[2,1].axis('off')
# adjust spacing
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0.5, wspace = 0.4)
plt.margins(0,0)
# save figure
fig.savefig('../output/figures/MOA_gs.png',bbox_inches = 'tight',
    pad_inches = 0.02,dpi=300)
