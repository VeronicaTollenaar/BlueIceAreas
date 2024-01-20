# estimate performance in validation and test squares
# import packages
import numpy as np
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import geopandas
from model import UNET
from utils import (perf_metrics,
                   load_checkpoint
                   )
from dataset import BIADataset
from torch.utils.data import DataLoader
import xarray as xr
from rasterio import features
import affine
from shapely.geometry import Point

path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

#%%
# estimate performance in test areas
def test_clean(loader, 
               model, 
               device="cuda",
    ):
    # set model to evaluation mode
    model.eval()
    
    # preset total performance metrics for both validation squares with clean labels
    tp_val_264 = 0
    fp_val_264 = 0
    fn_val_264 = 0
    tn_val_264 = 0
    
    tp_val_409 = 0
    fp_val_409 = 0
    fn_val_409 = 0
    tn_val_409 = 0
    
    # open all test squares
    test_sqs_path = '../data/test_squares.shp'
    test_sqs = geopandas.read_file(test_sqs_path)
    
    # define test squares
    sq_264 = test_sqs[test_sqs['id_square']=='264']['geometry'].iloc[0]
    sq_409 = test_sqs[test_sqs['id_square']=='409']['geometry'].iloc[0]
   
    # loop over all tiles
    with torch.no_grad():
        for idx, (data_test, targets_test, filename) in enumerate(loader):
            # calculate performance metrics for hand labelled test data
            # extract coordinates of lowerleft corner from filename
            ll_x = float(filename[0].split('_', 3)[0])
            ll_y = float(filename[0].split('_', 3)[1][:-3])
            point_ll = geopandas.GeoSeries([Point(ll_x,ll_y)])
            # check what square the tile is in
            if point_ll.within(sq_264).values == True:
                sq = '264'       
            if point_ll.within(sq_409).values == True:
                sq = '409'
            if (sq == '264' or sq=='409'):
                # send data and targets to device
                data_test = data_test.to(device=device)
                targets_test = targets_test.unsqueeze(1).to(device=device) 

                # apply trained model to validation data
                predictions_test = torch.sigmoid(model(data_test))
                # transform predictions (probabilities) to binary predictions
                predictions_binary_test = (predictions_test > 0.5).float()

                # cacluate performance metrics (ignores targets of -1)
                tp_tensor = torch.logical_and(predictions_binary_test == 1.0, targets_test == 1.0)
                fp_tensor = torch.logical_and(predictions_binary_test == 1.0, targets_test == 0.0)
                fn_tensor = torch.logical_and(predictions_binary_test == 0.0, targets_test == 1.0)
                tn_tensor = torch.logical_and(predictions_binary_test == 0.0, targets_test == 0.0)
            
                # sum values --> separately for both squares
                if sq == '264':
                    tp_val_264 += tp_tensor.sum()
                    fp_val_264 += fp_tensor.sum()
                    fn_val_264 += fn_tensor.sum()
                    tn_val_264 += tn_tensor.sum()
                
                if sq == '409':
                    tp_val_409 += tp_tensor.sum()
                    fp_val_409 += fp_tensor.sum()
                    fn_val_409 += fn_tensor.sum()
                    tn_val_409 += tn_tensor.sum()
                
        # calculate performance metrics for total
        precision_264, accuracy_264, recall_264, iu_264 = perf_metrics(tp_val_264, fp_val_264, tn_val_264, fn_val_264)
        precision_409, accuracy_409, recall_409, iu_409 = perf_metrics(tp_val_409, fp_val_409, tn_val_409, fn_val_409)
    
    return(precision_264, accuracy_264, recall_264, iu_264, 
           precision_409, accuracy_409, recall_409, iu_409)

# %%
# set loader and device
device = "cuda" if torch.cuda.is_available() else "cpu"
# object to load
test_ds = BIADataset(
    image_dir='../train_tiles/test_images_clean',
    target_dir='../train_tiles/test_targets_clean',
    transform_p=0.0,
    normalize_elevation=True,
)
# define dataloader test data
test_loader = DataLoader(
    test_ds,
    batch_size=1,
    num_workers=0,
    pin_memory=True,
    shuffle=False,
)

# predefine dataframe to fill with values
test_results = pd.DataFrame(columns=[
                                'precision_264',
                                'precision_409',
                                'recall_264',
                                'recall_409'])
# run over all 5 trained CNNs
for run in range(0,5,1):
    # import model
    if run == 0:
        name_run = f'hp_opt_1'
    else:
        name_run = f'final_model_{run}'
    model = UNET(in_channels=10, out_channels=1).to(device)
    # import trained parameters
    load_checkpoint(name_run, model)
    model.eval()
    precision_264, accuracy_264, recall_264, iu_264, precision_409, accuracy_409, recall_409, iu_409 = test_clean(test_loader,model,device)
    df_line = pd.DataFrame({'precision_264':precision_264.item(),
                            'precision_409':precision_409.item(),
                            'recall_264':recall_264.item(),
                            'recall_409':recall_409.item()}, index = [name_run])
    test_results = pd.concat([test_results,df_line])
    print((accuracy_264+accuracy_409)/2)
#%%
# append f1 score to dataframe
test_results['f1_264'] = (2*test_results['precision_264']*test_results['recall_264'])/(test_results['precision_264']+test_results['recall_264'])
test_results['f1_409'] = (2*test_results['precision_409']*test_results['recall_409'])/(test_results['precision_409']+test_results['recall_409'])
# append average scores to dataframe
test_results['average_precision'] = (test_results['precision_264']+test_results['precision_409'])/2
test_results['average_recall'] = (test_results['recall_264']+test_results['recall_409'])/2
test_results['average_f1'] = (test_results['f1_264']+test_results['f1_409'])/2
# print mean and std (Table 1)
print(test_results.mean())
print(test_results.std())
# %%
# calculate reference performance
# open data tiles clean labels and noisy labels
filenames = os.listdir('../train_tiles/test_targets_clean')

# preset total performance metrics for both validation squares with clean labels
tp_val_264 = 0
fp_val_264 = 0
fn_val_264 = 0
tn_val_264 = 0

tp_val_409 = 0
fp_val_409 = 0
fn_val_409 = 0
tn_val_409 = 0
# open all test squares
test_sqs_path = '../data/test_squares.shp'
test_sqs = geopandas.read_file(test_sqs_path)

# define test squares
sq_264 = test_sqs[test_sqs['id_square']=='264']['geometry'].iloc[0]
sq_409 = test_sqs[test_sqs['id_square']=='409']['geometry'].iloc[0]

# calculate performance metrics
for filename in filenames:
    # extract coordinates of lowerleft corner from filename
    ll_x = float(filename.split('_', 3)[0])
    ll_y = float(filename.split('_', 3)[1][:-3])
    point_ll = geopandas.GeoSeries([Point(ll_x,ll_y)]) 
    # 
    if point_ll.within(sq_264).values == True:
        sq = '264'       
    if point_ll.within(sq_409).values == True:
        sq = '409'
    
    if (sq == '264' or sq=='409'):
        clean_labels = xr.open_dataset(f'../train_tiles/test_targets_clean/{filename}').rename({'target':'clean_target'})
        noisy_labels = xr.open_dataset(f'../train_tiles/test_targets/{filename}').rename({'target':'noisy_target'})
        # merge two datasets
        labels = xr.merge([clean_labels,noisy_labels])

        # calculate tp, fp, tn, fn
        tp = sum(sum(xr.where(((labels['clean_target']==1) & (labels['noisy_target']==1)),1,0)))
        fp = sum(sum(xr.where(((labels['clean_target']==0) & (labels['noisy_target']==1)),1,0)))
        tn = sum(sum(xr.where(((labels['clean_target']==0) & (labels['noisy_target']==0)),1,0)))
        fn = sum(sum(xr.where(((labels['clean_target']==1) & (labels['noisy_target']==0)),1,0)))
  
        # sum values --> separately for both squares
        if sq == '264':
            tp_val_264 += tp.values
            fp_val_264 += fp.values
            fn_val_264 += fn.values
            tn_val_264 += tn.values
        
        if sq == '409':
            tp_val_409 += tp.values
            fp_val_409 += fp.values
            fn_val_409 += fn.values
            tn_val_409 += tn.values
        
# calculate performance metrics for total
precision_264, accuracy_264, recall_264, iu_264 = perf_metrics(tp_val_264, fp_val_264, tn_val_264, fn_val_264)
precision_409, accuracy_409, recall_409, iu_409 = perf_metrics(tp_val_409, fp_val_409, tn_val_409, fn_val_409)
#%%
f1_ref_264 = (2*precision_264*recall_264)/(precision_264+recall_264)
f1_ref_409 = (2*precision_409*recall_409)/(precision_409+recall_409)
f1_ref_test = (f1_ref_264 + f1_ref_409)/2

prec_ref_test = (precision_264+precision_409)/2
recall_ref_test = (recall_264+recall_409)/2
accuracy_ref_test = (accuracy_264+accuracy_409)/2
# %%
# compare to existing labels [VALIDATION DATA]
# open data tiles clean labels and noisy labels
filenames = os.listdir('../train_tiles/val_targets_clean')

# preset total performance metrics for both validation squares with clean labels
tp_val_246 = 0
fp_val_246 = 0
fn_val_246 = 0
tn_val_246 = 0

tp_val_278 = 0
fp_val_278 = 0
fn_val_278 = 0
tn_val_278 = 0
# open all validation squares
val_sqs_path = '../data/validation_squares.shp'
val_sqs = geopandas.read_file(val_sqs_path)

# define validation squares
sq_246 = val_sqs[val_sqs['id_square']=='246']['geometry'].iloc[0]
sq_278 = val_sqs[val_sqs['id_square']=='278']['geometry'].iloc[0]

# calculate performance metrics
for filename in filenames:
    print(filename)
    # extract coordinates of lowerleft corner from filename
    ll_x = float(filename.split('_', 3)[0])
    ll_y = float(filename.split('_', 3)[1][:-3])
    point_ll = geopandas.GeoSeries([Point(ll_x,ll_y)]) 
    # check what square the tile is in
    if point_ll.within(sq_246).values == True:
        sq = '246'       
    if point_ll.within(sq_278).values == True:
        sq = '278'
    # open tile with hand labels ("clean target") and with existing labels ("noiy target")
    if (sq == '246' or sq=='278'):
        clean_labels = xr.open_dataset(f'../train_tiles/val_targets_clean/{filename}').rename({'target':'clean_target'})
        noisy_labels = xr.open_dataset(f'../train_tiles/val_targets/{filename}').rename({'target':'noisy_target'})
        # merge two datasets
        labels = xr.merge([clean_labels,noisy_labels])

        # calculate tp, fp, tn, fn
        tp = sum(sum(xr.where(((labels['clean_target']==1) & (labels['noisy_target']==1)),1,0)))
        fp = sum(sum(xr.where(((labels['clean_target']==0) & (labels['noisy_target']==1)),1,0)))
        tn = sum(sum(xr.where(((labels['clean_target']==0) & (labels['noisy_target']==0)),1,0)))
        fn = sum(sum(xr.where(((labels['clean_target']==1) & (labels['noisy_target']==0)),1,0)))
  
        # sum values --> separately for both squares
        if sq == '246':
            tp_val_246 += tp.values
            fp_val_246 += fp.values
            fn_val_246 += fn.values
            tn_val_246 += tn.values
        
        if sq == '278':
            tp_val_278 += tp.values
            fp_val_278 += fp.values
            fn_val_278 += fn.values
            tn_val_278 += tn.values
        
# calculate performance metrics for total
precision_246, accuracy_246, recall_246, iu_246 = perf_metrics(tp_val_246, fp_val_246, tn_val_246, fn_val_246)
precision_278, accuracy_278, recall_278, iu_278 = perf_metrics(tp_val_278, fp_val_278, tn_val_278, fn_val_278)
# calculate f1 score per tile
f1_ref_246 = (2*precision_246*recall_246)/(precision_246+recall_246)
f1_ref_278 = (2*precision_278*recall_278)/(precision_278+recall_278)
# calculate average performance metrics (Table 1)
f1_ref_val = (f1_ref_246 + f1_ref_278)/2
prec_ref_val = (precision_246+precision_278)/2
recall_ref_val = (recall_246+recall_278)/2
accuracy_ref_val = (accuracy_246+accuracy_278)/2

#%%
# estimate performance in validation areas
def test_clean(loader, 
               model, 
               device="cuda",
    ):
    # set model to evaluation mode
    model.eval()
    
    # preset total performance metrics for both validation squares with clean labels
    tp_val_246 = 0
    fp_val_246 = 0
    fn_val_246 = 0
    tn_val_246 = 0
    
    tp_val_278 = 0
    fp_val_278 = 0
    fn_val_278 = 0
    tn_val_278 = 0
    
    # open all test squares
    val_sqs_path = '../data/val_squares.shp'
    val_sqs = geopandas.read_file(val_sqs_path)
    
    # define test squares
    sq_246 = test_sqs[test_sqs['id_square']=='246']['geometry'].iloc[0]
    sq_278 = test_sqs[test_sqs['id_square']=='278']['geometry'].iloc[0]
   
    # loop over all tiles
    with torch.no_grad():
        for idx, (data_test, targets_test, filename) in enumerate(loader):
            # calculate performance metrics for hand labelled test data
            # extract coordinates of lowerleft corner from filename
            ll_x = float(filename[0].split('_', 3)[0])
            ll_y = float(filename[0].split('_', 3)[1][:-3])
            point_ll = geopandas.GeoSeries([Point(ll_x,ll_y)])
            # check what square the tile is in
            if point_ll.within(sq_246).values == True:
                sq = '246'       
            if point_ll.within(sq_278).values == True:
                sq = '278'
            if (sq == '246' or sq=='278'):
                # send data and targets to device
                data_test = data_test.to(device=device)
                targets_test = targets_test.unsqueeze(1).to(device=device) 

                # apply trained model to validation data
                predictions_test = torch.sigmoid(model(data_test))
                # transform predictions (probabilities) to binary predictions
                predictions_binary_test = (predictions_test > 0.5).float()

                # cacluate performance metrics (ignores targets of -1)
                tp_tensor = torch.logical_and(predictions_binary_test == 1.0, targets_test == 1.0)
                fp_tensor = torch.logical_and(predictions_binary_test == 1.0, targets_test == 0.0)
                fn_tensor = torch.logical_and(predictions_binary_test == 0.0, targets_test == 1.0)
                tn_tensor = torch.logical_and(predictions_binary_test == 0.0, targets_test == 0.0)
            
                # sum values --> separately for both squares
                if sq == '246':
                    tp_val_246 += tp_tensor.sum()
                    fp_val_246 += fp_tensor.sum()
                    fn_val_246 += fn_tensor.sum()
                    tn_val_246 += tn_tensor.sum()
                
                if sq == '278':
                    tp_val_278 += tp_tensor.sum()
                    fp_val_278 += fp_tensor.sum()
                    fn_val_278 += fn_tensor.sum()
                    tn_val_278 += tn_tensor.sum()
                
        # calculate performance metrics for total
        precision_246, accuracy_246, recall_246, iu_246 = perf_metrics(tp_val_246, fp_val_246, tn_val_246, fn_val_246)
        precision_278, accuracy_278, recall_278, iu_278 = perf_metrics(tp_val_278, fp_val_278, tn_val_278, fn_val_278)
    
    return(precision_246, accuracy_246, recall_246, iu_246, 
           precision_278, accuracy_278, recall_278, iu_278)

# %%
# set loader and device
device = "cuda" if torch.cuda.is_available() else "cpu"
# object to load
val_ds = BIADataset(
    image_dir='../train_tiles/val_images_clean',
    target_dir='../train_tiles/val_targets_clean',
    transform_p=0.0,
    normalize_elevation=True,
)
# define dataloader test data
val_loader = DataLoader(
    val_ds,
    batch_size=1,
    num_workers=0,
    pin_memory=True,
    shuffle=False,
)

# predefine dataframe to fill with values
val_results = pd.DataFrame(columns=[
                                'precision_246',
                                'precision_278',
                                'recall_246',
                                'recall_278'])
# run over all 5 trained CNNs
for run in range(0,5,1):
    # import model
    if run == 0:
        name_run = f'hp_opt_1'
    else:
        name_run = f'final_model_{run}'
    model = UNET(in_channels=10, out_channels=1).to(device)
    # import trained parameters
    load_checkpoint(name_run, model)
    model.eval()
    precision_246, accuracy_246, recall_246, iu_246, precision_278, accuracy_278, recall_278, iu_278 = test_clean(test_loader,model,device)
    df_line = pd.DataFrame({'precision_246':precision_246.item(),
                            'precision_278':precision_278.item(),
                            'recall_246':recall_246.item(),
                            'recall_278':recall_278.item()}, index = [name_run])
    val_results = pd.concat([test_results,df_line])
#%%
# append f1 score to dataframe
val_results['f1_246'] = (2*val_results['precision_246']*val_results['recall_246'])/(val_results['precision_246']+val_results['recall_246'])
val_results['f1_278'] = (2*val_results['precision_278']*val_results['recall_278'])/(val_results['precision_278']+val_results['recall_278'])
# append average scores to dataframe
val_results['average_precision'] = (val_results['precision_264']+val_results['precision_409'])/2
val_results['average_recall'] = (val_results['recall_264']+val_results['recall_409'])/2
val_results['average_f1'] = (val_results['f1_264']+val_results['f1_409'])/2
# print mean and std (Table 1)
print(val_results.mean())
print(val_results.std())
#%%
# compare to post processed/ensemble averaged results
# read in final BIA map
BIA_map = xr.open_dataset('../output/final_map.nc')
resolution = 200

ll_x_main = BIA_map.x.min().values
ur_y_main = BIA_map.y.max().values

BIA_map.attrs['transform'] = affine.Affine(resolution, 0.0, ll_x_main-(resolution/2), 0.0, -1*resolution, ur_y_main+(resolution/2))
#%%
# open BIAs (="noisy" labels)
BIAs_path = r'../data/BlueIceAreas.shx'
BIAs_raw = geopandas.read_file(BIAs_path)

# rasterize polygons: create mask where only pixels whose center is within the polygon will be burned in.
ShapeMask = features.geometry_mask(BIAs_raw.geometry,
                                        out_shape=(len(BIA_map.y), len(BIA_map.x)),
                                        transform=BIA_map.transform,
                                        invert=True)

# assign coordinates and create a data array
ShapeMask = xr.DataArray(ShapeMask, coords={"y":BIA_map.y[::-1],
                                                "x":BIA_map.x},
                            dims=("y", "x"))
# flip shapemask upside down
ShapeMask= ShapeMask[::-1]
# Create Data Array with zeros
zeros_tomask = xr.zeros_like(BIA_map['mean'])
# apply Mask to zeros_tomask --> 1 = BIA, 0 = no BIA
BIAs_mask = xr.where((ShapeMask == True),1,zeros_tomask)
# add BIAs_mask to dataset
BIA_map['BIAs'] = BIAs_mask
#del(BIAs_mask, ShapeMask, zeros_tomask)

# open hand labels
sq_246_path = r'../data/handlabeled_sq246.shx'
sq_265_path = r'../data/handlabeled_sq265.shx'
sq_278_path = r'../data/handlabeled_sq278.shx'
sq_264_path = r'../data/handlabelled_sq264.shx'
sq_409_path = r'../data/handlabelled_sq409.shx'
sq_246 = geopandas.read_file(sq_246_path)
sq_265 = geopandas.read_file(sq_265_path)
sq_278 = geopandas.read_file(sq_278_path)
sq_264 = geopandas.read_file(sq_264_path)
sq_409 = geopandas.read_file(sq_409_path)
# merge handlabeled tiles
handlabels_merged = geopandas.GeoDataFrame(pd.concat([sq_246, sq_265, sq_278, sq_264, sq_409]))

# rasterize polygons: create mask where only pixels whose center is within the polygon will be burned in.
ShapeMask_clean = features.geometry_mask(handlabels_merged.geometry,
                                        out_shape=(len(BIA_map.y), len(BIA_map.x)),
                                        transform=BIA_map.transform,
                                        invert=True)

# assign coordinates and create a data array
ShapeMask_clean = xr.DataArray(ShapeMask_clean, coords={"y":BIA_map.y[::-1],
                                                "x":BIA_map.x},
                            dims=("y", "x"))
# flip shapemask upside down
ShapeMask_clean= ShapeMask_clean[::-1]
# Create Data Array with zeros
zeros_tomask = xr.zeros_like(BIA_map['mean'])
# apply Mask to zeros_tomask --> 1 = BIA, 0 = no BIA
BIAs_mask_clean = xr.where((ShapeMask_clean == True),1,zeros_tomask)

# add BIAs_mask to dataset
BIA_map['BIAs_clean'] = BIAs_mask_clean
del(BIAs_mask_clean, ShapeMask_clean, zeros_tomask)

# create targets for missing values (equal to -1) for 1. ice boundaries 2. input data
# read in ice boundaries
ice_boundaries_path = r'../data/IceBoundaries_Antarctica_v2.shx'
ice_boundaries_raw = geopandas.read_file(ice_boundaries_path)
# merge ice boundaries
ice_boundaries = ice_boundaries_raw['geometry'].unary_union
# rasterize polygons: create mask where only pixels whose center is within the polygon will be burned in.
ShapeMask = features.geometry_mask([ice_boundaries],
                                      out_shape=(len(BIA_map.y), len(BIA_map.x)),
                                      transform=BIA_map.transform,
                                      invert=True)
# assign coordinates and create a data array
ShapeMask = xr.DataArray(ShapeMask , coords={"y":BIA_map.y[::-1],
                                             "x":BIA_map.x},
                         dims=("y", "x"))
# flip shapemask upside down
ShapeMask= ShapeMask[::-1]
# Create Data Array with zeros
zeros_tomask = xr.zeros_like(BIA_map['mean'])
# apply Mask to zeros_tomask --> -1 = outside ice boundaries, 0 = inside ice boundaries
ice_boundaries_mask = xr.where((ShapeMask == True),zeros_tomask,-1)
# add missing data to dataset
BIA_map['missing_data'] = ice_boundaries_mask
del(ice_boundaries_mask,ShapeMask,zeros_tomask)

# combine mask of BIAs with the missing_data mask
BIA_map['target'] = xr.where(BIA_map['missing_data']==-1, -1, BIA_map['BIAs'])
BIA_map['target_clean'] = xr.where(BIA_map['missing_data']==-1, -1, BIA_map['BIAs_clean'])
BIA_map = BIA_map.drop(['BIAs','missing_data','BIAs_clean'])

#%%
# function to assign data to tile
def sel_area(square,dataset):
    min_x, min_y, max_x, max_y = square.exterior.bounds
    # obtain arreas with values for x and y coordinates
    mask_x = (dataset.x >= min_x) & (dataset.x < max_x)
    mask_y = (dataset.y >= min_y) & (dataset.y < max_y)
    # crop image based on arrays with values for x and y coordinates
    img_cropped = dataset.where(mask_x & mask_y, drop=True)
    return img_cropped
# loop over each square area
# select area based on extent (read in validation and test squares)
# open all validation squares
val_sqs_path = '../data/validation_squares.shp'
val_sqs = geopandas.read_file(val_sqs_path)
# define validation squares
sq_246 = val_sqs[val_sqs['id_square']=='246']['geometry'].iloc[0]
sq_278 = val_sqs[val_sqs['id_square']=='278']['geometry'].iloc[0]
# open all test squares
test_sqs_path = '../data/test_squares.shp'
test_sqs = geopandas.read_file(test_sqs_path)
# define test squares
sq_264 = test_sqs[test_sqs['id_square']=='264']['geometry'].iloc[0]
sq_409 = test_sqs[test_sqs['id_square']=='409']['geometry'].iloc[0]

# define empty dataframe for postprocessed results
pp_results = pd.DataFrame(columns=[
                                'precision',
                                'recall',
                                'f1',
                                'f1_ref'])

sq_list = ['sq_246','sq_278','sq_264','sq_409']
# loop over squares
for idx, square in enumerate([sq_246,sq_278,sq_264,sq_409]):
    print(sq_list[idx])
    preds = sel_area(square,BIA_map)

    # calculate performance metrics
    tp = sum(sum(xr.where(((preds['target_clean']==1) & (preds['mean']>=0.5)),1,0))).values
    fp = sum(sum(xr.where(((preds['target_clean']==0) & (preds['mean']>=0.5)),1,0))).values
    tn = sum(sum(xr.where(((preds['target_clean']==0) & (preds['mean']<0.5)),1,0))).values
    fn = sum(sum(xr.where(((preds['target_clean']==1) & (preds['mean']<0.5)),1,0))).values
    # calculate reference metrics
    tp_ref = sum(sum(xr.where(((preds['target_clean']==1) & (preds['target']==1)),1,0))).values
    fp_ref = sum(sum(xr.where(((preds['target_clean']==0) & (preds['target']==1)),1,0))).values
    tn_ref = sum(sum(xr.where(((preds['target_clean']==0) & (preds['target']==0)),1,0))).values
    fn_ref = sum(sum(xr.where(((preds['target_clean']==1) & (preds['target']==0)),1,0))).values
    # calculate performance metrics
    precision, accuracy, recall, iu = perf_metrics(tp, fp, tn, fn)
    precision_ref, accuracy_ref, recall_ref, iu_ref = perf_metrics(tp_ref, fp_ref, tn_ref, fn_ref)
    f1 = (2*precision*recall)/(precision+recall)
    f1_ref = (2*precision_ref*recall_ref)/(precision_ref+recall_ref)
    # print results (not reported in study)
    print(np.round(precision,2),np.round(recall,2),np.round(f1,2),np.round(precision_ref,2),np.round(recall_ref,2),np.round(f1_ref,2))
    df_line = pd.DataFrame({'precision':precision,
                            'recall':recall,
                            'f1':f1,
                            'f1_ref':f1_ref}, index = [sq_list[idx]])
    pp_results = pd.concat([pp_results,df_line])

# %%
# calculate ROC curve of validation data to see if 0.5 is a valid threhsold
# estimate performance in validation areas
def val_clean(loader, 
              model, 
              device="cuda",
              threshold=0.5,
    ):
    # set model to evaluation mode
    model.eval()
    
    # preset total performance metrics for both validation squares with clean labels
    tp_val_246 = 0
    fp_val_246 = 0
    fn_val_246 = 0
    tn_val_246 = 0
    
    tp_val_278 = 0
    fp_val_278 = 0
    fn_val_278 = 0
    tn_val_278 = 0
    
    # open all val squares
    val_sqs_path = '../data/validation_squares.shp'
    val_sqs = geopandas.read_file(val_sqs_path)
    
    # define val squares
    sq_246 = val_sqs[val_sqs['id_square']=='246']['geometry'].iloc[0]
    sq_278 = val_sqs[val_sqs['id_square']=='278']['geometry'].iloc[0]

    # loop over all tiles
    with torch.no_grad():
        for idx, (data_val, targets_val, filename) in enumerate(loader):
            # calculate performance metrics for hand labelled validation data
            # extract coordinates of lowerleft corner from filename
            ll_x = float(filename[0].split('_', 3)[0])
            ll_y = float(filename[0].split('_', 3)[1][:-3])
            point_ll = geopandas.GeoSeries([Point(ll_x,ll_y)])
            sq = 'none'
            # check what square the tile is in
            if point_ll.within(sq_246).values == True:
                sq = '246'       
            if point_ll.within(sq_278).values == True:
                sq = '278'
            if (sq == '246' or sq=='278'):
                # send data and targets to device
                data_val = data_val.to(device=device)
                targets_val = targets_val.unsqueeze(1).to(device=device) 

                # apply trained model to validation data
                predictions_val = torch.sigmoid(model(data_val))
                # transform predictions (probabilities) to binary predictions
                predictions_binary_val = (predictions_val > threshold).float()

                # cacluate performance metrics (ignores targets of -1)
                tp_tensor = torch.logical_and(predictions_binary_val == 1.0, targets_val == 1.0)
                fp_tensor = torch.logical_and(predictions_binary_val == 1.0, targets_val == 0.0)
                fn_tensor = torch.logical_and(predictions_binary_val == 0.0, targets_val == 1.0)
                tn_tensor = torch.logical_and(predictions_binary_val == 0.0, targets_val == 0.0)
            
                # sum values --> separately for both squares
                if sq == '246':
                    tp_val_246 += tp_tensor.sum()
                    fp_val_246 += fp_tensor.sum()
                    fn_val_246 += fn_tensor.sum()
                    tn_val_246 += tn_tensor.sum()
                
                if sq == '278':
                    tp_val_278 += tp_tensor.sum()
                    fp_val_278 += fp_tensor.sum()
                    fn_val_278 += fn_tensor.sum()
                    tn_val_278 += tn_tensor.sum()
        
        # calculate performance metrics for squares
        precision_246, accuracy_246, recall_246, iu_246 = perf_metrics(tp_val_246, fp_val_246, tn_val_246, fn_val_246)
        precision_278, accuracy_278, recall_278, iu_278 = perf_metrics(tp_val_278, fp_val_278, tn_val_278, fn_val_278)
        # calculate f1 score for squares
        f1_246 = (2*precision_246*recall_246)/(precision_246+recall_246)
        f1_278 = (2*precision_278*recall_278)/(precision_278+recall_278)
        # calculate tp/fp rates
        tp_rate_246 = tp_val_246/(tp_val_246+fn_val_246)
        fp_rate_246 = fp_val_246/(tn_val_246+fp_val_246)
        tp_rate_278 = tp_val_278/(tp_val_278+fn_val_278)
        fp_rate_278 = fp_val_278/(tn_val_278+fp_val_278)
        
    # return(precision_246, recall_246,
    #        precision_278, recall_278)
    return(tp_rate_246,fp_rate_246,tp_rate_278,fp_rate_278)
    #return((f1_246+f1_278)/2)

# set loader and device
device = "cuda" if torch.cuda.is_available() else "cpu"
# object to load
val_ds = BIADataset(
    image_dir='../train_tiles/val_images_clean',
    target_dir='../train_tiles/val_targets_clean',
    transform_p=0.0,
    normalize_elevation=True,
)
# define dataloader test data
val_loader = DataLoader(
    val_ds,
    batch_size=1,
    num_workers=0,
    pin_memory=True,
    shuffle=False,
)


# define threhsolds
thresholds = np.linspace(0.3,0.7,9)

# run over all 5 trained CNNs
for run in range(0,5,1):
    # predefine dataframe to fill with values
    val_ROC = pd.DataFrame(columns=[
                                'tp_246',
                                'tp_278',
                                'fp_246',
                                'fp_278'])
    # import model
    if run == 0:
        name_run = f'hp_opt_1'
    else:
        name_run = f'final_model_{run}'
    model = UNET(in_channels=10, out_channels=1).to(device)
    # import trained parameters
    load_checkpoint(name_run, model)
    # set model to evaluation mode
    model.eval()
    # loop over different thresholds
    for threshold in thresholds:
        print(threshold)
        tp_rate_246,fp_rate_246,tp_rate_278,fp_rate_278 = val_clean(val_loader,model,device,threshold)
        # append values to dataframe
        df_line = pd.DataFrame({'tp_246':tp_rate_246.item(),
                                'tp_278':tp_rate_278.item(),
                                'fp_246':fp_rate_246.item(),
                                'fp_278':fp_rate_278.item()}, index = [f'{str(threshold)}'])
        val_ROC = pd.concat([val_ROC,df_line])
    # plot ROC curve
    plt.scatter((val_ROC['fp_246']+val_ROC['fp_278'])/2,
              (val_ROC['tp_246']+val_ROC['tp_278'])/2,label=name_run)
# plot legend and save figure
plt.legend()
plt.savefig('../output/figures/F1_score.png')

    