# script to generate tiles for training, validation, and testing
import gc
gc.collect()
# import packages
import numpy as np
import xarray as xr
import os
import pandas as pd

import geopandas
from rasterio import features

from shapely.geometry.polygon import Polygon
import random

from multiprocessing import Pool

from xrspatial import slope
from xrspatial import aspect

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

#%%
# load squares that have been generated before or not
load_squares = True

# define 200m-resolution grid
resolution = 200 #m
# define extent of grid
ll_x_main = -2716100
ll_y_main = -2600100
ur_x_main = 2836100
ur_y_main = 2390100
# calculate number of entries
n_x = ((ur_x_main - ll_x_main)/resolution) + 1
n_y = ((ur_y_main - ll_y_main)/resolution) + 1
# create large empty dataset (ds) to store all values
ds_main = xr.Dataset(data_vars=dict(
                        zeros=(["y","x"],np.zeros((int(n_y),int(n_x))).astype('float32')),
                        )
                    )
# define coordinates (xmin, xmax, resolution) (ymin,ymax,resolution)
x_coords_main = np.arange(ll_x_main,ur_x_main+resolution,resolution).astype('int32')
y_coords_main = np.arange(ll_y_main,ur_y_main+resolution,resolution)[::-1].astype('int32')
# assign coordinates to large empty dataset
ds_main['x'] = x_coords_main
ds_main['y'] = y_coords_main

# align different datasets to continent-wide grid

# ELEVATION
# open TDX DEM
TDX_path = r'../data/TDM_merged.tif'
TDX_raw = xr.open_rasterio(TDX_path)

# convert DataArray to DataSet
TDX_ds = TDX_raw.drop('band')[0].to_dataset(name='DEM_ellipsoid')

# interpolate TDX DEM to the main grid
interpolated_TDX = TDX_ds.interp_like(ds_main)

# open geoid
geoid_path = r'../data/EIGEN-6C4_HeightAnomaly_10km.tif'
geoid_raw = xr.open_rasterio(geoid_path)
geoid_raw.attrs['nodatavals'] = (np.nan,)

# convert DataArray to DataSet
geoid_ds = geoid_raw.drop('band')[0].to_dataset(name='geoid')
# reset nodatavalues to nan (to avoid artifacts in interpolation)
geoid_ds = geoid_ds.where(geoid_ds['geoid'] != -9999.)

# interpolate geoid height to the main grid
interpolated_geoid = geoid_ds.interp_like(ds_main)

# calculate orthometric height
TDX_ortho = (interpolated_TDX.DEM_ellipsoid - interpolated_geoid.geoid).astype('float32')
TDX_ortho_ds = TDX_ortho.to_dataset(name='DEM')

# merge data to continent-wide dataset
ds_main = xr.merge([ds_main,TDX_ortho_ds])

# delete intermediate variables
del(TDX_path, TDX_raw, TDX_ds, interpolated_TDX, 
    geoid_path, geoid_raw, geoid_ds, interpolated_geoid,
    TDX_ortho, TDX_ortho_ds)
# drop zeros from dataset
ds_main = ds_main.drop('zeros')

# calculate slope and aspect (not used in final model)
ds_main['slope'] = slope(ds_main.DEM)
ds_main['aspect'] = aspect(ds_main.DEM)

# correct aspect wrt true north
north_direction = (90 - 180*np.arctan2(ds_main.y,ds_main.x)/(np.pi)) % 360 #--> solar angle in rad, aspect in degrees
corrected_aspect = (abs(((ds_main['aspect'] - north_direction + 180) % 360 ) - 180)).astype('float32')

ds_main['corrected_aspect'] = corrected_aspect

# delete intermediate variables
ds_main = ds_main.drop('aspect')
del(north_direction, corrected_aspect)

# RADAR
# open radar data
RS2_path = r'../data/RS2_32bit_100m_mosaic_HH_clip.tif'
RS2_raw = xr.open_rasterio(RS2_path)

# convert DataArray to DataSet
RS2_ds = RS2_raw.drop('band')[0].to_dataset(name='radar')

# reset nodatavalues to nan (to avoid artifacts in interpolation)
RS2_ds = RS2_ds.where(RS2_ds['radar'] != 0.)

# interpolate RS2 to the main grid
interpolated_radar = (RS2_ds.interp_like(ds_main)).astype('float32')

# merge data to continent-wide dataset
ds_main = xr.merge([ds_main,interpolated_radar])

# delete intermediate variables
del(RS2_path,RS2_raw,RS2_ds,interpolated_radar)

# open MODIS mosaic (MOA)
MOA_path = r'../data/moa125_2009_hp1_v02.0.tif'
MOA_raw = xr.open_rasterio(MOA_path)

# convert DataArray to DataSet
MOA_ds = MOA_raw.drop('band')[0].to_dataset(name='modis')

# reset nodatavalues to nan (to avoid artifacts in interpolation)
MOA_ds = MOA_ds.where(MOA_ds['modis'] != 0)

# interpolate MODIS to the main grid
interpolated_moa = (MOA_ds.interp_like(ds_main)).astype('float32')

# merge data to continent-wide dataset
ds_main = xr.merge([ds_main,interpolated_moa])

# delete intermediate variables
del(MOA_path, MOA_raw, MOA_ds, interpolated_moa)

# open MODIS multiband data
MOD_path = r'../data/merged_bands_composite3031080910.tif'
MOD_raw = xr.open_rasterio(MOD_path)
MOD_raw.attrs['nodatavals'] = (np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)
# loop over bands and append them to dataset
MOD_ds = xr.Dataset()
for i in range(1,8):
    MOD_1band_ds = MOD_raw[MOD_raw.band==i].drop('band')[0].to_dataset(name=f'MOD_B{i}').astype('float32')
    # reset nodatavalues to nan (to avoid artifacts in interpolation)
    MOD_1band_ds = MOD_1band_ds.where(MOD_1band_ds[f'MOD_B{i}'] != 0)
    MOD_ds = xr.merge([MOD_ds,MOD_1band_ds])

# interpolate MODIS multiband data to main grid
MOD_interpolated = MOD_ds.interp_like(ds_main).astype('float32')

# add data to other variables
ds_main = xr.merge([ds_main,MOD_interpolated])

# delete intermediate variables
del(MOD_interpolated, MOD_ds, MOD_1band_ds, MOD_raw,MOD_path)


# export ds_main
ds_main.attrs['transform'] = (resolution, 0.0, ll_x_main-(resolution/2), 0.0, -1*resolution, ur_y_main+(resolution/2))
ds_main.attrs['res'] = resolution
#ds_main.to_netcdf('../data/dataset_alignedtogrid.nc')

#%%
# open BIAs ("noisy" labels)
BIAs_path = r'../data/BlueIceAreas.shx'
BIAs_raw = geopandas.read_file(BIAs_path)
# calculate union of blue ice areas
BIAs_union = BIAs_raw['geometry'].unary_union

# open coastlines
CL_path = r'../data/ADD_Coastline_low_res_polygon.shx'
CL_raw = geopandas.read_file(CL_path)

# open ice boundaries
ice_boundaries_path = r'../data/IceBoundaries_Antarctica_v2.shx'
ice_boundaries_raw = geopandas.read_file(ice_boundaries_path)
# merge ice boundaries
ice_boundaries = ice_boundaries_raw['geometry'].unary_union

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


#%%
# generate locations of training and validation areas (squares) and generate 
# locations of tiles within these areas

# define n_pixels for each tile
n_pixels = 512
resolution = ds_main.res

# define directory to save images
train_dir = f'../train_tiles'
pred_dir = f'../pred_tiles'

# define size of areas to split up data in training and validation abd test (in m)
size_sq = 250000. #m

# generate random squares (e.g., Figure S2)
if load_squares == False:
    # create square areas that intersect with coastlines and overlap with BIAs
    # calculate bounds of coastlines
    minx_CL = np.floor(CL_raw.geometry.bounds.minx.min())
    miny_CL = np.floor(CL_raw.geometry.bounds.miny.min())
    maxx_CL = np.ceil(CL_raw.geometry.bounds.maxx.max())
    maxy_CL = np.ceil(CL_raw.geometry.bounds.maxy.max())
    
    # calculate number of squares
    n_images_x = np.floor((maxx_CL-minx_CL)/(size_sq))
    n_images_y = np.floor((maxy_CL-miny_CL)/(size_sq))
    
    # calculate residual to center residual symettrically
    resid_x = size_sq - (maxx_CL-minx_CL) % (size_sq)
    resid_y = size_sq - (maxy_CL-miny_CL) % (size_sq)
    
    # calculate lowerleft corners
    ll_x = np.arange(minx_CL - np.ceil(resid_x/2), maxx_CL, size_sq)
    ll_y = np.arange(miny_CL - np.ceil(resid_y/2), maxy_CL, size_sq)
    
    # create geodataframe to store rectangles
    all_sqs = geopandas.GeoDataFrame(columns=['id_square','square'], geometry='square')
    id_sq = 0
    # loop through all defined lowerleft corners to append to geodataframe
    for idx_x, ll_xi in enumerate(ll_x):
        for idx_y, ll_yi in enumerate(ll_y):
            id_sq += 1
            single_sq = Polygon([(ll_xi, ll_yi),
                                (ll_xi+size_sq, ll_yi),
                                (ll_xi+size_sq, ll_yi+size_sq),
                                (ll_xi, ll_yi+size_sq),
                                (ll_xi,ll_yi)])
            all_sqs = all_sqs.append({'id_square': id_sq, 'square': single_sq}, ignore_index=True)
    
    # check if squares intersects with the continents' outlines
    all_sqs['intersects_coastlines'] = geopandas.GeoSeries(all_sqs['square']).intersects(CL_raw['geometry'].unary_union)
    sqs = all_sqs[all_sqs.intersects_coastlines==True].reset_index(drop=True)
    
    # check if squares intersects with the BIA (="noisy") outlines
    sqs['intersects_BIAs'] = geopandas.GeoSeries(sqs['square']).intersects(BIAs_union)
    
    # select squares for training/validation/test
    sqs_shuffeled = sqs.sample(frac=1,random_state=30).reset_index(drop=True)
    
    # number of squares to have 20% of the area for validation and 20% for test
    n_sqs = len(sqs)
    n_sqs_val_test = np.ceil(n_sqs*0.2)
    
    # split up squares into train, validation, and test
    sqs_train = sqs_shuffeled[:int(-2*n_sqs_val_test)].reset_index(drop=True)
    sqs_val   = sqs_shuffeled[int(-2*n_sqs_val_test):int(-1*n_sqs_val_test)].reset_index(drop=True)
    sqs_test  = sqs_shuffeled[int(-1*n_sqs_val_test):].reset_index(drop=True)
    
    # drop squares without blue ice areas (according to "noisy" labels)
    sqs_train = sqs_train[sqs_train.intersects_BIAs==True].reset_index(drop=True)
    sqs_val   = sqs_val[sqs_val.intersects_BIAs==True].reset_index(drop=True)
    sqs_test  = sqs_test[sqs_test.intersects_BIAs==True].reset_index(drop=True)

    # export squares
    sqs_train.crs = 'epsg:3031'
    sqs_val.crs = 'epsg:3031'
    sqs_test.crs = 'epsg:3031'
    sqs_train.set_geometry(col='square',inplace=True)
    sqs_val.set_geometry(col='square',inplace=True)
    sqs_test.set_geometry(col='square',inplace=True)
    sqs_train.to_file('../data/train_squares.shp')
    sqs_val.to_file('../data/validation_squares.shp')
    sqs_test.to_file('../data/test_squares.shp')

# load squares that are defined earlier
if load_squares == True:
    sqs_train_path = r'../data/train_squares.shx'
    sqs_train = geopandas.read_file(sqs_train_path).rename(columns={'geometry':'square'}).set_geometry(col='square')
    
    sqs_val_path = r'../data/validation_squares.shx'
    sqs_val = geopandas.read_file(sqs_val_path).rename(columns={'geometry':'square'}).set_geometry(col='square')
    
    sqs_test_path = r'../data/test_squares.shx'
    sqs_test = geopandas.read_file(sqs_test_path).rename(columns={'geometry':'square'}).set_geometry(col='square')

# define sets of squares with handlabels (i.e., in training, validation, test squares)
sqs_train_noiseornot = sqs_train[sqs_train['id_square']=='265']
sqs_val_clean = sqs_val[(sqs_val['id_square']=='246') | (sqs_val['id_square']=='278')]
sqs_test_clean = sqs_test[(sqs_test['id_square']=='264') | (sqs_test['id_square']=='409')]
    
#%%
# define function that outputs polygon equalling the size of a tile when 
# providing coordinates of lowerleft corner
def poly(ll_x,ll_y, n_pixels=n_pixels, resolution=resolution):
    size_tile = n_pixels * resolution
    tile_poly = Polygon([(ll_x, ll_y),
                    (ll_x+size_tile, ll_y),
                    (ll_x+size_tile, ll_y+size_tile),
                    (ll_x, ll_y+size_tile),
                    (ll_x, ll_y)])
    return tile_poly

# fill up squares with "gridded" tiles
# caluclate number of tiles per square (in 1 dimension)
n_tiles_per_sq = np.ceil(size_sq/(n_pixels*resolution)) # in 1 dimension
# calculate length of overlap of tiles
length_overlap = n_tiles_per_sq*(n_pixels*resolution) - size_sq

# function that creates tiles within a given square
def tiles_in_square(ll_x_sq, ll_y_sq, id_square,
                    n_tiles_per_sq=n_tiles_per_sq,
                    n_pixels=n_pixels,
                    resolution=resolution,
                    length_overlap=length_overlap):
    tiles_in_sq = geopandas.GeoDataFrame(columns=['id_tile','tile'], geometry='tile')
    for tile_n_x in range(int(n_tiles_per_sq)):
        for tile_n_y in range(int(n_tiles_per_sq)):
            tiles_in_sq = tiles_in_sq.append({'id_tile': id_square, 'tile': poly(ll_x_sq + tile_n_x * (n_pixels*resolution -  length_overlap/(n_tiles_per_sq -1)),
                                                         ll_y_sq + tile_n_y * (n_pixels*resolution -  length_overlap/(n_tiles_per_sq -1)))},
                             ignore_index=True)
    return tiles_in_sq

# function to loop over squares to create tiles
def create_tiles(squares):
    tiles = geopandas.GeoDataFrame(columns=['id_tile','tile'], geometry='tile')    
    for j in range(len(squares)):
        ll_x_sq = squares['square'].iloc[j].exterior.coords[0][0]
        ll_y_sq = squares['square'].iloc[j].exterior.coords[0][1]
        tiles = tiles.append(tiles_in_square(ll_x_sq,ll_y_sq,squares['id_square'].iloc[j]))
    return tiles

# loop over training squares to create tiles
tiles_train = create_tiles(sqs_train)
tiles_train_noiseornot = create_tiles(sqs_train_noiseornot)

# loop over validation squares to create tiles
tiles_val = create_tiles(sqs_val).reset_index(drop=True)
tiles_val_clean = create_tiles(sqs_val_clean).reset_index(drop=True)
    
# loop over test squares to create tiles
tiles_test = create_tiles(sqs_test).reset_index(drop=True)
tiles_test_clean = create_tiles(sqs_test_clean).reset_index(drop=True)

# function to generate randomly shifted tiles
random.seed(10)
def create_random_tiles(number_extra_tiles, sqs_train=sqs_train):
    # predefine empty geodataframe to append tiles
    tiles_shifted = geopandas.GeoDataFrame(columns=['id_tile','tile'], geometry='tile')
    # create union of training squares
    sqs_train_union = sqs_train.unary_union
    # calculate bounds of training squares
    (minx, miny, maxx, maxy) = sqs_train_union.bounds
    while len(tiles_shifted) < number_extra_tiles:
        extra_tile = poly(np.round(random.uniform(minx, maxx)), np.round(random.uniform(miny, maxy)))
        if extra_tile.within(sqs_train_union):
            tiles_shifted = tiles_shifted.append({'id_tile': 1000, 'tile':extra_tile},
                                                 ignore_index=True)
        del(extra_tile)
    return tiles_shifted

# generate and append random tiles to training data (i.e., randomly shifted tiles)
tiles_shifted = create_random_tiles(len(tiles_train))
tiles_train = tiles_train.append(tiles_shifted).reset_index(drop=True)

#%%
# estimate the overlap of tiles in the validation data to later mask out these overlapping areas
# create geodataframe to store rectangles
nooverlap_tiles_val = geopandas.GeoDataFrame(columns=['id_square','nooverlap_tile'], geometry='nooverlap_tile')

# loop through validation squares
for index, row in sqs_val.iterrows():
    if load_squares == False:
        nooverlap_ll_x = np.linspace(row.square.exterior.coords[0][0],
                                   row.square.exterior.coords[1][0],
                                   int(n_tiles_per_sq), endpoint=False)
        nooverlap_ll_y = np.linspace(row.square.exterior.coords[0][1],
                                   row.square.exterior.coords[3][1],
                                   int(n_tiles_per_sq), endpoint=False)
    if load_squares == True: # through loading the squares from the shapefiles, the order of the coordinates is reversed
        nooverlap_ll_x = np.linspace(row.square.exterior.coords[0][0],
                                   row.square.exterior.coords[2][0],
                                   int(n_tiles_per_sq), endpoint=False)
        nooverlap_ll_y = np.linspace(row.square.exterior.coords[0][1],
                                   row.square.exterior.coords[1][1],
                                   int(n_tiles_per_sq), endpoint=False)
    # loop through all defined lowerleft corners to append to geodataframe
    for idx_x, ll_xi in enumerate(nooverlap_ll_x):
        for idx_y, ll_yi in enumerate(nooverlap_ll_y):
            single_tile = Polygon([(ll_xi, ll_yi),
                            (ll_xi+size_sq/n_tiles_per_sq, ll_yi),
                            (ll_xi+size_sq/n_tiles_per_sq, ll_yi+size_sq/n_tiles_per_sq),
                            (ll_xi, ll_yi+size_sq/n_tiles_per_sq),
                            (ll_xi,ll_yi)])
            nooverlap_tiles_val = nooverlap_tiles_val.append({'id_square': row.id_square,
                                                              'nooverlap_tile': single_tile}, 
                                                             ignore_index=True)
# estimate the overlap of tiles in the test data to later mask out these overlapping areas
# create geodataframe to store rectangles
nooverlap_tiles_test = geopandas.GeoDataFrame(columns=['id_square','nooverlap_tile'], geometry='nooverlap_tile')

# loop through validation squares
for index, row in sqs_test.iterrows():
    if load_squares == False:
        nooverlap_ll_x = np.linspace(row.square.exterior.coords[0][0],
                                   row.square.exterior.coords[1][0],
                                   int(n_tiles_per_sq), endpoint=False)
        nooverlap_ll_y = np.linspace(row.square.exterior.coords[0][1],
                                   row.square.exterior.coords[3][1],
                                   int(n_tiles_per_sq), endpoint=False)
    if load_squares == True: # through loading the squares from the shapefiles, the order of the coordinates is reversed
        nooverlap_ll_x = np.linspace(row.square.exterior.coords[0][0],
                                   row.square.exterior.coords[2][0],
                                   int(n_tiles_per_sq), endpoint=False)
        nooverlap_ll_y = np.linspace(row.square.exterior.coords[0][1],
                                   row.square.exterior.coords[1][1],
                                   int(n_tiles_per_sq), endpoint=False)
    # loop through all defined lowerleft corners to append to geodataframe
    for idx_x, ll_xi in enumerate(nooverlap_ll_x):
        for idx_y, ll_yi in enumerate(nooverlap_ll_y):
            single_tile = Polygon([(ll_xi, ll_yi),
                            (ll_xi+size_sq/n_tiles_per_sq, ll_yi),
                            (ll_xi+size_sq/n_tiles_per_sq, ll_yi+size_sq/n_tiles_per_sq),
                            (ll_xi, ll_yi+size_sq/n_tiles_per_sq),
                            (ll_xi,ll_yi)])
            nooverlap_tiles_test = nooverlap_tiles_test.append({'id_square': row.id_square,
                                                              'nooverlap_tile': single_tile}, 
                                                             ignore_index=True)

# create Prediction tiles (i.e., tiles that cover the ENTIRE continent)
# calculate bounds of coastlines
minx_CL = np.floor(CL_raw.geometry.bounds.minx.min())
miny_CL = np.floor(CL_raw.geometry.bounds.miny.min())
maxx_CL = np.ceil(CL_raw.geometry.bounds.maxx.max())
maxy_CL = np.ceil(CL_raw.geometry.bounds.maxy.max())

# calculate residual to center residual symettrically
resid_x = size_sq - (maxx_CL-minx_CL) % (size_sq)
resid_y = size_sq - (maxy_CL-miny_CL) % (size_sq)

# calculate lowerleft corners
ll_x_pred = np.arange(minx_CL - np.ceil(resid_x/2), maxx_CL, (resolution*n_pixels)//2)
ll_y_pred = np.arange(miny_CL - np.ceil(resid_y/2), maxy_CL, (resolution*n_pixels)//2)

# calculate coastline unary union
CL_union = CL_raw['geometry'].unary_union
# generate prediction tiles
tiles_pred = geopandas.GeoDataFrame(columns=['id_tile','tile'], geometry='tile')
for idx, tile_n_x in enumerate(ll_x_pred):
    for idy, tile_n_y in enumerate(ll_y_pred):
        # create polygon of tile
        tile_pred = poly(tile_n_x, tile_n_y, n_pixels=n_pixels, resolution=resolution)
        # check if tile intersects with coastlines
        if np.sum(CL_union.intersects(tile_pred))>0:
            tiles_pred = tiles_pred.append({'id_tile': 'pred', 'tile': tile_pred},
                            ignore_index=True)
# print number of tiles and save shapefile
print(len(tiles_pred))
tiles_pred.crs = 'epsg:3031'
tiles_pred.set_geometry(col='tile',inplace=True)
tiles_pred.to_file('../data/prediction_tiles.shp')

#%%
# function to assign data to tile
def sel_area(min_x,min_y,size,resolution,dataset):
    # caluclate maximum value for x and y coordinate
    max_x = min_x + resolution*size
    max_y = min_y + resolution*size
    # obtain arrays with values for x and y coordinates
    mask_x = (ds_main.x >= min_x) & (ds_main.x < max_x)
    mask_y = (ds_main.y >= min_y) & (ds_main.y < max_y)
    # crop image based on arrays with values for x and y coordinates
    img_cropped = dataset.where(mask_x & mask_y, drop=True)
    return img_cropped

# create mask of BIAs (="noisy" labels)
# rasterize polygons: create mask where only pixels whose center is within the polygon will be burned in.
ShapeMask = features.geometry_mask(BIAs_raw.geometry,
                                        out_shape=(len(ds_main.y), len(ds_main.x)),
                                        transform=ds_main.transform,
                                        invert=True)

# assign coordinates and create a data array
ShapeMask = xr.DataArray(ShapeMask, coords={"y":ds_main.y,
                                                "x":ds_main.x},
                            dims=("y", "x"))
# Create Data Array with zeros
zeros_tomask = xr.zeros_like(ds_main['DEM'])
# apply Mask to zeros_tomask --> 1 = BIA, 0 = no BIA
BIAs_mask = xr.where((ShapeMask == True),1,zeros_tomask)
# add BIAs_mask to dataset
ds_main['BIAs'] = BIAs_mask
del(BIAs_mask, ShapeMask, zeros_tomask)

# creat mask of BIAs for hand labelled data
# rasterize polygons: create mask where only pixels whose center is within the polygon will be burned in.
ShapeMask_clean = features.geometry_mask(handlabels_merged.geometry,
                                        out_shape=(len(ds_main.y), len(ds_main.x)),
                                        transform=ds_main.transform,
                                        invert=True)

# assign coordinates and create a data array
ShapeMask_clean = xr.DataArray(ShapeMask_clean, coords={"y":ds_main.y,
                                                "x":ds_main.x},
                            dims=("y", "x"))
# Create Data Array with zeros
zeros_tomask = xr.zeros_like(ds_main['DEM'])
# apply Mask to zeros_tomask --> 1 = BIA, 0 = no BIA
BIAs_mask_clean = xr.where((ShapeMask_clean == True),1,zeros_tomask)
# add BIAs_mask to dataset
ds_main['BIAs_clean'] = BIAs_mask_clean
del(BIAs_mask_clean, ShapeMask_clean, zeros_tomask)

# create targets for missing values (equal to -1) for 1. ice boundaries 2. input data

# 1. create mask of ice boundaries
# rasterize polygons: create mask where only pixels whose center is within the polygon will be burned in.
ShapeMask = features.geometry_mask([ice_boundaries],
                                      out_shape=(len(ds_main.y), len(ds_main.x)),
                                      transform=ds_main.transform,
                                      invert=True)
# assign coordinates and create a data array
ShapeMask = xr.DataArray(ShapeMask , coords={"y":ds_main.y,
                                             "x":ds_main.x},
                         dims=("y", "x"))

# Create Data Array with zeros
zeros_tomask = xr.zeros_like(ds_main['DEM'])
# apply Mask to zeros_tomask --> -1 = outside ice boundaries, 0 = inside ice boundaries
ice_boundaries_mask = xr.where((ShapeMask == True),zeros_tomask,-1)
# add missing data to dataset
ds_main['missing_data'] = ice_boundaries_mask
del(ice_boundaries_mask,ShapeMask,zeros_tomask)

# # check missing data values
# total_values = sum(sum(xr.where(((ds_main['missing_data']==0)),1,0)))

# missing_DEM = sum(sum(xr.where(((np.isnan(ds_main['DEM']))&(ds_main['missing_data']==0)),1,0)))
# print(missing_DEM/total_values)

# missing_radar = sum(sum(xr.where(((np.isnan(ds_main['radar']))&(ds_main['missing_data']==0)),1,0)))
# print(missing_radar/total_values)

# missing_moa = sum(sum(xr.where(((np.isnan(ds_main['modis']))&(ds_main['missing_data']==0)),1,0)))
# print(missing_moa/total_values)

# for i in range(1,8):
#     missing_MOD = sum(sum(xr.where(((np.isnan(ds_main[f'MOD_B{i}']))&(ds_main['missing_data']==0)),1,0)))
#     print(i, missing_MOD/total_values)

#%%
# 2. set values where DEM data is missing equal to -1 in the missing data variable 
# (we choose the derivative slope here, as nans in the slope are more abundant than
#  in the DEM through the slope algorithm 
# (https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-slope-works.htm))
ds_main['missing_data'] = xr.where(np.isnan(ds_main['slope']), -1, ds_main['missing_data'])
# 2. set values where RADAR data is missing equal to -1 in the missing data variable
ds_main['missing_data'] = xr.where(np.isnan(ds_main['radar']), -1, ds_main['missing_data'])
# 2. set values where MOA data is missing equal to -1 in the missing data variable
ds_main['missing_data'] = xr.where(np.isnan(ds_main['modis']), -1, ds_main['missing_data'])
# 2. set values where MODIS composite data is missing equal to -1 in the missing data variable
for i in range(1,8):
    ds_main['missing_data'] = xr.where(np.isnan(ds_main[f'MOD_B{i}']), -1, ds_main['missing_data'])

# combine mask of BIAs with the missing_data mask
ds_main['target'] = xr.where(ds_main['missing_data']==-1, -1, ds_main['BIAs'])
ds_main['target_clean'] = xr.where(ds_main['missing_data']==-1, -1, ds_main['BIAs_clean'])
ds_main = ds_main.drop(['BIAs','missing_data'])

#%%
# 3. replace all NaNs with 0 for the elevation data (!!!)
ds_main['DEM'] = xr.where(np.isnan(ds_main['DEM']), 0., ds_main['DEM'])
ds_main['slope'] = xr.where(np.isnan(ds_main['slope']), 0., ds_main['slope'])
ds_main['corrected_aspect'] = xr.where(np.isnan(ds_main['corrected_aspect']), 0., ds_main['corrected_aspect'])

# 3. replace all NaNs with the minimum value of Radar for the RADAR data (!!!)
radar_fill = ds_main['radar'].min().values
ds_main['radar'] = xr.where(np.isnan(ds_main['radar']), radar_fill, ds_main['radar'])

# 3. replace all NaNs with 0 for the MODIS composite data
for i in range(1,8):
    ds_main[f'MOD_B{i}'] = xr.where(np.isnan(ds_main[f'MOD_B{i}']), 0., ds_main[f'MOD_B{i}'])

# 3. replace all NaNs with the minimum value of MOA for the MOA data
moa_fill = ds_main['modis'].min().values
ds_main['modis'] = xr.where(np.isnan(ds_main['modis']), moa_fill, ds_main['modis'])
#%%
# 4. standardize all data
list_keys = list(ds_main.keys())[:12]
for key in list_keys:
    mu = np.nanmean(ds_main[key].where(ds_main['target']!=-1).values.flatten())
    sigma = np.nanstd(ds_main[key].where(ds_main['target']!=-1).values.flatten())
    ds_main[f'{key}_norm'] = ((ds_main[key]-mu)/sigma).astype('float32') # alternative would be min-max normalization
    ds_main = ds_main.drop(f'{key}')
#%%
# TO DEL?
# # least-squares to find 2d plane that fits data
# # observations = offset + slope_x*x + slope_y*y
# # observations = A*[offset, slope_x, slope_y]

# # create artificial coordinates
# xy_artificial = np.linspace((-0.5*n_pixels*resolution/1000.)+0.5*resolution/1000., 
#                             0.5*n_pixels*resolution/1000.-0.5*resolution/1000.,
#                             n_pixels)
# xv, yv = np.meshgrid(xy_artificial, xy_artificial)
# xs = np.ravel(xv)
# ys = np.ravel(yv)
# # predefine matrix A (containing 3 columns, 1: ones, 2: x-coordinates, 3: y-coordinates)
# A = np.zeros((int(n_pixels**2),3))
# # column_n = 0
# A[:,0] = 1
# A[:,1] = np.ravel(xs)
# A[:,2] = np.ravel(ys)


# make directories within folder
try:
    os.mkdir(train_dir+'/train_images')
    os.mkdir(train_dir+'/train_targets')
    os.mkdir(train_dir+'/val_images')
    os.mkdir(train_dir+'/val_targets')
    os.mkdir(train_dir+'/test_images')
    os.mkdir(train_dir+'/test_targets')
except OSError:
    pass 
try:  
    os.mkdir(train_dir+'/train_images_noiseornot')
    os.mkdir(train_dir+'/train_targets_noiseornot')
    os.mkdir(train_dir+'/val_images_clean')
    os.mkdir(train_dir+'/val_targets_clean')
    os.mkdir(train_dir+'/test_images_clean')
    os.mkdir(train_dir+'/test_targets_clean')
except OSError:
    pass

#%%
# function to create training tiles
def createTRAINtiles(index):
    tile_ = tiles_train.iloc[index]
    # check if BIAs (="noisy" labels) in tile
    # if BIA in tile continue
    if np.sum(BIAs_union.intersects(tile_.tile))>0:
        # create tile with all features
        lowerleft_x = tile_.tile.exterior.coords[0][0]
        lowerleft_y = tile_.tile.exterior.coords[0][1]
        # crop data to tile extent
        data_cropped = sel_area(lowerleft_x,
                                lowerleft_y,
                                n_pixels,
                                resolution,
                                ds_main)
        # check if extent of data covers entire tile
        if (data_cropped.sizes['y']<512) or (data_cropped.sizes['x']<512):
            print(data_cropped.sizes)
            print(lowerleft_x, lowerleft_y)
            pass
        else:
            # define name of image
            savename = f'{int(data_cropped.x.min().values)}_{int(data_cropped.y.min().values)}'
            # rotate or flip (both set to 0 for final model)
            k = 0
            toflip = 0
            # create list of dataset variables
            data_vars = list(data_cropped.keys())
            # for all data variables flip and rotate
            for data_var in data_vars:
                if toflip>0:
                    data_cropped[data_var].values = np.flip(np.rot90(data_cropped[data_var].values, k=k), axis=1)
                else:
                    data_cropped[data_var].values = np.rot90(data_cropped[data_var].values, k=k)
            # save data and targets as .nc
            data_cropped[['DEM_norm','slope_norm','corrected_aspect_norm','radar_norm','modis_norm',
                            'MOD_B1_norm','MOD_B2_norm','MOD_B3_norm','MOD_B4_norm',
                            'MOD_B5_norm','MOD_B6_norm','MOD_B7_norm']].to_netcdf(f'{train_dir}/train_images/{savename}.nc')
            data_cropped['target'].to_netcdf(f'{train_dir}/train_targets/{savename}.nc')

        del(data_cropped)

# function to create tiles in training area where we know whether labels are noisy or not
def createTRAINtiles_noiseornot(index):
    tile_ = tiles_train_noiseornot.iloc[index]
    # check if BIAs (="noisy" labels) in tile
    # if BIA in tile continue
    if np.sum(BIAs_union.intersects(tile_.tile))>0:
        # create tile with all features
        lowerleft_x = tile_.tile.exterior.coords[0][0]
        lowerleft_y = tile_.tile.exterior.coords[0][1]
        # crop data to tile extent
        data_cropped = sel_area(lowerleft_x,
                                lowerleft_y,
                                n_pixels,
                                resolution,
                                ds_main)
        # check if extent of data covers entire tile
        if (data_cropped.sizes['y']<512) or (data_cropped.sizes['x']<512):
            print(data_cropped.sizes)
            print(lowerleft_x, lowerleft_y)
            pass
        else:
            # define name of image
            savename = f'{int(data_cropped.x.min().values)}_{int(data_cropped.y.min().values)}'
            # create alternate target
            # -1 = missing data
            # 0 = no blue ice, noise and clean labels agree
            # 1 = blue ice, noise and clean labels agree
            # 2 = no blue ice (clean labels), noise and clean labels disagree
            # 3 = blue ice (clean labels), noise and clean labels disagree
            data_cropped['noiseornot'] = xr.where(((data_cropped['target']!=data_cropped['target_clean']) & (data_cropped['target_clean']==0.0)),2.,data_cropped['target_clean'])
            data_cropped['noiseornot'] = xr.where(((data_cropped['target']!=data_cropped['target_clean']) & (data_cropped['target_clean']==1.0)),3.,data_cropped['noiseornot'])
            data_cropped = data_cropped.drop(['target','target_clean'])
            data_cropped = data_cropped.rename({'noiseornot':'target'})
            # rotate or flip (both set to 0 for final model)
            k = 0
            toflip = 0
            # create list of dataset variables
            data_vars = list(data_cropped.keys())
            # for all data variables flip and rotate
            for data_var in data_vars:
                if toflip>0:
                    data_cropped[data_var].values = np.flip(np.rot90(data_cropped[data_var].values, k=k), axis=1)
                else:
                    data_cropped[data_var].values = np.rot90(data_cropped[data_var].values, k=k)
            # save data and targets as .nc
            data_cropped[['DEM_norm','slope_norm','corrected_aspect_norm','radar_norm','modis_norm',
                            'MOD_B1_norm','MOD_B2_norm','MOD_B3_norm','MOD_B4_norm',
                            'MOD_B5_norm','MOD_B6_norm','MOD_B7_norm']].to_netcdf(f'{train_dir}/train_images_noiseornot/{savename}.nc')
            data_cropped['target'].to_netcdf(f'{train_dir}/train_targets_noiseornot/{savename}.nc')

        del(data_cropped)

# function to create validation tiles
def createVALtiles(index):
    tile_ = tiles_val.iloc[index]
    # check if BIAs (="noisy" labels) in tile
    # if BIA in tile continue
    if np.sum(BIAs_union.intersects(tile_.tile))>0:
        # create tile with all features
        lowerleft_x = tile_.tile.exterior.coords[0][0]
        lowerleft_y = tile_.tile.exterior.coords[0][1]
        # crop data to tile extent
        data_cropped = sel_area(lowerleft_x,
                                lowerleft_y,
                                n_pixels,
                                resolution,
                                ds_main)
        # define name of image
        savename = f'{int(data_cropped.x.min().values)}_{int(data_cropped.y.min().values)}'
        # find nooverlap_tile that defines the extent of the overlap between different tiles
        selected_nooverlap_tile = nooverlap_tiles_val[nooverlap_tiles_val.within(tile_.tile)==True].nooverlap_tile.iloc[0]
        # mask data with nooverlap tile
        minx_tomask = selected_nooverlap_tile.exterior.coords[0][0]
        maxx_tomask = selected_nooverlap_tile.exterior.coords[1][0]
        miny_tomask = selected_nooverlap_tile.exterior.coords[0][1]
        maxy_tomask = selected_nooverlap_tile.exterior.coords[3][1]
        # obtain arrays with values for x and y coordinates
        mask_x = (data_cropped.x >= minx_tomask) & (data_cropped.x < maxx_tomask)
        mask_y = (data_cropped.y < maxy_tomask) & (data_cropped.y >= miny_tomask) # y coordinates range from large to small
        # crop image based on arrays with values for x and y coordinates
        data_cropped['target'] = data_cropped['target'].where((mask_y & mask_x),-1)
        # check if extent of data covers entire tile
        if (data_cropped.sizes['y']<512) or (data_cropped.sizes['x']<512):
            print(data_cropped.sizes)
            print(lowerleft_x, lowerleft_y)
            pass
        else:
            # rotate or flip (both set to 0 for final model)
            k = 0
            toflip = 0
            # create list of dataset variables
            data_vars = list(data_cropped.keys())
            # for all data variables flip and rotate
            for data_var in data_vars:
                if toflip>0:
                    data_cropped[data_var].values = np.flip(np.rot90(data_cropped[data_var].values, k=k), axis=1)
                    toflip_attr = 1
                else:
                    data_cropped[data_var].values = np.rot90(data_cropped[data_var].values, k=k)
                    toflip_attr = 0
            # add attributes on flipping and rotating
            data_cropped['target'].attrs['flip'] = toflip_attr
            data_cropped['target'].attrs['rotate'] = k
            # save data and target as .nc
            data_cropped[['DEM_norm','slope_norm','corrected_aspect_norm','radar_norm','modis_norm',
                            'MOD_B1_norm','MOD_B2_norm','MOD_B3_norm','MOD_B4_norm',
                            'MOD_B5_norm','MOD_B6_norm','MOD_B7_norm']].to_netcdf(f'{train_dir}/val_images/{savename}.nc')
            data_cropped['target'].to_netcdf(f'{train_dir}/val_targets/{savename}.nc')

        del(data_cropped)

# function to create validation tiles with hand labels as targets
def createVALtiles_clean(index):
    tile_ = tiles_val_clean.iloc[index]
    # check if BIAs in tile
    # if BIA in tile continue
    if np.sum(BIAs_union.intersects(tile_.tile))>0:
        # create tile with all features
        lowerleft_x = tile_.tile.exterior.coords[0][0]
        lowerleft_y = tile_.tile.exterior.coords[0][1]
        # crop data to tile extent
        data_cropped = sel_area(lowerleft_x,
                                lowerleft_y,
                                n_pixels,
                                resolution,
                                ds_main)
        # define name of image
        savename = f'{int(data_cropped.x.min().values)}_{int(data_cropped.y.min().values)}'
        # find nooverlap_tile that defines the extent of the overlap between different tiles
        selected_nooverlap_tile = nooverlap_tiles_val[nooverlap_tiles_val.within(tile_.tile)==True].nooverlap_tile.iloc[0]
        # mask data with nooverlap tile
        minx_tomask = selected_nooverlap_tile.exterior.coords[0][0]
        maxx_tomask = selected_nooverlap_tile.exterior.coords[1][0]
        miny_tomask = selected_nooverlap_tile.exterior.coords[0][1]
        maxy_tomask = selected_nooverlap_tile.exterior.coords[3][1]
        # obtain arrays with values for x and y coordinates
        mask_x = (data_cropped.x >= minx_tomask) & (data_cropped.x < maxx_tomask)
        mask_y = (data_cropped.y < maxy_tomask) & (data_cropped.y >= miny_tomask) # y coordinates range from large to small
        # crop image based on arrays with values for x and y coordinates
        data_cropped['target_clean'] = data_cropped['target_clean'].where((mask_y & mask_x),-1)
        # check if extent of data covers entire tile
        if (data_cropped.sizes['y']<512) or (data_cropped.sizes['x']<512):
            print(data_cropped.sizes)
            print(lowerleft_x, lowerleft_y)
            pass
        else:
            # rename target_clean to target
            data_cropped = data_cropped.drop(['target'])
            data_cropped = data_cropped.rename({'target_clean':'target'})
            # rotate or flip (both set to 0 for final model)
            k = 0
            toflip = 0
            # create list of dataset variables
            data_vars = list(data_cropped.keys())
            # for all data variables flip and rotate
            for data_var in data_vars:
                if toflip>0:
                    data_cropped[data_var].values = np.flip(np.rot90(data_cropped[data_var].values, k=k), axis=1)
                    toflip_attr = 1
                else:
                    data_cropped[data_var].values = np.rot90(data_cropped[data_var].values, k=k)
                    toflip_attr = 0
            # add attributes on flipping and rotating
            data_cropped['target'].attrs['flip'] = toflip_attr
            data_cropped['target'].attrs['rotate'] = k
            # save data and target as .nc
            data_cropped[['DEM_norm','slope_norm','corrected_aspect_norm','radar_norm','modis_norm',
                            'MOD_B1_norm','MOD_B2_norm','MOD_B3_norm','MOD_B4_norm',
                            'MOD_B5_norm','MOD_B6_norm','MOD_B7_norm']].to_netcdf(f'{train_dir}/val_images_clean/{savename}.nc')
            data_cropped['target'].to_netcdf(f'{train_dir}/val_targets_clean/{savename}.nc')
        del(data_cropped)

# function to create validation tiles
def createTESTtiles(index):
    tile_ = tiles_test.iloc[index]
    # check if BIAs in tile
    # if BIA in tile continue
    if np.sum(BIAs_union.intersects(tile_.tile))>0:
        # create tile with all features
        lowerleft_x = tile_.tile.exterior.coords[0][0]
        lowerleft_y = tile_.tile.exterior.coords[0][1]
        # crop data to tile extent
        data_cropped = sel_area(lowerleft_x,
                                lowerleft_y,
                                n_pixels,
                                resolution,
                                ds_main)
        # define name of image
        savename = f'{int(data_cropped.x.min().values)}_{int(data_cropped.y.min().values)}'
        # find nooverlap_tile that defines the extent of the overlap between different tiles
        selected_nooverlap_tile = nooverlap_tiles_test[nooverlap_tiles_test.within(tile_.tile)==True].nooverlap_tile.iloc[0]
        # mask data with nooverlap tile
        minx_tomask = selected_nooverlap_tile.exterior.coords[0][0]
        maxx_tomask = selected_nooverlap_tile.exterior.coords[1][0]
        miny_tomask = selected_nooverlap_tile.exterior.coords[0][1]
        maxy_tomask = selected_nooverlap_tile.exterior.coords[3][1]
        # obtain arrays with values for x and y coordinates
        mask_x = (data_cropped.x >= minx_tomask) & (data_cropped.x < maxx_tomask)
        mask_y = (data_cropped.y < maxy_tomask) & (data_cropped.y >= miny_tomask) # y coordinates range from large to small
        # crop image based on arrays with values for x and y coordinates
        data_cropped['target'] = data_cropped['target'].where((mask_y & mask_x),-1)
        # check if extent of data covers entire tile
        if (data_cropped.sizes['y']<512) or (data_cropped.sizes['x']<512):
            print(data_cropped.sizes)
            print(lowerleft_x, lowerleft_y)
            pass
        else:
            # rotate or flip (both set to 0 for final model)
            k = 0
            toflip = 0
            # create list of dataset variables
            data_vars = list(data_cropped.keys())
            # for all data variables flip and rotate
            for data_var in data_vars:
                if toflip>0:
                    data_cropped[data_var].values = np.flip(np.rot90(data_cropped[data_var].values, k=k), axis=1)
                    toflip_attr = 1
                else:
                    data_cropped[data_var].values = np.rot90(data_cropped[data_var].values, k=k)
                    toflip_attr = 0
            # add attributes on flipping and rotating
            data_cropped['target'].attrs['flip'] = toflip_attr
            data_cropped['target'].attrs['rotate'] = k
            # save data and target as .nc
            data_cropped[['DEM_norm','slope_norm','corrected_aspect_norm','radar_norm','modis_norm',
                            'MOD_B1_norm','MOD_B2_norm','MOD_B3_norm','MOD_B4_norm',
                            'MOD_B5_norm','MOD_B6_norm','MOD_B7_norm']].to_netcdf(f'{train_dir}/test_images/{savename}.nc')
            data_cropped['target'].to_netcdf(f'{train_dir}/test_targets/{savename}.nc')
        del(data_cropped)

# function to create test tiles with hand labels as targets
def createTESTtiles_clean(index):
    tile_ = tiles_test_clean.iloc[index]
    # check if BIAs in tile
    # if BIA in tile continue
    if np.sum(BIAs_union.intersects(tile_.tile))>0:
        # create tile of all features
        lowerleft_x = tile_.tile.exterior.coords[0][0]
        lowerleft_y = tile_.tile.exterior.coords[0][1]
        # crop data to tile extent
        data_cropped = sel_area(lowerleft_x,
                                lowerleft_y,
                                n_pixels,
                                resolution,
                                ds_main)
        # define name of image
        savename = f'{int(data_cropped.x.min().values)}_{int(data_cropped.y.min().values)}'
        # find nooverlap_tile that defines the extent of the overlap between different tiles
        selected_nooverlap_tile = nooverlap_tiles_test[nooverlap_tiles_test.within(tile_.tile)==True].nooverlap_tile.iloc[0]
        # mask data with nooverlap tile
        minx_tomask = selected_nooverlap_tile.exterior.coords[0][0]
        maxx_tomask = selected_nooverlap_tile.exterior.coords[1][0]
        miny_tomask = selected_nooverlap_tile.exterior.coords[0][1]
        maxy_tomask = selected_nooverlap_tile.exterior.coords[3][1]
        # obtain arrays with values for x and y coordinates
        mask_x = (data_cropped.x >= minx_tomask) & (data_cropped.x < maxx_tomask)
        mask_y = (data_cropped.y < maxy_tomask) & (data_cropped.y >= miny_tomask) # y coordinates range from large to small
        # crop image based on arrays with values for x and y coordinates
        data_cropped['target_clean'] = data_cropped['target_clean'].where((mask_y & mask_x),-1)
        # check if extent of data covers entire tile
        if (data_cropped.sizes['y']<512) or (data_cropped.sizes['x']<512):
            print(data_cropped.sizes)
            print(lowerleft_x, lowerleft_y)
            pass
        else:
            # rename target_clean to target
            data_cropped = data_cropped.drop(['target'])
            data_cropped = data_cropped.rename({'target_clean':'target'})
            # rotate or flip (both set to 0 for final model)
            k = 0
            toflip = 0
            # create list of dataset variables
            data_vars = list(data_cropped.keys())
            # for all data variables flip and rotate
            for data_var in data_vars:
                if toflip>0:
                    data_cropped[data_var].values = np.flip(np.rot90(data_cropped[data_var].values, k=k), axis=1)
                    toflip_attr = 1
                else:
                    data_cropped[data_var].values = np.rot90(data_cropped[data_var].values, k=k)
                    toflip_attr = 0
            # add attributes on flipping and rotating
            data_cropped['target'].attrs['flip'] = toflip_attr
            data_cropped['target'].attrs['rotate'] = k
            # save data and target as .nc
            data_cropped[['DEM_norm','slope_norm','corrected_aspect_norm','radar_norm','modis_norm',
                            'MOD_B1_norm','MOD_B2_norm','MOD_B3_norm','MOD_B4_norm',
                            'MOD_B5_norm','MOD_B6_norm','MOD_B7_norm']].to_netcdf(f'{train_dir}/test_images_clean/{savename}.nc')
            data_cropped['target'].to_netcdf(f'{train_dir}/test_targets_clean/{savename}.nc')
        del(data_cropped)

# function to create test prediction tiles (no targets)
def createPREDtiles(index):
    tile_ = tiles_pred.iloc[index]
    #create tile of all features
    lowerleft_x = tile_.tile.exterior.coords[0][0]
    lowerleft_y = tile_.tile.exterior.coords[0][1]
    # crop data to tile extent
    data_cropped = sel_area(lowerleft_x,
                            lowerleft_y,
                            n_pixels,
                            resolution,
                            ds_main)
    # check if extent of data covers entire tile
    if (data_cropped.sizes['y']<512) or (data_cropped.sizes['x']<512):
        print(data_cropped.sizes)
        print(lowerleft_x, lowerleft_y)
        pass
    else:
        # define name of image
        savename = f'{int(data_cropped.x.min().values)}_{int(data_cropped.y.min().values)}'
        # rotate or flip (both set to 0 for final model)
        k = 0
        toflip = 0
        # create list of dataset variables
        data_vars = list(data_cropped.keys())
        # for all data variables flip and rotate
        for data_var in data_vars:
            if toflip>0:
                data_cropped[data_var].values = np.flip(np.rot90(data_cropped[data_var].values, k=k), axis=1)
                toflip_attr=1
            else:
                data_cropped[data_var].values = np.rot90(data_cropped[data_var].values, k=k)
                toflip_attr=0
        data_cropped.attrs['flip'] = toflip_attr
        data_cropped.attrs['rotate'] = k
        # save data and targets as .nc
        data_cropped[['DEM_norm','slope_norm','corrected_aspect_norm','radar_norm','modis_norm',
                        'MOD_B1_norm','MOD_B2_norm','MOD_B3_norm','MOD_B4_norm',
                        'MOD_B5_norm','MOD_B6_norm','MOD_B7_norm']].to_netcdf(f'{pred_dir}/{savename}.nc')
    del(data_cropped)


#%%
# create tiles in parallel processes (depending on RAM available on machine)
print(len(tiles_train),' training tiles to make')
p = Pool(processes=2)
p.map(createTRAINtiles, np.arange(0,len(tiles_train),1))
p.close()
print('train data done')

print(len(tiles_val),' validation tiles to make')
p2 = Pool(processes=2)
p2.map(createVALtiles, np.arange(0,len(tiles_val),1))
p2.close()
print('validation data done')

print(len(tiles_test),' test tiles to make')
p = Pool(processes=2)
p.map(createTESTtiles, np.arange(0,len(tiles_test),1))
p.close()
print('test data done')

print(len(tiles_val_clean),' clean validation tiles to make')
p2 = Pool(processes=2)
p2.map(createVALtiles_clean, np.arange(0,len(tiles_val_clean),1))
p2.close()
print('validation data done')

print(len(tiles_train_noiseornot),' noise or not training tiles to make')
p = Pool(processes=2)
p.map(createTRAINtiles_noiseornot, np.arange(0,len(tiles_train_noiseornot),1))
p.close()
print('train data done')

print(len(tiles_test_clean),' clean test tiles to make')
p = Pool(processes=2)
p.map(createTESTtiles_clean, np.arange(0,len(tiles_test_clean),1))
p.close()
print('test data done')

print(len(tiles_pred),' prediction tiles to make')
p3 = Pool(processes=2)
p3.map(createPREDtiles, np.arange(0,len(tiles_pred),1))
p3.close()
print('prediction data done')

