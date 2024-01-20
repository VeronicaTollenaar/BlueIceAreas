# merge 40 BIA maps into a single BIA map
# import packages
import xarray as xr
import numpy as np
import rasterio
import geopandas
import rasterio.features
import affine
import os
# set directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)
#%%
# read in 40 final maps
filenames = os.listdir(f'../output')
filenames = [filename for filename in filenames if filename.startswith(f'preds')]
# check length of filenames (should be equal to 40)
print(f'number of BIA predictions is {len(filenames)}')

# %%
# predefine chunksizes (needed to manage RAM)
ch_x = 9254.
ch_y = 6238.
# use first BIA map as sample to predefine continent-wide dataset
sample_data = xr.open_dataset(f'../output/{filenames[0]}',chunks={'x': ch_x, 'y': ch_y})
# predefine continent-wide dataset (same size as datasets)
final_map = sample_data.drop(['all_preds','weights'])
final_map['mean'] = (('y','x'),np.zeros(np.shape(sample_data['all_preds'])))
final_map['std'] = (('y','x'),np.zeros(np.shape(sample_data['all_preds'])))

# define chunks
chunks_y,chunks_x = sample_data['all_preds'].chunks
del(sample_data)
# loop over chuncks
for ys,chunk_y in enumerate(chunks_y):
    for xs,chunk_x in enumerate(chunks_x):
        print(f'chunk {ys*len(chunks_x)+xs+1} // {len(chunks_y)*len(chunks_x)} ...')
        # open first file
        file_path = f'../output/{filenames[0]}'
        preds = xr.open_dataset(file_path,chunks={'x': ch_x, 'y': ch_y})['all_preds']
        preds_chunk_concat = preds[ys*chunk_y:(ys+1)*chunk_y,xs*chunk_x:(xs+1)*chunk_x].load().to_dataset(name=f'{filenames[0]}')
        
        # loop over filenames
        for filename in filenames[1:]:
            file_path = f'../output/{filename}'
            preds = xr.open_dataset(file_path,chunks={'x': ch_x, 'y': ch_y})['all_preds']
            preds_chunk = preds[ys*chunk_y:(ys+1)*chunk_y,xs*chunk_x:(xs+1)*chunk_x]
            # merge all files in chunk to one dataset
            preds_chunk_concat[f'{filename}'] = (('y','x'),preds_chunk.values)
        # caculate mean and std and assign values to final map
        final_map['mean'][ys*chunk_y:(ys+1)*chunk_y,xs*chunk_x:(xs+1)*chunk_x] = preds_chunk_concat.to_array().mean(dim='variable')
        final_map['std'][ys*chunk_y:(ys+1)*chunk_y,xs*chunk_x:(xs+1)*chunk_x] = preds_chunk_concat.to_array().std(dim='variable')
#%%
# mask out values outside the continent's coastlines
# open iceboundaries (quantarctica measures ice boundaries)
ice_boundaries_path = r'../data/IceBoundaries_Antarctica_v2.shx'
ice_boundaries_raw = geopandas.read_file(ice_boundaries_path)

# create union of ice boundaries
ice_boundaries = ice_boundaries_raw['geometry'].unary_union

# set transform parameter
resolution = final_map.x[1].values-final_map.x[0].values
ll_x_main = final_map.x.min().values
ur_y_main = final_map.y.max().values
final_map.attrs['transform'] = affine.Affine(resolution, 0.0, ll_x_main-(resolution/2), 0.0, -1*resolution, ur_y_main+(resolution/2))

# mask out values outside ice_boundaries
ShapeMask = rasterio.features.geometry_mask([ice_boundaries],
                                      out_shape=(len(final_map.y),
                                                 len(final_map.x)),
                                      transform=final_map.transform,
                                      invert=True,
                                      all_touched=False)
ShapeMask = xr.DataArray(ShapeMask, 
                         dims=({"y":final_map["y"][::-1], "x":final_map["x"]}))
# flip shapemask upside down
ShapeMask= ShapeMask[::-1]
map_masked = final_map.where((ShapeMask == True),drop=True)
# update transform parameters
resolution = map_masked.x[1].values-map_masked.x[0].values
ll_x_main = map_masked.x.min().values
ur_y_main = map_masked.y.max().values
del(map_masked.attrs['transform'])
map_masked.attrs['transform'] = affine.Affine(resolution, 0.0, ll_x_main-(resolution/2), 0.0, -1*resolution, ur_y_main+(resolution/2))

# export data
save_path = '../output/final_map.nc'
map_masked.to_netcdf(path=save_path)
