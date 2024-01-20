# visualize map and analyze continent-wide mapping (Figures 1, 2, and S8, S9, S11, S12, S13)
# import packages
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import geopandas
from rasterio import features
import affine
from shapely.geometry import Polygon
from shapely.geometry import Point
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
from pyproj import Transformer
from scipy.ndimage import generic_filter
import matplotlib
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
# set directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)
#%%

# import BIA predictions
BIA_map = xr.open_dataset('../output/final_map.nc')
# update transform parameters BIA_map
resolution = 200
ll_x_main = BIA_map.x.min().values
ur_y_main = BIA_map.y.max().values
BIA_map.attrs['transform'] = affine.Affine(resolution, 0.0, ll_x_main-(resolution/2), 0.0, -1*resolution, ur_y_main+(resolution/2))

# import BIAs of Hui et al., 2014 (="noisy" labels)
BIAs_path = r'../data/BlueIceAreas.shx'
BIAs_hui = geopandas.read_file(BIAs_path)

# rasterize polygons: create mask where only pixels whose center is within the polygon will be burned in.
ShapeMask = features.geometry_mask(BIAs_hui.geometry,
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
BIA_map['BIAs_hui'] = BIAs_mask
del(BIAs_mask, ShapeMask, zeros_tomask)

# calculate number of positive observations
# NB no difference between > and >=
n_pos_obs = np.sum(np.sum(xr.where(BIA_map['mean']>=0.5,1,0))) 
# calculate area of positive observations
area_pos_obs = n_pos_obs*0.2*0.2

# calculate total number of observations (over continent)
n_obs = np.sum(np.sum(xr.where(~np.isnan(BIA_map['mean']),1,0)))
# cacluate total area
#area_total = n_obs*0.2*0.2

# open ice boundaries
ice_boundaries_path = r'../data/IceBoundaries_Antarctica_v2.shx'
ice_boundaries_raw = geopandas.read_file(ice_boundaries_path)
# merge ice boundaries
ice_boundaries = ice_boundaries_raw['geometry'].unary_union
area_total = (ice_boundaries.area)*1e-6

#%%
# calculate number of obs at BIAs of Hui et al., 2014
n_BIAs_hui = np.sum(np.sum(xr.where((BIA_map['BIAs_hui']==1) & (~(np.isnan(BIA_map['mean']))),1,0)))
# cacluate total area of BIAs of Hui et al.
area_BIAs_hui = n_BIAs_hui*0.2*0.2

# print numbers
print('percentage BIA this study:', np.round(100*area_pos_obs.values/area_total,2))
print('percentage BIA Hui et al.:', np.round(100*area_BIAs_hui.values/area_total,2))

# compare our estimate to Hui et al., 2014
# observations that were previously thought to be NO blue ice
n_newice = np.sum(np.sum(xr.where((BIA_map['BIAs_hui']==0) & (BIA_map['mean']>=0.5),1,0)))
area_newice = n_newice*0.2*0.2
print('percentage of our BIA map that was previously thought to be no blue ice:', 
      np.round(100*area_newice.values/area_pos_obs.values,2))

# observations that were previously thought to be YES blue ice
n_noice = np.sum(np.sum(xr.where((BIA_map['BIAs_hui']==1) & (BIA_map['mean']<0.5),1,0)))
area_noice = n_noice*0.2*0.2
print("percentage of Hui's map that is now no blue ice:",
      np.round(100*area_noice.values/area_BIAs_hui.values,2))

#%% 
# uncertainty analysis
# calculate number of positive observations when +1std and when -1std
# NB no difference between > and >=
n_pos_obs_max = np.sum(np.sum(xr.where(BIA_map['mean']+BIA_map['std']>=0.5,1,0)))
n_pos_obs_min = np.sum(np.sum(xr.where(BIA_map['mean']-BIA_map['std']>=0.5,1,0)))
# calculate area of positive observations
area_pos_obs_max = n_pos_obs_max*0.2*0.2
area_pos_obs_min = n_pos_obs_min*0.2*0.2
print(f'range BIA extent this study (percentage): {np.round(100*area_pos_obs_min.values/area_total,2)} to \
      {np.round(100*area_pos_obs_max.values/area_total,2)}%')
print(f'range BIA extent this study: {np.round(area_pos_obs_min.values,0)} to \
      {np.round(area_pos_obs_max.values,0)} km')
#%%
# check how uncertainties influence estimates per region
# mask out region with a polygon (area of interest)
# 4 areas of interest:
# 1. grounding line (+- 20km)
# 2. mountainous terrain
# 3. higher latitudes (>82.5S)
# 4. high elevation

# open/create polygon data
# open grounding line data (data from measures v2 (part of quantarcitca))
gr_line_path = r'../data/GroundingLine_Antarctica_v2.shx'
gr_line_raw = geopandas.read_file(gr_line_path)
# create unary union
gr_line = gr_line_raw['geometry'].unary_union
# total area around grounding line (in km2)
aoi_gr_line = gr_line.buffer(20*1e3).difference(gr_line.buffer(-20*1e3)) 
#%%
# create binary mask to generate shapefile of mountainous terrain in QGIS
# import data
elevation_differences = xr.open_dataset('../output/elevation_differences.nc')
elev_differences_binary = xr.where(elevation_differences>900,1,0)
elev_differences_binary.to_netcdf('../output/elev_differences_binary.nc')
#%%
# steps to process in QGIS
# 1. save netcdf with crs
# 2. vectorize, and select polygons with values 1, save as shapefile
# 3. buffer polygons twice (once +10, once -10) to obtain valid geometries

# open mountainous terrain data (processed in QGIS)
mt_terrain_path = r'../data/mountainous_terrain.shx'
mt_terrain_raw = geopandas.read_file(mt_terrain_path)
# create unary union
aoi_mt_terrain = mt_terrain_raw['geometry'].unary_union

#%%
# create shapely polygon marking the latitude of 82.5 S
high_lat4326 = Point([(0,-82.5)])
high_lat4326gdf = geopandas.GeoDataFrame( 
    geometry=[high_lat4326],
    crs='epsg:4326'
)
# transform polygon to epsg:3031 to extract radius
radius = high_lat4326gdf.to_crs('epsg:3031').geometry.y

centerpoint = Point([0,0])
aoi_high_lat = centerpoint.buffer(radius)

#%%
# open elevation data
TDX_orhto_path = r'../data/TDM_merged_ortho.nc'
TDX_ortho_ds = xr.open_dataset(TDX_orhto_path)
TDX_ortho_binary = xr.where(TDX_ortho_ds>1500,1,0)
TDX_ortho_binary.to_netcdf('../output/TDX_ortho_binary.nc')
#%%
# open high elevation areas (processed in QGIS)
high_elev_path = r'../data/1500m_aoi.shx'
high_elev_raw = geopandas.read_file(high_elev_path)
# create unary union
aoi_high_elev = high_elev_raw['geometry'].unary_union

#%%
# define function that calculates uncertainties per area of interest
def uncertainties_per_region(polygon):
    ShapeMask = features.geometry_mask([polygon],
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
    BIAs_inAOI_mean = xr.where((ShapeMask == True),BIA_map['mean'],zeros_tomask)
    BIAs_inAOI_min = xr.where((ShapeMask == True),BIA_map['mean']-BIA_map['std'],zeros_tomask)
    BIAs_inAOI_max = xr.where((ShapeMask == True),BIA_map['mean']+BIA_map['std'],zeros_tomask)
    # calculate number of positive observations in area of interest
    n_pos_obs_AOI = sum(sum(BIAs_inAOI_mean>=0.5))
    n_pos_obs_AOI_min = sum(sum(BIAs_inAOI_min>=0.5))
    n_pos_obs_AOI_max = sum(sum(BIAs_inAOI_max>=0.5))
    # print estimates of min-max extent in area of interest
    print(f'percentage of blue ice located in AOI: {np.round(100*(n_pos_obs_AOI.values/n_pos_obs.values),2)}%')
    print(f'range of blue ice estimates in AOI: {np.round(100*(n_pos_obs_AOI_min.values/n_pos_obs_AOI.values),2)}% to {np.round(100*(n_pos_obs_AOI_max.values/n_pos_obs_AOI.values),2)}%')

# calculate uncertainties per region
print('grounding line +- 20 km')
uncertainties_per_region(aoi_gr_line)
print('mountainous terrain')
uncertainties_per_region(aoi_mt_terrain)
print('high latitudes')
uncertainties_per_region(aoi_high_lat)
print('high elevation')
uncertainties_per_region(aoi_high_elev)

#%%
# process the BIA predictions to obtain polygons instead of rasterized observations
# restructure the 2d raster into a dataframe (df) with each row representing a pixel
df = BIA_map.to_dataframe().reset_index()
# select only the lines where the mean prediction is >=0.5
pos_obs = df[df['mean']>=0.5].reset_index()

# define a function to create polygons from the center of the pixels
# y and x represent the center of the pixel, so to create polygons out of it we have to define a square around it of 200 by 200 meter
def square(center_x,center_y,resolution):
    min_x = center_x - resolution/2
    min_y = center_y - resolution/2
    max_x = center_x + resolution/2
    max_y = center_y + resolution/2
    poly = Polygon([(min_x,min_y),(max_x,min_y),(max_x,max_y),(min_x,max_y),(min_x,min_y)])
    return(poly)

# create geodataframe with square polygons for each pixel
gdf = geopandas.GeoDataFrame(
    pos_obs['mean'], 
    geometry=[square(x_,y_,200) for (x_,y_) in (pos_obs[['x','y']].values)],
    crs='epsg:3031'
)

# merge neighbouring polygons into larger polygons
gdf_merged = gdf.geometry.buffer(0.1).unary_union.buffer(-0.1) # this is done with .buffer(0.1) and .buffer(-0.1) to merge polygons that only share a corner
# create a dataframe from the merged polygons
gdf_merged_df = geopandas.GeoDataFrame(
    geometry=[gdf_merged],
    crs='epsg:3031'
)

# export the generated polygons to a file
gdf_merged_df.to_file('../output/shapefile_preds_nosmoothing.shp')

# convert the single multipolygon to many individual polygons
BIAs = gdf_merged_df.explode(index_parts=True)
# check the number of individual polygons
print('number of BIAs:', len(BIAs))

# delete variables that are not necessary
del(df, gdf, gdf_merged, gdf_merged_df)

#%%
# repeat for +1 std and -1 std
df = BIA_map.to_dataframe().reset_index()

pos_obs_max = df[df['mean']+df['std']>=0.5].reset_index()
pos_obs_min = df[df['mean']-df['std']>=0.5].reset_index()

# create geodataframe with square polygons for each pixel
gdf_max = geopandas.GeoDataFrame(
    pos_obs_max['mean'], 
    geometry=[square(x_,y_,200) for (x_,y_) in (pos_obs_max[['x','y']].values)],
    crs='epsg:3031'
)
gdf_min = geopandas.GeoDataFrame(
    pos_obs_min['mean'], 
    geometry=[square(x_,y_,200) for (x_,y_) in (pos_obs_min[['x','y']].values)],
    crs='epsg:3031'
)

# merge neighbouring polygons into larger polygons
gdf_merged_max = gdf_max.geometry.buffer(0.1).unary_union.buffer(-0.1)
gdf_merged_min = gdf_min.geometry.buffer(0.1).unary_union.buffer(-0.1)
# create a dataframe from the merged polygons
gdf_merged_max_df = geopandas.GeoDataFrame(
    geometry=[gdf_merged_max],
    crs='epsg:3031'
)
# create a dataframe from the merged polygons
gdf_merged_min_df = geopandas.GeoDataFrame(
    geometry=[gdf_merged_min],
    crs='epsg:3031'
)

# convert the single multipolygon to many individual polygons
BIAs_max = gdf_merged_max_df.explode(index_parts=True)
BIAs_min = gdf_merged_min_df.explode(index_parts=True)
# check the number of individual polygons
print('number of BIAs:', len(BIAs_max))
print('number of BIAs:', len(BIAs_min))

# delete variables that are not necessary
del(df, gdf_min, gdf_max, gdf_merged_min, gdf_merged_max, gdf_merged_min_df,gdf_merged_max_df)

#%%
# define smoothing function to smooth individual polygons
# based on the codes shared on https://stackoverflow.com/questions/47068504/where-to-find-python-implementation-of-chaikins-corner-cutting-algorithm/47255374#47255374
def smoothing(polygon,
            buffer=10, # in meters
            fact=0.75, # value between 0.5 and 1
            tolerance=250, # in meters
            refinements=5 # number of iterations
            ):
    # draw a small buffer around the polygon to account for the areal loss through rounding the edges
    polygon = polygon.buffer(buffer)
    # simplify the polygon
    polygon = polygon.simplify(tolerance=tolerance)
    boundaries = polygon.boundary
    # smooth polygon iteratively, try except for cases with multipolygons
    try:
        coords = np.array(list(polygon.boundary.coords))
        # loop over refinements
        for _ in range(refinements):
            L = coords.repeat(2, axis=0)
            R = np.empty_like(L)
            R[0] = L[0]
            R[2::2] = L[1:-1:2]
            R[1:-1:2] = L[2::2]
            R[-1] = L[-1]
            coords = L * fact + R * (1-fact)
            # to account for the fact that it is a polygon remove the first and last coordinates
            coords = coords[1:-1]
            # and append the new first coordinate to the rest
            coords = np.concatenate([coords,[coords[0]]])
        poly_to_return = Polygon(coords)
    except NotImplementedError:
        to_cut_from_poly = []
        coords = np.array(list(polygon.boundary[0].coords))
        for _ in range(refinements):
            L = coords.repeat(2, axis=0)
            R = np.empty_like(L)
            R[0] = L[0]
            R[2::2] = L[1:-1:2]
            R[1:-1:2] = L[2::2]
            R[-1] = L[-1]
            coords = L * fact + R * (1-fact)
            # to account for the fact that it is a polygon remove the first and last coordinates
            coords = coords[1:-1]
            # and append the new first coordinate to the rest
            coords = np.concatenate([coords,[coords[0]]])
        poly_to_return = Polygon(coords)
        for j in range(1,len(boundaries)):
            coords = np.array(list(polygon.boundary[j].coords))
            for _ in range(refinements):
                L = coords.repeat(2, axis=0)
                R = np.empty_like(L)
                R[0] = L[0]
                R[2::2] = L[1:-1:2]
                R[1:-1:2] = L[2::2]
                R[-1] = L[-1]
                coords = L * fact + R * (1-fact)
                # to account for the fact that it is a polygon remove the first and last coordinates
                coords = coords[1:-1]
                # and append the new first coordinate to the rest
                coords = np.concatenate([coords,[coords[0]]])
            # cut from polygon
            poly_to_return = poly_to_return.difference(Polygon(coords))
    return poly_to_return



#%%
# apply smoothing algorithm to clustered BIAs
BIAs_smoothed = [smoothing(BIAs.iloc[j].geometry, buffer=6, fact=0.75, tolerance=175, refinements=int(5)) for j in range(len(BIAs))]
# restructure as geopandas dataframe
BIAs_smoothed_dict = [{'geometry': poly} for poly in BIAs_smoothed]
gdf_smoothed_BIAs = geopandas.GeoDataFrame(
    BIAs_smoothed_dict,
    crs='epsg:3031'
)
# save as shapefile
gdf_smoothed_BIAs.to_file('../output/smoothed_BIAs.shp')
# check area of smoothed BIAs compared to area of raw BIAs
print(100*sum(gdf_smoothed_BIAs.area)/sum(BIAs.area) - 100)
del(BIAs_smoothed,BIAs_smoothed_dict)

#%%
# apply smoothing to maximum and minimum estimate of BIA extent
BIAs_smoothed_max = [smoothing(BIAs_max.iloc[j].geometry, buffer=6, fact=0.75, tolerance=175, refinements=int(5)) for j in range(len(BIAs_max))]
BIAs_smoothed_max_dict = [{'geometry': poly} for poly in BIAs_smoothed_max]
gdf_smoothed_BIAs_max = geopandas.GeoDataFrame(
    BIAs_smoothed_max_dict,
    crs='epsg:3031'
)
gdf_smoothed_BIAs_max.to_file('../output/smoothed_BIAs_max.shp')

BIAs_smoothed_min = [smoothing(BIAs_min.iloc[j].geometry, buffer=6, fact=0.75, tolerance=175, refinements=int(5)) for j in range(len(BIAs_min))]
BIAs_smoothed_min_dict = [{'geometry': poly} for poly in BIAs_smoothed_min]
gdf_smoothed_BIAs_min = geopandas.GeoDataFrame(
    BIAs_smoothed_min_dict,
    crs='epsg:3031'
)
gdf_smoothed_BIAs_min.to_file('../output/smoothed_BIAs_min.shp')

#%%
# visualize BIA predictions
# define colorbar (from 0 to 0.5 and from 0.5 to 1)
noblueice = cm.get_cmap('Greys', 256)
blueice = cm.get_cmap('Blues', 256)
newcolors = noblueice(np.linspace(0.3,0.6,256))
newcolors = np.concatenate([newcolors,blueice(np.linspace(0.5,1,256))])
newcmp = ListedColormap(newcolors)
cmap = newcmp
norm_BIA = mpl.colors.Normalize(vmin=0,vmax=1)
# define figure orientation
fig_orientation = 'portrait'
# define subaxes for overview map and subpanels
if fig_orientation == 'portrait':
    fig = plt.figure(figsize=(18/2.54, 24/2.54))
    gs = fig.add_gridspec(40, 30) # rows, columns
    # overview map
    ax_map = fig.add_subplot(gs[10:30,:20])# rows, columns
    # insets
    ax_ins1 = fig.add_subplot(gs[0:10,0:10])
    ax_ins2 = fig.add_subplot(gs[0:10,10:20])
    ax_ins3 = fig.add_subplot(gs[0:10,20:30])
    ax_ins4 = fig.add_subplot(gs[10:20,20:30])
    ax_ins5 = fig.add_subplot(gs[20:30,20:30])
    ax_ins6 = fig.add_subplot(gs[30:40,0:10])
    ax_ins7 = fig.add_subplot(gs[30:40,10:20])
    ax_ins8 = fig.add_subplot(gs[30:40,20:30])
if fig_orientation == 'landscape':
    fig = plt.figure(figsize=(24/2.54, 18/2.54))
    gs = fig.add_gridspec(30, 40) # rows, columns
    # overview map
    ax_map = fig.add_subplot(gs[10:30,10:30])# rows, columns
    # insets
    ax_ins1 = fig.add_subplot(gs[0:10,10:20])
    ax_ins2 = fig.add_subplot(gs[0:10,20:30])
    ax_ins3 = fig.add_subplot(gs[0:10,30:40])
    ax_ins4 = fig.add_subplot(gs[10:20,30:40])
    ax_ins5 = fig.add_subplot(gs[20:30,30:40])
    ax_ins6 = fig.add_subplot(gs[0:10,0:10])
    ax_ins7 = fig.add_subplot(gs[10:20,0:10])
    ax_ins8 = fig.add_subplot(gs[20:30,0:10])

# plot overview map
xr.plot.imshow(BIA_map['mean'],cmap=cmap,ax=ax_map,
               norm=norm_BIA,add_colorbar=False)
# set axes equal
ax_map.axis('equal')
# set subaxis for colorbar
ax_sub = inset_axes(ax_map, width=1.9, height=0.3, 
                        loc='center',bbox_to_anchor=((-1.2e6,-2.32e6)),
                        bbox_transform=ax_map.transData)
# add colorbar
cb = mpl.colorbar.ColorbarBase(ax_sub, cmap=cmap,
                                orientation='horizontal',
                                ticks=[0,0.5,1],norm=norm_BIA)
cb.ax.tick_params(labelsize=9)
for axis_ in ['top','bottom','left','right']:
    ax_sub.spines[axis_].set_linewidth(0.01)
    cb.ax.spines[axis_].set_linewidth(0.1)
cb.ax.xaxis.set_label_position('top')
cb.ax.set_xlabel('prediction values', fontsize=9)
cb.ax.annotate('BLUE ICE',xy=(0.75,0.5),horizontalalignment='center',
               verticalalignment='center',
               fontsize=8,weight='bold',annotation_clip=False)
cb.ax.annotate('NO BLUE ICE',xy=(0.25,0.5),horizontalalignment='center',
               verticalalignment='center',
               fontsize=8,weight='bold',annotation_clip=False)
# annotate overview map
ax_map.annotate('A',xy=(0.02,0.87),xycoords='axes fraction',fontsize=18,weight='bold',path_effects=[pe.withStroke(linewidth=2, foreground="white")])
# set axes off
ax_map.axis('off')
# plot graticules
# open data
longs_path = r'../data/60dg_longitude_clipped.shx'
longs_raw = geopandas.read_file(longs_path,encoding='utf-8')
lats_path = r'../data/10dg_latitude_clipped.shx'
lats_raw = geopandas.read_file(lats_path,encoding='utf-8')
# plot data
longs_raw.plot(ax=ax_map,color='k',linewidth=0.3,zorder=1)
lats_raw.plot(ax=ax_map,color='k',linewidth=0.3,zorder=1)
# plot annotations of graticules
longs_annot = longs_raw.boundary.explode(index_parts=True)[1::2]
for x, y, label in zip(longs_annot.geometry.x, longs_annot.geometry.y, longs_raw.Longitude):
    ax_map.annotate(label, xy=(x, y), xytext=(0, 0), fontsize=6, textcoords='offset points', va='center',ha='center',
                path_effects=[pe.withStroke(linewidth=0.8, foreground="white")])
ax_map.annotate(lats_raw.iloc[0].Latitude,xy=(0,2.1e6),textcoords='data',ha='center',fontsize=6,
            path_effects=[pe.withStroke(linewidth=0.8, foreground="white")])
ax_map.annotate(lats_raw.iloc[1].Latitude,xy=(0,1.1e6),textcoords='data',ha='center',fontsize=6,
            path_effects=[pe.withStroke(linewidth=0.8, foreground="white")])
# plot scalebar
scalebar = AnchoredSizeBar(ax_map.transData,
                           1e6, '1000 km', 
                           loc='lower right',
                           pad=2.14,#0.01, #0.0005
                           color='black',
                           frameon=False, 
                           size_vertical=(4e6)/60,
                           fontproperties=fm.FontProperties(size=9),
                           label_top=True,
                           sep=1)
ax_map.add_artist(scalebar)

# function to plot an inset
def plot_inset(center, # center coordinates (m)
               extent, # extent of horizontal axis in m
               ratio, # vertical/horizontal ratio
               axis, # sub axis to plot
               title, # string of label
               panel, # letter of panel
               loc_scalebar='lower left', # location of scalebar
               ):
    # define extent
    cent_x,cent_y = center
    min_x = cent_x - extent/2
    max_x = cent_x + extent/2
    min_y = cent_y - (extent*ratio)/2
    max_y = cent_y + (extent*ratio)/2
    # obtain arrays with values for x and y coordinates
    mask_x = (BIA_map.x >= min_x) & (BIA_map.x < max_x)
    mask_y = (BIA_map.y >= min_y) & (BIA_map.y < max_y)
    # crop image based on arrays with values for x and y coordinates
    img_cropped = BIA_map.where(mask_x & mask_y, drop=True)
    # plot data
    xr.plot.imshow(img_cropped['mean'],cmap=cmap,norm=norm_BIA,ax=axis,add_colorbar=False)
    # add scalebar
    len_scalebar = extent/6
    label_scalebar = f'{int(len_scalebar*1e-3)} km'
    scalebar = AnchoredSizeBar(axis.transData,
                            len_scalebar, label_scalebar, 
                            loc=loc_scalebar, 
                            pad=0.5, #0.0005
                            color='black',
                            frameon=False, 
                            size_vertical=extent/60,
                            fontproperties=fm.FontProperties(size=9),
                            label_top=True,
                            sep=1)
    axis.add_artist(scalebar)
    # switch off axes
    axis.axis('off')
    axis.annotate(title,xy=(0.15,0.91),xycoords='axes fraction',fontsize=9,weight='bold',path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    # plot extent of inset on main map
    ext_main_map = Polygon([[min_x,min_y],[max_x,min_y],[max_x,max_y],[min_x,max_y]])
    ax_map.plot(*ext_main_map.exterior.xy,color='k')
    # plot letter to annotate panel
    axis.annotate(panel,xy=(0.02,0.87),xycoords='axes fraction',fontsize=18,weight='bold',path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# plot insets
plot_inset(center=(0.45e6,2e6), extent=300e3, ratio=1, axis=ax_ins1, title='SCHIRMACHER OASIS',panel='B')
plot_inset(center=(0.78e6,1.8e6), extent=300e3, ratio=1, axis=ax_ins2, title='SOR RONDANE MTS',panel='C')
plot_inset(center=(1.15e6,1.60e6), extent=240e3, ratio=1, axis=ax_ins3, title='YAMATO',panel='D',loc_scalebar='lower right')
plot_inset(center=(1.68e6,0.74e6), extent=300e3, ratio=1, axis=ax_ins4, title='PRINCE CHARLES MTS',panel='E')
plot_inset(center=(2.545e6,-0.49e6), extent=180e3, ratio=1, axis=ax_ins5, title='APFEL GLACIER',panel='F')
plot_inset(center=(-0.25e6,-0.32e6), extent=240e3, ratio=1, axis=ax_ins6, title='QUEEN MAUD MTS',panel='G')
plot_inset(center=(0.21e6,-0.63e6), extent=360e3, ratio=1, axis=ax_ins7, title='QUEEN ALEXANDRA RANGE',panel='H')
plot_inset(center=(0.53e6,-1.33e6), extent=120e3, ratio=1, axis=ax_ins8, title='ALLAN HILLS',panel='I',loc_scalebar='lower center')

# plot settings
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0.5, wspace = 0.5)
# save figure
fig.savefig(f'../output/figures/BIA_map_{fig_orientation}.png',bbox_inches = 'tight',
            facecolor='white',
    pad_inches = 0,dpi=300)
#%%
# visualize  uncertainties
# define colorbar
c_uncertainties = cm.get_cmap('Oranges', 256)
newcolors = c_uncertainties(np.linspace(0,1,256))
newcmp = ListedColormap(newcolors)
cmap = newcmp
norm = mpl.colors.Normalize(vmin=0, vmax=BIA_map['std'].max())

# define figure orientation
fig_orientation = 'portrait'
# define subaxes for overview map and subpanels
if fig_orientation == 'portrait':
    fig = plt.figure(figsize=(18/2.54, 24/2.54))
    gs = fig.add_gridspec(40, 30) # rows, columns
    # overview map
    ax_map = fig.add_subplot(gs[10:30,:20])# rows, columns
    # insets
    ax_ins1 = fig.add_subplot(gs[0:10,0:10])
    ax_ins2 = fig.add_subplot(gs[0:10,10:20])
    ax_ins3 = fig.add_subplot(gs[0:10,20:30])
    ax_ins4 = fig.add_subplot(gs[10:20,20:30])
    ax_ins5 = fig.add_subplot(gs[20:30,20:30])
    ax_ins6 = fig.add_subplot(gs[30:40,0:10])
    ax_ins7 = fig.add_subplot(gs[30:40,10:20])
    ax_ins8 = fig.add_subplot(gs[30:40,20:30])
if fig_orientation == 'landscape':
    fig = plt.figure(figsize=(24/2.54, 18/2.54))
    gs = fig.add_gridspec(30, 40) # rows, columns
    # overview map
    ax_map = fig.add_subplot(gs[10:30,10:30])# rows, columns
    # insets
    ax_ins1 = fig.add_subplot(gs[0:10,10:20])
    ax_ins2 = fig.add_subplot(gs[0:10,20:30])
    ax_ins3 = fig.add_subplot(gs[0:10,30:40])
    ax_ins4 = fig.add_subplot(gs[10:20,30:40])
    ax_ins5 = fig.add_subplot(gs[20:30,30:40])
    ax_ins6 = fig.add_subplot(gs[0:10,0:10])
    ax_ins7 = fig.add_subplot(gs[10:20,0:10])
    ax_ins8 = fig.add_subplot(gs[20:30,0:10])

# plot overview map
xr.plot.imshow(BIA_map['std'],cmap=cmap,norm=norm,
               ax=ax_map,add_colorbar=False)
# set axes equal
ax_map.axis('equal')
# set subaxis for colorbar
ax_sub = inset_axes(ax_map, width=1.9, height=0.15, 
                        loc='center',bbox_to_anchor=((-1.2e6,-2.32e6)),
                        bbox_transform=ax_map.transData)
# add colorbar
cb = mpl.colorbar.ColorbarBase(ax_sub, cmap=cmap,
                               norm=norm,
                                orientation='horizontal',
                                ticks=[0,0.1,0.2,0.3])
cb.ax.tick_params(labelsize=9)
for axis_ in ['top','bottom','left','right']:
    ax_sub.spines[axis_].set_linewidth(0.01)
    cb.ax.spines[axis_].set_linewidth(0.1)
cb.ax.xaxis.set_label_position('top')
cb.ax.set_xlabel('std of prediction values', fontsize=9)
cb.ax.add_patch(Rectangle((0,1),0.08,0.22,facecolor='none', edgecolor='k',
                         label='BIA outlines',clip_on=False))
cb.ax.annotate('BIA outlines',xy=(0.095,1),fontsize=9,annotation_clip=False)
# annotate overview map
ax_map.annotate('A',xy=(0.02,0.87),xycoords='axes fraction',fontsize=18,weight='bold',path_effects=[pe.withStroke(linewidth=2, foreground="white")])
# set axes off
ax_map.axis('off')
# plot graticules
# open data
longs_path = r'../data/60dg_longitude_clipped.shx'
longs_raw = geopandas.read_file(longs_path,encoding='utf-8')
lats_path = r'../data/10dg_latitude_clipped.shx'
lats_raw = geopandas.read_file(lats_path,encoding='utf-8')
# plot data
longs_raw.plot(ax=ax_map,color='k',linewidth=0.3,zorder=1)
lats_raw.plot(ax=ax_map,color='k',linewidth=0.3,zorder=1)
# plot annotations of graticules
longs_annot = longs_raw.boundary.explode(index_parts=True)[1::2]
for x, y, label in zip(longs_annot.geometry.x, longs_annot.geometry.y, longs_raw.Longitude):
    ax_map.annotate(label, xy=(x, y), xytext=(0, 0), fontsize=6, textcoords='offset points', va='center',ha='center',
                path_effects=[pe.withStroke(linewidth=0.8, foreground="white")])
ax_map.annotate(lats_raw.iloc[0].Latitude,xy=(0,2.1e6),textcoords='data',ha='center',fontsize=6,
            path_effects=[pe.withStroke(linewidth=0.8, foreground="white")])
ax_map.annotate(lats_raw.iloc[1].Latitude,xy=(0,1.1e6),textcoords='data',ha='center',fontsize=6,
            path_effects=[pe.withStroke(linewidth=0.8, foreground="white")])
# plot scalebar
scalebar = AnchoredSizeBar(ax_map.transData,
                           1e6, '1000 km', 
                           loc='lower right',
                           pad=2.14,#0.01, #0.0005
                           color='black',
                           frameon=False, 
                           size_vertical=(4e6)/60,
                           fontproperties=fm.FontProperties(size=9),
                           label_top=True,
                           sep=1)
ax_map.add_artist(scalebar)

# function to plot an inset
def plot_inset(center, # center coordinates (m)
               extent, # extent of horizontal axis in m
               ratio, # vertical/horizontal ratio
               axis, # sub axis to plot
               title, # string of label
               panel, # letter of panel
               loc_scalebar='lower left', # location of scalebar
               ):
    # define extent
    cent_x,cent_y = center
    min_x = cent_x - extent/2
    max_x = cent_x + extent/2
    min_y = cent_y - (extent*ratio)/2
    max_y = cent_y + (extent*ratio)/2
    # obtain arrays with values for x and y coordinates
    mask_x = (BIA_map.x >= min_x) & (BIA_map.x < max_x)
    mask_y = (BIA_map.y >= min_y) & (BIA_map.y < max_y)
    # crop image based on arrays with values for x and y coordinates
    img_cropped = BIA_map.where(mask_x & mask_y, drop=True)
    # plot data
    xr.plot.imshow(img_cropped['std'],cmap=cmap,norm=norm,
                   ax=axis,add_colorbar=False)
    gdf_smoothed_BIAs.plot(ax=axis,linewidth=0.8,
                facecolor="none",edgecolor='k')
    axis.set_xlim([min_x,max_x])
    axis.set_ylim([min_y,max_y])
    # add scalebar
    len_scalebar = extent/6
    label_scalebar = f'{int(len_scalebar*1e-3)} km'
    scalebar = AnchoredSizeBar(axis.transData,
                            len_scalebar, label_scalebar, 
                            loc=loc_scalebar, 
                            pad=0.5, #0.0005
                            color='black',
                            frameon=False, 
                            size_vertical=extent/60,
                            fontproperties=fm.FontProperties(size=9),
                            label_top=True,
                            sep=1)
    axis.add_artist(scalebar)
    # switch off axes
    axis.axis('off')
    axis.annotate(title,xy=(0.15,0.91),xycoords='axes fraction',fontsize=9,weight='bold',path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    # plot extent of inset on main map
    ext_main_map = Polygon([[min_x,min_y],[max_x,min_y],[max_x,max_y],[min_x,max_y]])
    ax_map.plot(*ext_main_map.exterior.xy,color='k')
    # plot letter to annotate panel
    axis.annotate(panel,xy=(0.02,0.87),xycoords='axes fraction',fontsize=18,weight='bold',path_effects=[pe.withStroke(linewidth=2, foreground="white")])


# plot insets
plot_inset(center=(0.45e6,2e6), extent=300e3, ratio=1, axis=ax_ins1, title='SCHIRMACHER OASIS',panel='B')
plot_inset(center=(0.78e6,1.8e6), extent=300e3, ratio=1, axis=ax_ins2, title='SOR RONDANE MTS',panel='C')
plot_inset(center=(1.15e6,1.60e6), extent=240e3, ratio=1, axis=ax_ins3, title='YAMATO',panel='D',loc_scalebar='lower right')
plot_inset(center=(1.68e6,0.74e6), extent=300e3, ratio=1, axis=ax_ins4, title='PRINCE CHARLES MTS',panel='E')
plot_inset(center=(2.545e6,-0.49e6), extent=180e3, ratio=1, axis=ax_ins5, title='APFEL GLACIER',panel='F')
plot_inset(center=(-0.25e6,-0.32e6), extent=240e3, ratio=1, axis=ax_ins6, title='QUEEN MAUD MTS',panel='G')
plot_inset(center=(0.21e6,-0.63e6), extent=360e3, ratio=1, axis=ax_ins7, title='QUEEN ALEXANDRA RANGE',panel='H')
plot_inset(center=(0.53e6,-1.33e6), extent=120e3, ratio=1, axis=ax_ins8, title='ALLAN HILLS',panel='I',loc_scalebar='lower center')

# plot settings
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0.5, wspace = 0.5)
# save figure
fig.savefig(f'../output/figures/BIA_map_uncertainties_{fig_orientation}.png',bbox_inches = 'tight',
            facecolor='white',
    pad_inches = 0,dpi=300)

#%%
# calculate area of individual BIAs
# copy BIAs
BIAs_cp = BIAs.copy()

# reset index
BIAs_cp = BIAs_cp.reset_index(level=[0]).drop(columns='level_0')

# calculate area of individual BIAs
BIAs_cp['area'] = BIAs_cp.area*1e-6
# plot histogram of BIAs area
plt.hist(BIAs_cp['area'])

#%%
# create geodataframe with Points for each pixel
gdf = geopandas.GeoDataFrame(
    pos_obs['mean'], 
    geometry=[Point((x_,y_)) for (x_,y_) in (pos_obs[['x','y']].values)],
    crs='epsg:3031'
)
#%%
# calculate some statistics

# 1. elevation of BIAs
# import elevation data and correct elevation data for geoid
# open TDX DEM
TDX_path = r'../data/TDM_merged.tif'
TDX_raw = xr.open_rasterio(TDX_path)

# convert DataArray to DataSet
TDX_ds = TDX_raw.drop('band')[0].to_dataset(name='DEM_ellipsoid')

# open geoid
geoid_path = r'../data/EIGEN-6C4_HeightAnomaly_10km.tif'
geoid_raw = xr.open_rasterio(geoid_path)
geoid_raw.attrs['nodatavals'] = (np.nan,)

# convert DataArray to DataSet
geoid_ds = geoid_raw.drop('band')[0].to_dataset(name='geoid')
# reset nodatavalues to nan (to avoid artifacts in interpolation)
geoid_ds = geoid_ds.where(geoid_ds['geoid'] != -9999.)

# interpolate geoid height to same grid as elevation data
interpolated_geoid = geoid_ds.interp_like(TDX_ds)

# calculate orthometric height
TDX_ortho = (TDX_ds.DEM_ellipsoid - interpolated_geoid.geoid).astype('float32')
TDX_ortho_ds = TDX_ortho.to_dataset(name='DEM')

del(TDX_raw,TDX_ds,geoid_raw,geoid_ds,interpolated_geoid,TDX_ortho)

TDX_ortho_ds.to_netcdf(r'../data/TDM_merged_ortho.nc')

# OR open corrected elevation data directly
TDX_orhto_path = r'../data/TDM_merged_ortho.nc'
TDX_ortho_ds = xr.open_dataset(TDX_orhto_path)

# extract values at BIA locations
DEM_at_BIA = TDX_ortho_ds.interp(x=gdf.geometry.x.to_xarray(),
                   y=gdf.geometry.y.to_xarray())

elev_at_BIA = DEM_at_BIA.to_dataframe()[['x','y','DEM']]
gdf['DEM'] = elev_at_BIA['DEM']
print(f'50% of blue ice is located below {gdf.DEM.median()} meter')

# export file
gdf.to_file('../output/gdf_bias_elev.shp')
#%%

# 2. how much % of blue ice is located at latitudes > 82.5?
transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326")
lat, lon = transformer.transform(gdf.geometry.x,gdf.geometry.y)

gdf['lat'] = lat

#%%
# 3. distance to coast --> maybe use individual pixels for this to avoid having influence of small blue ice patches ("most blue ice is located near the coast")
# open ice boundaries
ice_boundaries_path = r'../data/IceBoundaries_Antarctica_v2.shx'
ice_boundaries_raw = geopandas.read_file(ice_boundaries_path)
# merge ice boundaries
ice_boundaries = ice_boundaries_raw['geometry'].unary_union
# calculate distance to ice boundary
gdf['dist_ice_boundary'] = [ice_boundaries.boundary.distance(gdf.iloc[i].geometry)*1e-3 for i in range(len(gdf))]
# export file (so that individual cells of script can run)
gdf.to_file('../output/gdf_bias_elevation_lat_iceboundary.shp')
#%%
# import gdf
gdf = geopandas.read_file('../output/gdf_bias_elevation_lat_iceboundary.shx')
#%%
# 4. calculate distance to grounding line
# open grounding line data (data from measures v2 (part of quantarcitca))
gr_line_path = r'../data/GroundingLine_Antarctica_v2.shx'
gr_line_raw = geopandas.read_file(gr_line_path)
# merge ice boundaries
gr_line = gr_line_raw['geometry'].unary_union
# calculate distance to grounding line
gdf['dist_gr_line'] = [gr_line.boundary.distance(gdf.iloc[i].geometry)*1e-3 for i in range(len(gdf))]
# export data
gdf.to_file('../output/gdf_bias_elevation_lat_iceboundary_grline.shp')

#%%
# 5. how much blue ice is in mountainous terrain?
# definition mountainous terrain:
# https://www.easa.europa.eu/en/document-library/easy-access-rules/online-publications/easy-access-rules-standardised-european?page=4
# i.e.,  changes of terrain elevation exceed 900 m (3 000 ft) within a distance of 18,5 km
# calculate mountainous area:
# 1. circular kernel to find places within 18.5 km of a single point
# 2. take min and max in this area, subtract and check if difference > 900 m

# distance is 93 pixels on each side (21 * 0.09 * 10 = 18.9 km) + 1 center pixel --> 43 pixels
kernel = np.zeros((43,43))
mid = (len(kernel)-1)/2
radius = (len(kernel)-1)/2
for i in range(len(kernel)):
    for j in range(len(kernel)):
        if np.sqrt(((i-mid)**2 + (j-mid)**2)) <= radius:
            kernel[i,j]=1
plt.imshow(kernel)  

# coarsen DEM to speed up calculations (this is considered in defining the kernel)
DEM_coarsened = TDX_ortho_ds.DEM.values[::10,::10]

# calculate max elevation within kernel
elevation_max = generic_filter(DEM_coarsened,
                                    np.max, footprint = kernel)
# calculate min elevation within kernel
elevation_min = generic_filter(DEM_coarsened,
                                    np.min, footprint = kernel)
# store elevation differences in Dataset
elevation_differences = xr.Dataset({
               "elevation_diff": (("y", "x"), elevation_max-elevation_min)},
               coords={"x": TDX_ortho_ds.x.values[::10], 
                       "y": TDX_ortho_ds.y.values[::10]})
# export dataset
elevation_differences.to_netcdf('../output/elevation_differences.nc')
#%%
# import dataset
elevation_differences = xr.open_dataset('../output/elevation_differences.nc')
# calculate elevation difference for each BIA pixel
mount_at_BIA = elevation_differences.interp(x=gdf.geometry.x.to_xarray(),
                   y=gdf.geometry.y.to_xarray())
# store as dataframe
mount_at_BIA = mount_at_BIA.to_dataframe()[['x','y','elevation_diff']]
# append to geodataframe
gdf['elev_diff'] = mount_at_BIA['elevation_diff']
# export data
gdf.to_file('../output/gdf_bias_elevation_lat_iceboundary_grline_mountain.shp')
#%%
# import geodataframe with all values (calculated above)
gdf = geopandas.read_file('../output/gdf_bias_elevation_lat_iceboundary_grline_mountain.shx')

# -------------------------- #
# print calculated statistics
# distance to the ice boundary (in km) (n. 3)
plt.hist(gdf['dist_ice_b'])
print(len(gdf[gdf['dist_ice_b']<50])/len(gdf))
plt.show()
# distance to the grounding line (in km) (n. 4)
plt.hist(gdf['dist_gr_li'])
print(f'50 % of blue ice is within {gdf["dist_gr_li"].quantile(0.5)} km of the grounding line')
print(len(gdf[gdf['dist_gr_li']<20])/len(gdf))
# total area around grounding line (in km2) (n. 4)
area_gr_line = gr_line.buffer(20*1e3).difference(gr_line.buffer(-20*1e3)) 
area_gr_line_mask_boundaries = ice_boundaries.intersection(area_gr_line)
# total area blue ice in 20 km from grounding line (n. 4)
area_BIA_gr_line = len(gdf[gdf['dist_gr_li']<20])*200*200
# print percentage of area near grounding line that exposes blue ice
print(f'{100*(area_BIA_gr_line/area_gr_line_mask_boundaries.area)}% of the area near the grounding line exposes blue ice')
#%%
# print percentage of blue ice in mountainous terrain (definition: in 18.5 km radius more than 900m elevation difference) (n. 5)
print(f"{100*len(gdf[gdf['elev_diff']>900])/len(gdf)}% of blue ice is located in mountainous terrain")
#%%
# total area mountainous terrain
# mask out values outside ice boundaries
# update transform parameters elevation_differences
resolution_elev = 900
ll_x_elev = elevation_differences.x.min().values
ur_y_elev = elevation_differences.y.max().values
elevation_differences.attrs['transform'] = affine.Affine(resolution_elev, 0.0, ll_x_elev-(resolution_elev/2), 0.0, -1*resolution_elev, ur_y_elev+(resolution_elev/2))
# create mask
ShapeMask = features.geometry_mask([ice_boundaries],
                                        out_shape=(len(elevation_differences.y), len(elevation_differences.x)),
                                        transform=elevation_differences.transform,
                                        invert=True)
# assign coordinates and create a data array
ShapeMask = xr.DataArray(ShapeMask, coords={"y":elevation_differences.y,
                                                "x":elevation_differences.x},
                            dims=("y", "x"))
# Create Data Array with zeros
zeros_tomask = xr.zeros_like(elevation_differences['elevation_diff'])
# apply Mask to zeros_tomask --> 1 = BIA, 0 = no BIA
elev_mask = xr.where((ShapeMask == True),elevation_differences['elevation_diff'],zeros_tomask)
# add elev_mask to dataset
elevation_differences['elevation_diff_masked'] = elev_mask
# estimate area mountainous terrain
area_mountainous_terrain = sum(sum(xr.where(elevation_differences['elevation_diff_masked']>900,1,0))).values*0.9*0.9
# print percentage of mountainous terrain that exposes blue ice
area_BIA_mt_terr = len(gdf[gdf['elev_diff']>900])*0.2*0.2
print(f'{100*(area_BIA_mt_terr/area_mountainous_terrain)}% of mountainous terrain exposes blue ice')

#%%
# elevation of blue ice (n. 1)
print(f"{100*(len(gdf[gdf['DEM']>1500])/len(gdf))}% of blue ice is at elevations higher than 1500m")

#%%
# calculate what percentage of blue ice is located at latitudes south of 82.5S (n. 2)
print(f"{100*sum(gdf['lat']<-82.5)/len(gdf)}% of blue ice is located at latitudes > 82.5S")

# calculate what percentage of high elevation blue ice is located at latitudes south of 82.5S (n. 1 & 2)
perc_high_elev_lat = 100*len(gdf[(gdf['lat']<-82.5)&(gdf['DEM']>1500)])/len(gdf[gdf['DEM']>1500])
print(f"{perc_high_elev_lat}% of high elevation blue ice is located at latitudes >82.5S")

#%%
# calculate distance to BIA of Hui et al. for "new" blue ice
newice = pos_obs[pos_obs['BIAs_hui']==0]
# create geodataframe with Points for each pixel
gdf_newice = geopandas.GeoDataFrame(
    newice['mean'], 
    geometry=[Point((x_,y_)) for (x_,y_) in (newice[['x','y']].values)],
    crs='epsg:3031'
)
# simplify BIAs of Hui et al., 2014 (to speed up calculations)
BIAs_hui_simplified = BIAs_hui.simplify(tolerance=100)
# unary union of shapefile
BIAs_hui_uu = BIAs_hui_simplified.unary_union
# calculate distance from newice to hui's blue ice
gdf_newice['dist_hui'] = [BIAs_hui_uu.boundary.distance(gdf_newice.iloc[i].geometry)*1e-3 for i in range(len(gdf_newice))]
# export data
gdf_newice.to_file('../output/gdf_newice.shp')
# plot histogram
plt.hist(gdf_newice['dist_hui'],bins=20)
plt.xlabel('distance in km')
plt.title('distance to hui BIAs')
plt.savefig('../output/figures/dist_huiBIAs.png')
# print statistics
print(len(gdf_newice[gdf_newice['dist_hui']<1])/len(gdf_newice))


#%%
# spectral fingerprint of blue ice (Figure 2)
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

# save BIA predictions as dataframe
df = BIA_map.to_dataframe().reset_index()
# set number of random samples
n_samples = 10000
# select rows that are certainly blue ice etc, certainly no blue ice, and in the regions of disagreement
bia_cert = df[(df['mean']>=0.5)&(df['BIAs_hui']==1)].sample(n=n_samples,random_state=0)
nobia_cert = df[(df['mean']<0.5)&(df['BIAs_hui']==0)].sample(n=n_samples,random_state=1)
bia_new = df[(df['mean']>=0.5)&(df['BIAs_hui']==0)].sample(n=n_samples,random_state=2)
nobia_new = df[(df['mean']<0.5)&(df['BIAs_hui']==1)].sample(n=n_samples,random_state=3)
# export different point selections as .csv
bia_cert.to_csv('../output/spect_fing_bia.csv',index_label='index')
nobia_cert.to_csv('../output/spect_fing_nobia.csv',index_label='index')
bia_new.to_csv('../output/spect_fing_aodii.csv',index_label='index')
nobia_new.to_csv('../output/spect_fing_aodi.csv',index_label='index')
#%%
# MODIS bandwidths (https://modis.gsfc.nasa.gov/about/specifications.php#2) in nm
MODIS_bw = np.array([(670+620)/2,
            (876+841)/2,
            (479+459)/2,
            (565+545)/2,
            (1250+1230)/2,
            (1652+1628)/2,
            (2155+2105)/2])
# --> order from small to large is [2,3,0,1,4,5,6]
#%%
# extract MODIS values at BIA locations
MOD_at_BIA = MOD_ds.interp(x=bia_cert.x.to_xarray(),
                   y=bia_cert.y.to_xarray())
# format to dataframe
MOD_at_BIA_df = MOD_at_BIA.to_dataframe()[['x','y','MOD_B1','MOD_B2','MOD_B3','MOD_B4','MOD_B5','MOD_B6','MOD_B7']]

# extract values at no BIA locations
MOD_at_noBIA = MOD_ds.interp(x=nobia_cert.x.to_xarray(),
                   y=nobia_cert.y.to_xarray())
# format to dataframe
MOD_at_noBIA_df = MOD_at_noBIA.to_dataframe()[['x','y','MOD_B1','MOD_B2','MOD_B3','MOD_B4','MOD_B5','MOD_B6','MOD_B7']]

#%%
# extract values at new BIA locations
MOD_at_BIAnew = MOD_ds.interp(x=bia_new.x.to_xarray(),
                   y=bia_new.y.to_xarray())
# format to dataframe
MOD_at_BIAnew_df = MOD_at_BIAnew.to_dataframe()[['x','y','MOD_B1','MOD_B2','MOD_B3','MOD_B4','MOD_B5','MOD_B6','MOD_B7']]

# extract values at new noBIA locations
MOD_at_noBIAnew = MOD_ds.interp(x=nobia_new.x.to_xarray(),
                   y=nobia_new.y.to_xarray())
# format to dataframe
MOD_at_noBIAnew_df = MOD_at_noBIAnew.to_dataframe()[['x','y','MOD_B1','MOD_B2','MOD_B3','MOD_B4','MOD_B5','MOD_B6','MOD_B7']]

#%%
# plot spectral fingerprints
# define colors
c_BIA = '#85CFEE'
c_noBIA = 'k'
c_AODi = '#F7941D'
c_AODii = '#9E1F63'
# plot figure
fig,axs = plt.subplots(2,1,figsize=(9/2.54, 11/2.54))
# plot spectral fingerprints MODIS for the four different regions
axs[0].plot(MODIS_bw[[2,3,0,1,4,5,6]],MOD_at_noBIA_df.median()[2:][[2,3,0,1,4,5,6]]/10000,color=c_noBIA,label='AOA: No blue ice',
         linewidth=2,zorder=5)
axs[0].plot(MODIS_bw[[2,3,0,1,4,5,6]],MOD_at_noBIAnew_df.median()[2:][[2,3,0,1,4,5,6]]/10000,color=c_AODi,label='AOD: Type I',
         linewidth=2,linestyle=(0,(5,1)),zorder=6)
axs[0].plot(MODIS_bw[[2,3,0,1,4,5,6]],MOD_at_BIAnew_df.median()[2:][[2,3,0,1,4,5,6]]/10000,color=c_AODii,label='AOD: Type II',
         linewidth=2,linestyle=(0,(3,1,1,1)),zorder=7)
axs[0].plot(MODIS_bw[[2,3,0,1,4,5,6]],MOD_at_BIA_df.median()[2:][[2,3,0,1,4,5,6]]/10000,color=c_BIA,label='AOA: Blue ice',
         linewidth=2,zorder=8)
# plot uncertainty bands (25th and 75th percentile)
alpha_all = 0.5
uncert_interval = 0.25
lw_fill = 0.8
mpl.rcParams['hatch.linewidth'] = 0.3
axs[0].fill_between(MODIS_bw[[2,3,0,1,4,5,6]],y1=MOD_at_BIA_df.quantile(uncert_interval)[2:][[2,3,0,1,4,5,6]]/10000,
                 y2=MOD_at_BIA_df.quantile(0.5+uncert_interval)[2:][[2,3,0,1,4,5,6]]/10000,
                 facecolor="none",edgecolor=c_BIA,linewidth=lw_fill,alpha=alpha_all,zorder=8,hatch="......")
axs[0].fill_between(MODIS_bw[[2,3,0,1,4,5,6]],y1=MOD_at_noBIA_df.quantile(uncert_interval)[2:][[2,3,0,1,4,5,6]]/10000,
                 y2=MOD_at_noBIA_df.quantile(0.5+uncert_interval)[2:][[2,3,0,1,4,5,6]]/10000,
                 facecolor="none",edgecolor=c_noBIA,linewidth=lw_fill,alpha=alpha_all,zorder=5,hatch='\\\\\\\\\\\\')
axs[0].fill_between(MODIS_bw[[2,3,0,1,4,5,6]],y1=MOD_at_BIAnew_df.quantile(uncert_interval)[2:][[2,3,0,1,4,5,6]]/10000,
                 y2=MOD_at_BIAnew_df.quantile(0.5+uncert_interval)[2:][[2,3,0,1,4,5,6]]/10000,
                 facecolor="none",edgecolor=c_AODii,linewidth=lw_fill,alpha=alpha_all,zorder=7,hatch="------")
axs[0].fill_between(MODIS_bw[[2,3,0,1,4,5,6]],y1=MOD_at_noBIAnew_df.quantile(uncert_interval)[2:][[2,3,0,1,4,5,6]]/10000,
                 y2=MOD_at_noBIAnew_df.quantile(0.5+uncert_interval)[2:][[2,3,0,1,4,5,6]]/10000,
                 facecolor="none",edgecolor=c_AODi,linewidth=lw_fill,alpha=alpha_all,zorder=6,hatch="||||||")

# plot labels
axs[0].set_title('MODIS')
axs[0].set_xlabel('Wavelength (nm)')
axs[0].set_ylabel('Reflectance')

# plot RGB colors
red_spectr = [620,670]
green_spectr = [545,565]
blue_spectr = [459,479]
nir_spectr = [841,876]
h_patch = 0
thickness_patch = 0.1
axs[0].add_patch(matplotlib.patches.Rectangle((red_spectr[0],h_patch),red_spectr[1]-red_spectr[0],thickness_patch,
                                               facecolor='red',clip_on=False))
axs[0].add_patch(matplotlib.patches.Rectangle((green_spectr[0],h_patch),green_spectr[1]-green_spectr[0],thickness_patch,
                                               facecolor='green',clip_on=False))
axs[0].add_patch(matplotlib.patches.Rectangle((blue_spectr[0],h_patch),blue_spectr[1]-blue_spectr[0],thickness_patch,
                                               facecolor='blue',clip_on=False))
axs[0].add_patch(matplotlib.patches.Rectangle((nir_spectr[0],h_patch),nir_spectr[1]-nir_spectr[0],thickness_patch,
                                               facecolor='orange',clip_on=False))

# import Landsat data (retrieved from GEE, https://code.earthengine.google.com/a9ca4bd4fd1f11a9a4ad2e4cd91fd843)
bia_refl = pd.read_csv(r'../data/bia_reflectances.csv')
nobia_refl = pd.read_csv(r'../data/nobia_reflectances.csv')
aodi_refl = pd.read_csv(r'../data/aodi_reflectances.csv')
aodii_refl = pd.read_csv(r'../data/aodii_reflectances.csv')

# define bands of Landsat (https://developers.google.com/earth-engine/datasets/catalog/USGS_LIMA_SR#bands)
B1 = 1e3*(0.45 + 0.52)/2 # nm 	
B2 = 1e3*(0.52 + 0.60)/2 # nm
B3 = 1e3*(0.63 + 0.69)/2 # nm
B4 = 1e3*(0.77 + 0.90)/2 # nm 	
B5 = 1e3*(1.55 + 1.75)/2 # nm 	
B7 = 1e3*(2.08 + 2.35)/2 # nm 	
B8 = 1e3*(0.52 + 0.90)/2 # nm --> panchromatic band, not use for spectr fingerprint

BWs = [B1, B2, B3, B4, B5, B7]
Bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7']

# select only places where reflectance values are available
bia_toplot = bia_refl[bia_refl['B1']>0][Bands]/10000
nobia_toplot = nobia_refl[nobia_refl['B1']>0][Bands]/10000
aodi_toplot = aodi_refl[aodi_refl['B1']>0][Bands]/10000
aodii_toplot = aodii_refl[aodii_refl['B1']>0][Bands]/10000

# plot spectral fingerprints Landsat for four different regions
axs[1].plot(BWs,nobia_toplot.median(),color=c_noBIA,label='AOA: No blue ice',
         linewidth=2,zorder=5)
axs[1].plot(BWs,aodi_toplot.median(),color=c_AODi,label='AOD: Type I',
         linewidth=2,linestyle=(0,(5,1)),zorder=6)
axs[1].plot(BWs,aodii_toplot.median(),color=c_AODii,label='AOD: Type II',
         linewidth=2,linestyle=(0,(3,1,1,1)),zorder=7)
axs[1].plot(BWs,bia_toplot.median(),color=c_BIA,label='AOA: Blue ice',
         linewidth=2,zorder=8)

# plot uncertainty bands (25th and 75th percentile)
axs[1].fill_between(BWs,y1=bia_toplot.quantile(0.5-uncert_interval),
                 y2=bia_toplot.quantile(0.5+uncert_interval),facecolor="none",edgecolor=c_BIA,
                 linewidth=lw_fill,alpha=alpha_all,zorder=8,hatch="......",label='AOA: Blue ice')
axs[1].fill_between(BWs,y1=nobia_toplot.quantile(0.5-uncert_interval),
                 y2=nobia_toplot.quantile(0.5+uncert_interval),facecolor="none",edgecolor=c_noBIA,
                 linewidth=lw_fill,alpha=alpha_all,zorder=5,hatch='\\\\\\\\\\\\',label='AOA: No blue ice')
axs[1].fill_between(BWs,y1=aodii_toplot.quantile(0.5-uncert_interval),
                 y2=aodii_toplot.quantile(0.5+uncert_interval),facecolor="none",edgecolor=c_AODii,
                 linewidth=lw_fill,alpha=alpha_all,zorder=7,hatch="------")
axs[1].fill_between(BWs,y1=aodi_toplot.quantile(0.5-uncert_interval),
                 y2=aodi_toplot.quantile(0.5+uncert_interval),facecolor="none",edgecolor=c_AODi,
                 linewidth=lw_fill,alpha=alpha_all,zorder=6,hatch="||||||")

# add legend (manually)
p_noBIA = matplotlib.patches.Patch(facecolor="none",edgecolor=c_noBIA,linewidth=lw_fill,alpha=alpha_all,hatch="\\\\\\\\\\\\")
l_noBIA = mlines.Line2D([], [], color=c_noBIA, linewidth=2)
p_aodi = matplotlib.patches.Patch(facecolor="none",edgecolor=c_AODi,linewidth=lw_fill,alpha=alpha_all,hatch="||||||")
l_aodi = mlines.Line2D([], [], color=c_AODi,linestyle=(0,(5,1)), linewidth=2)
p_aodii = matplotlib.patches.Patch(facecolor="none",edgecolor=c_AODii,linewidth=lw_fill,alpha=alpha_all,hatch="------")
l_aodii = mlines.Line2D([], [], color=c_AODii,linestyle=(0,(3,1,1,1)), linewidth=2)
p_BIA = matplotlib.patches.Patch(facecolor="none",edgecolor=c_BIA,linewidth=lw_fill,alpha=alpha_all,hatch="......")
l_BIA = mlines.Line2D([], [], color=c_BIA, linewidth=2)
axs[0].legend([(p_noBIA,l_noBIA),(p_aodi,l_aodi),(p_aodii,l_aodii),(p_BIA, l_BIA)],['AOA: No blue ice','AOD: Type I', 'AOD: Type II','AOA: Blue ice'])

# plot labels
axs[1].set_title('Landsat')
axs[1].set_xlabel('Wavelength (nm)')
axs[1].set_ylabel('Reflectance')

# plot RGB colors
red_spectr = [630,690]
green_spectr = [520,600]
blue_spectr = [450,520]
nir_spectr = [770,900]
h_patch = 0
thickness_patch = 0.1
axs[1].add_patch(matplotlib.patches.Rectangle((red_spectr[0],h_patch),red_spectr[1]-red_spectr[0],thickness_patch,
                                               facecolor='red',clip_on=False))
axs[1].add_patch(matplotlib.patches.Rectangle((green_spectr[0],h_patch),green_spectr[1]-green_spectr[0],thickness_patch,
                                               facecolor='green',clip_on=False))
axs[1].add_patch(matplotlib.patches.Rectangle((blue_spectr[0],h_patch),blue_spectr[1]-blue_spectr[0],thickness_patch,
                                               facecolor='blue',clip_on=False))
axs[1].add_patch(matplotlib.patches.Rectangle((nir_spectr[0],h_patch),nir_spectr[1]-nir_spectr[0],thickness_patch,
                                               facecolor='orange',clip_on=False))
# set axes limits
axs[0].set_xlim([400,2300])
axs[1].set_xlim([400,2300])
axs[0].set_ylim([0,1])
axs[1].set_ylim([0,1])
# adjust spacing
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0.4, wspace = 0)
plt.margins(0,0)
# save figure
fig.savefig('../output/figures/spectral_fingerprints.png',bbox_inches = 'tight',
    pad_inches = 0.02,dpi=300)

#%%
# plot histograms of Landsat data (Figure S13)
# define bandwiths
B1 = '450 - 520 nm \n(blue)' 	
B2 = '520 - 600 nm \n(green)'
B3 = '630 - 690 nm \n(red)'
B4 = '770 - 900 nm \n(near infrared)'
B5 = '1550 - 1750 nm \n(shortwave infrared 1)' 	
B7 = '2080 - 2350 nm \n(shortwave infrared 2)'	
B8 = '520 - 900 nm \n(panchromatic)'

BWs_labels = [B1, B2, B3, B4, B5, B7, B8]
Bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7','B8']
# select only places where reflectance values are available
bia_toplot = bia_refl[bia_refl['B1']>0][Bands]/10000
nobia_toplot = nobia_refl[nobia_refl['B1']>0][Bands]/10000
aodi_toplot = aodi_refl[aodi_refl['B1']>0][Bands]/10000
aodii_toplot = aodii_refl[aodii_refl['B1']>0][Bands]/10000

# plot figure
fig,axs = plt.subplots(2,4,figsize=(18/2.54, 9/2.54))
# function to plot histogram
def plot_hist(ax,band,label_n):
    # concat all to calculate the 1st and 99th perecentile for extents of plot
    all_values = pd.concat([nobia_toplot[band],bia_toplot[band],
               aodi_toplot[band],aodii_toplot[band]])
    # define limits of plot
    xmin = all_values.quantile(0.01)
    xmax = all_values.quantile(0.99)
    # define bins
    bins = np.linspace(xmin,xmax,50)
    # plot histogram
    ax.hist(nobia_toplot[band],color=c_noBIA,bins=bins,label='AOA: No blue ice');
    ax.hist(bia_toplot[band],color=c_BIA,bins=bins,alpha=0.5,label='AOA: Blue ice');
    ax.hist(aodi_toplot[band],color=c_AODi,bins=bins,histtype='step',label='AOD: Type I');
    ax.hist(aodii_toplot[band],color=c_AODii,bins=bins,histtype='step',label='AOD: Type II');
    # set limits and labels
    ax.set_xlim([xmin,xmax])
    ax.set_title(f'{band}, {BWs_labels[label_n]}',fontsize=10)

# plot histograms for all different bands
plot_hist(axs[0,0],'B1',0)
plot_hist(axs[0,1],'B2',1)
plot_hist(axs[0,2],'B3',2)
plot_hist(axs[0,3],'B4',3)
plot_hist(axs[1,0],'B5',4)
plot_hist(axs[1,1],'B7',5)
plot_hist(axs[1,2],'B8',6)

# set labels, plot legend
axs[1,0].set_xlabel('Reflectance',fontsize=9)
axs[0,0].set_ylabel('Count',fontsize=9)
axs[1,0].set_ylabel('Count',fontsize=9)
axs[1,2].legend(bbox_to_anchor=(1.15, 0.7))
axs[1,3].axis('off')
axs[1,2].annotate('Landsat data',xy=(1.86,0.7),
                annotation_clip=False,
                xycoords='axes fraction',
                ha='center')
# adjust spacing
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0.5, wspace = 0.4)
plt.margins(0,0)
# save figure
fig.savefig('../output/figures/spectral_histo_Landsat.png',bbox_inches = 'tight',
    pad_inches = 0.02,dpi=300)

#%%
# plot histograms of MODIS data
# define bandwiths
B1m = '620 - 670 nm \n(red)'
B2m = '841 - 876 nm \n(near infrared)'
B3m = '459 - 479 nm \n(blue)'
B4m = '545 - 565 nm \n(green)'
B5m = '1230 - 1250 nm \n(shortwave infrared 1)'
B6m = '1628 - 1652 nm \n(shortwave infrared 2)'
B7m = '2105 - 2155 nm \n(shortwave infrared 3)'

BWs_labels = [B1m, B2m, B3m, B4m, B5m, B6m, B7m]
Band_names = ['MOD_B1','MOD_B2','MOD_B3','MOD_B4','MOD_B5','MOD_B6','MOD_B7']
# plot figure
fig,axs = plt.subplots(2,4,figsize=(18/2.54, 9/2.54))
# function to plot histogram
def plot_hist(ax,band_n):
    # concat all to calculate the 1st and 99th perecentile for extents of plot
    nobia_toplot = MOD_at_noBIA_df[Band_names[band_n]]/10000
    bia_toplot = MOD_at_BIA_df[Band_names[band_n]]/10000
    aodi_toplot = MOD_at_noBIAnew_df[Band_names[band_n]]/10000
    aodii_toplot = MOD_at_BIAnew_df[Band_names[band_n]]/10000
    all_values = pd.concat([nobia_toplot,bia_toplot,
               aodi_toplot,aodii_toplot])
    # define limits of plot
    xmin = all_values.quantile(0.01)
    xmax = all_values.quantile(0.99)
    # define bins
    bins = np.linspace(xmin,xmax,50)
    # plot histogram
    ax.hist(nobia_toplot,color=c_noBIA,bins=bins,label='AOA: No blue ice');
    ax.hist(bia_toplot,color=c_BIA,bins=bins,alpha=0.5,label='AOA: Blue ice');
    ax.hist(aodi_toplot,color=c_AODi,bins=bins,histtype='step',label='AOD: Type I');
    ax.hist(aodii_toplot,color=c_AODii,bins=bins,histtype='step',label='AOD: Type II');
    # set limits and labels
    ax.set_xlim([xmin,xmax])
    ax.set_title(f'{Band_names[band_n][-2:]}, {BWs_labels[band_n]}',fontsize=10)

# plot histograms for all different bands
plot_hist(axs[0,0],0)
plot_hist(axs[0,1],1)
plot_hist(axs[0,2],2)
plot_hist(axs[0,3],3)
plot_hist(axs[1,0],4)
plot_hist(axs[1,1],5)
plot_hist(axs[1,2],6)

# set labels, plot legend
axs[1,0].set_xlabel('Reflectance',fontsize=9)
axs[0,0].set_ylabel('Count',fontsize=9)
axs[1,0].set_ylabel('Count',fontsize=9)
axs[1,2].legend(bbox_to_anchor=(1.15, 0.7))
axs[1,3].axis('off')
axs[1,2].annotate('MODIS data',xy=(1.83,0.7),
                annotation_clip=False,
                xycoords='axes fraction',
                ha='center')
# adjust spacing
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0.5, wspace = 0.4)
plt.margins(0,0)
# save figure
fig.savefig('../output/figures/spectral_histo_Modis.png',bbox_inches = 'tight',
    pad_inches = 0.02,dpi=300)
#%%
# Additional analysis for review:
# draw 600m buffer around smoothed BIAs
buffered_smoothed_BIAs = gdf_smoothed_BIAs.exterior.buffer(600) # exterior for buffering around edge (!!!)
#%%
# merge elevation on grid of BIA_map
interpolated_ortho_height = TDX_ortho_ds.interp_like(BIA_map)
BIA_map_elev = xr.merge([BIA_map,interpolated_ortho_height])
#%%
# loop over random subset of 500 smoothed BIAs to select downstream pixels around edge
# first define empty dataframe to store selected pixels
downstream_pixels = pd.DataFrame(columns=['x','y','mean','median','std','BIAs_hui','DEM'])
upstream_pixels = pd.DataFrame(columns=['x','y','mean','median','std','BIAs_hui','DEM'])
# define random seed
np.random.seed(seed=10)
index_random = np.random.randint(0,len(buffered_smoothed_BIAs),size=800)
# drop duplicates
index_random_unique = list(set(index_random))

#%%
for i in index_random_unique[:500]: #select first 500 BIAs
    # select buffer around single BIA
    selected_BIA = buffered_smoothed_BIAs.iloc[i]
    # rasterize selected buffer
    ShapeMask = features.geometry_mask([selected_BIA],
                                            out_shape=(len(BIA_map_elev.y), len(BIA_map_elev.x)),
                                            transform=BIA_map_elev.transform,
                                            invert=True)
    # assign coordinates and create a data array
    ShapeMask = xr.DataArray(ShapeMask[::-1], coords={"y":BIA_map_elev.y,
                                                    "x":BIA_map_elev.x},
                                dims=("y", "x"))
    # select pixels within buffer
    selected_pixels1 = BIA_map_elev.where((ShapeMask == True),drop=True).to_dataframe()
    # drop rows where no predictions
    selected_pixels2 = selected_pixels1[~np.isnan(selected_pixels1['mean'])].reset_index()
    # select 25% pixels with lowest elevation and 25% pixels with highest elevation
    selected_pixels3 = selected_pixels2[selected_pixels2['DEM']<selected_pixels2['DEM'].quantile(0.25)]
    selected_pixels4 = selected_pixels2[selected_pixels2['DEM']>selected_pixels2['DEM'].quantile(0.75)]
    # append selected pixels to large dataframe of which to compile the spectral fingerprint later
    downstream_pixels = pd.concat([downstream_pixels,selected_pixels3])
    upstream_pixels = pd.concat([upstream_pixels,selected_pixels4])
    # delete unneccesary variables
    del(selected_pixels1,selected_pixels2,selected_pixels3,selected_pixels4)

#%%
# separate pixels around downstream edge that are classified as BIA or not
downstream_pixels_BIA = downstream_pixels[downstream_pixels['mean']>0.5]
downstream_pixels_noBIA = downstream_pixels[downstream_pixels['mean']<0.5]
# separate pixels around upstream edge that are classified as BIA or not
upstream_pixels_BIA = upstream_pixels[upstream_pixels['mean']>0.5]
upstream_pixels_noBIA = upstream_pixels[upstream_pixels['mean']<0.5]
#%%
# extract MODIS values at downstream pixel locations (BIA)
MOD_at_dsBIA = MOD_ds.interp(x=downstream_pixels_BIA.x.to_xarray(),
                   y=downstream_pixels_BIA.y.to_xarray())
# format to dataframe
MOD_at_dsBIA_df = MOD_at_dsBIA.to_dataframe()[['x','y','MOD_B1','MOD_B2','MOD_B3','MOD_B4','MOD_B5','MOD_B6','MOD_B7']]

# extract values at downstream pixel locations (no BIA)
MOD_at_dsnoBIA = MOD_ds.interp(x=downstream_pixels_noBIA.x.to_xarray(),
                   y=downstream_pixels_noBIA.y.to_xarray())
# format to dataframe
MOD_at_dsnoBIA_df = MOD_at_dsnoBIA.to_dataframe()[['x','y','MOD_B1','MOD_B2','MOD_B3','MOD_B4','MOD_B5','MOD_B6','MOD_B7']]

#%%
# plot values at downstream edge
# define colors
c_BIA = '#85CFEE'
c_noBIA = 'k'
c_AODi = '#F7941D'
c_AODii = '#9E1F63'
# plot figure
fig,axs = plt.subplots(1,1,figsize=(11/2.54, 11/2.54))
# plot spectral fingerprints
axs.plot(MODIS_bw[[2,3,0,1,4,5,6]],MOD_at_noBIA_df.median()[2:][[2,3,0,1,4,5,6]]/10000,color=c_noBIA,label='No blue ice',
         linewidth=2,zorder=5)
axs.plot(MODIS_bw[[2,3,0,1,4,5,6]],MOD_at_dsnoBIA_df.median()[2:][[2,3,0,1,4,5,6]]/10000,color=c_AODi,label='Downstream edge\n(no blue ice)',
         linewidth=2,linestyle=(0,(5,1)),zorder=6)
axs.plot(MODIS_bw[[2,3,0,1,4,5,6]],MOD_at_dsBIA_df.median()[2:][[2,3,0,1,4,5,6]]/10000,color=c_AODii,label='Downstream edge\n(blue ice)',
         linewidth=2,linestyle=(0,(3,1,1,1)),zorder=7)
axs.plot(MODIS_bw[[2,3,0,1,4,5,6]],MOD_at_BIA_df.median()[2:][[2,3,0,1,4,5,6]]/10000,color=c_BIA,label='Blue ice',
         linewidth=2,zorder=8)
# plot uncertainty bands (25th and 75th percentile)
alpha_all = 0.5
uncert_interval = 0.25
lw_fill = 0.8
mpl.rcParams['hatch.linewidth'] = 0.3
axs.fill_between(MODIS_bw[[2,3,0,1,4,5,6]],y1=MOD_at_BIA_df.quantile(uncert_interval)[2:][[2,3,0,1,4,5,6]]/10000,
                 y2=MOD_at_BIA_df.quantile(0.5+uncert_interval)[2:][[2,3,0,1,4,5,6]]/10000,
                 facecolor="none",edgecolor=c_BIA,linewidth=lw_fill,alpha=alpha_all,zorder=8,hatch="......")
axs.fill_between(MODIS_bw[[2,3,0,1,4,5,6]],y1=MOD_at_noBIA_df.quantile(uncert_interval)[2:][[2,3,0,1,4,5,6]]/10000,
                 y2=MOD_at_noBIA_df.quantile(0.5+uncert_interval)[2:][[2,3,0,1,4,5,6]]/10000,
                 facecolor="none",edgecolor=c_noBIA,linewidth=lw_fill,alpha=alpha_all,zorder=5,hatch='\\\\\\\\\\\\')
axs.fill_between(MODIS_bw[[2,3,0,1,4,5,6]],y1=MOD_at_dsBIA_df.astype('float').quantile(uncert_interval)[2:][[2,3,0,1,4,5,6]]/10000,
                 y2=MOD_at_dsBIA_df.astype('float').quantile(0.5+uncert_interval)[2:][[2,3,0,1,4,5,6]]/10000,
                 facecolor="none",edgecolor=c_AODii,linewidth=lw_fill,alpha=alpha_all,zorder=7,hatch="------")
axs.fill_between(MODIS_bw[[2,3,0,1,4,5,6]],y1=MOD_at_dsnoBIA_df.astype('float').quantile(uncert_interval)[2:][[2,3,0,1,4,5,6]]/10000,
                 y2=MOD_at_dsnoBIA_df.astype('float').quantile(0.5+uncert_interval)[2:][[2,3,0,1,4,5,6]]/10000,
                 facecolor="none",edgecolor=c_AODi,linewidth=lw_fill,alpha=alpha_all,zorder=6,hatch="||||||")

# plot legend, set labels
axs.legend(loc='upper right')
axs.set_title('MODIS')
axs.set_xlabel('Wavelength (nm)')
axs.set_ylabel('Reflectance')

# plot RGB colors
red_spectr = [620,670]
green_spectr = [545,565]
blue_spectr = [459,479]
nir_spectr = [841,876]
h_patch = 0
thickness_patch = 0.1
axs.add_patch(matplotlib.patches.Rectangle((red_spectr[0],h_patch),red_spectr[1]-red_spectr[0],thickness_patch,
                                               facecolor='red',clip_on=False))
axs.add_patch(matplotlib.patches.Rectangle((green_spectr[0],h_patch),green_spectr[1]-green_spectr[0],thickness_patch,
                                               facecolor='green',clip_on=False))
axs.add_patch(matplotlib.patches.Rectangle((blue_spectr[0],h_patch),blue_spectr[1]-blue_spectr[0],thickness_patch,
                                               facecolor='blue',clip_on=False))
axs.add_patch(matplotlib.patches.Rectangle((nir_spectr[0],h_patch),nir_spectr[1]-nir_spectr[0],thickness_patch,
                                               facecolor='orange',clip_on=False))
# adjust spacing
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0.5, wspace = 0.4)
plt.margins(0,0)
# save figure
fig.savefig('../output/figures/downstream_edge.png',bbox_inches = 'tight',
    pad_inches = 0.02,dpi=300)

#fig.savefig('../output/figures/downstream_edge.png')
#%%
# extract values at upstream pixel locations (BIA)
MOD_at_usBIA = MOD_ds.interp(x=upstream_pixels_BIA.x.to_xarray(),
                   y=upstream_pixels_BIA.y.to_xarray())
# format to dataframe
MOD_at_usBIA_df = MOD_at_usBIA.to_dataframe()[['x','y','MOD_B1','MOD_B2','MOD_B3','MOD_B4','MOD_B5','MOD_B6','MOD_B7']]

# extract values at upstream pixel locations (no BIA)
MOD_at_usnoBIA = MOD_ds.interp(x=upstream_pixels_noBIA.x.to_xarray(),
                   y=upstream_pixels_noBIA.y.to_xarray())
# format to dataframe
MOD_at_usnoBIA_df = MOD_at_usnoBIA.to_dataframe()[['x','y','MOD_B1','MOD_B2','MOD_B3','MOD_B4','MOD_B5','MOD_B6','MOD_B7']]

#%%
# plot values at upstream edge
# define colors
c_BIA = '#85CFEE'
c_noBIA = 'k'
c_AODi = '#F7941D'
c_AODii = '#9E1F63'
#%%
# plot figure (Figure S8 and S9)
fig,axs = plt.subplots(1,1,figsize=(11/2.54, 11/2.54))
# plot spectral fingerprints
axs.plot(MODIS_bw[[2,3,0,1,4,5,6]],MOD_at_noBIA_df.median()[2:][[2,3,0,1,4,5,6]]/10000,color=c_noBIA,label='No blue ice',
         linewidth=2,zorder=5)
axs.plot(MODIS_bw[[2,3,0,1,4,5,6]],MOD_at_usnoBIA_df.median()[2:][[2,3,0,1,4,5,6]]/10000,color=c_AODi,label='Upstream edge\n(no blue ice)',
         linewidth=2,linestyle=(0,(5,1)),zorder=6)
axs.plot(MODIS_bw[[2,3,0,1,4,5,6]],MOD_at_usBIA_df.median()[2:][[2,3,0,1,4,5,6]]/10000,color=c_AODii,label='Upstream edge\n(blue ice)',
         linewidth=2,linestyle=(0,(3,1,1,1)),zorder=7)
axs.plot(MODIS_bw[[2,3,0,1,4,5,6]],MOD_at_BIA_df.median()[2:][[2,3,0,1,4,5,6]]/10000,color=c_BIA,label='Blue ice',
         linewidth=2,zorder=8)
# plot uncertainty bands (25th and 75th percentile)
alpha_all = 0.5
uncert_interval = 0.25
lw_fill = 0.8
mpl.rcParams['hatch.linewidth'] = 0.3
axs.fill_between(MODIS_bw[[2,3,0,1,4,5,6]],y1=MOD_at_BIA_df.quantile(uncert_interval)[2:][[2,3,0,1,4,5,6]]/10000,
                 y2=MOD_at_BIA_df.quantile(0.5+uncert_interval)[2:][[2,3,0,1,4,5,6]]/10000,
                 facecolor="none",edgecolor=c_BIA,linewidth=lw_fill,alpha=alpha_all,zorder=8,hatch="......")
axs.fill_between(MODIS_bw[[2,3,0,1,4,5,6]],y1=MOD_at_noBIA_df.quantile(uncert_interval)[2:][[2,3,0,1,4,5,6]]/10000,
                 y2=MOD_at_noBIA_df.quantile(0.5+uncert_interval)[2:][[2,3,0,1,4,5,6]]/10000,
                 facecolor="none",edgecolor=c_noBIA,linewidth=lw_fill,alpha=alpha_all,zorder=5,hatch='\\\\\\\\\\\\')
axs.fill_between(MODIS_bw[[2,3,0,1,4,5,6]],y1=MOD_at_usBIA_df.astype('float').quantile(uncert_interval)[2:][[2,3,0,1,4,5,6]]/10000,
                 y2=MOD_at_usBIA_df.astype('float').quantile(0.5+uncert_interval)[2:][[2,3,0,1,4,5,6]]/10000,
                 facecolor="none",edgecolor=c_AODii,linewidth=lw_fill,alpha=alpha_all,zorder=7,hatch="------")
axs.fill_between(MODIS_bw[[2,3,0,1,4,5,6]],y1=MOD_at_usnoBIA_df.astype('float').quantile(uncert_interval)[2:][[2,3,0,1,4,5,6]]/10000,
                 y2=MOD_at_usnoBIA_df.astype('float').quantile(0.5+uncert_interval)[2:][[2,3,0,1,4,5,6]]/10000,
                 facecolor="none",edgecolor=c_AODi,linewidth=lw_fill,alpha=alpha_all,zorder=6,hatch="||||||")

# plot legend, set labels
axs.legend(loc='upper right')
axs.set_title('MODIS')
axs.set_xlabel('Wavelength (nm)')
axs.set_ylabel('Reflectance')

# plot RGB colors
red_spectr = [620,670]
green_spectr = [545,565]
blue_spectr = [459,479]
nir_spectr = [841,876]
h_patch = 0
thickness_patch = 0.1
axs.add_patch(matplotlib.patches.Rectangle((red_spectr[0],h_patch),red_spectr[1]-red_spectr[0],thickness_patch,
                                               facecolor='red',clip_on=False))
axs.add_patch(matplotlib.patches.Rectangle((green_spectr[0],h_patch),green_spectr[1]-green_spectr[0],thickness_patch,
                                               facecolor='green',clip_on=False))
axs.add_patch(matplotlib.patches.Rectangle((blue_spectr[0],h_patch),blue_spectr[1]-blue_spectr[0],thickness_patch,
                                               facecolor='blue',clip_on=False))
axs.add_patch(matplotlib.patches.Rectangle((nir_spectr[0],h_patch),nir_spectr[1]-nir_spectr[0],thickness_patch,
                                               facecolor='orange',clip_on=False))
# adjust spacing
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0.5, wspace = 0.4)
plt.margins(0,0)
# save figure
fig.savefig('../output/figures/upstream_edge.png',bbox_inches = 'tight',
    pad_inches = 0.02,dpi=300)

#%%
# compare downstream and upstream edge
# plot figure
fig,axs = plt.subplots(1,1,figsize=(11/2.54, 11/2.54))

# plot spectral fingerprints
axs.plot(MODIS_bw[[2,3,0,1,4,5,6]],MOD_at_usnoBIA_df.median()[2:][[2,3,0,1,4,5,6]]/10000,color=c_AODi,label='Upstream edge\n(no blue ice)',
         linewidth=2,linestyle=(0,(5,1)),zorder=6)
axs.plot(MODIS_bw[[2,3,0,1,4,5,6]],MOD_at_dsnoBIA_df.median()[2:][[2,3,0,1,4,5,6]]/10000,color='k',label='Downstream edge\n(no blue ice)',
         linewidth=2,linestyle=(0,(3,1,1,1)),zorder=7)
# plot legend and labels
axs.legend(loc='upper right')
axs.set_title('MODIS')
axs.set_xlabel('Wavelength (nm)')
axs.set_ylabel('Reflectance')

# plot RGB colors
red_spectr = [620,670]
green_spectr = [545,565]
blue_spectr = [459,479]
nir_spectr = [841,876]
h_patch = 0
thickness_patch = 0.1
axs.add_patch(matplotlib.patches.Rectangle((red_spectr[0],h_patch),red_spectr[1]-red_spectr[0],thickness_patch,
                                               facecolor='red',clip_on=False))
axs.add_patch(matplotlib.patches.Rectangle((green_spectr[0],h_patch),green_spectr[1]-green_spectr[0],thickness_patch,
                                               facecolor='green',clip_on=False))
axs.add_patch(matplotlib.patches.Rectangle((blue_spectr[0],h_patch),blue_spectr[1]-blue_spectr[0],thickness_patch,
                                               facecolor='blue',clip_on=False))
axs.add_patch(matplotlib.patches.Rectangle((nir_spectr[0],h_patch),nir_spectr[1]-nir_spectr[0],thickness_patch,
                                               facecolor='orange',clip_on=False))
# adjust spacing
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0.5, wspace = 0.4)
plt.margins(0,0)
# save figure
fig.savefig('../output/figures/edges_compared.png',bbox_inches = 'tight',
    pad_inches = 0.02,dpi=300)

