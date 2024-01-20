# generates Figure S1 to visualize new MODIS composite
# import packages
import numpy as np
import xarray as xr
import os
import geopandas
from shapely.geometry.polygon import Polygon
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import rasterio.features
import rasterio.mask
# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)
#%%
# define region
xmin = 240000
xmax = 300000
ymin = -770000
ymax = -680000

# import LIMA
LIMA = xr.open_rasterio('../data/LIMA/LIMA_Mosaic.jp2')
# crop LIMA to region of interest
LIMA_sel = LIMA.sel(x=slice(xmin, xmax),y=slice(ymax,ymin))

# import GEE composite (e.g., https://code.earthengine.google.com/110de2bd189a0327dedb6d0124c86660)
GEE_composite = xr.open_rasterio('../data/perc50_modis.tif')
GEE_sel = GEE_composite.sel(x=slice(xmin, xmax),y=slice(ymax,ymin))
for i in range(GEE_sel.shape[0]):
    GEE_sel[i] = GEE_sel[i]/GEE_sel[i].max()

# import MODIS composite
MODIS_composite = xr.open_rasterio('../data/merged_bands_composite3031080910.tif')
MODIS_sel = MODIS_composite.sel(x=slice(xmin, xmax),y=slice(ymax,ymin))
MODIS_sel = MODIS_sel.astype('float')
for i in range(MODIS_sel.shape[0]):
    MODIS_sel[i] = MODIS_sel[i].astype('float')/float(MODIS_sel[i].values.max())
    
#%%
# coarsen MODIS for more rapid visualization
MODIS_coarsened = MODIS_composite.coarsen(x=4,y=4,
                                          boundary='trim').mean()

#%%
# normalize values
MODIS_coarsened = MODIS_coarsened.astype('float')
for i in range(MODIS_coarsened.shape[0]):
    MODIS_coarsened[i] = MODIS_coarsened[i].astype('float')/float(MODIS_coarsened[i].values.max())

#%%
# plot data
# font settings
font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 10}
plt.rc('font', **font)
# plot figure
fig, (ax0,ax1,ax2,ax3) = plt.subplots(1,4,figsize=(18/2.54, 10/2.54),gridspec_kw={'width_ratios': [1, 1, 1, 1.8]})
# plot LIMA
ax0.imshow(LIMA_sel[0:3].values.transpose(1,2,0))
ax0.set_title('LIMA')
# plot GEE composite
ax1.imshow(GEE_sel[[1,3,2]].values.transpose(1,2,0))
ax1.set_title('GEE composite')
# plot MODIS composite
ax2.imshow(MODIS_sel[[1,3,2]].values.transpose(1,2,0))
ax2.set_title('MODIS composite')

# plot scalebars
# scalebar is calculated based on GEE data on 500 m resolution: 1 pixel is 500 m
scalebar = AnchoredSizeBar(ax1.transData, 
                           20000/500, '20 km', 
                           loc='lower right', 
                           pad=0.5, #0.0005
                           color='black',
                           frameon=False, 
                           size_vertical=1.8,
                           fontproperties=fm.FontProperties(size=9),
                           label_top=True,
                           sep=1)
ax2.add_artist(scalebar)

# plot coastlines
# open iceboundaries (quantarctica measures ice boundaries)
ice_boundaries_path = r'../data/IceBoundaries_Antarctica_v2.shx'
ice_boundaries_raw = geopandas.read_file(ice_boundaries_path)
# create union of ice boundaries
ice_boundaries_all = geopandas.GeoSeries(ice_boundaries_raw.unary_union)
# plot ice boundaries
#ice_boundaries_all.boundary.plot(ax=ax3, color='k',zorder=0, linewidth=0.6) #bfcbe3

ice_boundaries = ice_boundaries_raw['geometry'].unary_union

# set transform parameter MODIS coarsened
resolution = MODIS_coarsened.x[1].values-MODIS_coarsened.x[0].values
ll_x_main = MODIS_coarsened.x.min().values
ur_y_main = MODIS_coarsened.y.max().values
MODIS_coarsened.attrs['transform'] = (resolution, 0.0, ll_x_main-(resolution/2), 0.0, -1*resolution, ur_y_main+(resolution/2))

# mask out values outside ice_boundaries
ShapeMask = rasterio.features.geometry_mask([ice_boundaries],
                                      out_shape=(len(MODIS_coarsened.y),
                                                 len(MODIS_coarsened.x)),
                                      transform=MODIS_coarsened.transform,
                                      invert=True,
                                      all_touched=False)

ShapeMask = xr.DataArray(ShapeMask, 
                         dims=({"y":MODIS_coarsened["y"][::-1], "x":MODIS_coarsened["x"]}))
# flip shapemask upside down
#ShapeMask= ShapeMask[::-1]
MODIS_masked = MODIS_coarsened.where((ShapeMask == True),drop=True)

# set nan values to be transparent
MODIS_toplot = MODIS_masked[[1,3,2]].values.transpose(1,2,0)
MODIS_alpha = np.ones(np.shape(MODIS_toplot)[0:2])
MODIS_nans = np.nan_to_num(MODIS_toplot[:,:,0], nan=-1)
MODIS_alpha[MODIS_nans == -1] = 0
MODIS_alpha = np.expand_dims(MODIS_alpha,axis=2)
MODIS_concat = np.concatenate((MODIS_toplot,MODIS_alpha),axis=2)

# plot continent-wide MODIS composite
ax3.imshow(MODIS_concat,extent=[MODIS_masked.x.min(),
                                MODIS_masked.x.max(),
                                MODIS_masked.y.min(),
                                MODIS_masked.y.max()])
ax3.set_title('Median MODIS composite')

# plot inset on main map
ins_ax = Polygon([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])
ax3.plot(*ins_ax.exterior.xy,color='k',linewidth=0.5)

# switch off all axes
ax0.xaxis.set_visible(False)
ax0.yaxis.set_visible(False)
ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)
ax2.xaxis.set_visible(False)
ax2.yaxis.set_visible(False)
ax3.axis('off')

# annotate subpanels
ax0.annotate('A',xy=(0.04,0.90),xycoords='axes fraction',fontsize=18,weight='bold',path_effects=[pe.withStroke(linewidth=2, foreground="white")])
ax1.annotate('B',xy=(0.04,0.90),xycoords='axes fraction',fontsize=18,weight='bold',path_effects=[pe.withStroke(linewidth=2, foreground="white")])
ax2.annotate('C',xy=(0.04,0.90),xycoords='axes fraction',fontsize=18,weight='bold',path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# adjust spacing
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0.05)
plt.margins(0.1,0.1)
# save figure
fig.savefig('../figures/MODIS_composite.png',bbox_inches = 'tight',
    pad_inches = 0,dpi=300)


