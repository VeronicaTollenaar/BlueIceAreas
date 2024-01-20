# visualize uncertainties of BIA predictions related to the inclusion of radar data (Figure S7)
import numpy as np
import os
import matplotlib.pyplot as plt
import xarray as xr
import geopandas
import affine
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
from pyproj import Transformer
import matplotlib
from matplotlib.patches import Rectangle
import rasterio
from rasterio.plot import show
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

# read in smoothed BIAs
smoothed_BIAs_path = r'../output/smoothed_BIAs.shx'
gdf_smoothed_BIAs = geopandas.read_file(smoothed_BIAs_path)

# read in radar data
RS2_path = r'../data/RS2_32bit_100m_mosaic_HH_clip.tif'
RS2_raw = xr.open_rasterio(RS2_path)
# convert DataArray to DataSet
RS2_ds = RS2_raw.drop('band')[0].to_dataset(name='radar')

#%%
# visualize uncertainties
# plot figure
fig,axs = plt.subplots(4,4,figsize=(18/2.54, 19.5/2.54))

# define colorbar for uncertainties
c_uncertainties = cm.get_cmap('Oranges', 256)
newcolors = c_uncertainties(np.linspace(0,1,256))
newcmp = ListedColormap(newcolors)
cmap = newcmp

# define colorbar (from 0 to 0.5 and from 0.5 to 1) for BIAmap
noblueice = cm.get_cmap('Greys', 256)
blueice = cm.get_cmap('Blues', 256)
newcolors_BIA = noblueice(np.linspace(0.3,0.6,256))
newcolors_BIA = np.concatenate([newcolors_BIA,blueice(np.linspace(0.5,1,256))])
newcmp_BIA = ListedColormap(newcolors_BIA)
cmap_BIA = newcmp_BIA
norm_BIA = mpl.colors.Normalize(vmin=0,vmax=1)

# define colorbar for radar data
c_radar = cm.get_cmap('binary_r', 256)
newcolors_radar = c_radar(np.linspace(0,1,256))
newcmp_radar = ListedColormap(newcolors_radar)
cmap_radar = newcmp_radar
norm_radar = mpl.colors.Normalize(vmin=-28, vmax=10)

# define function that plots uncertainties and BIA map for defined regions
def plot_uncertainties(cent_x,cent_y,extent,axis_uncert,axis_BIAmap,loc_scalebar,
                       vmin=0,vmax=BIA_map['std'].max()):
    # define normalization of colors
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # define extent: cent_x, cent_y are center coordinates of plotted data
    min_x = cent_x - extent/2
    max_x = cent_x + extent/2
    min_y = cent_y - extent/2
    max_y = cent_y + extent/2
    # obtain arrays with values for x and y coordinates
    mask_x = (BIA_map.x >= min_x) & (BIA_map.x < max_x)
    mask_y = (BIA_map.y >= min_y) & (BIA_map.y < max_y)
    # crop image based on arrays with values for x and y coordinates
    img_cropped = BIA_map.where(mask_x & mask_y, drop=True)
    # plot data
    xr.plot.imshow(img_cropped['std'],cmap=cmap,norm=norm,
                   ax=axis_uncert,add_colorbar=False)
    gdf_smoothed_BIAs.plot(ax=axis_uncert,linewidth=0.8,
                facecolor="none",edgecolor='k')
    xr.plot.imshow(img_cropped['mean'],cmap=cmap_BIA,norm=norm_BIA,
                   ax=axis_BIAmap,add_colorbar=False)
    # set extent of axes
    axis_uncert.set_xlim([min_x,max_x])
    axis_uncert.set_ylim([min_y,max_y])
    axis_BIAmap.set_xlim([min_x,max_x])
    axis_BIAmap.set_ylim([min_y,max_y])
    # add scalebar
    len_scalebar = extent/6
    label_scalebar = f'{int(len_scalebar*1e-3)} km'
    scalebar = AnchoredSizeBar(axis_BIAmap.transData,
                            len_scalebar, label_scalebar, 
                            loc=loc_scalebar, 
                            pad=0.7, #0.5
                            color='black',
                            frameon=False, 
                            size_vertical=extent/60,
                            fontproperties=fm.FontProperties(size=9),
                            label_top=True,
                            sep=1)
    axis_BIAmap.add_artist(scalebar)
    # switch off axes
    axis_uncert.axis('off')
    axis_BIAmap.axis('off')
    
# function that plots radar data
def plot_radar(cent_x,cent_y,extent,axis_data,data_toplot,subdata,
               cmap_toplot,norm_toplot):
    # define extent
    min_x = cent_x - extent/2
    max_x = cent_x + extent/2
    min_y = cent_y - extent/2
    max_y = cent_y + extent/2
    # obtain arrays with values for x and y coordinates
    mask_x = (data_toplot.x >= min_x) & (data_toplot.x < max_x)
    mask_y = (data_toplot.y >= min_y) & (data_toplot.y < max_y)
    # crop image based on arrays with values for x and y coordinates
    img_cropped = data_toplot.where(mask_x & mask_y, drop=True)
    # plot data
    xr.plot.imshow(img_cropped[subdata],cmap=cmap_toplot,norm=norm_toplot,
                   ax=axis_data,add_colorbar=False)
    gdf_smoothed_BIAs.plot(ax=axis_data,linewidth=0.8,
                facecolor="none",edgecolor='k')
    # set extent of axes
    axis_data.set_xlim([min_x,max_x])
    axis_data.set_ylim([min_y,max_y])
    # switch off axes
    axis_data.axis('off')

# function that opens LIMA images for a given location
def openJPGgivenbounds(xmin,ymin):
    # define arrays referring to names of LIMA subtiles
    cirref_xs = np.arange(-2700000, 2700000, 150000)
    cirref_ys = np.arange(-2700000, 2700000, 150000)
    # select names of LIMA subtiles corresponding to given coordinates
    cirref_x = cirref_xs[cirref_xs<xmin][-1]
    cirref_y = cirref_ys[cirref_ys<ymin][-1]
    # define list of zeros to make sure names include zeros
    name_zeros = list('0000000')
    # ensure a minus is included in namestring when given coordinates are negative
    if cirref_x < 0:
        name_zeros[7-len(str(abs(cirref_x))):] = list(str(abs(cirref_x)))
        name_x_abs = ''.join(list(name_zeros))
        name_x = '-'+name_x_abs
        name_zeros = list('0000000')
    else:
        name_zeros[7-len(str(abs(cirref_x))):] = list(str(abs(cirref_x)))
        name_x_abs = ''.join(list(name_zeros))
        name_x = '+'+name_x_abs
        name_zeros = list('0000000')
    if cirref_y < 0:
        name_zeros[7-len(str(abs(cirref_y))):] = list(str(abs(cirref_y)))
        name_y_abs = ''.join(list(name_zeros))
        name_y = '-'+name_y_abs
        name_zeros = list('0000000')
    else:
        name_zeros[7-len(str(abs(cirref_y))):] = list(str(abs(cirref_y)))
        name_y_abs = ''.join(list(name_zeros))
        name_y = '+'+name_y_abs
        name_zeros = list('0000000')
    # define full name of LIMA subtile
    name = 'CIRREF_'+'x'+name_x+'y'+name_y
    # try open LIMA subtile (for high latitudes no data exists)
    try:
        img = rasterio.open('../data/LIMA/'+name+'.jpg')
    except:
        print('no high resolution background image')
        img = 0
    # return image and lowerleft coordinates of image
    return(img,cirref_x,cirref_y)

# function to plot LIMA data
def plot_lima(cent_x,cent_y,extent,axis_data):
    # define extent
    min_x = cent_x - extent/2
    max_x = cent_x + extent/2
    min_y = cent_y - extent/2
    max_y = cent_y + extent/2
    # define array to open images
    img_open_x = np.arange(min_x,max_x+0.14e6,0.15e6)
    img_open_y = np.arange(min_y,max_y+0.14e6,0.15e6)
    # set limits of axes
    axis_data.set_xlim([min_x,max_x])
    axis_data.set_ylim([min_y,max_y])
    # loop over images that need to be opened and plotted to cover the given area
    for xs in img_open_x:
        for ys in img_open_y:
            backgr,_x,_y = openJPGgivenbounds(xs,ys)
            show(backgr.read(),ax=axis_data,transform=backgr.transform)
    # plot BIA outlines
    gdf_smoothed_BIAs.plot(ax=axis_data,linewidth=0.8,
                facecolor="none",edgecolor='k')        
    # hide axes
    axis_data.axis('off')

# function to calculate approx coordinates of locations
def coords_label(cent_x, cent_y):
    transformer = Transformer.from_crs("EPSG:3031","EPSG:4326")
    lat, lon = transformer.transform(cent_x,cent_y)
    if lon > 0:
        return_str = f'{-1*np.round(lat,2)}°S, {np.round(lon,2)}°E'
    if lon < 0:
        return_str = f'{-1*np.round(lat,2)}°S, {-1*np.round(lon,2)}°W'
    return(return_str)

# function to add northarrow
def plot_arrow(x_center,y_center,extent,ax_arrow,upper_left=False):
    # set arrowlength
    arrow_length = 0.045*extent
    # calculate additional length of arraw
    x_add = arrow_length*x_center/((x_center**2 + y_center**2)**0.5)
    y_add = arrow_length*y_center/((x_center**2 + y_center**2)**0.5)
    # calculate location of arrow
    x_arrow = x_center + extent/2.28
    y_arrow = y_center + extent/2.28
    if upper_left:
        # calculate location of arrow
        x_arrow = x_center - extent/2.28
        y_arrow = y_center + extent/2.28
    # plot arrow
    ax_arrow.arrow(x_arrow-x_add,y_arrow-y_add,2*x_add,2*y_add,color='k',
                width=0.005*extent,length_includes_head=True) #width=40
#%%
# plot uncertainties
# spacing labels
lab_h = 0.12
lab_v = 1.022
lab_hrel = 180 # larger number means closer spacing

# EXAMPLE 1
# define extent of example
cent_x1 = 795e3
cent_y1 = 1720e3
extent1 = 25e3
# define name of area
area = f'Nansen C icefield ({coords_label(cent_x1,cent_y1)})'
ax_n = 0

# plot uncertainties
plot_uncertainties(cent_x1,cent_y1,extent1,axs[ax_n,1],axs[ax_n,0],loc_scalebar='lower right',
                   vmin=0,vmax=0.3)
# plot other data
plot_radar(cent_x1,cent_y1,extent1,axs[ax_n,2],RS2_ds,'radar',
               cmap_radar,norm_radar)   
plot_lima(cent_x1,cent_y1,extent1,axs[ax_n,3])
# plot location info
axs[ax_n,0].annotate(xy=(lab_h,lab_v),xycoords='axes fraction',
                     text=area,rotation=0,annotation_clip=False,
                     style='italic',fontsize=9)
# plot north arrow
plot_arrow(cent_x1,cent_y1,extent1,axs[ax_n,3])

# EXAMPLE 2
# define extent of example
cent_x2 = 616e3
cent_y2 = -1929e3
extent2 = 72e3
# define name of area
area = f'Sullivan Glacier ({coords_label(cent_x2,cent_y2)})'
ax_n = 1

# plot uncertainties
plot_uncertainties(cent_x2,cent_y2,extent2,axs[ax_n,1],axs[ax_n,0],loc_scalebar='lower left',
                   vmin=0,vmax=0.3,wind_scour=False)
# plot other data
plot_radar(cent_x2,cent_y2,extent2,axs[ax_n,2],RS2_ds,'radar',
               cmap_radar,norm_radar)
plot_lima(cent_x2,cent_y2,extent2,axs[ax_n,3])
# plot location info
axs[ax_n,0].annotate(xy=(lab_h,lab_v),xycoords='axes fraction',
                     text=area,rotation=0,annotation_clip=False,
                     style='italic',fontsize=9)
# plot north arrow
plot_arrow(cent_x2,cent_y2,extent2,axs[ax_n,3])

# EXAMPLE 3
# define extent of example
cent_x3 = -374e3
cent_y3 = 1900e3
extent3 = 60e3#20e3
# define name of area
area = f'Riiser-Larsen ice shelf grounding zone ({coords_label(cent_x3,cent_y3)})'
ax_n = 2

#plot uncertainties
plot_uncertainties(cent_x3,cent_y3,extent3,axs[ax_n,1],axs[ax_n,0],loc_scalebar='lower left',
                   vmin=0,vmax=0.3)
#plot other data
plot_radar(cent_x3,cent_y3,extent3,axs[ax_n,2],RS2_ds,'radar',
               cmap_radar,norm_radar)   
plot_lima(cent_x3,cent_y3,extent3,axs[ax_n,3])
# plot location info
axs[ax_n,0].annotate(xy=(lab_h,lab_v),xycoords='axes fraction',
                     text=area,rotation=0,annotation_clip=False,
                     style='italic',fontsize=9)
# plot north arrow
plot_arrow(cent_x3,cent_y3,extent3,axs[ax_n,3])

# EXAMPLE 4
# define extent of example
cent_x4 = 814e3
cent_y4 = 1792e3
extent4 = 28e3#20e3
# define name of area
area = f'Jenningsbreen ({coords_label(cent_x4,cent_y4)})'
ax_n = 3

# plot uncertainties
plot_uncertainties(cent_x4,cent_y4,extent4,axs[ax_n,1],axs[ax_n,0],loc_scalebar='lower left',
                    vmin=0,vmax=0.3)
# plot other data
plot_radar(cent_x4,cent_y4,extent4,axs[ax_n,2],RS2_ds,'radar',
               cmap_radar,norm_radar)   
plot_lima(cent_x4,cent_y4,extent4,axs[ax_n,3])
# plot location info
axs[ax_n,0].annotate(xy=(lab_h,lab_v),xycoords='axes fraction',
                     text=area,rotation=0,annotation_clip=False,
                     style='italic',fontsize=9)
# plot north arrow
plot_arrow(cent_x2,cent_y2,extent2,axs[ax_n,3],upper_left=True)

# plot colorbars
ax_sub_cb_BIA = inset_axes(axs[3,0], width=1.4, height=0.15,
                        loc='center',bbox_to_anchor=((0.5,-0.2)),
                        bbox_transform=axs[3,0].transAxes)
cb = mpl.colorbar.ColorbarBase(ax_sub_cb_BIA, cmap=cmap_BIA,
                                orientation='horizontal',
                                ticks=[0,0.5,1],
                                norm=norm_BIA)
cb.ax.tick_params(labelsize=9)
cb.ax.xaxis.set_label_position('top')
cb.ax.set_xlabel('Prediction values', fontsize=9)

ax_sub_cb_radar = inset_axes(axs[3,2], width=1.4, height=0.15, 
                        loc='center',bbox_to_anchor=((0.5,-0.2)),
                        bbox_transform=axs[3,2].transAxes)
cb = mpl.colorbar.ColorbarBase(ax_sub_cb_radar, cmap=cmap_radar,
                                orientation='horizontal',
                                norm=norm_radar)
cb.ax.tick_params(labelsize=9)
cb.ax.xaxis.set_label_position('top')
cb.ax.set_xlabel('Radar backscatter (dB)', fontsize=9)

ax_sub_cb_uncertainties = inset_axes(axs[3,1], width=1.4, height=0.15, 
                        loc='center',bbox_to_anchor=((0.5,-0.2)),
                        bbox_transform=axs[3,1].transAxes)
cb = mpl.colorbar.ColorbarBase(ax_sub_cb_uncertainties, cmap=cmap,
                                orientation='horizontal',
                                norm = mpl.colors.Normalize(vmin=0, vmax=0.3),
                                ticks=[0,0.3])
cb.ax.tick_params(labelsize=9)
cb.ax.xaxis.set_label_position('top')
cb.ax.set_xlabel('Estimated uncertainties', fontsize=9)

axs[3,3].annotate(xy=(0.5,-0.2),xycoords='axes fraction',
                     text='Landsat Image \nMosaic of Antarctica',
                     ha='center',
                     rotation=0,annotation_clip=False,fontsize=9)


# plot settings
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0.18, wspace = 0.05)



# annotate panels
def annot_panels(axs,text_):
    axs.annotate(text_,xy=(0.01,0.88),xycoords='axes fraction',fontsize=18,
                 weight='bold',
                 path_effects=[pe.withStroke(linewidth=1, foreground="white")],
                 annotation_clip=False)
    # create lightgray boxes behind each example
    axs.add_patch(Rectangle((-0.05,-0.02),4.25,1.12,facecolor='whitesmoke', 
                             edgecolor='k',clip_on=False,
                             linewidth=0.4,
                             transform=axs.transAxes,
                             zorder=0))
letters = ['A','B','C','D','E']
numbers =['i','ii','iii','iv']
for row in range(4):
    text_ = letters[row]
    annot_panels(axs[row,0],text_)

# save figure
fig.savefig(f'../output/figures/BIA_map_uncertainties_SARexamples.png',bbox_inches = 'tight',
            facecolor='white',
    pad_inches = 0.01,dpi=300)

