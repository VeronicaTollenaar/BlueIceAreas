# plot overview of datasplit (training, validation, and testing; Figure S2)
# import packages
import numpy as np
import os
import geopandas
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)
#%%
# define directory of data
data_dir = f'train_tiles'
# define resolution
res = 200 #m
# define size of images
n_pix = 512

# define colors
c_train = '#940065'
c_val = '#FA990A'
c_test = '#00C797'
c_bia = '#466595'

# plot figure
fig, ax1 = plt.subplots(1,1,figsize=(9/2.54,9/2.54))

# open iceboundaries (quantarctica measures ice boundaries)
ice_boundaries_path = r'../data/IceBoundaries_Antarctica_v2.shx'
ice_boundaries_raw = geopandas.read_file(ice_boundaries_path)
ice_boundaries_all = geopandas.GeoSeries(ice_boundaries_raw.unary_union)
# plot iceboundaries
ice_boundaries_all.plot(ax=ax1, color='#CED0D6')

# open BIAs (="noisy" labels)
BIAs_path = r'../data/BlueIceAreas.shx'
BIAs_raw = geopandas.read_file(BIAs_path)

# plot all BIAs (="noisy" labels)
BIAs_raw['geometry'].plot(ax=ax1,color=c_bia,zorder=2)

# open train, validation, test areas
sqs_train_path = r'../data/train_squares.shx'
sqs_train = geopandas.read_file(sqs_train_path).rename(columns={'geometry':'square'}).set_geometry(col='square')
    
sqs_val_path = r'../data/validation_squares.shx'
sqs_val = geopandas.read_file(sqs_val_path).rename(columns={'geometry':'square'}).set_geometry(col='square')

sqs_test_path = r'../data/test_squares.shx'
sqs_test = geopandas.read_file(sqs_test_path).rename(columns={'geometry':'square'}).set_geometry(col='square')

# add negative buffer to polygons to avoid overlapping lines
bufsize=-6000.
sqs_train['square_buf'] = sqs_train.buffer(bufsize)
sqs_val['square_buf'] = sqs_val.buffer(bufsize)
sqs_test['square_buf'] = sqs_test.buffer(bufsize)

# plot train, validation, test areas
# set transparency
transp_ = 0.25
sqs_train['square_buf'].plot(ax=ax1, color=c_train,alpha=transp_)
sqs_val['square_buf'].plot(ax=ax1, color=c_val,alpha=transp_)
sqs_test['square_buf'].plot(ax=ax1, color=c_test,alpha=transp_)

# highlight handlabelled areas
sqs_train[sqs_train['id_square']=='265']['square_buf'].boundary.plot(ax=ax1, color='k')
sqs_val[sqs_val['id_square']=='278']['square_buf'].boundary.plot(ax=ax1, color='k')
sqs_val[sqs_val['id_square']=='246']['square_buf'].boundary.plot(ax=ax1, color='k')
sqs_test[sqs_test['id_square']=='264']['square_buf'].boundary.plot(ax=ax1, color='k')
sqs_test[sqs_test['id_square']=='409']['square_buf'].boundary.plot(ax=ax1, color='k')

# create insets to show tiles
idx_sq1 = '265'
axins1 = ax1.inset_axes([0.1,-0.12,0.28,0.28])

# read list of train images
list_imgs_train_all = os.listdir(f'../{data_dir}/train_images')
try: 
    list_imgs_train_all.remove('.DS_Store')
except ValueError:
    pass

# create lists of lower left coordinates based on the file names
x_coords_train = [float(img.split('_', 1)[0]) for img in list_imgs_train_all]
y_coords_train = [float(img.split('_', 1)[1][:-3]) for img in list_imgs_train_all]

# plot train tiles
for img in range(len(list_imgs_train_all)):
    train_patch = axins1.add_patch(Rectangle((x_coords_train[img],y_coords_train[img]),
                            res*n_pix, res*n_pix,
                            facecolor=c_train,
                            fill=True,
                            edgecolor='k',
                            lw=0.5, 
                            alpha=0.3))
# set limits of axes
axins1.set_xlim(sqs_train[sqs_train.id_square==idx_sq1]['square'].iloc[0].exterior.coords[0][0],
                sqs_train[sqs_train.id_square==idx_sq1]['square'].iloc[0].exterior.coords[2][0])
axins1.set_ylim(sqs_train[sqs_train.id_square==idx_sq1]['square'].iloc[0].exterior.coords[0][1],
                sqs_train[sqs_train.id_square==idx_sq1]['square'].iloc[0].exterior.coords[2][1])
axins1.set_aspect('equal', adjustable='box')
# remove labels on axes
axins1.set_xticklabels([])
axins1.set_yticklabels([])
axins1.tick_params(left = False, bottom = False)
# set labels
axins1.set_xlabel(r'$\leftarrow$ 250 km $\rightarrow$',labelpad=-5,fontsize=9) #longleftarrow, longrightarrow
axins1.set_ylabel(r'$\leftarrow$ 250 km $\rightarrow$',labelpad=-5,fontsize=9)
axins1.set_title(r'Training tiles in' "\n" f' Prince Albert Mts', path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                 fontsize=9)
axins1.set_facecolor('none')
# create zoom lines to inset axis
box, c1 = ax1.indicate_inset_zoom(axins1, edgecolor="black",linewidth=0)
plt.setp(c1,linewidth=0.3)

# create insets to show validation tiles
idx_sq2 = '278'
axins2 = ax1.inset_axes([0.86,0.57,0.28,0.28])
# read list of val images
list_imgs_val_all = os.listdir(f'../{data_dir}/val_images')
# create lists of lower left coordinates based on file names
x_coords_val = [float(img.split('_', 1)[0]) for img in list_imgs_val_all]
y_coords_val = [float(img.split('_', 1)[1][:-3]) for img in list_imgs_val_all]
# plot test tiles
for img in range(len(list_imgs_val_all)):
    val_patch = axins2.add_patch(Rectangle((x_coords_val[img],y_coords_val[img]),
                            res*n_pix, res*n_pix,
                            facecolor=c_val,
                            fill=True,
                            edgecolor='k',
                            lw=0.5,
                            alpha=0.3))
# set limits of axes
axins2.set_xlim(sqs_val[sqs_val.id_square==idx_sq2]['square'].iloc[0].exterior.coords[0][0],
                sqs_val[sqs_val.id_square==idx_sq2]['square'].iloc[0].exterior.coords[2][0])
axins2.set_ylim(sqs_val[sqs_val.id_square==idx_sq2]['square'].iloc[0].exterior.coords[0][1],
                sqs_val[sqs_val.id_square==idx_sq2]['square'].iloc[0].exterior.coords[2][1])
axins2.set_aspect('equal', adjustable='box')
# remove labels on axes
axins2.set_xticklabels([])
axins2.set_yticklabels([])
axins2.tick_params(left = False, bottom = False)
# set labels
axins2.yaxis.set_label_position("right")
axins2.set_xlabel(r'$\leftarrow$ 250 km $\rightarrow$',labelpad=-5,fontsize=9)
axins2.set_ylabel(r'$\leftarrow$ 250 km $\rightarrow$',fontsize=9)
axins2.set_title(r'Validation tiles in' "\n" f' S\u00F8r Rondane Mts (W)',path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                 fontsize=9)
axins2.set_facecolor('none')
# create zoom lines to inset axis
box, c1  = ax1.indicate_inset_zoom(axins2, edgecolor="black",linewidth=0)
plt.setp(c1,linewidth=0.3)

# plot individual tile
axins_tile = ax1.inset_axes([1.01,0.32,0.112,0.112])
# select single patch and calculate dimensions
x_val_area = sqs_val[sqs_val.id_square=='278'].square.centroid.x
y_val_area = sqs_val[sqs_val.id_square=='278'].square.centroid.y
dist_tile_center = (np.array(x_coords_val) - x_val_area.values + res*n_pix/2)**2 + (np.array(y_coords_val) - y_val_area.values + res*n_pix/2)**2
x_patch = x_coords_val[np.argmin(dist_tile_center)]
y_patch = y_coords_val[np.argmin(dist_tile_center)]
w_patch = res*n_pix
h_patch = res*n_pix
# plot individual tile
patch_single_plot = axins_tile.add_patch(Rectangle((x_patch,y_patch),
                     w_patch,h_patch,
                            facecolor=c_val,
                            fill=True,
                            edgecolor='k',
                            lw=0.5, 
                            alpha=0.3))
# set limits of axes
axins_tile.set_xlim([x_patch,x_patch + w_patch])
axins_tile.set_ylim([y_patch,y_patch + h_patch])
axins_tile.set_aspect('equal', adjustable='box')
# remove labels on axes
axins_tile.set_xticklabels([])
axins_tile.set_yticklabels([])
axins_tile.tick_params(left = False, bottom = False)
# set labels
axins_tile.set_xlabel(r'$\leftarrow$ $\rightarrow$' "\n" '100 km',labelpad=-5,fontsize=9) #longleftarrow, longrightarrow
axins_tile.set_ylabel('100 km' "\n" r'$\leftarrow$ $\rightarrow$' ,labelpad=-5,fontsize=9)
axins_tile.set_title(r'Single tile', path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                 fontsize=9)
axins_tile.set_facecolor('none')
# create zoom lines to inset axis
box, c1 = axins2.indicate_inset_zoom(axins_tile, edgecolor="black",linewidth=0)
plt.setp(c1,linewidth=0.3)

# add legend
bias_patch = mpatches.Patch(color=c_bia)
train_patch_legend = mpatches.Patch(color=c_train,alpha=transp_,linewidth=0)
val_patch_legend = mpatches.Patch(color=c_val,alpha=transp_,linewidth=0)
test_patch_legend = mpatches.Patch(color=c_test,alpha=transp_,linewidth=0)
handlabels = mpatches.Patch(edgecolor='k',fill=False)
legend = ax1.legend(handles=[train_patch_legend, val_patch_legend, test_patch_legend, handlabels, bias_patch],
                    labels=['Training areas','Validation areas','Test areas','Areas with handlabels','BIAs (Hui et al., 2014)'],
                    loc='lower left', 
                    bbox_to_anchor=(0.6,-0.2)
                    )
# set background of legend transparent
frame = legend.get_frame()
frame.set_facecolor('None')
frame.set_edgecolor('None')
# add scalebar
scalebar = AnchoredSizeBar(ax1.transData,
                           1000000, '1000 km', 
                           loc = 'lower left',
                           bbox_to_anchor=(0.735,0.2),
                           bbox_transform=ax1.transAxes,
                           pad=0.1,
                           color='black',
                           frameon=False,
                           size_vertical=50000,
                           label_top=True,
                           sep=2.5
                           )
ax1.add_artist(scalebar)

# adjust ticks of overview plot and set title
ax1.set_yticks([])
ax1.set_xticks([])
plt.title('Data split into different areas and tiles',x=0.59)
plt.axis('off')

# adjust spacing of plot
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
# save figure
fig.savefig(f'../output/figures/overview_testtrain_areas2.png',bbox_inches = 'tight',pad_inches = 0.01,dpi=300,facecolor='white')
plt.show()
