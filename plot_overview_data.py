# plot overview of input data (Figure S6)
# import packages
import xarray as xr
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)
# %%
# import data for one specific tile
path_data='../train_tiles/train_images_noiseornot/'
path_targets='../train_tiles/train_targets_noiseornot/'
img='542100_-1436100.nc'

data_ = xr.open_dataset(path_data+img)
targets_ = xr.open_dataset(path_targets+img)
#%%
# plot settings
font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 10}
plt.rc('font', **font)
# plot figure
fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2,4,figsize=(18/2.54,9/2.54))
# select modis bands available in multiband composite
modis_bands = [name for name in list(data_.keys()) if name.startswith('MOD_B')]
# set offset for modis data
pos0 = ax1.get_position()
pos0.x0 = pos0.x0 - 0.015*3
pos0.x1 = pos0.x1 - 0.015*3
ax1.set_position(pos0)
# set limits of colorbar (identical for all different data as the data are normalized)
glob_vmin = -3
glob_vmax = 3

# loop over modis bands to plot seven bands
for band_n, modis_band in enumerate(modis_bands[::-1]):
    pos = ax1.get_position()
    pos.x0 = pos.x0 + 0.015*band_n
    pos.y0 = pos.y0 - 0.07*band_n
    pos.x1 = pos.x1 + 0.015*band_n
    pos.y1 = pos.y1 - 0.07*band_n
    new_ax = fig.add_axes(pos)
    data_[modis_band].plot(ax=new_ax,add_colorbar=False,
                            vmin=glob_vmin,vmax=glob_vmax,cmap='Greys_r')
    new_ax.set_aspect('equal', 'box')
    new_ax.get_xaxis().set_visible(False)
    new_ax.get_yaxis().set_visible(False)
    new_ax.set_title(f'band {7-band_n}',y=1.0, pad=-11.5,fontsize=9)
ax1.axis('off')
ax1.set_title('Modis composite',x=0.28)
ax1.set_zorder(1)
# annotate subpanel
ax1.annotate('A',xy=(-0.18,0.82),fontsize=18,
            weight='bold',path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            annotation_clip=False)

# plot modis mosaic
data_['modis_norm'].plot(ax=ax2,add_colorbar=False,
                        vmin=glob_vmin,vmax=glob_vmax,cmap='Greys_r')
ax2.set_aspect('equal', 'box')
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
ax2.set_title('Modis mosaic')
# annotate subpanel
ax2.annotate('B',xy=(0.05,0.82),xycoords='axes fraction',fontsize=18,weight='bold',path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# plot radar data
data_['radar_norm'].plot(ax=ax3,add_colorbar=False,
                        vmin=glob_vmin,vmax=glob_vmax,cmap='Greys_r')
ax3.set_aspect('equal', 'box')
ax3.get_xaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)
ax3.set_title('Radar backscatter')
# annotate subpanel
ax3.annotate('C',xy=(0.05,0.82),xycoords='axes fraction',fontsize=18,weight='bold',path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# plot elevation data
dem_plt = data_['DEM_norm'].plot(ax=ax4,add_colorbar=False,
                                vmin=glob_vmin,vmax=glob_vmax,cmap='Greys_r')
ax4.set_aspect('equal', 'box')
ax4.get_xaxis().set_visible(False)
ax4.get_yaxis().set_visible(False)
ax4.set_title('Elevation')
ax4.set_zorder(0)
ax4t = fig.add_axes(ax4.get_position())
ax4t.set_zorder(2)
ax4t.axis('off')
# annotate subpanel
ax4.annotate('D',xy=(0.05,0.82),xycoords='axes fraction',fontsize=18,weight='bold',path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# hide unused axes
ax5.axis('off')
ax6.axis('off')

# plot labels (hand labels and "noisy" labels)
# -1 = missing data
# 0 = no blue ice, noise and clean labels agree
# 1 = blue ice, noise and clean labels agree
# 2 = no blue ice (clean labels), noise and clean labels disagree
# 3 = blue ice (clean labels), noise and clean labels disagree
targets_clean = targets_['target']
targets_clean = xr.where(targets_clean==3,1,targets_clean)
targets_clean = xr.where(targets_clean==2,0,targets_clean)
targets_clean.plot(ax=ax7,add_colorbar=False,
                levels=2, vmin=1, vmax=2, colors=['snow','lightblue'])
ax7.set_aspect('equal', 'box')
ax7.get_xaxis().set_visible(False)
ax7.get_yaxis().set_visible(False)
ax7.set_title('Hand labels')
ax7.set_zorder(0)
# annotate subpanel
ax7.annotate('E',xy=(0.05,0.82),xycoords='axes fraction',fontsize=18,weight='bold',path_effects=[pe.withStroke(linewidth=2, foreground="white")])

targets_noise = targets_['target']
targets_noise = xr.where(targets_noise==2,1,targets_noise)
targets_noise = xr.where(targets_noise==3,0,targets_noise)
ax8plt = targets_noise.plot(ax=ax8,add_colorbar=False,levels=2,
                    vmin=1, vmax=2,colors=['snow','lightblue'])
ax8plt.set_zorder(0)
ax8.set_aspect('equal', 'box')
ax8.get_xaxis().set_visible(False)
ax8.get_yaxis().set_visible(False)
ax8.set_title('Noisy labels')
# annotate subpanel
ax8.annotate('F',xy=(0.05,0.82),xycoords='axes fraction',fontsize=18,weight='bold',path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# plot legend
legend_ax = fig.add_axes([0.355, 0.16, 0.145, 0.045])
legend_0 = Rectangle((0.01,0.1),0.44,0.88,facecolor='snow',edgecolor='k',linewidth=0.8)
legend_1 = Rectangle((0.55,0.1),0.44,0.88,facecolor='lightblue',edgecolor='k',linewidth=0.8)
legend_ax.add_patch(legend_0)
legend_ax.add_patch(legend_1)
legend_ax.axis('off')
legend_ax.annotate('blue ice',xy=(0.8,-0.6),ha='center',annotation_clip=False,fontsize=9)
legend_ax.annotate('other',xy=(0.25,-0.6),ha='center',annotation_clip=False,fontsize=9)

# plot colorbar
cbar_ax = fig.add_axes([0.355, 0.37, 0.145, 0.035])
cb = fig.colorbar(dem_plt, cax=cbar_ax,orientation='horizontal')
cb.set_label('normalized values',fontsize=9,labelpad=-40)

# adjust spacing
plt.subplots_adjust(wspace=0)
plt.margins(0,0)
# save figure
fig.savefig(f'../output/figures/overview_inputdata.png',bbox_inches = 'tight',
    pad_inches = 0,dpi=300,facecolor='white')
plt.show()

