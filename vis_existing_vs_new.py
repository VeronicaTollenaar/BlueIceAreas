# visualize existing labels vs BIA outlines generated in this study and handlabels
# import packages
import numpy as np
import os
import matplotlib.pyplot as plt
import geopandas
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
from rasterio.plot import show
import rasterio
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
path = 'bia'
#path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)
# %%
# function to open LIMA data
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

#%%
# parameter to plot LIMA data or not (faster without)
plt_lima = True
# define figure
fig = plt.figure(figsize=(18/2.54, 13/2.54))
gs = fig.add_gridspec(30, 30)
# subaxes overview map
ax_map = fig.add_subplot(gs[15:,10:20])# rows, columns
# subaxes insets
ax_ins1 = fig.add_subplot(gs[:15,:10])
ax_ins2 = fig.add_subplot(gs[:15,10:20])
ax_ins3 = fig.add_subplot(gs[:15,20:])
ax_ins4 = fig.add_subplot(gs[15:,:10])
ax_ins5 = fig.add_subplot(gs[15:,20:])

# set global colors
c_handlabels = '#FA990A'
c_bia = '#466595'

# plot overview map
# open iceboundaries (quantarctica measures ice boundaries)
ice_boundaries_path = r'../data/IceBoundaries_Antarctica_v2.shx'
ice_boundaries_raw = geopandas.read_file(ice_boundaries_path)
ice_boundaries_all = geopandas.GeoSeries(ice_boundaries_raw.unary_union)
# plot ice boundaries
ice_boundaries_all.plot(ax=ax_map, color='#CED0D6')
# open BIAs of Hui et al., 2014
BIAs_path = r'../data/BlueIceAreas.shx'
BIAs_raw = geopandas.read_file(BIAs_path)
# plot all BIAs
BIAs_raw['geometry'].plot(ax=ax_map,color=c_bia,zorder=2)
# hide frame around map
ax_map.xaxis.set_visible(False)
ax_map.yaxis.set_visible(False)
ax_map.spines['top'].set_visible(False)
ax_map.spines['right'].set_visible(False)
ax_map.spines['bottom'].set_visible(False)
ax_map.spines['left'].set_visible(False)
# plot label subpanel
ax_map.annotate('E',xy=(0.03,0.82),xycoords='axes fraction',fontsize=18,weight='bold',path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# open train, validation, test areas
sqs_train_path = r'../data/train_squares.shx'
sqs_train = geopandas.read_file(sqs_train_path).rename(columns={'geometry':'square'}).set_geometry(col='square')  
sqs_val_path = r'../data/validation_squares.shx'
sqs_val = geopandas.read_file(sqs_val_path).rename(columns={'geometry':'square'}).set_geometry(col='square')
sqs_test_path = r'../data/test_squares.shx'
sqs_test = geopandas.read_file(sqs_test_path).rename(columns={'geometry':'square'}).set_geometry(col='square')
# select handlabeled areas and plot outlines of tiles on overview map
sqs_train[sqs_train['id_square']=='265']['square'].boundary.plot(ax=ax_map, color='k')
sqs_val[sqs_val['id_square']=='278']['square'].boundary.plot(ax=ax_map, color='k')
sqs_val[sqs_val['id_square']=='246']['square'].boundary.plot(ax=ax_map, color='k')
sqs_test[sqs_test['id_square']=='264']['square'].boundary.plot(ax=ax_map, color='k')
sqs_test[sqs_test['id_square']=='409']['square'].boundary.plot(ax=ax_map, color='k')

# function to plot insets
def plot_inset(ax_ins,name_region,n_region,poly,subpanel,loc_legend,type_region,color_region):
    # set limits of axes
    ax_ins.set_xlim([poly.bounds.minx.values,poly.bounds.maxx.values])
    ax_ins.set_ylim([poly.bounds.miny.values,poly.bounds.maxy.values])
    # plot existing BIA outlines (of Hui et al., 2014)
    plt.rcParams['hatch.linewidth'] = 0.5
    plt.rcParams['hatch.color'] = c_bia
    # simplify BIA outlines for visual purposes
    BIAs_raw.simplify(500).plot(ax=ax_ins,linewidth=0.6,
                facecolor="none",edgecolor=c_bia)
    # open handlabels (try except to accomodate for spelling of filename)
    try:
        handlabels_path = rf'../data/handlabeled_sq{n_region}.shx'
        handlabels_raw = geopandas.read_file(handlabels_path)
    except:
        handlabels_path = rf'../data/handlabelled_sq{n_region}.shx'
        handlabels_raw = geopandas.read_file(handlabels_path)
    # plot handlabels
    handlabels_raw.simplify(500).plot(ax=ax_ins,linewidth=0.6,
                facecolor="none",edgecolor=c_handlabels)
    # load and plot background image
    # caluculate how many background images to open (is slow)
    if plt_lima == True:
        img_open_x = np.arange(poly.bounds.minx.values,poly.bounds.maxx.values+0.14e6,0.15e6)
        img_open_y = np.arange(poly.bounds.miny.values,poly.bounds.maxy.values+0.14e6,0.15e6)
        for xs in img_open_x:
            for ys in img_open_y:
                backgr,_x,_y = openJPGgivenbounds(xs,ys)
                show(backgr.read(),ax=ax_ins,transform=backgr.transform)
    # hide axes
    ax_ins.xaxis.set_visible(False)
    ax_ins.yaxis.set_visible(False)
    # plot scalebar
    scalebar = AnchoredSizeBar(ax_ins.transData,
                            50000, '50 km', 
                            loc=loc_legend, 
                            pad=0.5,
                            color='black',
                            frameon=False, 
                            size_vertical=(poly.bounds.maxx.values-poly.bounds.minx.values)/90,
                            fontproperties=fm.FontProperties(size=9),
                            label_top=True,
                            sep=1)
    ax_ins.add_artist(scalebar)
    # annotate subpanels
    ax_ins.annotate(subpanel,xy=(0.03,0.87),xycoords='axes fraction',fontsize=18,weight='bold',path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    title_annot = f"{name_region} $\it{{({type_region})}}$"
    ax_ins.annotate(title_annot,xy=(0.01,1.025),xycoords='axes fraction',fontsize=9,weight='bold',path_effects=[pe.withStroke(linewidth=2, foreground="white")])
# define colors of different type of regions (not used finally)
c_train = '#940065'
c_val = '#FA990A'
c_test = '#00C797'
# plot insets
plot_inset(ax_ins=ax_ins1,
           name_region='McMurdo Dry Valleys',
           n_region='246',
           poly=sqs_val[sqs_val['id_square']=='246']['square'],
           subpanel='A',
           loc_legend='lower left',
           type_region='Validation',
           color_region=c_val)

plot_inset(ax_ins=ax_ins2,
           name_region='S\u00F8r Rondane Mts (W)',
           n_region='278',
           poly=sqs_val[sqs_val['id_square']=='278']['square'],
           subpanel='B',
           loc_legend='lower left',
           type_region='Validation',
           color_region=c_val)

plot_inset(ax_ins=ax_ins3,
           name_region='Denman/Apfel Glacier',
           n_region='409',
           poly=sqs_test[sqs_test['id_square']=='409']['square'],
           subpanel='C',
           loc_legend='lower left',
           type_region='Test',
           color_region=c_test)

plot_inset(ax_ins=ax_ins4,
           name_region='Prince Albert Mts',
           n_region='265',
           poly=sqs_train[sqs_train['id_square']=='265']['square'],
           subpanel='D',
           loc_legend='lower right',
           type_region='Training',
           color_region=c_train)

plot_inset(ax_ins=ax_ins5,
           name_region='Victoria Land (West)',
           n_region='264',
           poly=sqs_test[sqs_test['id_square']=='264']['square'],
           subpanel='F',
           loc_legend='lower right',
           type_region='Test',
           color_region=c_test)

# plot legend
legend_elements = [Patch(facecolor='none', edgecolor=c_bia,
                         label='Hui et al., 2014'),
                   Patch(facecolor='none', edgecolor=c_handlabels,
                         label='Handlabels')]
ax_map.legend(handles=legend_elements, 
              bbox_to_anchor=(0.5, -0.21),loc='lower center',borderaxespad=0)

# adjust margins/spacing
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0.3)
plt.margins(0,0)

# adjust location of main map plot
top_align = ax_ins4.get_position().y1 + 0.05
pos_map = ax_map.get_position()
pos_map.y1 = top_align
ax_map.set_position(pos_map)
# save figure
fig.savefig('../output/figures/handlabels.png',bbox_inches = 'tight',
    pad_inches = 0.02,dpi=300,facecolor='white')
# %%
# plot handlabels vs generated BIA outlines in this study
# open our BIAs
ourBIAs_path = r'../output/smoothed_BIAs.shx'
ourBIAs_raw = geopandas.read_file(ourBIAs_path)
ourBIAs_uu = ourBIAs_raw.unary_union
# define figure
fig = plt.figure(figsize=(18/2.54, 13/2.54))
gs = fig.add_gridspec(30, 30)
# subaxes overview map
ax_map = fig.add_subplot(gs[15:,10:20])# rows, columns
# subaxes insets
ax_ins1 = fig.add_subplot(gs[:15,:10])
ax_ins2 = fig.add_subplot(gs[:15,10:20])
ax_ins3 = fig.add_subplot(gs[:15,20:])
ax_ins4 = fig.add_subplot(gs[15:,:10])
ax_ins5 = fig.add_subplot(gs[15:,20:])

# set global colors
c_ourbia = 'k'

# plot overview map
# open iceboundaries (quantarctica measures ice boundaries)
ice_boundaries_path = r'../data/IceBoundaries_Antarctica_v2.shx'
ice_boundaries_raw = geopandas.read_file(ice_boundaries_path)
ice_boundaries_all = geopandas.GeoSeries(ice_boundaries_raw.unary_union)
# plot ice boundaries
ice_boundaries_all.plot(ax=ax_map, color='#CED0D6')
# plot all BIAs
ourBIAs_raw['geometry'].plot(ax=ax_map,color=c_ourbia,zorder=2)
# hide frame around map
ax_map.xaxis.set_visible(False)
ax_map.yaxis.set_visible(False)
ax_map.spines['top'].set_visible(False)
ax_map.spines['right'].set_visible(False)
ax_map.spines['bottom'].set_visible(False)
ax_map.spines['left'].set_visible(False)
# plot label subpanel
ax_map.annotate('E',xy=(0.03,0.82),xycoords='axes fraction',fontsize=18,weight='bold',path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# select handlabeled areas and plot outlines of tiles on overview map
sqs_train[sqs_train['id_square']=='265']['square'].boundary.plot(ax=ax_map, color='k')
sqs_val[sqs_val['id_square']=='278']['square'].boundary.plot(ax=ax_map, color='k')
sqs_val[sqs_val['id_square']=='246']['square'].boundary.plot(ax=ax_map, color='k')
sqs_test[sqs_test['id_square']=='264']['square'].boundary.plot(ax=ax_map, color='k')
sqs_test[sqs_test['id_square']=='409']['square'].boundary.plot(ax=ax_map, color='k')

# function to plot insets
def plot_inset(ax_ins,name_region,n_region,poly,subpanel,loc_legend,type_region,color_region):
    # set limits of axes
    ax_ins.set_xlim([poly.bounds.minx.values,poly.bounds.maxx.values])
    ax_ins.set_ylim([poly.bounds.miny.values,poly.bounds.maxy.values])
    
    # plot our BIA outlines
    ourBIAs_raw.plot(ax=ax_ins,linewidth=0.7,
                facecolor="none",edgecolor=c_ourbia)
    # open handlabels (try except to accomodate for spelling of filename)
    try:
        handlabels_path = rf'../data/handlabeled_sq{n_region}.shx'
        handlabels_raw = geopandas.read_file(handlabels_path)
    except:
        handlabels_path = rf'../data/handlabelled_sq{n_region}.shx'
        handlabels_raw = geopandas.read_file(handlabels_path)
    # plot handlabels
    handlabels_raw.plot(ax=ax_ins,linewidth=0.3,
                facecolor="none",edgecolor=c_handlabels)
    # load and plot background image
    # caluculate how many background images to open (is slow)
    if plt_lima == True:
        img_open_x = np.arange(poly.bounds.minx.values,poly.bounds.maxx.values+0.14e6,0.15e6)
        img_open_y = np.arange(poly.bounds.miny.values,poly.bounds.maxy.values+0.14e6,0.15e6)
        for xs in img_open_x:
            for ys in img_open_y:
                backgr,_x,_y = openJPGgivenbounds(xs,ys)
                show(backgr.read(),ax=ax_ins,transform=backgr.transform)

    # hide axes
    ax_ins.xaxis.set_visible(False)
    ax_ins.yaxis.set_visible(False)
    # plot scalebar
    scalebar = AnchoredSizeBar(ax_ins.transData,
                            50000, '50 km', 
                            loc=loc_legend, 
                            pad=0.5, #0.0005
                            color='black',
                            frameon=False, 
                            size_vertical=(poly.bounds.maxx.values-poly.bounds.minx.values)/90,
                            fontproperties=fm.FontProperties(size=9),
                            label_top=True,
                            sep=1)
    ax_ins.add_artist(scalebar)
    # annotate subpanels
    ax_ins.annotate(subpanel,xy=(0.03,0.87),xycoords='axes fraction',fontsize=18,weight='bold',path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    title_annot = f"{name_region} $\it{{({type_region})}}$"
    ax_ins.annotate(title_annot,xy=(0.01,1.025),xycoords='axes fraction',fontsize=9,weight='bold',path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    
# define colors of different type of regions (not used finally)
c_train = '#940065'
c_val = '#FA990A'
c_test = '#00C797'
# plot insets
plot_inset(ax_ins=ax_ins1,
           name_region='McMurdo Dry Valleys',
           n_region='246',
           poly=sqs_val[sqs_val['id_square']=='246']['square'],
           subpanel='A',
           loc_legend='lower left',
           type_region='Validation',
           color_region=c_val)

plot_inset(ax_ins=ax_ins2,
           name_region='S\u00F8r Rondane Mts (W)',
           n_region='278',
           poly=sqs_val[sqs_val['id_square']=='278']['square'],
           subpanel='B',
           loc_legend='lower left',
           type_region='Validation',
           color_region=c_val)

plot_inset(ax_ins=ax_ins3,
           name_region='Denman/Apfel Glacier',
           n_region='409',
           poly=sqs_test[sqs_test['id_square']=='409']['square'],
           subpanel='C',
           loc_legend='lower left',
           type_region='Test',
           color_region=c_test)

plot_inset(ax_ins=ax_ins4,
           name_region='Prince Albert Mts',
           n_region='265',
           poly=sqs_train[sqs_train['id_square']=='265']['square'],
           subpanel='D',
           loc_legend='lower right',
           type_region='Training',
           color_region=c_train)

plot_inset(ax_ins=ax_ins5,
           name_region='Victoria Land (West)',
           n_region='264',
           poly=sqs_test[sqs_test['id_square']=='264']['square'],
           subpanel='F',
           loc_legend='lower right',
           type_region='Test',
           color_region=c_test)

# plot legend
legend_elements = [Patch(facecolor='none', edgecolor=c_ourbia,
                         label='Our BIA outlines'),
                   Patch(facecolor='none', edgecolor=c_handlabels,
                         label='Handlabels')]
ax_map.legend(handles=legend_elements, 
              bbox_to_anchor=(0.5, -0.21),loc='lower center',borderaxespad=0)

# adjust margins/spacing
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0.3)
plt.margins(0,0)

# adjust location of main map plot
top_align = ax_ins4.get_position().y1 + 0.05
pos_map = ax_map.get_position()
pos_map.y1 = top_align
ax_map.set_position(pos_map)
# save figure
fig.savefig('../output/figures/handlabels_vs_ourBIAs.png',bbox_inches = 'tight',
    pad_inches = 0.02,dpi=300,facecolor='white')

