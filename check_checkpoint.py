# check training results (e.g., learning curve)
# import packages
import os
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)
import torch
import matplotlib.pyplot as plt
import numpy as np
#%%
# define name of model run
name_run = 'final_model_1_withoutSAR'
# load checkpoint
checkpoint = torch.load(f"../output/{name_run}_perf_metrics.pth.tar",map_location=torch.device('cpu'))

# plot figure
fig, ax1 = plt.subplots(1,1,figsize=(12/2.54,8/2.54))
# plot training loss
plt.plot(checkpoint['epoch'],checkpoint['loss_train'],label='train',
         color='#AD0092')
# plot validation loss
plt.plot(checkpoint['epoch'],checkpoint['loss_val'],label='validation',
         color='#FAAF00',alpha=1)
# plot legend and labels
plt.legend(loc='upper left')
plt.title('Train/validation loss (DEM + Modis Mosaique)')
plt.xlabel('epoch')
plt.ylabel('loss')
# set limits
plt.ylim([0,0.6])
# adjust spacing
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
# save figure
fig.savefig(f'../output/learning_curves/trainvalloss_{name_run}.png',bbox_inches = 'tight',
    pad_inches = 0,dpi=300)
#%%
# plot figure
fig, ax1 = plt.subplots(1,1,figsize=(12/2.54,8/2.54))
# plot validation loss of "noisy" labels
val_noisy = np.array(checkpoint['val_noisy'])
# plot performance metrics
plt.plot(checkpoint['epoch'],val_noisy[:,1],label='accuracy',
          color='#015973')
plt.plot(checkpoint['epoch'],val_noisy[:,0],label='precision',
         color='#ADAC8B')
plt.plot(checkpoint['epoch'],val_noisy[:,2],label='recall',
         color='#DB6600')
plt.plot(checkpoint['epoch'],val_noisy[:,3],label='intersection over union',
         color='#00BCFA')
# plot legend and labels
plt.legend(loc='lower right')
plt.title('Performance metrics validation data (DEM + Modis Mosaique)')
plt.xlabel('epoch')
plt.ylabel('performance metric')
# set limits
plt.ylim([0,1])
# adjust spacing
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
# annotate plot with F1 score of "noisy" validation data
prec = val_noisy[:,0]
recall = val_noisy[:,2]
F1 = 2*(prec*recall)/(prec+recall)
plt.annotate(f'F1: {np.round(F1.max(),3):.3f} (epoch {F1.argmax()})', (25, 0.55))
print(f'F1: {np.round(F1.max(),3):.3f} (epoch {F1.argmax()})')
# save figure
fig.savefig(f'../output/learning_curves/performance_metrics_validation_{name_run}.png',bbox_inches = 'tight',
    pad_inches = 0,dpi=300)

#%%
# check performaance on hand labeled validation data (two squares)
sq_246 = np.array(checkpoint['val_clean_246'])
prec = sq_246[:,0]
recall = sq_246[:,2]
F1_246 = 2*(prec*recall)/(prec+recall)

sq_278 = np.array(checkpoint['val_clean_278']) #precision_278, accuracy_278, recall_278, iu_278
prec = sq_278[:,0]
recall = sq_278[:,2]
F1_278 = 2*(prec*recall)/(prec+recall)

# plot figure
fig, ax1 = plt.subplots(1,1,figsize=(12/2.54,8/2.54))
# plot performance on hand labeled validation data over time
plt.plot(checkpoint['epoch'],F1_246,label='square 246',
          color='#015973')
plt.plot(checkpoint['epoch'],F1_278,label='square 278',
          color='#ADAC8B')
# plot reference F1 score (Table 1)
plt.hlines(0.56,0,checkpoint['epoch'][-1],
          color='#015973', linestyle = '--', label='reference F1 score')
plt.hlines(0.64,0,checkpoint['epoch'][-1],
          color='#ADAC8B', linestyle = '--', label='reference F1 score')
# plot legend and labels
plt.legend(loc='lower right')
plt.title(f'F1 score clean validation data {name_run}')
plt.xlabel('epoch')
plt.ylabel('performance metric')
# set limits
plt.ylim([0,1])
# adjust spacing
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
# save figure
fig.savefig(f'../output/learning_curves/F1_clean_validation_{name_run}.png',bbox_inches = 'tight',
    pad_inches = 0,dpi=300)
#%%
# check performance in areas of disagreement between hand labels and "noisy" labels in training data
perc_noiseornot = np.array(checkpoint['perc_noiseornot'])
# plot figure
fig, ax1 = plt.subplots(1,1,figsize=(12/2.54,8/2.54))
# plot percentage of correct predictions (according to hand labels)
plt.plot(checkpoint['epoch'],(perc_noiseornot)*100,color='k',linewidth=2)
# plot legend and labels
plt.legend(loc='lower right')
plt.title('Percentage of correct predictions of areas with noisy labels in training tiles')
plt.xlabel('epoch')
plt.ylabel('percentage correct predictions')
# set limits
plt.ylim([0,101])
# adjust spacing
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
# save figure
fig.savefig(f'../output/learning_curves/perc_noiseornot_{name_run}.png',bbox_inches = 'tight',
    pad_inches = 0,dpi=300,facecolor='white')
