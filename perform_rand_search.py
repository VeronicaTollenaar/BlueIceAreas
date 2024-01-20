# run random search for hyper parameter optimizing and compare results
# import packages
import numpy as np
import json
import subprocess
import os
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
# set directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)
#%%

# tell MKL to use GNU threading layer instead of Inel one
os.environ['MKL_THREADING_LAYER'] = 'GNU'

#run trainingscript in terminal
# 20 runs
np.random.seed(seed=1)#0
# generate parameter settings
for n in range(1,31,1):
    name_run = f'hp_opt_{n}'
    power_learning_rate = np.round(np.random.rand(1)*2 - 5,1)
    learning_rate = 10**(power_learning_rate) # between 10-5 and 10-3
    learning_rate_decay_mult =  np.round(np.random.rand(1)*0.9 + 0.1,1) # between 0.1 and 1
    weight_decay = np.round(np.random.rand(1)*1e-4,5)# betweeen 0 and 10-4
    norm_elevation = bool(np.round(np.random.rand(1))) # true or false
    # create parameter setting .json script
    fh = open(f"parameter_settings_hp_opt_{n}.json", "w") # hp_opt = hyper parameter optimization
    fh.write(json.dumps({"name_run":  name_run,
                        "load_model": False,
                        "learning_rate": float(learning_rate),
                        "num_epochs": 100,
                        "learning_rate_decay_mult": float(learning_rate_decay_mult),
                        "learning_rate_decay_step": 20,
                        "weight_decay": float(weight_decay),
                        "data_directory": "train_tiles",
                        "batch_size": 20,
                        "save_train_images": False,
                        "manual_seed": int(n),
                        "loss_function": "BCE",
                        "weight_loss": 1.0,
                        "save_validation_imgs_interval": 0,
                        "save_validation_imgs_number": 0,
                        "save_clean_validation_imgs_interval": 0,
                        "batchnorm": True,
                        "probability_dropout": 0.0,
                        "probability_augmentation": 1.0,
                        "in_channels": 10,
                        "norm_elevation": norm_elevation}, indent=4))
    fh.close()

for n in range(1,31,1):
    # open terminal to run script
    print(f'Hyper parameter optimization run {n}')
    cmd = f'nohup python -u train.py -c parameter_settings_hp_opt_{n}.json &'
    subprocess.Popen(cmd, shell=True).wait()
    # wait for the process to finish before moving on to the next iteration
    time.sleep(5200)

#%%
#load results
#create empty dataframe
results = pd.DataFrame(columns=['learning_rate',
                                'epoch_max_perf',
                                'learning_rate_decay_mult',
                                'weight_decay',
                                'norm_elevation',
                                'f1_val_clean'])
#%%
for run in range(1,31,1):
    name_run = f'hp_opt_{run}'
    epoch = torch.load(f"../output/{name_run}.pth.tar",map_location=torch.device('cpu'))['epoch']
    perf_metrics = torch.load(f"../output/{name_run}_perf_metrics.pth.tar",map_location=torch.device('cpu'))['f1_val_clean'][epoch].item()
    filename = open(f'parameter_settings_hp_opt_{run}.json')
    param_settings = json.load(filename)
    learning_rate = param_settings['learning_rate']
    learning_rate_decay_mult = param_settings['learning_rate_decay_mult']
    weight_decay = param_settings['weight_decay']
    norm_elevation = param_settings['norm_elevation']
    df_line = pd.DataFrame({'learning_rate':learning_rate,
                            'epoch_max_perf':epoch,
                            'learning_rate_decay_mult':learning_rate_decay_mult,
                            'weight_decay':weight_decay,
                            'norm_elevation':norm_elevation,
                            'f1_val_clean':perf_metrics}, index = [name_run])

    results = pd.concat([results,df_line])
# %%
# plot relations
plt.scatter(results['learning_rate'],results['f1_val_clean'])
plt.xscale('log')
plt.title('learning rate')
plt.show()

plt.scatter(results['epoch_max_perf'],results['f1_val_clean'])
plt.title('epoch maximum performace')
plt.show()

plt.scatter(results[results['epoch_max_perf']>20]['learning_rate_decay_mult'],results[results['epoch_max_perf']>20]['f1_val_clean'])
plt.title('learning rate decay multiplication factor (for model runs with >20 epochs)')
plt.show()

plt.scatter(results['weight_decay'],results['f1_val_clean'])
plt.title('weight decay')
plt.show()

plt.scatter(results['norm_elevation'],results['f1_val_clean'])
plt.title('normalize elevation')
plt.show()

# # %%
results_sorted = results.sort_values('f1_val_clean',ascending=False)
# select settings according to best performing hyper parameter run
learning_rate = results_sorted.iloc[0]['learning_rate']
learning_rate_decay_mult = results_sorted.iloc[0]['learning_rate_decay_mult']
weight_decay = results_sorted.iloc[0]['weight_decay']
norm_elevation = results_sorted.iloc[0]['norm_elevation']

# %%
# generate parameter settings for final model
for n in range(1,5,1):
    name_run = f'final_model_{n}'
    # create parameter setting .json script
    fh = open(f"parameter_settings_final_model_{n}.json", "w") # hp_opt = hyper parameter optimization
    fh.write(json.dumps({"name_run":  name_run,
                        "load_model": False,
                        "learning_rate": float(learning_rate),
                        "num_epochs": 100,
                        "learning_rate_decay_mult": float(learning_rate_decay_mult),
                        "learning_rate_decay_step": 20,
                        "weight_decay": float(weight_decay),
                        "data_directory": "train_tiles",
                        "batch_size": 20,
                        "save_train_images": False,
                        "manual_seed": 1, #int(n*100),
                        "loss_function": "BCE",
                        "weight_loss": 1.0,
                        "save_validation_imgs_interval": 0,
                        "save_validation_imgs_number": 0,
                        "save_clean_validation_imgs_interval": 0,
                        "batchnorm": True,
                        "probability_dropout": 0.0,
                        "probability_augmentation": 1.0,
                        "in_channels": 10,
                        "norm_elevation": norm_elevation}, indent=4))
    fh.close()
#%%
for n in range(0,1,1):
    # open terminal to run script
    print(f'Final model run {n}')
    cmd = f'nohup python -u train.py -c parameter_settings_final_model_{n}.json &'
    subprocess.Popen(cmd, shell=True).wait()
    # wait for the process to finish before moving on to the next iteration
    time.sleep(5200)

#%%
# load results
# create empty dataframe
results = pd.DataFrame(columns=['learning_rate',
                                'epoch_max_perf',
                                'learning_rate_decay_mult',
                                'weight_decay',
                                'norm_elevation',
                                'f1_val_clean',
                                'precision_246',
                                'precision_278',
                                'recall_246',
                                'recall_278'])

for run in range(0,5,1):
    if run == 0:
        name_run = f'hp_opt_1'
    else:
        name_run = f'final_model_{run}'
    epoch = torch.load(f"../output/{name_run}.pth.tar",map_location=torch.device('cpu'))['epoch']
    perf_metrics = torch.load(f"../output/{name_run}_perf_metrics.pth.tar",map_location=torch.device('cpu'))['f1_val_clean'][epoch].item()
    precision_246, accuracy_246, recall_246, iu_246 = torch.load(f"../output/{name_run}_perf_metrics.pth.tar",map_location=torch.device('cpu'))['val_clean_246'][epoch]
    precision_278, accuracy_278, recall_278, iu_278 = torch.load(f"../output/{name_run}_perf_metrics.pth.tar",map_location=torch.device('cpu'))['val_clean_278'][epoch]
    filename = open(f'parameter_settings_final_model_{run}.json')
    param_settings = json.load(filename)
    learning_rate = param_settings['learning_rate']
    learning_rate_decay_mult = param_settings['learning_rate_decay_mult']
    weight_decay = param_settings['weight_decay']
    norm_elevation = param_settings['norm_elevation']
    df_line = pd.DataFrame({'learning_rate':learning_rate,
                            'epoch_max_perf':epoch,
                            'learning_rate_decay_mult':learning_rate_decay_mult,
                            'weight_decay':weight_decay,
                            'norm_elevation':norm_elevation,
                            'f1_val_clean':perf_metrics,
                            'precision_246':precision_246.item(),
                            'precision_278':precision_278.item(),
                            'recall_246':recall_246.item(),
                            'recall_278':recall_278.item()}, index = [name_run])

    results = pd.concat([results,df_line])

# calculate F1 score per area (redundant with Performance_ValTestTiles.py script)
results['f1_246'] = (2*results['precision_246']*results['recall_246'])/(results['precision_246']+results['recall_246'])
results['f1_278'] = (2*results['precision_278']*results['recall_278'])/(results['precision_278']+results['recall_278'])
results['average_precision'] = (results['precision_246']+results['precision_278'])/2
results['average_recall'] = (results['recall_246']+results['recall_278'])/2
results['check'] = (results['f1_246']+results['f1_278'])/2
print(results.mean())
print(results.std())

# %%
# plot learning curve for the different models
fig, ax = plt.subplots(figsize=(9/2.54, 9/2.54))
# add subaxes
ax1 = ax.twinx()
# predefine emtpy arrays to store data
epochs_all = np.empty(0)
perc_noiseornot_all = np.empty(0)
f1_all = np.empty(0)
perc_noiseornot_sel_epoch = []

# plot learning curves for the 5 different models
for run in range(0,5,1):
    if run == 0:
        name_run = f'hp_opt_1'
    else:
        name_run = f'final_model_{run}'
    # load performance metrics
    perf_metrics = torch.load(f"../output/{name_run}_perf_metrics.pth.tar",map_location=torch.device('cpu'))
    # extract loss, epoch, perc_noiseornot and f1 from peformance metrics
    loss_train = perf_metrics['loss_train']
    epochs = np.array(perf_metrics['epoch'])+1
    perc_noiseornot = perf_metrics['perc_noiseornot']
    f1_val = perf_metrics['f1_val_clean']
    print(perc_noiseornot[np.argmax(f1_val)])
    perc_noiseornot_sel_epoch.append(perc_noiseornot[np.argmax(f1_val)])
    
    # concatenate values to a big array that stores data
    epochs_all = np.concatenate([epochs_all,epochs])
    perc_noiseornot_all = np.concatenate([perc_noiseornot_all,perc_noiseornot])
    f1_all = np.concatenate([f1_all,f1_val])

    # plot learning curve
    ax.plot(epochs,loss_train,color='k',linewidth=0.5)
    
    # plot percentage of noise or not
    ax1.plot(epochs,[i*100 for i in perc_noiseornot], color='orangered',linewidth=0.5)
    # plot f1 score validation data
    ax1.plot(epochs,[i*100 for i in f1_val],color='darkslategray',linewidth=0.5)
    # set limits
    ax1.set_ylim([0.,100.])
    # plot maximum performance (=model as used to predict)
    ax1.scatter(epochs[np.argmax(f1_val)],np.max(f1_val)*100,marker='*',color='darkslategray')

# set labels
ax.set_xlabel('epoch')
ax.set_ylabel('Binary cross entropy loss')
ax1.set_ylabel('F1 score handlabeled validation (%)',color='darkslategray')
ax1.annotate('Correct predictions in AOD (%)',rotation='90',xy=(72,50),
             horizontalalignment='center',
             verticalalignment='center',
             color="orangered",
             weight='bold',
             annotation_clip=False)

# adjust spacing
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
plt.margins(0,0)
# save figure
fig.savefig('../output/figures/learning_curve_plot.png',bbox_inches = 'tight',
    pad_inches = 0.02,dpi=300)


