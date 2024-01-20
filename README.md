Overview of scripts:

BIA_map_vis_and_analyze.py - visualizes map and analyzes continent-wide mapping (e.g., Figures 1, 2, and S8, S9, S11, S12, S13)
BIA_map_vis_radar.py -  visualizes uncertainties of BIA predictions related to the inclusion of radar data (Figure S7)
BIA_map_vis_uncertainties.py - visualizes uncertainties of BIA predictions (Figure 3)
check_checkpoint.py - checks training results (e.g., learning curve)
check_num_params.py - checks number of parameters of model
continentwide_predictions.py - generates continent-wide predictions, plots Figure S5
DataPreparation.py - generates data tiles for training, validation, testing, and for continent-wide predictions
dataset.py - reads in data in dataloader and applies different data augmentations
download_data.py - function to download data from MODIS (based on https://www.moonbooks.org/Articles/How-to-download-a-file-from-NASA-LAADS-DAAC-using-python-/)
explore_handlabels_MOAgrainsize.py - compares handlabels to MOA grainsize data (Figure S10)
merge_daily_passes.py - merges daily composites of MODIS into a multi-day composite
merge_ensemble_predictions.py - merges 40 BIA maps into a single BIA map
model.py - defines CNN model, script based on U-Net implementaion on https://github.com/aladdinpersson/Machine-Learning-Collection
parameter_settings_final_model_1.json - example of parameter settings file to train the CNN
perform_rand_search.py - runs random search for hyper parameter optimizing and compares results
Performance_ValTestTiles.py - estimates performance in validation and test squares
plot_MODISnewcomposite.py - generates Figure S1 to visualize new MODIS composite
plot_overview_data.py - plots overview of input data (Figure S6)
plot_overview_tiling.py - plots overview of datasplit (training, validation, and testing; Figure S2)
python_wrapper_functions.py - defines different functions to reproject and process MODIS data
python_wrapper_main.py - python wrapper to download and reproject MODIS data automatically
QA_tomask.py - processes quality bands of MODIS to mask out e.g., cloudy observations
train.py - trains CNN, script based on U-Net implementaion on https://github.com/aladdinpersson/Machine-Learning-Collection
utils.py - different functions used for training CNN
vis_existing_vs_new.py - visualizes existing labels vs BIA outlines generated in this study and handlabels (Figures S3 and S4)
