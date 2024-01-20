# check number of parameters of model
# set working directory
import os
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

# import packages
import torch
from model import UNET
from utils import load_checkpoint

#%%
model_run = f"hp_opt_1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# import model
model = UNET(in_channels=10, out_channels=1).to(DEVICE)
# load trained model
load_checkpoint(model_run, model)
# set model in evaluation mode
model.eval()

#%%
# function to calculate the number of parameters
def pytorch_total_params(model):
    params = sum(p.numel() for p in model.parameters())
    return params

#%%
# calculate number of parameters
total_params = pytorch_total_params(model)

