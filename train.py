# train CNN, script based on U-Net implementaion on https://github.com/aladdinpersson/Machine-Learning-Collection
# import packages
import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    validation,
    validation_clean,
    check_overfitting_noisylabels,
)
import torchvision
import argparse
from distutils.util import strtobool
import json
import random

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)
import matplotlib.pyplot as plt
plt.close('all')
#%%
# define parameters

# configure parser (set up so that arguments can be defined in json file)
# based on https://medium.com/swlh/efficient-python-user-interfaces-combining-json-with-argparse-8bff716f31e4 
# and https://gist.github.com/matthewfeickert/3b7d30e408fe4002aac728fc911ced35, both consulted on 5/7/2022)
config_parser = argparse.ArgumentParser(description="train BIA classifier")
config_parser.add_argument("--config_file","-c", dest="config_file",
    type=str, default=None, help="Configuration JSON file")
args,unkown = config_parser.parse_known_args()
parser = argparse.ArgumentParser(parents=[config_parser], add_help=False)

# define arguments and default values
parser.add_argument("--name_run", "-name", type=str, default="latest_run", 
                    help="name of model run")
parser.add_argument("--load_model", "-load_model", type=lambda x: bool(strtobool(x)), default=False, 
                    help="load model True or False, loaded model should have the same name as the model run name")
parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4, help="Learning rate (float)")
parser.add_argument("--num_epochs", "-num_epochs", type=int, default=2, help="Number of epochs (int)")
parser.add_argument("--learning_rate_decay_mult", "-lr_mult", type=float, default=1.0, 
                    help="reduces the learning rate over time, value indicates multiplicative factor of leanring rate decay")
parser.add_argument("--learning_rate_decay_step", "-lr_step", type=int, default=10, 
                    help="reduces the learning rate over time, value indicates after how many epochs the learning rate is reduced")
parser.add_argument("--weight_decay", "-weight_decay", type=float, default=0.0, help="weight decay, can be used in Adam optimizer")
parser.add_argument("--data_directory", "-data_directory", type=str, default="traintest", help="define training and testing data directory")
parser.add_argument("--batch_size", "-batch_size", type=int, default=8, help="batch size, depends on input data and GPU memory")
parser.add_argument("--save_train_images", "-save_train_imgs", type=lambda x: bool(strtobool(x)), default=False, 
                    help="Save images during training and validation")
parser.add_argument("--manual_seed","-seed", type=int, default=0, 
                    help="torch.manual_seed ensures reproducibility of results")
parser.add_argument("--loss_function", "-loss_fn", type=str, default="BCE",
                    help="define loss function, BCE, weighted_BCE, L1, weighted_L1")
parser.add_argument("--weight_loss", "-weight_loss", type=float, default=1.,
                    help="define positive weight for weighted loss functions")
parser.add_argument("--save_validation_imgs_interval", "-save_val_imgs_int", type=int, default=0,
                    help="save validation (noisy labels) images every x epochs")
parser.add_argument("--save_validation_imgs_number", "-save_val_imgs_num", type=int, default=0,
                    help="save x validation images (approximately)")
parser.add_argument("--save_clean_validation_imgs_interval", "-save_clean_val_imgs_int", type=int, default=0,
                    help="save validation (clean labels) images every x epochs")
parser.add_argument("--batchnorm", "-batchnorm", type=lambda x: bool(strtobool(x)), default=True, 
                    help="implement batchnorm")
parser.add_argument("--probability_dropout", "-p_dropout", type=float, default=0.,
                    help="apply dropout with probability p")
parser.add_argument("--probability_augmentation", "-p_augm", type=float, default=0.,
                    help="probability of augmentation")
parser.add_argument("--in_channels", "-n_channels", type=int, default=2, 
                    help="number of input channels (depends on data)")
parser.add_argument("--norm_elevation", "-norm_elev", type=lambda x: bool(strtobool(x)), default=True,  
                    help="subtract mean of elevation per tile")

if args.config_file is not None:
    if '.json' in args.config_file:
        config = json.load(open(args.config_file))
        parser.set_defaults(**config)
        [
            parser.add_argument(arg)
            for arg in [arg for arg in unkown if arg.startswith('--')]
            if arg.split('--')[-1] in config
        ]
args = parser.parse_args()
print(args)

# device dependent settings
PIN_MEMORY = True
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print('GPU?', torch.cuda.is_available())
# set seed
torch.manual_seed(args.manual_seed)
random.seed(args.manual_seed)

# data directories
# train noisy labels
TRAIN_IMG_DIR = f"../{args.data_directory}/train_images/"
TRAIN_TARGET_DIR = f"../{args.data_directory}/train_targets/"
# train clean/hand labels
TRAIN_HL_IMG_DIR = f"../{args.data_directory}/train_images_noiseornot/"
TRAIN_HL_TARGET_DIR = f"../{args.data_directory}/train_targets_noiseornot/"

# validation noisy labels
VAL_IMG_DIR = f"../{args.data_directory}/val_images/"
VAL_TARGET_DIR = f"../{args.data_directory}/val_targets/"
# validation clean/hand labels
VAL_HL_IMG_DIR = f"../{args.data_directory}/val_images_clean/"
VAL_HL_TARGET_DIR = f"../{args.data_directory}/val_targets_clean/"

# train function (does one epoch of training)
def train_fn(loader,
             model, 
             optimizer, 
             loss_fn, 
             scheduler):
    # progress bar
    loop = tqdm(loader)
    # preset training loss and accuracy
    loss_train = 0.0
    accuracy_train = 0.0
    # loop over each batch
    for batch_idx, (data, 
                    targets, 
                    corners) in enumerate(loop):
        # send data to GPU
        data = data.to(device=DEVICE)
        
        # save targets as png
        if args.save_train_images:
            torchvision.utils.save_image(targets.float().unsqueeze(1),f"../output/train_images/img_{batch_idx}.png")
        # send targets to GPU
        targets = targets.unsqueeze(1).to(device=DEVICE)
        
        # perform forward pass
        # get predictions by applying model to data
        predictions = model(data)
        # calculate loss using defined loss function (targets==-1 are no data entries)
        loss = loss_fn(predictions[targets>=0], targets[targets>=0])
        # break if loss is not finite
        assert torch.isfinite(loss)
        
        # perform backward pass
        # zero all gradients of previous
        optimizer.zero_grad()
        # backpropagation
        loss.backward()
        # update weights
        optimizer.step()

        # calculate loss and accuracy to save for plotting
        # transform predictions to binary values (1=blue ice, 0=not blue ice)
        with torch.no_grad():
            predictions_binary = (torch.sigmoid(predictions) > 0.5).float()
            # save predictions as png
            if args.save_train_images:
                torchvision.utils.save_image(predictions_binary, f"../output/train_images/pred_{batch_idx}.png")
            # add loss of item to total training loss
            loss_train = loss_train + loss.item()
            # calculate number of correct predictions
            num_correct = (predictions_binary[targets>=0] == targets[targets>=0]).sum()
            # calculate number of predictions
            num_pixels = torch.numel(predictions_binary[targets>=0])
            # add accuracy to total training accuracy
            accuracy_train = accuracy_train + (num_correct/num_pixels)
            # ensure num_correct and num_pixels are overwritten for every batch
            del num_correct, num_pixels
        
        # update progress bar
        loop.set_postfix(loss=loss.item())
    # update learning rate
    scheduler.step()
    # return training loss and training accuracy
    return loss_train/len(loop), accuracy_train/len(loop)
        
        
# main function
def main():
    # send model to GPU
    model = UNET(in_channels=args.in_channels, out_channels=1, batchnorm=args.batchnorm, p_dropout=args.probability_dropout).to(DEVICE)
    
    # define loss function
    if args.loss_function == "BCE":
        loss_fn = nn.BCEWithLogitsLoss()
    
    if args.loss_function == "weighted_BCE":
        weight = torch.tensor(args.weight_loss)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)

    if args.loss_function == "L1":
        def loss_fn(predictions, targets):
            L1 = nn.L1Loss()
            preds = torch.sigmoid(predictions)
            loss_L1 = L1(preds, targets)
            return(loss_L1)
        
    if args.loss_function == "weighted_L1":
        def loss_fn(predictions, targets):
            preds = torch.sigmoid(predictions)
            loss_L1_noBIA = (preds[targets==0] - targets[targets==0]).abs()
            loss_L1_BIA = args.weight_loss*(preds[targets==1] - targets[targets==1]).abs()
            loss_L1 = torch.cat([loss_L1_noBIA,loss_L1_BIA]).mean()
            return(loss_L1) 
    
    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # define scheduler (that reduces learning rate over time)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.learning_rate_decay_step, gamma=args.learning_rate_decay_mult)
    # define data loaders
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_TARGET_DIR,
        VAL_IMG_DIR,
        VAL_TARGET_DIR,
        args.batch_size,
        args.probability_augmentation,
        0.0,
        NUM_WORKERS,
        PIN_MEMORY,
        shuffle_=True,
        norm_elevation=args.norm_elevation,
    )
    train_loader_noiseornot, val_loader_clean = get_loaders(
        TRAIN_HL_IMG_DIR,
        TRAIN_HL_TARGET_DIR,
        VAL_HL_IMG_DIR,
        VAL_HL_TARGET_DIR,
        1,
        0.0,
        0.0,
        NUM_WORKERS,
        PIN_MEMORY,
        shuffle_=False,
        norm_elevation=args.norm_elevation,
    )
    # check if existing model needs to be loaded
    if args.load_model:
        load_checkpoint(args.name_run, model)

    # define empty lists to save values per epoch (depending on whether continue training or new training)
    if args.load_model==True:
        trained_model = torch.load(f"../output/{args.name_run}.pth.tar")
        perf_metrics = torch.load(f"../output/{args.name_run}_perf_metrics.pth.tar")
        epoch_save = perf_metrics['epoch']
        loss_train_save = perf_metrics['loss_train']
        accuracy_train_save = perf_metrics['accuracy_train']
        loss_val_save = perf_metrics['loss_val']
        val_save = perf_metrics['val_noisy']
        val_clean_246_save = perf_metrics['val_clean_246']
        val_clean_278_save = perf_metrics['val_clean_278']
        epoch_init = epoch_save[-1] + 1
        f1_val_clean_save = perf_metrics['f1_val_clean']
        perc_noiseornot_save = perf_metrics['perc_noiseornot']

    else:
        epoch_save = []
        loss_train_save = []
        accuracy_train_save = []
        loss_val_save = []
        val_save = []
        val_clean_246_save = []
        val_clean_278_save = []
        epoch_init = 0
        perc_noiseornot_save = []
        f1_val_clean_save = []

    # set counter to abort training if no better performance of validation data is reached during 20 epochs
    count_nomax = 0    
    # loop over epochs
    for epoch in range(epoch_init, int(args.num_epochs + epoch_init)):
        print(f'epoch {epoch}')
        # apply train function to training data, calculate performance metrics training data
        loss_sum_train, accuracy_sum_train = train_fn(train_loader, model, optimizer, loss_fn, scheduler)
        
        # save epoch
        epoch_save.append(epoch)

        # append training loss and training accuracy
        loss_train_save.append(loss_sum_train)
        accuracy_train_save.append(accuracy_sum_train)
        
        # calculate performance metrics valdiation data and handlabelled data (summed over all samples) and save predicitons to folder
        # (functions loaded from utils.py)
        loss_val, precision_val, accuracy_val, recall_val, iu_val = validation(val_loader, model, loss_fn, 
            epoch=epoch, batchsize=1, epoch_start=epoch_init, epoch_stop=int(args.num_epochs+epoch_init),
            folder="../output/saved_images/", device=DEVICE,save_img_interval=args.save_validation_imgs_interval,
            save_img_number=args.save_validation_imgs_number)
        # calculate performance metrics validation data with clean targets
        # (functions loaded from utils.py)
        precision_246, accuracy_246, recall_246, iu_246, \
            precision_278, accuracy_278, recall_278, iu_278 = validation_clean(val_loader_clean, model, epoch,
                    epoch_start=epoch_init, epoch_stop=int(args.num_epochs+epoch_init),
                    save_img_interval=args.save_clean_validation_imgs_interval, folder="../output/saved_images/", device=DEVICE)
        # calculate percentage of noisy training data that is predicted correctly
        # (functions loaded from utils.py)
        perc_noiseornot = check_overfitting_noisylabels(train_loader_noiseornot, model, device="cuda")
        f1_246 = 2*(precision_246*recall_246)/(precision_246+recall_246)
        f1_278 = 2*(precision_278*recall_278)/(precision_278+recall_278)
        # calculate mean of performace in handlabelled validation tiles
        f1_val_clean_new = (f1_246 + f1_278)/2.

        print(f'F1 in square 246 = {f1_246}')
        print(f'F1 in square 278 = {f1_278}')
        print(f'percentage correctly predicted despite wrong label: {perc_noiseornot}')
        
        # append performance metrics to lists
        loss_val_save.append(loss_val)
        val_save.append((precision_val, accuracy_val, recall_val, iu_val))
        val_clean_246_save.append((precision_246, accuracy_246, recall_246, iu_246))
        val_clean_278_save.append((precision_278, accuracy_278, recall_278, iu_278))
        perc_noiseornot_save.append(perc_noiseornot)
        f1_val_clean_save.append(f1_val_clean_new)
        
        # save model and performance metrics
        perf_metrics = {
            "epoch": epoch_save,
            "loss_train":  loss_train_save,
            "accuracy_train": accuracy_train_save,
            "loss_val": loss_val_save,
            "val_noisy": val_save,
            "val_clean_246": val_clean_246_save,
            "val_clean_278": val_clean_278_save,
            "last_lr": scheduler.get_last_lr(),
            "weight_decay": args.weight_decay,
            "perc_noiseornot": perc_noiseornot_save,
            "f1_val_clean": f1_val_clean_save
        }
        save_checkpoint(perf_metrics,filename=f'{args.name_run}_perf_metrics')
        
        # only save trained model if performance on clean validation data maximizes
        if f1_val_clean_new == max(f1_val_clean_save):
            trained_model = {
                "epoch": epoch_save[-1],
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(trained_model,filename=args.name_run)
            count_nomax = 0
        # if previous epochs are not outperformed: add 1 to the counter that will eventually break the loop
        else:
            count_nomax = count_nomax + 1
        # stop training if no new maximum of performance on clean validation data has been reached in 20 epochs
        if count_nomax == 20:
            print('20 epochs without new best performance - breaking the loop')
            break

if __name__ == "__main__":
    main()

