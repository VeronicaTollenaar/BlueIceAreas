# utils: different functions used for training CNN
# script based on U-Net implementaion on https://github.com/aladdinpersson/Machine-Learning-Collection
# import packages
import torch
import torchvision
from dataset import BIADataset
from torch.utils.data import DataLoader
import geopandas
from shapely.geometry import Point

# function to save checkpoint
def save_checkpoint(state, filename="my_checkpoint"):
    print("=> Saving checkpoint")
    torch.save(state,f'../output/{filename}.pth.tar')

# function to load checkpoint
def load_checkpoint(model_filename, model, state_dict="state_dict"):
    checkpoint = torch.load(f'../output/{model_filename}.pth.tar')
    print('=> Loading checkpoint')
    model.load_state_dict(checkpoint[state_dict])

# function to import dataloaders (further defined in dataset.py)
def get_loaders(
    train_dir,
    train_targetdir,
    val_dir,
    val_targetdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
    shuffle_=True,
    norm_elevation=True
    ):
    # object to load
    train_ds = BIADataset(
        image_dir=train_dir,
        target_dir=train_targetdir,
        transform_p=train_transform,
        normalize_elevation=norm_elevation,
    )
    # define dataloader train data
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle_,
    )
    # object to load 
    val_ds = BIADataset(
        image_dir=val_dir,
        target_dir=val_targetdir,
        transform_p=val_transform,
        normalize_elevation=norm_elevation,
    )
    # define dataloader validation data
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    
    return train_loader, val_loader

# function to check the performance in areas of disagreement between noisy labels and clean labels
def check_overfitting_noisylabels(loader_noiseornot, model, device="cuda"):
    # of the noisy area: how much % agrees with clean labels, how much with noise?
    perc_clean = 0.
    correct_pos_total = 0.
    correct_neg_total = 0.
    n_noise_total = 0.
    
    model.eval()

    with torch.no_grad():
        for image, target, lowerleft in loader_noiseornot:
            image = image.to(device)
            # targets of noiseornot:
            # -1 = missing data
            # 0 = no blue ice, noise and clean labels agree
            # 1 = blue ice, noise and clean labels agree
            # 2 = no blue ice (clean labels), noise and clean labels disagree
            # 3 = blue ice (clean labels), noise and clean labels disagree
            target = target.unsqueeze(1).to(device)
            preds = torch.sigmoid(model(image))
            preds = (preds > 0.5).float()
            n_noise = torch.logical_or(target == 2.0, target == 3.0)
            correct_pos = torch.logical_and(target == 3.0, preds == 1.0)
            correct_neg = torch.logical_and(target == 2.0, preds == 0.0)
            correct_pos_total = correct_pos_total + correct_pos.sum()
            correct_neg_total = correct_neg_total + correct_neg.sum()
            n_noise_total = n_noise_total + n_noise.sum()
        perc_clean = (correct_pos_total + correct_neg_total)/ n_noise_total
    model.train()
    return(perc_clean)

# function that calculates performance metrics given number of true positve, false positive, true negative, false negative
def perf_metrics(tp, fp, tn, fn):
    precision = (tp/(tp+fp))
    accuracy = ((tp+tn)/(tp+fp+fn+tn))
    recall = (tp/(tp+fn))
    iu = (tp/(tp+fp+fn))
    return precision, accuracy, recall, iu

# function that saves predictions as images
def validation(
        loader, 
        model, 
        loss_fn, 
        epoch, 
        batchsize, 
        epoch_start,
        epoch_stop,
        folder="../output/saved_images/", 
        device="cuda",
        save_img_interval=0,
        save_img_number=0,
    ):
    # set model to evaluation mode
    model.eval()
    # preset total loss
    loss_val = 0.0
    # preset true positive, false positive, false negative and true negative values for validation data (noisy labels)
    tp_val = 0
    fp_val = 0
    fn_val = 0
    tn_val = 0
    with torch.no_grad():
        for idx, (data_val, targets_val, filename) in enumerate(loader): #  masks_val, weightmaps_val
            # send data and targets to device
            data_val = data_val.to(device=device)
            targets_val = targets_val.unsqueeze(1).to(device=device) 
            
            # apply trained model to validation data
            predictions_val = model(data_val)
            # calculate loss of individual tile (ensure that there is data in the tile)
            if torch.numel(targets_val[targets_val>=0]) > 0:
                loss = loss_fn(predictions_val[targets_val>=0],
                                    targets_val[targets_val>=0])
                # sum loss of individual tile to total loss
                loss_val = loss_val + loss.item()
            
            # transform predictions (probabilities) to binary predictions
            predictions_binary_val = (torch.sigmoid(predictions_val) > 0.5).float()
            
            # cacluate true positive and false positive rates
            tp_tensor = torch.logical_and(predictions_binary_val == 1.0, targets_val == 1.0)
            fp_tensor = torch.logical_and(predictions_binary_val == 1.0, targets_val == 0.0)
            fn_tensor = torch.logical_and(predictions_binary_val == 0.0, targets_val == 1.0)
            tn_tensor = torch.logical_and(predictions_binary_val == 0.0, targets_val == 0.0)
            
            # sum values
            tp_val += tp_tensor.sum()
            fp_val += fp_tensor.sum()
            fn_val += fn_tensor.sum()
            tn_val += tn_tensor.sum()
            
            if save_img_interval > 0:
                epochs_tosave = range(epoch_start, epoch_stop, save_img_interval)
                # calculate stepsize that results in save_img_number saved images
                save_img_number_transf = int(len(loader)/save_img_number)
                n_tosave = range(0, len(loader), save_img_number_transf)
                
                # save targets as image
                if epoch==0:
                    if idx in n_tosave:
                        targets_img = targets_val.squeeze(1).squeeze(1).expand(3,-1,-1).clone() 
                        targets_img = -1*(targets_img -1)
                        targets_img[0,:,:][targets_img[2,:,:]==0] = 135/255  # #RGB of lightskyblue is 135,206,250
                        targets_img[1,:,:][targets_img[1,:,:]==0] = 206/255
                        targets_img[2,:,:][targets_img[2,:,:]==0] = 1
                        torchvision.utils.save_image(targets_img,
                            f"{folder}/targets_{filename[0][:-3]}.png")
                
                if epoch in epochs_tosave:
                    if idx in n_tosave:
                        # save predictions as image (per epoch)
                        predictions_img = predictions_binary_val.squeeze(1).expand(3,-1,-1).clone() 
                        predictions_img = -1*(predictions_img -1)
                        predictions_img[0,:,:][predictions_img[2,:,:]==0] = 135/255  # #RGB of lightskyblue is 135,206,250
                        predictions_img[1,:,:][predictions_img[1,:,:]==0] = 206/255
                        predictions_img[2,:,:][predictions_img[2,:,:]==0] = 1
                        torchvision.utils.save_image(predictions_img,
                                f"{folder}/pred_{filename[0][:-3]}_epoch{epoch}.png")

        # calculate performance metrics for total
        precision_val, accuracy_val, recall_val, iu_val = perf_metrics(tp_val, fp_val, tn_val, fn_val)
    
    # reset model to training mode    
    model.train()
    return(loss_val/len(loader), precision_val, accuracy_val, recall_val, iu_val)

# function to estimate performance of model on hand labelled validation data
def validation_clean(
                    loader, 
                    model, 
                    epoch,
                    epoch_start,
                    epoch_stop,
                    save_img_interval=0,
                    folder="../output/saved_images/", 
                    device="cuda",
    ):
    # set model to evaluation mode
    model.eval()
    
    # preset total performance metrics for both validation squares (= areas of 250 by 250 km) with hand labels
    tp_val_246 = 0
    fp_val_246 = 0
    fn_val_246 = 0
    tn_val_246 = 0
    
    tp_val_278 = 0
    fp_val_278 = 0
    fn_val_278 = 0
    tn_val_278 = 0
    
    # open all validation squares
    val_sqs_path = '../data/validation_squares.shp'
    val_sqs = geopandas.read_file(val_sqs_path)
    
    # define validation squares
    sq_246 = val_sqs[val_sqs['id_square']=='246']['geometry'].iloc[0]
    sq_278 = val_sqs[val_sqs['id_square']=='278']['geometry'].iloc[0]
   
    with torch.no_grad():
        for idx, (data_val, targets_val, filename) in enumerate(loader): 
            
            # calculate performance metrics for clean validation
            # extract coordinates of lowerleft corner from filename
            ll_x = float(filename[0].split('_', 3)[0])
            ll_y = float(filename[0].split('_', 3)[1][:-3])
            point_ll = geopandas.GeoSeries([Point(ll_x,ll_y)]) 
            if point_ll.within(sq_246).values == True:
                sq = '246'       
            if point_ll.within(sq_278).values == True:
                sq = '278'
            
            if (sq == '246' or sq=='278'):
                # send data and targets to device
                data_val = data_val.to(device=device)
                targets_val = targets_val.unsqueeze(1).to(device=device)

                # apply trained model to validation data
                predictions_val = torch.sigmoid(model(data_val))
                # transform predictions (probabilities) to binary predictions
                predictions_binary_val = (predictions_val > 0.5).float()

                # cacluate performance metrics (ignores targets of -1)
                tp_tensor = torch.logical_and(predictions_binary_val == 1.0, targets_val== 1.0)
                fp_tensor = torch.logical_and(predictions_binary_val == 1.0, targets_val == 0.0)
                fn_tensor = torch.logical_and(predictions_binary_val == 0.0, targets_val == 1.0)
                tn_tensor = torch.logical_and(predictions_binary_val == 0.0, targets_val == 0.0)
            
                # sum values --> separately for both squares
                if sq == '246':
                    tp_val_246 += tp_tensor.sum()
                    fp_val_246 += fp_tensor.sum()
                    fn_val_246 += fn_tensor.sum()
                    tn_val_246 += tn_tensor.sum()
                
                if sq == '278':
                    tp_val_278 += tp_tensor.sum()
                    fp_val_278 += fp_tensor.sum()
                    fn_val_278 += fn_tensor.sum()
                    tn_val_278 += tn_tensor.sum()
                
                if save_img_interval > 0:
                    epochs_tosave = range(epoch_start, epoch_stop, save_img_interval)
                    # save targets as image
                    if epoch==0:
                        targets_img = targets_val.squeeze(1).squeeze(1).expand(3,-1,-1).clone() 
                        targets_img = -1*(targets_img -1)
                        targets_img[0,:,:][targets_img[2,:,:]==0] = 135/255  # #RGB of lightskyblue is 135,206,250
                        targets_img[1,:,:][targets_img[1,:,:]==0] = 206/255
                        targets_img[2,:,:][targets_img[2,:,:]==0] = 1
                        torchvision.utils.save_image(targets_img,
                            f"{folder}/targets_clean_{filename[0][:-3]}.png")
                    
                    if epoch in epochs_tosave:
                        # save predictions as image (per epoch)
                        predictions_img = predictions_binary_val.squeeze(1).expand(3,-1,-1).clone() 
                        predictions_img = -1*(predictions_img -1)
                        predictions_img[0,:,:][predictions_img[2,:,:]==0] = 135/255  # #RGB of lightskyblue is 135,206,250
                        predictions_img[1,:,:][predictions_img[1,:,:]==0] = 206/255
                        predictions_img[2,:,:][predictions_img[2,:,:]==0] = 1
                        torchvision.utils.save_image(predictions_img,
                                f"{folder}/pred_clean_{filename[0][:-3]}_epoch{epoch}.png")

        # calculate performance metrics for total
        precision_246, accuracy_246, recall_246, iu_246 = perf_metrics(tp_val_246, fp_val_246, tn_val_246, fn_val_246)
        precision_278, accuracy_278, recall_278, iu_278 = perf_metrics(tp_val_278, fp_val_278, tn_val_278, fn_val_278)
    
    # reset model to training mode    
    model.train()
    
    return(precision_246, accuracy_246, recall_246, iu_246, 
           precision_278, accuracy_278, recall_278, iu_278)