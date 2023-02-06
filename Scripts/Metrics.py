import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import Configs



def pixel_accuracy(output, mask):
    '''
    Calculate the ratio betweeen the sum of for pixels thats were corrected in prediction mask.
    '''
    with torch.no_grad():
        output   = torch.argmax(F.softmax(output, dim = 1), dim = 1) #Get the location of pixel in predicted mask
        correct  = torch.eq(output, mask).int()                      #Comapare pixels between predicted mask and ground truth mask (= bool tensor of True's and False's)
        accuracy = float(correct.sum()) / float(correct.numel())     #Ratio between the sum of correct pixels and the count of all pixels in the mask
        
    return accuracy



def meanIoU(pred_mask, mask,
            num_classes:int = Configs.num_classes,
            smooth:bool     = False):
    
    '''
    Calculates the mean overlap area between the predicted and ground truth segmentation maps
    and divides it by the sum of their areas.
    The IoU ranges from 0 (no overlap) to 1 (perfect overlap).
    '''
    
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim = 1)    
        pred_mask = torch.argmax(pred_mask, dim = 1)
        pred_mask = pred_mask.contiguous().view(-1)     #Flatten the prediced mask tensor
        mask      = mask.contiguous().view(-1)          #Flatten the truth mask tensor
 
        iou_per_class = []
    
        for clas in range(0, num_classes):              #loop per pixel class
            true_class = pred_mask == clas                 #output looks like: tensor([False, True, False, ..., False, False, False])
            true_label = mask == clas

            if true_label.long().sum().item() == 0:     # 0 meens theres no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union     = torch.logical_or(true_class, true_label).sum().float().item()
                
                if smooth:
                    smooth = 1e-10
                    iou = (intersect + smooth) / (union + smooth)
                else:
                    iou = intersect / union
                
                iou_per_class.append(iou)
                    
                
        
        meanIoU = np.nanmean(iou_per_class)            #Compute the mean of iou_per_class list while ignoring the None values
                
    return meanIoU



def predict_image_and_score(model, Dataset, device, output_path:str):
    
    '''

    Predict an image and saving the predicted mask file,
    Calculates it's mean Intersect over Union (meanIoU) score and it's Pixel Accuracy (pixel_accuracy) score.
    Returns a pandas DataFrame with metrics results for each predicted image in the given Dataset.

    parameters:
    -----------
      - model       : Pytorch model.

      - Dataset     : Pytorch Dataset class object (test dataset to predict).

      - device      : Device name

      - output_path : Folder path to save the masks images as files (.png files)
    
    '''
    
    model.eval()
    model.to(device)
    
    meanIoU_scores        = []
    pixel_accuracy_scores = []
    
    for i in tqdm(range(1, Dataset.__len__())):
        model.eval()
        model.to(device)
        
        image = Dataset[i][0].to(device)           
        
        mask  = Dataset[i][1].to(device)
        
        
        with torch.no_grad():
            image = image.unsqueeze(0)
            
            mask = mask.unsqueeze(0)
            
            pred_mask            = model(image)
            meanIoU_score        = meanIoU(pred_mask, mask)
            pixel_accuracy_score = pixel_accuracy(pred_mask, mask)
            
            meanIoU_scores.append(meanIoU_score)
            pixel_accuracy_scores.append(pixel_accuracy_score)
            
            pred_mask = torch.argmax(pred_mask, dim=1)
            pred_mask = pred_mask.cpu().squeeze(0)
            plt.imsave(f'{output_path}{i}.png', pred_mask)
            
            
 
    
    metrics = pd.DataFrame({'Mean IoU'       : meanIoU_scores, 'Pixel Accuracy' : pixel_accuracy_scores})
    metrics['Image'] = metrics.index + 1
    
    return metrics
