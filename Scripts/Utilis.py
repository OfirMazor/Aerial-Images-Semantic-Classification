import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Configs import Configs



def create_df(path:str):
    '''
    
    Returns DataFrame of images (jpg/png) in folder.
    
    parameters:
    - path : the path to folder of images.
    
    '''
    name = []
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                name.append(file.split('.')[0])
    
    df = pd.DataFrame({'id_str': name}, index = np.arange(0, len(name)))
    df['id_int'] = df['id_str'].astype(int)
    
    return df




def plot_random_image(images_df:pd.DataFrame):
    
    id_list = sorted(images_df['id_int'].values.tolist())
    random_id = random.choice(id_list)
    print(f'Showing image ID: {random_id}')
        
    image_id_str = images_df[images_df['id_int'] == random_id]['id_str'].item()
    random_image = Image.open(Configs.images_path + image_id_str + '.jpg')

    mask_id_str = masks_df[masks_df['id_int'] == random_id]['id_str'].item()
    random_mask = Image.open(Configs.masks_path + mask_id_str + '.png')

    
    plt.figure()
    fig, axarr = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(25, 25))

    axarr[0].imshow(random_image)
    axarr[0].set_title(f'Original RGB image {np.asarray(random_image).shape}')

    axarr[1].imshow(random_mask)
    axarr[1].set_title(f'Masked image {np.asarray(random_mask).shape}')

    axarr[2].imshow(random_image)
    axarr[2].imshow(random_mask, alpha=0.6)
    axarr[2].set_title('Overlay image')

    plt.show()

    
    
def plot_history(history:dict):
    
    def plot_loss(history):
        plt.plot(history['valid_loss'], label ='Valid', marker = 'o')
        plt.plot(history['train_loss'], label ='Train', marker = 'o')
        plt.title('Loss per Epoch')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(), plt.grid()
        plt.show()

    def plot_score(history):
        plt.plot(history['train_meanIoU'], label='Train meanIoU', marker='*')
        plt.plot(history['valid_meanIoU'], label='Valid meanIoU', marker='*')
        plt.title('Mean Intersect-over-Union per Epoch')
        plt.ylabel('Mean IoU')
        plt.xlabel('Epoch')
        plt.legend(), plt.grid()
        plt.show()

    def plot_accuracy(history):
        plt.plot(history['train_accuracies'], label='Train Accuracy', marker='*')
        plt.plot(history['valid_accuracies'], label='Valid Accuracy', marker='*')
        plt.title('Pixel Accuracy per Epoch')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(), plt.grid()
        plt.show()
        
    plot_loss(history);
    plot_score(history);
    plot_accuracy(history);



def get_lr(optimizer):
  for group in optimizer.param_groups: 
    return group['lr']



def plot_random_prediction(Dataset, prediction_folder:str, metrics_df : pd.DataFrame = None):
    random_number = np.random.randint(1, Dataset.__len__())
    '''
    Plot a random image, it's label mask and it's prediction mask. 
    Parameters:
    -----------
        - Dataset           : torch.Dataset class object (test set)
        - prediction_folder : Folder path that contains the prediction masks files.
        - metrics_df        : DataFrame with the metrics informations of masks (optional). default is None.
    '''
    
    # Generate random number of image file name
    random_number = np.random.randint(1, Dataset.__len__())
    print(f'Showing observation number: {random_number}')
    
    # Get the random image and mask from torch Dataset
    #              Image                                                           Mask
    random_image = Dataset[random_number][0]               ;        random_mask = Dataset[random_number][1] ;
    random_image = random_image.to('cpu').permute(1,2,0)   ;        random_mask = random_mask.to('cpu')     ;

    # Get the prediction mask metrics values
    if metrics_df is not None:
        meanIoU  = metrics_df[metrics_df['Image'] == random_number]['Mean IoU'].values
        accuracy = metrics_df[metrics_df['Image'] == random_number]['Pixel Accuracy'].values

    else:
        pass

        # Get the prediction mask 
    pred_mask = Image.open(f'{prediction_folder}{random_number}.png')

      # Figure settings
    plt.figure()
    fig, axarr = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(25, 25))

    axarr[0].imshow(random_image)
    axarr[0].set_title(f'Original image {np.asarray(random_image).shape}')

    axarr[1].imshow(random_mask)
    axarr[1].set_title(f'Masked image {np.asarray(random_mask).shape}')

    axarr[2].imshow(pred_mask)
    if metrics_df is not None:
        axarr[2].set_title(f'Predicted mask image {np.asarray(random_mask).shape} \n with pixel accuracy:{Accuracy} & mean IoU: {meanIoU}')
    else:
        axarr[2].set_title(f'Predicted mask image {np.asarray(random_mask).shape}')


    plt.show();
