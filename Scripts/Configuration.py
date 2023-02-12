import os
import pandas as pd
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class Configs:
    '''Set values for parameters & variables'''
    
    ### General ###
    main_folder      = r'/content/Aerial-Images-Semantic-Classification/'
    num_classes      = pd.read_csv(r'/content/drive/MyDrive/Datasets/Drones Images/data/class_dict.csv')['name'].nunique() - 1 #subtruct the "unlabeled" class
                                    #main_folder + 'Data/class_dict.csv'
    images_path      = r'/content/drive/MyDrive/Datasets/Drones Images/data/RGB Images/'                 #main_folder + 'Data/RGB Images/'
    masks_path       = r'/content/drive/MyDrive/Datasets/Drones Images/data/Label Images/label_images/'  #main_folder + 'Data/Label Images/'
    images_count     = len(os.listdir(images_path))
    masks_count      = len(os.listdir(masks_path))
    num_bands        = 3 #RGB images
    device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed             = 100
    prediction_folder= main_folder + 'PredictedMasks/'  
    
    ### Data manegement ###
    test_size        = 0.1
    val_size         = 0.15
    batch_size       = 3
    train_val_count  = images_count - (test_size * images_count)
    val_size_count   = val_size * train_val_count
    
    ### Data Augmentation ###
    resize_height    = 704
    resize_width     = 1056
    normalize_mean   = (0.485, 0.456, 0.406)
    normalize_std    = (0.229, 0.224, 0.225)
    
    ### Model ###
    model_folder     = main_folder + 'ModelFolder/'
    encoder_name     = 'mobilenet_v2'
    encoder_weights  = 'imagenet'
    decoder_channels = [256, 128, 64, 32, 16]
    encoder_depth    = len(decoder_channels)
    activation       = None
    model            = smp.Unet(encoder_name     = encoder_name,
                                encoder_weights  = encoder_weights,
                                classes          = num_classes,
                                encoder_depth    = encoder_depth,
                                decoder_channels = decoder_channels,
                                activation       = activation).to(device)
        
    ### Training ###
    epochs           = 10
    criterion        = nn.CrossEntropyLoss()
    max_lr           = 1e-3
    weight_decay     = 1e-4
    steps_per_epoch  = int((train_val_count - val_size_count) / batch_size)
    
    optimizer        = torch.optim.AdamW(params       = model.parameters(),
                                         lr           = max_lr,
                                         weight_decay = weight_decay)
    
    scheduler        = torch.optim.lr_scheduler.OneCycleLR(optimizer       = optimizer,
                                                           max_lr          = max_lr,
                                                           epochs          = epochs,
                                                           steps_per_epoch = steps_per_epoch)
