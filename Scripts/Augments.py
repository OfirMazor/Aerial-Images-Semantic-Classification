import albumentations as A
from torchvision import transforms as T
import cv2

import Configs




train_augmentor = A.Compose([  
                             A.Resize(height        = Configs.resize_height,
                                      width         =  Configs.resize_width,
                                      interpolation = cv2.INTER_NEAREST),
    
                             A.HorizontalFlip(),
    
                             A.VerticalFlip(),
    
                             A.GridDistortion(p = 0.2),
    
                             A.RandomBrightnessContrast(brightness_limit = (0,0.5),
                                                        contrast_limit   = (0,0.5)),
                             A.GaussNoise()
                            ])



valid_augmentor = A.Compose([
                             A.Resize(height        = Configs.resize_height,
                                      width         = Configs.resize_width,
                                      interpolation = cv2.INTER_NEAREST),
    
                             A.HorizontalFlip(),
    
                             A.GridDistortion(p = 0.2)
                            ])




test_augmentor = A.Compose([
                            A.Resize(height        = Configs.resize_height,
                                     width         = Configs.resize_width,
                                     interpolation = cv2.INTER_NEAREST)
                           ])




normalizer_transform = T.Compose([
                                 T.ToTensor(),
                                 
                                 T.Normalize(Configs.normalize_mean,
                                             Configs.normalize_std)
                                 ])