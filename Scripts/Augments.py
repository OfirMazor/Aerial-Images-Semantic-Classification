import albumentations as A
from torchvision import transforms as T
import cv2
import Configs



def train_augmentor():
  train_augmentor = A.Compose([  
                               A.Resize(height        = Configs.resize_height,
                                        width         = Configs.resize_width,
                                        interpolation = cv2.INTER_NEAREST),

                               A.HorizontalFlip(),

                               A.VerticalFlip(),

                               A.GridDistortion(p = 0.2),

                               A.RandomBrightnessContrast(brightness_limit = (0,0.5),
                                                          contrast_limit   = (0,0.5)),
                               A.GaussNoise()
                              ])
 return train_augmentor








def valid_augmentor():
  valid_augmentor = A.Compose([
                               A.Resize(height        = Configs.resize_height,
                                        width         = Configs.resize_width,
                                        interpolation = cv2.INTER_NEAREST),

                               A.HorizontalFlip(),

                               A.GridDistortion(p = 0.2)
                              ])
return valid_augmentor







def test_augmentor():
  test_augmentor = A.Compose([ 
                              A.Resize(height        = Configs.resize_height,
                                       width         = Configs.resize_width,
                                       interpolation = cv2.INTER_NEAREST)
                             ])
return test_augmentor







def normalizer_transform():
  normalizer_transform = T.Compose([
                                   T.ToTensor(),

                                   T.Normalize(Configs.normalize_mean,
                                               Configs.normalize_std)
                                   ])
return normalizer_transform

