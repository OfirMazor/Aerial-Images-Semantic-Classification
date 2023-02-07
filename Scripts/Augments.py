import albumentations as A
from torchvision import transforms as T
import cv2
from Configuration import Configs



def train_augmentor():
  '''
  A composition of image auguments for training dataset.
  '''
  augmentor = A.Compose([  
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
  
  return augmentor








def valid_augmentor():
  '''
  A composition of image auguments for validation dataset.
  '''
  augmentor = A.Compose([
                         A.Resize(height        = Configs.resize_height,
                                  width         = Configs.resize_width,
                                  interpolation = cv2.INTER_NEAREST),

                         A.HorizontalFlip(),

                         A.GridDistortion(p = 0.2)
                         ])
  return augmentor







def test_augmentor():
  '''
  A composition of image auguments for testing dataset.
  '''
  augmentor = A.Compose([ 
                         A.Resize(height        = Configs.resize_height,
                                  width         = Configs.resize_width,
                                  interpolation = cv2.INTER_NEAREST)
                         ])
  
  return augmentor







def normalizer_transform():
  '''
  A transformations to normalize image pixels values .
  '''
  normalizer = T.Compose([
                          T.ToTensor(),

                          T.Normalize(Configs.normalize_mean,
                                      Configs.normalize_std)
                          ])
  
  return normalizer

