import cv2
from PIL import Image
import torch


class DroneDataset(Dataset):

    def __init__(self, image_path : str,
                       mask_path  : str,
                       data,
                       augmentor         = None,
                       normalizer        = None):
        '''
        Initialize the training dataset parameters.
        - image_path : Folder path to the images files. Image should be .jpg file.
        - mask_path  : Folder path to the labeled images files. Labeled image should be .png file.
        - data       : The selected images for dataset (use data split berfore).
        - augmentor  : A composision of auguments.
        - normalizer : A composision of transforms.
        '''
        
        self.image_path = image_path    ;    self.mask_path  = mask_path   ;    self.data = data   ;
        self.augmentor  = augmentor     ;    self.normalizer = normalizer  ;
        
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_path + self.data[idx] + '.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask  = cv2.imread(self.mask_path + self.data[idx] + '.png', cv2.IMREAD_GRAYSCALE)
        
        
        if self.augmentor is not None:
            aug   = self.augmentor(image = image, mask = mask)    
            image = Image.fromarray(aug['image'])
            mask  = aug['mask']
        elif self.augmentor is None:
            image = Image.fromarray(image)
            mask  = mask
            
            
        if self.normalizer is not None:
            image = self.normalizer(image)
            mask  = mask
        
        else:
            pass
        
        mask = torch.from_numpy(mask).long()
        
        
        
    return image, mask

