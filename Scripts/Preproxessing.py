import os
from Configuration import Configs
import pandas as pd
import numpy as np
import cv2


def df_to_RGB_dictionary(df:pd.DataFrame, RGB_columns:list):
  '''
  Convert the classes rgb data to a color map dictionary with classes number and rgb tuple.
  '''
  output = {}
  for index, row in df.iterrows():
        key = tuple(row[col] for col in RGB_columns)
        output[key] = index

  return output



def map_to_mask(input_rgb, out_mask_path, mapping_dict:dict):
  '''
  Convert RGB mask image to single channel mask image using a color map dictionary.
  Saves the new mask image in output folder with the original name of input rgm image.
  '''
  # Read the input RGB mask image
  rgb_mask = cv2.imread(input_rgb)

  # Create template array with the shape of the input RGB mak image
  single_channel_mask = np.zeros((rgb_mask.shape[0],
                                  rgb_mask.shape[1]),
                                  dtype=np.uint8)
  
  # Loop through all pixels in the RGB mask image
  for i in range(rgb_mask.shape[0]):
    for j in range(rgb_mask.shape[1]):
        # Check if the pixel color is in the RGB color mapping table
        if tuple(rgb_mask[i, j, :]) in mapping_dict:
            # Assign the corresponding label to the pixel in the single channel mask image
            single_channel_mask[i, j] = mapping_dict[tuple(rgb_mask[i, j, :])]

  #Save the single channel mask image
  out_mask_name = input_rgb.split('/')[-1]
  out_mask = out_mask_path + f'/{out_mask_name}'
  cv2.imwrite(out_mask, single_channel_mask)




# Preprocessing
class_csv      = Configs.data_folder + 'class_dict.csv'
input_rgb_path = Configs.masks_path
out_mask_path  = Configs.labels_path
class_df = pd.read_csv(class_csv)

mapping_dict = df_to_RGB_dictionary(df = class_df, RGB_columns = [' r', ' g', ' b'])

images_files = []


for root, dirs, files in os.walk(input_rgb_path):
  for file in files:
    if file.endswith('.png'):
      images_files.append(file)


for file in images_files:
  print('Mapping image to mask for', file)
  map_to_mask(input_rgb     = input_rgb_path + file,
              out_mask_path = out_mask_path,
              mapping_dict  = mapping_dict)
  
