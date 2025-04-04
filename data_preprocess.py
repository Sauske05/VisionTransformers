import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
class TinyImageDataset(Dataset):
    def __init__(self, df:pd.DataFrame,image_col:str = 'image_array',  label_col:str = 'label'):
        image_array_stacked:np.array = np.stack(df[image_col].tolist(), axis=0)
        label_col_stacked:np.array = np.stack(df[label_col].tolist(), axis=0)
        self.image_array:torch.tensor = torch.tensor(image_array_stacked, dtype=torch.float32)
        self.image_array = self.image_array / 255.0
        
        # Apply mean and std normalization (e.g., ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3)
        self.image_array = (self.image_array - mean) / std
        self.image_labels = torch.tensor(label_col_stacked, dtype=torch.float32)

    def __len__(self):
        return len(self.image_labels)
    
    def __getitem__(self, index):
        image = self.image_array[index]
        label = self.image_labels[index]
        return image, label
    

def load_data():
    splits = {'train': 'data/train-00000-of-00001-1359597a978bc4fa.parquet', 'valid': 'data/valid-00000-of-00001-70d52db3c749a935.parquet'}
    train_df:pd.DataFrame = pd.read_parquet("hf://datasets/zh-plus/tiny-imagenet/" + splits["train"])
    val_df:pd.DataFrame = pd.read_parquet("hf://datasets/zh-plus/tiny-imagenet/" + splits["valid"])
    return train_df, val_df

from PIL import Image
import io
def apply_fn(image_dict):
    byte_data = image_dict['bytes']
    img = Image.open(io.BytesIO(byte_data))

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # # Print the shape of the array (e.g., height, width, channels)
    # print(img_array.shape)
    return img_array

def check_image_array_shapes(df)-> pd.DataFrame:
    count:int = 0
    rows_to_drop = []
    expected_shape:tuple = (64, 64, 3)
    for idx, img_array in enumerate(df['image_array']):
        if img_array.shape != expected_shape:
            count +=1
            print(f'Assertion failed at index {idx}: Expected shape {expected_shape}, but got {img_array.shape}')
            rows_to_drop.append(idx)
            #raise Exception(f"Assertion failed at index {idx}: Expected shape {expected_shape}, but got {img_array.shape}")
    
    if rows_to_drop:
        df.drop(index=rows_to_drop, inplace=True)
        print(f"Dropped {len(rows_to_drop)} rows with mismatched shapes.")
    return df
    #return count

def clean_df(df):
    df['image_array'] = df['image'].apply(lambda x: apply_fn(x))
    try:
        df = check_image_array_shapes(df)
        df = df.reset_index()
        return df
        #print(result)
    except Exception as e:
        print(e)
    
