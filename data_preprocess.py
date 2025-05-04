import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import io
import logging

from torchvision.transforms import v2
logger = logging.getLogger(__name__)
logging.basicConfig(filename='distil.log', level=logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('distil.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)
# class TinyImageDataset(Dataset):
#     def __init__(self, df:pd.DataFrame,image_col:str = 'image_array',  label_col:str = 'label'):
#         print('Start')
#         image_array_stacked:np.array = np.stack(df[image_col].tolist(), axis=0)
#         label_col_stacked:np.array = np.stack(df[label_col].tolist(), axis=0)
#         #Here 5 is the number of augemented image we want per image.
#         #augmented_images = []
#         #print('Reaches here')
#         # for i in range(3):
#         #     augmented_images.append(self.transform(image_array_stacked))

#         self.image_array = np.stack(image_array_stacked, axis=  0)
#         print(f'Image array shape before augementation: {self.image_array.shape}')
#         self.image_array = self.image_array.reshape(-1,64,64,3)
#         print(f'Image array shape after augementation: {self.image_array.shape}')
#         self.image_array:torch.tensor = torch.tensor(self.image_array, dtype=torch.float32)

#         self.image_array = self.image_array / 255.0
#         # Apply mean and std normalization (e.g., ImageNet stats)
#         mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3)
#         std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3)
#         self.image_array = (self.image_array - mean) / std
#         self.image_labels = torch.tensor(label_col_stacked, dtype=torch.long)
#         #self.image_labels = self.image_labels.reshape(-1,1).expand(-1,1).reshape(-1,1).squeeze() #For data augmentation
#         print(f'The shape of image_labels: {self.image_labels.shape}')

#     def __len__(self):
#         return len(self.image_labels)
    
#     def transform(self, x):
#         transform_op = v2.Compose([
#             v2.RandomHorizontalFlip(p = 0.5),
#             v2.RandomVerticalFlip(p = 0.5), 
#             v2.RandomRotation(degrees=30),
#             v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),    
#             ])
#         return transform_op(x)


#     def __getitem__(self, index):
#         image = self.image_array[index]
#         label = self.image_labels[index]
#         return image, label

class TinyImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_col: str = 'image_array', label_col: str = 'label', data_type = 'train'):
        self.data_type = data_type
        self.images = df[image_col].tolist() 
        self.labels = df[label_col].values 
        self.transform_op = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=30),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
        self.num_augmentations = 5 

    def __len__(self):
        if self.data_type == 'train':
            return len(self.images) * self.num_augmentations
        else:
            return len(self.images)

    def __getitem__(self, index):
        if self.data_type == 'train':
            orig_idx = index // self.num_augmentations
            image = self.images[orig_idx]
            label = self.labels[orig_idx]
        else:
            image = self.images[index]
            label = self.labels[index]
       
        image = torch.tensor(image, dtype=torch.float32)
        if image.shape == (64, 64, 3):
            image = image.permute(2, 0, 1)
        elif image.shape == (3, 64, 64):
            pass 
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        image = image / 255.0

        if self.data_type == 'train':
            image = self.transform_op(image)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        return image, torch.tensor(label, dtype=torch.long)
def load_data():
    splits = {'train': 'data/train-00000-of-00001-1359597a978bc4fa.parquet', 'valid': 'data/valid-00000-of-00001-70d52db3c749a935.parquet'}
    train_df:pd.DataFrame = pd.read_parquet("hf://datasets/zh-plus/tiny-imagenet/" + splits["train"])
    val_df:pd.DataFrame = pd.read_parquet("hf://datasets/zh-plus/tiny-imagenet/" + splits["valid"])
    return train_df, val_df


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
            #logger.info(f'Assertion failed at index {idx}: Expected shape {expected_shape}, but got {img_array.shape}')
            rows_to_drop.append(idx)
            #raise Exception(f"Assertion failed at index {idx}: Expected shape {expected_shape}, but got {img_array.shape}")
    
    if rows_to_drop:
        df.drop(index=rows_to_drop, inplace=True)
        logger.info(f"Dropped {len(rows_to_drop)} rows with mismatched shapes.")
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
    
def load_dataloader():
    logger.info('Downlaoding data!')
    train_df, val_df = load_data()
    logger.info('Downlaoding data finished!')
    logger.info('Starting data preprocess!')
    clean_df(train_df)
    clean_df(val_df)
    logger.info('Data preprocess finished!')
    logger.info('Converting to dataset....')
    train_dataset = TinyImageDataset(train_df, data_type = 'train')
    val_dataset = TinyImageDataset(val_df, data_type = 'test')
    logger.info('Loaded to dataset')
    logger.info('Loading in Dataloaders...')

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, drop_last=True)
    logger.info('Dataloaders loaded')
    return train_dataloader, val_dataloader
