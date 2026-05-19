import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class GTSRBDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        """
        Args:
            data_dir (string): Directory with all the images (e.g., 'data/raw').
            csv_file (string): Path to the csv file with annotations (e.g., 'data/raw/Train.csv' or DataFrame).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        if isinstance(csv_file, str):
            self.data_frame = pd.read_csv(csv_file)
        else:
            self.data_frame = csv_file
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.data_dir, self.data_frame.iloc[idx]['Path'])
        image = Image.open(img_name).convert('RGB')
        
        # The class ID is typically in the 'ClassId' column
        class_id = int(self.data_frame.iloc[idx]['ClassId'])

        if self.transform:
            image = self.transform(image)

        return image, class_id

def get_dataloaders(data_dir, batch_size=64, val_split=0.2):
    """
    Returns train, val, and test dataloaders.
    """
    train_csv_path = os.path.join(data_dir, 'Train.csv')
    test_csv_path = os.path.join(data_dir, 'Test.csv')
    
    # Read train data to split into train and val
    full_train_df = pd.read_csv(train_csv_path)
    train_df, val_df = train_test_split(full_train_df, test_size=val_split, random_state=42, stratify=full_train_df['ClassId'])
    
    # Define transforms
    # GTSRB reference typically resizes to 32x32.
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629]) # GTSRB approx mean/std
    ])
    
    # Create datasets
    train_dataset = GTSRBDataset(data_dir=data_dir, csv_file=train_df, transform=transform)
    val_dataset = GTSRBDataset(data_dir=data_dir, csv_file=val_df, transform=transform)
    test_dataset = GTSRBDataset(data_dir=data_dir, csv_file=test_csv_path, transform=transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader
