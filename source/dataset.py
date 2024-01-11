import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
 
 
class CustomDataset(Dataset):
 
    def __init__(self, path):
        '''
        A class for your custom dataset.
        In the init you should do all the preprocessing and data loading
        that should be done only once, such as scaling, one-hot encoding, etc.
        Feel free to create auxiliary methods for this class.
 
        Args:
            Include here the arguments you need to initialize your dataset.
        '''

        df = pd.read_csv(path)
 
        # The num_data attribute is required for the __len__ method.
        # It is the total number of samples in your dataset.
        self.num_data = df.shape[0]
 
        self.X = df.drop('label', axis=1).values.reshape(-1, 28, 28)
        self.X = self.X / 255.0
        self.Y = df['label'].values
        self.Y = OneHotEncoder().fit_transform(self.Y.reshape(-1, 1)).toarray()
   
    def __len__(self):
        return self.num_data
   
    def __getitem__(self, idx):
        '''
        Must-have method for pytorch dataloader.
        Gets the data and the labels for a given index.
        You should load the data based on the idx and pair it with its labels.
        Here is where you should perform online data augmentation.
 
        Args:
            idx (int): index of the data. You don't need to worry about it.
 
        Returns:
            data (_type_): describe the data type.
            labels (_type_): describe the labels type.
        '''
 
        data = self.X[idx]
        labels = self.Y[idx]
       
        return data, labels