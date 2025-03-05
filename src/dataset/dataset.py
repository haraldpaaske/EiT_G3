import torch
import pandas as pd
from torch.utils.data import Dataset

class DataFrameDataset(Dataset):
    def __init__(self, json_path):
        """
        Initializes the dataset from a JSON file.

        Args:
        - json_path: Path to the dataset JSON file.
        """
        dataframe = pd.read_json(json_path)
        self.data = torch.tensor(dataframe.iloc[:, :6].values, dtype=torch.float32)  # Features (x, y)
        self.labels = torch.tensor(dataframe.iloc[:, 6:].values, dtype=torch.float32)  # Labels (theta1, theta2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
