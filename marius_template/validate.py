from modules import transform, DataFrameDataset
import torch
import torch.nn as nn
from model import kinematic_NN
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

model = kinematic_NN()
model.eval()
with torch.nograd:
    