import torch

# Import necessary packages.
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageOps
import random
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
import torchvision.models as models
from torchsummary import summary

GPU_name = torch.cuda.get_device_name()
print("Your GPU is {}!".format(GPU_name))


model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 11)

# summary(model, (3, 224, 224), device='cpu')
print(model)