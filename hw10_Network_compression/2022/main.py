
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset # "ConcatDataset" and "Subset" are possibly useful
from torchvision.datasets import DatasetFolder, VisionDataset
from torchsummary import summary
from tqdm.auto import tqdm
import random
from utils import seed
import inspect


device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = {
    'dataset_root': './food11-hw13',
    'save_dir': './outputs',
    'exp_name': "simple_baseline",
    'batch_size': 64,
    'lr': 2e-4,
    'seed': 20220013,
    'loss_fn_type': 'KD', # simple baseline: CE, medium baseline: KD. See the Knowledge_Distillation part for more information.
    'weight_decay': 2e-5,
    'grad_norm_max': 10,
    'n_epochs': 500, # train more steps to pass the medium baseline.
    'patience': 100,
}

if __name__ == '__main__':
    myseed = cfg['seed']  # set a random seed for reproducibility

    save_path = os.path.join(cfg['save_dir'], cfg['exp_name']) # create saving directory
    os.makedirs(save_path, exist_ok=True)

    # define simple logging functionality
    log_fw = open(f"{save_path}/log.txt", 'w') # open log file to save log outputs
    def log(text):     # define a logging function to trace the training process
        print(text)
        log_fw.write(str(text)+'\n')
        log_fw.flush()

    log(cfg)  # log your configs to the log file