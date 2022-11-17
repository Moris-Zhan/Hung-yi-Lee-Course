import numpy as np
import torch
import random

def seed(myseed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    random.seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)