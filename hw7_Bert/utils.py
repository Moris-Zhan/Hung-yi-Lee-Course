import imp
import numpy as np
import random
import torch
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

# Fix random seed for reproducibility
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_device(fp16_training):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Change "fp16_training" to True to support automatic mixed precision training (fp16)	
    if fp16_training:
        # !pip install accelerate==0.2.0
        from accelerate import Accelerator
        accelerator = Accelerator(fp16=True)
        device = accelerator.device
        # Documentation for the toolkit:  https://huggingface.co/docs/accelerate/
        return accelerator, device 
    else: return None, device

def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]    