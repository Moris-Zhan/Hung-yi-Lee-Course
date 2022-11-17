import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

import torchvision.models as models
# from baseline import StudentNet
from strong import StudentNet
from utils import load_teacherNet
from dataset import semi_dataloader, get_loader
from script import train

do_semi = True

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"


# Teacher Model Setting
if __name__ == "__main__":
    # Load teacherNet    
    teacher_net = load_teacherNet()
    
    # Initialize a model, and put it on the device specified.
    student_net = StudentNet()
    student_net = student_net.to(device)
    teacher_net = teacher_net.to(device)

    train_loader, valid_loader, test_loader, train_set = get_loader()
    # Whether to do pseudo label.
    if do_semi:
        train_loader = semi_dataloader(teacher_net, train_set)

    # The number of training epochs.
    n_epochs = 80
    train(teacher_net, student_net, train_loader, valid_loader)