import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

# baseline
class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()

        # ---------- TODO ----------
        # Modify your model architecture

        self.cnn = nn.Sequential(
        nn.Conv2d(3, 32, 3), 
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3),  
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),     

        nn.Conv2d(32, 64, 3), 
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),     

        nn.Conv2d(64, 100, 3), 
        nn.BatchNorm2d(100),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),

        # Here we adopt Global Average Pooling for various input size.
        nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
        nn.Linear(100, 11),
        )
      
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
