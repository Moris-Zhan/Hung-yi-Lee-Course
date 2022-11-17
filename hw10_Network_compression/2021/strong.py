import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

# strong
def dwpw_conv(in_chs, out_chs, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_chs, in_chs, kernel_size, stride, padding, groups=in_chs),
        nn.BatchNorm2d(in_chs),
        nn.ReLU(),
        nn.Conv2d(in_chs, out_chs, 1),
        nn.MaxPool2d(2),
    )

class StudentNet(nn.Module):
    def __init__(self):
      super(StudentNet, self).__init__()

      # ---------- TODO ----------
      # Modify your model architecture

      self.cnn =  nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 64, 3, 1, 0),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ),
            dwpw_conv(64, 128, 3, 1, 0),
            dwpw_conv(128, 256, 3, 1, 0),
           
            nn.Sequential(
                nn.Conv2d(256, 256, 3, 1, 0, groups=256),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 150, 1),
            ),
            # Here we adopt Global Average Pooling for various input size.
            nn.AdaptiveAvgPool2d((1, 1)),
      )
      self.fc = nn.Sequential(
        nn.Linear(150, 64),
        nn.ReLU(),
        nn.Linear(64, 11),
      )
      
    def forward(self, x):
      out = self.cnn(x)
      out = out.view(out.size()[0], -1)
      return self.fc(out)