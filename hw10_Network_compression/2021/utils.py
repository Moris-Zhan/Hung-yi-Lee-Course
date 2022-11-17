import torch
import torchvision.models as models
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_teacherNet():
    # Load teacherNet
    teacher_net = models.resnet18(pretrained=False)
    num_ftrs = teacher_net.fc.in_features
    teacher_net.fc = nn.Linear(num_ftrs, 11)

    teacher_net.load_state_dict(torch.load('./teacher_model.ckpt'))
    teacher_net.eval()
    return teacher_net