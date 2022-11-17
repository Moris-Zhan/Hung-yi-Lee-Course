import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_fn_kd(outputs, labels, teacher_outputs, alpha=0.5, temperature=20):
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha) 
    # ---------- TODO ----------
    # Complete soft loss in knowledge distillation
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    soft_loss = alpha * pow(temperature, 2) * kl_loss(F.log_softmax(outputs/temperature, dim=1), \
                                                 F.softmax(teacher_outputs/temperature, dim=1))
    return hard_loss + soft_loss