import torch
import torch.nn.functional as F
import numpy as np


def horizontal_flip(images, targets, num_htiles=4):
    images = torch.flip(images, [-1])
    targets[:, 1] = (num_htiles-1) - targets[:, 1]
    return images, targets
