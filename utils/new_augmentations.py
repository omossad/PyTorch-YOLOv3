import torch
import torch.nn.functional as F
import numpy as np


def horizontal_flip(images, targets, num_tiles=8):
    images = torch.flip(images, [-1])
    print('IMAGE FLIP')
    print(targets)
    targets[:, 1] = (num_tiles-1) - targets[:, 1]
    print(targets)
    return images, targets
