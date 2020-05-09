import torch
import torch.nn.functional as F
import numpy as np


def horisontal_flip(images, targets, num_tiles=8):
    images = torch.flip(images, [-1])
    print('IMAGE FLIP')
    print(targets)
    targets[:, 0] = num_tiles - targets[:, 0]
    return images, targets
