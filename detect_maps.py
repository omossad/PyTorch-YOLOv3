from __future__ import division

from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


def get_tile(t_x,t_y):
    W = 1920.0
    H = 1080.0
    num_tiles = 8
    w_tile = W/num_tiles
    h_tile = H/num_tiles
    x1 = t_x * w_tile
    y1 = t_y * h_tile
    return x1, y1, w_tile, h_tile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="/home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/ha_9/", help="path to dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--out_folder", type=str, default="/home/omossad/scratch/Gaming-Dataset/maps/fifa/ha_9/", help="path to dataset")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )


    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    bbox_colors = random.sample(colors, 4)
    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path) in enumerate(zip(imgs)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        label_file = path.replace('selected_frames','frame_labels')
        label_file = label_file.replace('.jpg','.txt')
        #print(label_file)
        f = open(label_file, "r")
        f_line = f.readline()
        x = min(int(f_line.split()[0]), 8)
        #print(x)
        y = min(int(f_line.split()[1]), 8)
        #print(y)

        color = bbox_colors[3]
        [x1, y1, box_w, box_h] = get_tile(x,y)
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(bbox)

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        filename = opt.out_folder + filename + '.png'
        plt.savefig(filename, bbox_inches="tight", pad_inches=0.0)
        plt.close()
