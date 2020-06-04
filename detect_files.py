from __future__ import division

from models import *
from utils.utils import *
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/base_model.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/tiny_yolo.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/annotations/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.0, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.0, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    #cmap = plt.get_cmap("tab20b")
    #colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    txy_labels_path = '/home/omossad/scratch/temp/roi/labels4x4/'
    t_labels_path = '/home/omossad/scratch/temp/roi/labels/'
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        img = np.array(Image.open(path))
        print("(%d) Image: '%s'" % (img_i, path))
        print(img_i)
        print(path)

        # Draw bounding boxes and labels of detections
        filename = path.split("/")[-1].split(".")[0]
        print(filename)
        t_file = open(t_labels_path + filename + '.txt', "r").read()
        #print(t_file)
        t_label = int(t_file)
        print(t_label)
        txy_file = open(txy_labels_path + filename + '.txt', "r").read()
        #print(txy_file)
        tx_label = int(txy_file.split()[0])
        ty_label = int(txy_file.split()[1])
        print(tx_label)
        print(ty_label)
        #f = open(f"output/{filename}.txt", "a")
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            W = img.shape[1]
            H = img.shape[0]
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            #bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                #if int(cls_pred) == 0:
                #: or int(cls_pred) == 32):
                #print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                x_c = (x1+x2)/2
                y_c = (y1+y2)/2
                box_w = x2 - x1
                box_h = y2 - y1

                to_write = str(int(cls_pred)) + " "
                #to_write = str(int(cls_pred)) + " "
                to_write = to_write + str(x_c.item()/W)    + " " + str(y_c.item()/H)    + " "
                to_write = to_write + str(box_w.item()/W) + " " + str(box_h.item()/H) + "\n"
                #print(to_write)
                    #f.write(to_write)

        #f.close()
