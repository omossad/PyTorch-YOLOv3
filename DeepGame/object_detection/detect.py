from __future__ import division

from models import *
from yolo_utils import *
from datasets import *

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
    parser.add_argument("--image_folder", type=str, default="C:\\Users\\omossad\\Desktop\\dataset\\model_data\\selected_frames\\pa_8\\", help="path to dataset")
    #parser.add_argument("--image_folder", type=str, default="D:\\Encoding\\encoding_files\\fifa\\frames\\gzpt\\", help="path to dataset")
    #parser.add_argument("--image_folder", type=str, default="D:\\Encoding\\encoding_files\\nhl\\frames\\gzpt\\", help="path to dataset")
    #parser.add_argument("--image_folder", type=str, default="D:\\Encoding\\encoding_files\\nba\\frames\\gzpt\\", help="path to dataset")

    parser.add_argument("--model_def", type=str, default="base_model.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="fifa.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.05, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.1, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--out_folder", type=str, default="C:\\Users\\omossad\\Desktop\\dataset\\model_data\\objects\\pa_8\\", help="path to output_folder")
    #parser.add_argument("--out_folder", type=str, default="C:\\Users\\omossad\\Desktop\\recorded_samples\\fifa\\model_data\\objects\\", help="path to output_folder")
    #parser.add_argument("--out_folder", type=str, default="D:\\Encoding\\encoding_files\\fifa\\model\\objects\\", help="path to output_folder")
    #parser.add_argument("--out_folder", type=str, default="D:\\Encoding\\encoding_files\\nhl\\model\\objects\\", help="path to output_folder")
    #parser.add_argument("--out_folder", type=str, default="D:\\Encoding\\encoding_files\\nba\\model\\objects\\", help="path to output_folder")

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
        imgs.extend(img_paths)
        img_detections.extend(detections)
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))
        img_name = path.split("\\")[-1].split(".")[0]
        img_name = opt.out_folder + img_name + '.pt'
        #print(img_name)
        # Create plot
        #img = np.array(Image.open(path))
        #plt.figure()
        #fig, ax = plt.subplots(1)
        #ax.imshow(img)
        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, (1080,1920))
            #detections = rescale_boxes(detections, opt.img_size, (1440,2560))
            #detections = rescale_boxes(detections, opt.img_size, (720,1280))
            #unique_labels = detections[:, -1].cpu().unique()
            #n_cls_preds = len(unique_labels)
            #bbox_colors = random.sample(colors, n_cls_preds)
            #print(detections)
        torch.save(detections, img_name)
