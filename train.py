from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, help="if specified starts from checkpoint model")
parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
opt = parser.parse_args()
print(opt)

logger = Logger("logs")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
train_path = data_config["train"]

# Initiate model
model = Darknet(opt.model_config_path).to(device)
model.apply(weights_init_normal)

# If specified we start from checkpoint
if opt.weights_path:
    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path)
    else:
        model.load_state_dict(torch.load(opt.weights_path))

model.train()

# Get dataloader
dataset = ListDataset(train_path)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, pin_memory=True
)

optimizer = torch.optim.Adam(model.parameters())

for epoch in range(opt.epochs):
    start_time = time.time()
    for batch_i, (_, imgs, targets) in enumerate(dataloader):

        batches_done = len(dataloader) * epoch + batch_i

        imgs = Variable(imgs.to(device))
        targets = Variable(targets.to(device), requires_grad=False)

        loss, outputs = model(imgs, targets)
        loss.backward()

        if batch_i % opt.gradient_accumulations:
            # Accumulates gradient before each step
            optimizer.step()
            optimizer.zero_grad()

        # ----------------
        #   Log progress
        # ----------------

        print("\n---- [Epoch %d/%d, Batch %d/%d] ----" % (epoch, opt.epochs, batch_i, len(dataloader)))

        # Log metrics at each YOLO layer
        for i in range(3):
            print(
                "[%s] [loss %f, x %f, y %f, w %f, h %f, conf %f, cls %f, cls_acc: %.2f%%, recall: %.5f, precision: %.5f]"
                % (
                    "Total" if i + 1 == 4 else "YOLO Layer %d" % (i + 1),
                    model.metrics[i]["loss"],
                    model.metrics[i]["x"],
                    model.metrics[i]["y"],
                    model.metrics[i]["w"],
                    model.metrics[i]["h"],
                    model.metrics[i]["conf"],
                    model.metrics[i]["cls"],
                    100 * model.metrics[i]["cls_acc"],
                    model.metrics[i]["recall"],
                    model.metrics[i]["precision"],
                )
            )

            # Tensorboard logging
            for name, metric in model.metrics[i].items():
                logger.scalar_summary(f"{name}_{i+1}", metric, batches_done)

        global_metrics = [("Total Loss", loss.item())]

        # Compute mAP every tenth iteration
        if opt.compute_map and batches_done % 10 == 0:

            # ---------------
            #   Compute mAP
            # ---------------

            # Get NMS output
            predictions = non_max_suppression(outputs, 0.5, 0.5)
            # Convert target coordinates to x1y1x2y2
            targets[:, :, 1:] = xywh2xyxy(targets[:, :, 1:])
            # Rescale to image dimension
            targets[:, :, 1:] *= opt.img_size
            # Get batch statistics used to compute metrics
            statistics = get_batch_statistics(predictions, targets, iou_threshold=0.5)
            true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*statistics))]
            # Compute metrics
            precision, recall, AP, f1, ap_class = ap_per_class(
                true_positives, pred_scores, pred_labels, list(range(int(data_config["classes"])))
            )
            global_metrics += [
                ("F1", f1.mean()),
                ("Recall", recall.mean()),
                ("Precision", precision.mean()),
                ("mAP", AP.mean()),
            ]

        # Log global metrics to Tensorboard
        for metric_name, metric in global_metrics:
            logger.scalar_summary(metric_name, metric, batches_done)

        # Print mAP and other global metrics
        print(" | ".join([f"{metric_name} {metric:f}" for metric_name, metric in global_metrics]))

        # Determine approximate time left for epoch
        epoch_batches_left = len(dataloader) - (batch_i + 1)
        time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
        print(f"---- ETA {time_left}")

        model.seen += imgs.size(0)

        torch.cuda.empty_cache()

    if epoch % opt.checkpoint_interval == 0:
        torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)