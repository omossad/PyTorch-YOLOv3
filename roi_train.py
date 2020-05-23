from __future__ import division

from roi_model import *
from utils.roi_utils import *
from utils.roi_datasets import *
from utils.parse_config import *
from roi_test import evaluate

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--base_model_def", type=str, default="config/base_model.cfg", help="path to base model definition file")
    parser.add_argument("--fine_model_def", type=str, default="config/fine_model.cfg", help="path to fine model definition file")
    parser.add_argument("--data_config", type=str, default="config/roi.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, default="checkpoints/tiny_yolo.pth", help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--multiscale_training", default=False, help="allow for multi-scale training")
    parser.add_argument("--conf_thres", type=float, default=0.3, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.2, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--htiles", type=int, default=5, help="number of horizontal tiles")
    parser.add_argument("--vtiles", type=int, default=3, help="number of vertical tiles")
    parser.add_argument("--classes", type=int, default=3, help="number of classes")

    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    base_model = Darknet(opt.base_model_def).to(device)
    base_model.apply(weights_init_normal)
    fine_model = ROI(opt.fine_model_def, opt.htiles, opt.vtiles, opt.classes, opt.img_size).to(device)
    print(fine_model.ROI)
    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            base_model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            base_model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(fine_model.parameters())

    metrics = [
        "loss_x",
        "loss_y",
        "loss",
        "acc_x",
        "acc_y",
        "acc",
    ]

    for epoch in range(opt.epochs):
        #print(model)
        base_model.eval()
        fine_model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            yolo_outputs  = base_model(imgs)
            #yolo_outputs = non_max_suppression(yolo_outputs, opt.conf_thres, opt.nms_thres)
            x_inpt, y_inpt = yolo_preprocessing(yolo_outputs, opt.conf_thres, opt.nms_thres, opt.htiles, opt.vtiles, opt.classes, opt.img_size)
            x_inpt = Variable(x_inpt.to(device))
            y_inpt = Variable(y_inpt.to(device))
            loss, output_x, output_y = fine_model(x_inpt, y_inpt, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"ROI Layer {i}" for i in range(len(fine_model.roi_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                #formats["grid_size"] = "%2d"
                #formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % roi.metrics.get(metric, 0) for roi in fine_model.roi_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, roi in enumerate(fine_model.roi_layers):
                    for name, metric in roi.metrics.items():
                        tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                #logger.list_of_scalars_summary(tensorboard_log, batches_done)

            #log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)
'''
        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                base_model,
                fine_model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            #logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            #ap_table = [["Index", "Class name", "AP"]]
            #for i, c in enumerate(ap_class):
            #    ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            #print(AsciiTable(ap_table).table)
            #print(f"---- mAP {AP.mean()}")

        if epoch % opt.checkpoint_interval == 0:
            torch.save(fine_model.state_dict(), f"checkpoints/roi_ckpt_%d.pth" % epoch)
'''
