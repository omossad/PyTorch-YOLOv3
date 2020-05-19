from __future__ import division

from new_model import *
#from utils.logger import *
from utils.new_utils import *
from utils.new_datasets import *
from utils.parse_config import *
from new_test import evaluate

from terminaltables import AsciiTable

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
    #parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--epochs", type=int, default=40, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=False, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)

    #logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights), strict=False)
        else:
            model.load_darknet_weights(opt.pretrained_weights)
    for name, param in model.named_parameters():
        #print(name)
        param.requires_grad = False

        #print('ROI')
    for name, param in model.roi_layer[0].named_parameters():
        param.requires_grad = True
        #if param.requires_grad:
        #    print('PARAM')
        #    print(name)
        #    print(param)

    # Get dataloader
    #model = nn.DataParallel(model)
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    #optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    metrics = [
    #    "grid_size",
    #    "loss",
    #    "x",
    #    "y",
    #    "w",
    #    "h",
    #    "conf",
    #    "cls",
    #    "cls_acc",
    #    "recall50",
    #    "recall75",
    #    "precision",
    #    "conf_obj",
    #    "conf_noobj",
        "loss_x",
        "loss_y",
        "loss",
        "acc_x",
        "acc_y",
        "acc",
    ]
    #model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
    for epoch in range(opt.epochs):
        #print(model)
        model.train()
        start_time = time.time()
        num_batches = len(dataloader)
        tot_loss_x = 0
        tot_loss_y = 0
        tot_loss = 0
        tot_acc_x = 0
        tot_acc_y = 0
        tot_acc = 0
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            #print('TARGET FILE')
            #print(targets)
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            #print('TARGET VAR')
            #print(targets)
            #loss, outputs = model(imgs, targets)
            loss, outputs_x, outputs_y = model(imgs, targets)
            #lossX, lossY, outputs_x, outputs_y = model(imgs, targets)
            #for name, param in model.roi_layer[0].named_parameters():
            #    param.requires_grad = True
            #for name, param in model.roi_layer[0].fc_net_y.named_parameters():
            #    print(name)
            #    param.requires_grad = False
            #for name, param in model.roi_layer[0].fc_net_x.named_parameters():
            #    param.requires_grad = True
            #lossX.backward()

            #if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
            #    optimizer.step()
            #    optimizer.zero_grad()

            #for name, param in model.roi_layer[0].named_parameters():
            #    param.requires_grad = True
            #for name, param in model.roi_layer[0].fc_net_x.named_parameters():
            #    param.requires_grad = False
            #lossY.backward()
            #loss = lossX + lossY
            loss.backward()
            #lossY.backward()
            #loss = lossX + lossY
            #loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            #metric_table = [["Metrics", *[f"ROI Layer {i}" for i in range(1)]]]
            #metric_table = [["Metrics", *["ROI Layer"]]]
            # Log metrics at each YOLO layer
            #print(model.Darknet)
            roi = model.roi_layer[0]
            for i, metric in enumerate(metrics):
                #print(metric)
                formats = {m: "%.6f" for m in metrics}
                #formats["grid_size"] = "%2d"
                #formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % roi.metrics.get(metric, 0)]
                #row_metrics = [formats[metric] % roi.metrics.get(metric, 0) for roi in model.roi_layer]
                #metric_table += [[metric, *row_metrics]]
                #print('METRIC')
                #print(metric)
                #print(*row_metrics)
                if i == 0:
                    tot_loss_x += float(*row_metrics)
                elif i == 1:
                    tot_loss_y += float(*row_metrics)
                elif i == 2:
                    tot_loss += float(*row_metrics)
                elif i == 3:
                    tot_acc_x += float(*row_metrics)
                elif i == 4:
                    tot_acc_y += float(*row_metrics)
                elif i == 5:
                    tot_acc += float(*row_metrics)

                # Tensorboard logging
                #tensorboard_log = []
                #for j, roi in enumerate(model.roi_layer):
            #for name, metric in roi.metrics.items():
                        #if name != "grid_size":
                #tensorboard_log += [(f"{name} ", metric)]
            #tensorboard_log += [("loss", loss.item())]
            #logger.list_of_scalars_summary(tensorboard_log, batches_done)

            #log_str += AsciiTable(metric_table).table
            #log_str += f"\nTotal loss {loss.item()}"
            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            #print(log_str)

            model.seen += imgs.size(0)
        print('Training Statistics for epoch ' + str(epoch) + '/' + str(opt.epochs))
        print('losses: '     + str(tot_loss_x/num_batches) + ', ' + str(tot_loss_y/num_batches) + ', ' + str(tot_loss/num_batches))
        print('accuracies: ' + str(tot_acc_x/num_batches)  + ', ' + str(tot_acc_y/num_batches)  + ', ' + str(tot_acc/num_batches))

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            tot_accuracy, x_acc, y_acc = evaluate(
                model,
                path=valid_path,
                conf_thres=0.3,
                nms_thres=0.2,
                img_size=opt.img_size,
                batch_size=8,
            )

            #precision, recall, AP, f1, ap_class = evaluate(
            #    model,
            #    path=valid_path,
            #    iou_thres=0.5,
            #    conf_thres=0.5,
            #    nms_thres=0.5,
            #    img_size=opt.img_size,
            #    batch_size=8,
            #)
            evaluation_metrics = [
                ("total_Accuracy", tot_accuracy.mean()),
                ("grid_x Accuracy", x_acc.mean()),
                ("grid_y Accuracy", y_acc.mean()),
            ]
            #print('accuracies: ' + str(x_acc.mean()) + ', ' + str(y_acc.mean()) + ', ' + str(tot_accuracy.mean()) + '\n')
            print('accuracies: ' + str(x_acc + ', ' + str(y_acc) + ', ' + str(tot_accuracy) + '\n')
            #logger.list_of_scalars_summary(evaluation_metrics, epoch)

            #evaluation_metrics = [
            #    ("val_precision", precision.mean()),
            #    ("val_recall", recall.mean()),
            #    ("val_mAP", AP.mean()),
            #    ("val_f1", f1.mean()),
            #]
            #logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            #ap_table = [["Index", "Class name", "AP"]]
            #for i, c in enumerate(ap_class):
            #    ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            #print(AsciiTable(ap_table).table)
            #print(f"---- mAP {AP.mean()}")

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
