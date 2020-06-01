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
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=5, help="size of each image batch")
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
    parser.add_argument("--conf_thres", type=float, default=0.1, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.1, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--htiles", type=int, default=4, help="number of horizontal tiles")
    parser.add_argument("--vtiles", type=int, default=4, help="number of vertical tiles")
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
    #fine_model_h = ROI(opt.fine_model_def, opt.htiles, opt.classes, 1, opt.img_size).to(device)
    #fine_model_v = ROI(opt.fine_model_def, opt.vtiles, opt.classes, 2, opt.img_size).to(device)
    fine_model = ROI(opt.fine_model_def, opt.htiles * opt.htiles, opt.classes, 1, opt.img_size).to(device)
    encoder = Encoder(48, 128)
    decoder = Decoder(128)
    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            base_model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            base_model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=False, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    #learning_rate = 1e-4
    #mom = 0.9
    #wd = 0.0005
    #optimizer_h = torch.optim.SGD(fine_model_h.parameters(), lr=0.001, momentum=0.9, weight_decay=0.005)
    #optimizer_v = torch.optim.SGD(fine_model_v.parameters(), lr=0.005, momentum=0.9)
    #optimizer = torch.optim.SGD(fine_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.005)
    #optimizer = torch.optim.Adam(fine_model.parameters(), lr=0.0001, weight_decay=0.0005)
    optimizer = torch.optim.Adam(fine_model.parameters(), lr=0.0001)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)
    #optimizer_h = torch.optim.Adam(fine_model_h.parameters(), lr=learning_rate, weight_decay=wd)
    #optimizer_v = torch.optim.Adam(fine_model_v.parameters(), lr=learning_rate, weight_decay=wd)
    #optimizer = torch.optim.Adam(fine_model.parameters(), lr=learning_rate)

    #metrics = [
    #    "loss_x",
    #    "loss_y",
    #    "loss",
    #    "acc_x",
    #    "acc_y",
    #    "acc",
    #]

    for epoch in range(opt.epochs):
        #print(model)
        train_accuracy = 0
        #train_accuracy_h = 0
        #train_accuracy_v = 0
        base_model.eval()
        #fine_model_h.train()
        #fine_model_v.train()
        #fine_model.train()
        encoder.train()
        decoder.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i
            imgs = Variable(imgs.to(device))
            #print(imgs.shape)
            targets = Variable(targets.to(device), requires_grad=False)
            yolo_outputs  = base_model(imgs)
            #print('YOLO OUTPUTS')
            #print(yolo_outputs.shape)
            #print(yolo_outputs)
            yolo_outputs = custom_nms(yolo_outputs, opt.conf_thres, opt.nms_thres)
            #x_inpt = yolo_preprocessing(yolo_outputs, opt.htiles, 0, opt.classes, opt.img_size)
            #y_inpt = yolo_preprocessing(yolo_outputs, opt.vtiles, 1, opt.classes, opt.img_size)
            x_inpt = yolo_single_tile(yolo_outputs, opt.htiles, opt.classes, opt.img_size)
            #print(x_inpt)
            #x_inpt = Variable(x_inpt.to(device))
            #print(x_inpt.shape)
            inputs = x_inpt.view(1, opt.batch_size, -1)
            #print(inputs.shape)
            inputs = inputs.transpose(1,0)
            #print(inputs.shape)
            #y_inpt = Variable(y_inpt.to(device))
            #loss_h, output_x, h_score = fine_model_h(x_inpt, targets)

            #loss, output, score = fine_model(x_inpt, targets)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            encoder.hidden = encoder.init_hidden(inputs.shape[1])
            #encoder.hidden = Variable(encoder.hidden.to(device))
            #print(encoder.hidden)
            inputs = Variable(inputs.to(device))
            hidden = encoder(inputs)
            _, pred_x = torch.max(hidden)
            #loss, score = decoder(targets, hidden)
            loss, score = decoder(targets, pred_x)
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            #print("Loss:", loss.item())
            #loss.backward()
            #optimizer_h.zero_grad()
            #loss_h.backward()
            #if batches_done % opt.gradient_accumulations:
            #    optimizer.step()
            #    optimizer.zero_grad()
                # Accumulates gradient before each step
            #optimizer_h.step()

            #loss_v, output_y, v_score = fine_model_v(y_inpt, targets)
            #optimizer_v.zero_grad()
            #loss_v.backward()
            #if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
            #optimizer_v.step()
            #overall_score = h_score * v_score
            #train_accuracy   += overall_score.mean()
            #train_accuracy_h += h_score.mean()
            #train_accuracy_v += v_score.mean()

            #train_accuracy += score.mean()

            train_accuracy += score/opt.batch_size
            #print('TRAIN ACC')
            #print(train_accuracy)
            #print('AVG')
            #print(train_accuracy/(batch_i+1))
            #print(train_accuracy_h/(batch_i+1))
            #print(train_accuracy_v/(batch_i+1))
            #print(train_accuracy/(batch_i+1))
            #loss = loss_h + loss_v
            #optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            #metric_table = [["Metrics", *[f"ROI Layer {i}" for i in range(len(fine_model.roi_layers))]]]
            '''
             Log metrics at each YOLO layer
            #for i, metric in enumerate(metrics):
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
            '''
            #log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            #print(log_str)
            #print ("\r Processing batch no {}".format(batch_i) + ' of ' + str(batch_i), end="")
            #print('\n')
            print ("\r batch {}".format(batch_i) + ' of ' + str(len(dataloader)) + ": loss: {}".format(loss.item()) + ' -  accuracy: {}'.format(str(train_accuracy.item()/(batch_i+1))), end="")

            base_model.seen += imgs.size(0)
        print('\n Epoch ' + str(epoch) + ' of ' + str(opt.epochs) + ' accuracy: ' + str(train_accuracy.item()/(batch_i+1)))
        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            test_score = evaluate(
                base_model,
                encoder,
                decoder,
                path=valid_path,
                conf_thres=opt.conf_thres,
                nms_thres=opt.nms_thres,
                img_size=opt.img_size,
                num_tiles=opt.htiles,
                classes=opt.classes,
                batch_size=opt.batch_size,
            )
            print('Test accuracy: ' + str(test_score.item()))
            print('\n')
'''
        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            test_score = evaluate(
                base_model,
                fine_model,
                path=valid_path,
                conf_thres=opt.conf_thres,
                nms_thres=opt.nms_thres,
                img_size=opt.img_size,
                num_tiles=opt.htiles,
                classes=opt.classes,
                batch_size=32,
            )
            print('Test accuracy: ' + str(test_score.item()))
            print('\n')
'''
'''
            precision, recall, AP, f1, ap_class = evaluate(
                base_model,
                fine_model_h,
                fine_model_v,
                path=valid_path,
                conf_thres=opt.conf_thres,
                nms_thres=opt.nms_thres,
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
