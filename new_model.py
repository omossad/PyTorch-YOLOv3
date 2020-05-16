from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils.parse_config import *
from utils.new_utils import build_targets, to_cpu, non_max_suppression, xyxy2xywh

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_size = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters

        elif module_def["type"] == "roi":
            img_size = int(hyperparams["height"])
            num_classes = int(module_def["classes"])
            num_tiles = int(module_def["tiles"])
            conf_thes = float(module_def["conf_thes"])
            nms_thes = float(module_def["nms_thes"])
            roi_layer = ROILayer(num_classes, num_tiles, img_size, conf_thes, nms_thes)
            modules.add_module(f"roi_{module_i}", roi_layer)

        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        #self.mse_loss = nn.MSELoss()
        #self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        #self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        #print('GRID')
        #print(self.grid_x)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    #def forward(self, x, targets=None, img_dim=None):
    def forward(self, x, img_dim=None):
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)
        #print('BEFORE')
        #print(x.shape)
        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
        #print('AFTER')
        #print(prediction.shape)
        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.
        #print('OUT')
        #print(prediction[..., 0].shape)
        #print(prediction[..., 1].shape)
        #print(prediction[..., 2].shape)
        #print(prediction[..., 3].shape)
        #print(prediction[..., 4].shape)
        #print(prediction[..., 5:].shape)
        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h
        #print('X DATA')
        #print(pred_boxes[..., 0].shape)
        #print(pred_boxes[..., 0])
        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )
        #print('OUT SHAPE')
        #print(output.shape)
        return output
        #if targets is None:
        #    return output, 0
        #else:
            #print('TARGETS')
            #print(targets.shape)
        #    iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
        #        pred_boxes=pred_boxes,
        #        pred_cls=pred_cls,
        #        target=targets,
        #        anchors=self.scaled_anchors,
        #        ignore_thres=self.ignore_thres,
        #    )
            #print('OBJ MASK')
            #print(obj_mask.shape)

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
        #    loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
        #    loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
        #    loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        #    loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
        #    loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
        #    loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
        #    loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
        #    loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
        #    total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
        #    cls_acc = 100 * class_mask[obj_mask].mean()
        #    conf_obj = pred_conf[obj_mask].mean()
        #    conf_noobj = pred_conf[noobj_mask].mean()
        #    conf50 = (pred_conf > 0.5).float()
        #    iou50 = (iou_scores > 0.5).float()
        #    iou75 = (iou_scores > 0.75).float()
        #    detected_mask = conf50 * class_mask * tconf
        #    precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
        #    recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
        #    recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

        #    self.metrics = {
        #        "loss": to_cpu(total_loss).item(),
        #        "x": to_cpu(loss_x).item(),
        #        "y": to_cpu(loss_y).item(),
        #        "w": to_cpu(loss_w).item(),
        #        "h": to_cpu(loss_h).item(),
        #        "conf": to_cpu(loss_conf).item(),
        #        "cls": to_cpu(loss_cls).item(),
        #        "cls_acc": to_cpu(cls_acc).item(),
        #        "recall50": to_cpu(recall50).item(),
        #        "recall75": to_cpu(recall75).item(),
        #        "precision": to_cpu(precision).item(),
        #        "conf_obj": to_cpu(conf_obj).item(),
        #        "conf_noobj": to_cpu(conf_noobj).item(),
        #        "grid_size": grid_size,
        #    }

        #    return output, total_loss

class ROILayer(nn.Module):
    """ROI layer"""

    def __init__(self, num_classes=3, num_tiles=4, img_dim=416, conf_thes=0.3, nms_thes=0.2):
        super(ROILayer, self).__init__()
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.num_tiles = num_tiles
        self.conf_thres = conf_thes
        self.nms_thres = nms_thes
        #self.mse_loss = nn.MSELoss()
        #self.loss_func = nn.BCEWithLogitsLoss()
        self.metrics = {}
        self.tile_size = self.img_dim // self.num_tiles
        self.loss_func = nn.CrossEntropyLoss()
        #self.fc_net = nn.Sequential(
        #    nn.Linear(self.num_classes * self.num_tiles * 2, 64),
            #nn.BatchNorm1d(1024),
        #    nn.ReLU(inplace=True),
        #    nn.Dropout(),
        #    nn.Linear(64, 32),
        #    nn.BatchNorm1d(32),
        #    nn.ReLU(inplace=True),
        #    nn.Dropout(),
        #)
        self.fc_out_x = nn.Sequential(
            nn.Linear(self.num_classes * self.num_tiles, 64),
            #nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, 32),
            #nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(32, 24),
            nn.ReLU(inplace=True),
            nn.Linear(24, self.num_tiles)
        )
        self.fc_out_y = nn.Sequential(
            nn.Linear(self.num_classes * self.num_tiles, 64),
            #nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, 32),
            #nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(32, 24),
            nn.ReLU(inplace=True),
            nn.Linear(24, self.num_tiles)
        )
        #self.fc_net = nn.Sequential(
        #    nn.Linear(self.num_classes * self.num_tiles * 2, 64),
            #nn.BatchNorm1d(64),
        #    nn.ReLU(inplace=True),
            #nn.Dropout(),
        #    nn.Linear(64, 32),
        #    nn.BatchNorm1d(32),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(32, 16),
            #nn.BatchNorm1d(32),
            #nn.ReLU(inplace=True),
            #nn.Dropout()
        #)
        #self.fc_out_x = nn.Sequential(
        #    nn.Linear(self.num_classes * self.num_tiles, 64),
            #nn.BatchNorm1d(64),
        #    nn.ReLU(inplace=True),
            #nn.Dropout(),
        #    nn.Linear(64, 32),
        #    nn.BatchNorm1d(32),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(32, self.num_tiles),
            #nn.BatchNorm1d(32),
            #nn.ReLU(inplace=True),
            #nn.Dropout()
        #)
        #self.fc_out_y = nn.Sequential(
        #    nn.Linear(self.num_classes * self.num_tiles, 64),
            #nn.BatchNorm1d(64),
        #    nn.ReLU(inplace=True),
            #nn.Dropout(),
        #    nn.Linear(64, 32),
        #    nn.BatchNorm1d(32),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(32, self.num_tiles),
            #nn.BatchNorm1d(32),
            #nn.ReLU(inplace=True),
            #nn.Dropout()
        #)
        #self.fc_net_y = nn.Sequential(
        #    nn.Linear(self.num_classes * self.num_tiles, 256),
        #    nn.BatchNorm1d(256),
        #    nn.ReLU(inplace=True),
        #    nn.Dropout(),
        #    nn.Linear(256, 128),
        #    nn.BatchNorm1d(128),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(128, self.num_tiles)
        #)

    def forward(self, x, targets=None, img_dim=None):
        #print('INPUT SHAPE')
        #print(x.shape)
        num_samples = x.size(0)
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor
        total_loss = 0
        objects = non_max_suppression(x, self.conf_thres, self.nms_thres)
        x_inpt = torch.zeros([num_samples, self.num_tiles, self.num_classes]).type(FloatTensor)
        y_inpt = torch.zeros([num_samples, self.num_tiles, self.num_classes]).type(FloatTensor)
        for image_i, image_pred in enumerate(objects):
            #print('OBJECTS')
            #print(image_pred)
            if image_pred is not None:
                num_pred = len(image_pred)
                #image_pred[..., :4] = xyxy2xywh(image_pred[..., :4])
                x_tiles = (image_pred[..., 0] // self.tile_size).int()
                y_tiles = (image_pred[..., 1] // self.tile_size).int()
                x_tiles_ = (image_pred[..., 3] // self.tile_size).int()
                y_tiles_ = (image_pred[..., 4] // self.tile_size).int()
                obj_class    = image_pred[..., 6].int()
                obj_conf     = image_pred[..., 4]
                for i in range(num_pred):
                    x_tile = max(x_tiles.data.tolist()[i], 0)
                    y_tile = max(y_tiles.data.tolist()[i], 0)
                    x_tile = min(x_tiles.data.tolist()[i], self.num_tiles-1)
                    y_tile = min(y_tiles.data.tolist()[i], self.num_tiles-1)
                    x_tile_ = max(x_tiles_.data.tolist()[i], 0)
                    y_tile_ = max(y_tiles_.data.tolist()[i], 0)
                    x_tile_ = min(x_tiles_.data.tolist()[i], self.num_tiles-1)
                    y_tile_ = min(y_tiles_.data.tolist()[i], self.num_tiles-1)
                    s_obj  = obj_class.data.tolist()[i]
                    s_conf = obj_conf.data.tolist()[i]
                    #print(str(x_coordinate.data.tolist()[i]) + ' ' + str(x_coordinate.data.tolist()[i]))
                    #print(str(image_i) + ' ' + str(x_tile) + ' ' + str(y_tile) + ' ' + str(s_obj) + ' ' + str(s_conf) + '\n')
                    x_inpt[image_i][x_tile][s_obj] += s_conf
                    y_inpt[image_i][y_tile][s_obj] += s_conf
                    if x_tile != x_tile_:
                        x_inpt[image_i][x_tile_][s_obj] += s_conf
                    if y_tile != y_tile_:
                        y_inpt[image_i][y_tile_][s_obj] += s_conf
                #if targets is None:
                #    print('INPUT RAW')
                #    print(image_pred)
                #    print('X')
                #    print(x_inpt[image_i])
                #    print('Y')
                #    print(y_inpt[image_i])
        #print('X before model')
        #print(x_inpt)
        #print('Y before model')
        #print(y_inpt.shape)
        #print('INPUT')
        #print(x_inpt)
        x = x_inpt.view(x_inpt.size(0), -1)
        y = y_inpt.view(y_inpt.size(0), -1)
        #x_cat = torch.cat((x_, y_), 1)
        #x_cat = self.fc_net(x_cat)
        x = self.fc_out_x(x)
        y = self.fc_out_y(y)

        #x_ = x_inpt.view(x_inpt.size(0), -1)
        #x = self.fc_net_x(x)
        #y_ = y_inpt.view(y_inpt.size(0), -1)
        #x_cat = torch.cat((x_, y_), 1)
        #x_cat = self.fc_net(x_cat)
        #x = self.fc_out_x(x_)
        #y = self.fc_out_y(y_)
        #x = x_cat[...,:8]
        #y = x_cat[...,8:]
        #x = x_cat[...,:self.num_tiles]
        #y = x_cat[...,self.num_tiles:]
        #y = self.fc_net_y(y)
        #print('X after MODEL')
        #print(x.shape)

        if targets is None:
        #if 1 == 2:
            return x,y, 0
        else:
            #print('RECEIVED TARGETS')
            #print(targets)
            #new_target = torch.zeros([num_samples, self.num_tiles])
            #new_target[..., 4] = 1
            x_label = targets[..., 1].view(num_samples,-1).type(LongTensor)
            y_label = targets[..., 2].view(num_samples,-1).type(LongTensor)
            #print('X TARGETS')
            #print(x_label)
            #y = torch.LongTensor(batch_size,3).random_() % nb_digits
            tx = torch.zeros([num_samples, self.num_tiles]).type(FloatTensor)
            ty = torch.zeros([num_samples, self.num_tiles]).type(FloatTensor)
            #y_onehot = torch.FloatTensor(batch_size, nb_digits)
            #y_onehot.zero_()
            #y_onehot.scatter_(1, y, 1)
            tx.scatter_(1, x_label, 1)
            ty.scatter_(1, y_label, 1)
            #tx = new_target.type(FloatTensor)
            #ty = new_target.type(FloatTensor)
            #print('ONE HOT TARGETS')
            #print(tx)


            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            #print('SHAPE of LABEL')
            #print(x.shape)

            #_, targets_x = tx.max(dim=1)
            #_, targets_y = ty.max(dim=1)
            #print('SHAPE of TARGET')
            #print(x.shape)
            #print(x)
            #print(y)
            #print(targets_x)
            #loss_x = self.loss_func(x, tx)
            #loss_y = self.loss_func(y, ty)
            #print('PREDICTED')
            #print(x)

            _, corr_x = torch.max(tx, 1)
            _, corr_y = torch.max(ty, 1)
            #print('LOSS')
            #print('X')
            #print(x)
            #print('sigmoid')
            #x = torch.sigmoid(x)
            #print('BEFORE SOFTMAX')
            #print(x)
            x = torch.softmax(x,1)
            #print('SIGMOID')
            #print(x)
            #y = torch.sigmoid(y)
            y = torch.softmax(y,1)
            _, pred_x = torch.max(x, 1)
            _, pred_y = torch.max(y, 1)
            #pre_lbl = torch.zeros([num_samples, self.num_tiles*self.num_tiles]).type(FloatTensor)
            #pre_lbl.scatter(1, pred_x*self.num_tiles + pred_y , 1)
            #print('PREDICTED LABEL')
            #print(pre_lbl)
            #cor_lbl = torch.zeros([num_samples, self.num_tiles*self.num_tiles]).type(FloatTensor)
            #cor_lbl.scatter(1, corr_x*self.num_tiles + corr_y , 1)
            #print('CORRECT LABEL')
            #print(cor_lbl)
            #print(x)
            #print('corr X')
            #print(corr_x)
            loss_x = self.loss_func(x, corr_x)
            loss_y = self.loss_func(y, corr_y)
            x_score = torch.eq(pred_x, corr_x).type(FloatTensor)
            y_score = torch.eq(pred_y, corr_y).type(FloatTensor)
            overall = x_score * y_score
            #overall = overall.bool()
            #overall = overall.type(FloatTensor)
            acc_x = x_score.mean()
            acc_y = y_score.mean()
            acc   = overall.mean()
            #print(pred)
            #print('ACTUAL')
            #print(tx)

            #print(corr)
            #loss_x = self.loss_func(x, targets_x)
            #loss_y = self.loss_func(y, targets_y)
            total_loss = loss_x + loss_y
            #total_loss = self.loss_func(pre_lbl,cor_lbl)
            self.metrics = {
                "loss_x": to_cpu(loss_x).item(),
                "loss_y": to_cpu(loss_y).item(),
                "loss"  : to_cpu(total_loss).item(),
                "acc_x" : to_cpu(acc_x).item(),
                "acc_y" : to_cpu(acc_y).item(),
                "acc"   : to_cpu(acc).item(),
            }
            #return x,y, loss_x, loss_y
            return x,y, total_loss



class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        #print("MODULES")
        #print(self.module_list)
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.roi_layer = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                #print('BEFORE')
                #print(x.shape)
                #x, layer_loss = module[0](x, targets, img_dim)
                x = module[0](x, img_dim)
                #loss += layer_loss
                yolo_outputs.append(x)
            elif module_def["type"] == "roi":
                #print(yolo_outputs)
                yolo_outputs = torch.cat(yolo_outputs, 1)
                roi_x, roi_y, roi_loss = module[0](yolo_outputs, targets)
                #roi_x, roi_y, roi_lossX, roi_lossY = module[0](yolo_outputs, targets)
                #print('ROI LOSS')
                #print(roi_loss)
            layer_outputs.append(x)
        #yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        #yolo_outputs = to_cpu(yolo_outputs)
        return (roi_x, roi_y) if targets is None else (roi_loss, roi_x, roi_y)
        #return (roi_x, roi_y) if targets is None else (roi_lossX, roi_lossY, roi_x, roi_y)
        #print('AFTER')
        #print(yolo_outputs.shape)
        #print(loss)
        #return (loss, yolo_outputs)
        #return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()
