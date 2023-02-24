import copy
import os
import pickle
import time
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms

import Configuration
from Utils import Logger
from Utils.torchvision_utils import draw_bounding_boxes

import numpy as np
import matplotlib
from matplotlib.cm import cmaps_listed

from Utils.yolov5.load_model import load_model
from Utils.yolov5.utils.augmentations import letterbox

color_palette = matplotlib.cm.get_cmap('viridis', 115).colors


# from Utils.util_yolov5.common import *
# from Utils.util_yolov5.experimental import *


class YoloV5:

    def __init__(self):  # model, input channels, number of classes
        super().__init__()

        # Use the GPU or the CPU, if a GPU is not available
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Load pretrained weights
        if os.path.exists(Configuration.OBJ_DETECTOR_PATH):
            self.model = load_model(Configuration.OBJ_DETECTOR_PATH)

        # Move model to the right device
        self.model.to(self.device)

        self.categories = [#'background',
               'alarmclock', 'aluminumfoil', 'apple', 'armchair', 'baseballbat', 'basketball', 'bathtub',
               'bathtubbasin', 'bed', 'blinds', 'book', 'boots', 'bottle', 'bowl', 'box', 'bread', 'butterknife',
               'cd', 'cabinet', 'candle', 'cellphone', 'chair', 'cloth', 'coffeemachine', 'coffeetable', 'countertop',
               'creditcard', 'cup', 'curtains', 'desk', 'desklamp', 'desktop', 'diningtable', 'dishsponge', 'dogbed',
               'drawer', 'dresser', 'dumbbell', 'egg', 'faucet', 'floorlamp', 'footstool', 'fork', 'fridge',
               'garbagebag', 'garbagecan', 'handtowel', 'handtowelholder', 'houseplant', 'kettle', 'keychain', 'knife',
               'ladle', 'laptop', 'laundryhamper', 'lettuce', 'lightswitch', 'microwave', 'mirror', 'mug', 'newspaper',
               'ottoman', 'painting', 'pan', 'papertowelroll', 'pen', 'pencil', 'peppershaker', 'pillow', 'plate',
               'plunger', 'poster', 'pot', 'potato', 'remotecontrol', 'safe', 'saltshaker', 'scrubbrush',
               'shelf', 'shelvingunit', 'showercurtain', 'showerdoor', 'showerglass', 'showerhead', 'sidetable', 'sink',
               'sinkbasin', 'soapbar', 'soapbottle', 'sofa', 'spatula', 'spoon', 'spraybottle', 'statue', 'stool',
               'stoveburner', 'stoveknob', 'tvstand', 'teddybear', 'television', 'tennisracket',
               'tissuebox', 'toaster', 'toilet', 'toiletpaper', 'toiletpaperhanger', 'tomato', 'towel', 'towelholder',
               'vacuumcleaner', 'vase', 'watch', 'wateringcan', 'window', 'winebottle']


    def predict(self, rgb_img):

        # Set model in evaluation mode
        self.model.eval()

        # Predict objects in the image
        rgb_img = Image.fromarray(rgb_img, mode="RGB")
        if rgb_img.width != 224 or rgb_img.height != 224:
            rgb_img_resized = rgb_img.resize((224, 224))
        else:
            rgb_img_resized = copy.deepcopy(rgb_img)


        bgr_img = np.array(rgb_img_resized)[:,:,::-1]  # RGB to BGR
        img_size = [224, 224]
        bgr_img = letterbox(bgr_img, img_size)[0]
        img = bgr_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        # img = np.array(rgb_img_resized).transpose((2,1,0))
        img = torch.from_numpy(img).to(self.device)
        # img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        pred_preprocess = self.model(img)[0]
        # NMS
        # pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False, max_det=1000)

        # list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        pred_preprocess = non_max_suppression(pred_preprocess)
        pred_preprocess = pred_preprocess[0].cpu().numpy()

        pred = dict()
        pred['boxes'] = np.array([[int(p[0]), int(p[1]), int(p[2]), int(p[3])] for p in pred_preprocess])
        pred['scores'] = np.array([p[4] for p in pred_preprocess])
        pred['labels'] = np.array([int(p[5]) for p in pred_preprocess])
        pred['boxes'] = torch.from_numpy(pred['boxes'])
        pred['scores'] = torch.from_numpy(pred['scores'])
        pred['labels'] = torch.from_numpy(pred['labels'])

        # Process predictions



        # Resize boxes to fit input image size
        # rgb_img.thumbnail((224, 224), Image.ANTIALIAS)
        rgb_img_tensor = torch.from_numpy(np.array(rgb_img).transpose((2,0,1))) # move channels to first tensor axis
        # bbox = self.resize_boxes(pred['boxes'], (224, 224), rgb_img_tensor.shape[-2:])
        # pred['boxes'] = bbox

        # Discard predictions with low scores
        idx = [i for i, score in enumerate(pred['scores']) if score > Configuration.OBJ_SCORE_THRSH]
        pred['boxes'] = pred['boxes'][idx]
        pred['labels'] = pred['labels'][idx]
        pred['scores'] = pred['scores'][idx]

        # Parse labels id to category names
        if self.device == torch.device('cpu'):
            color_labels = [color_palette[int(l)] for l in np.array(pred['labels'])]
            pred['labels'] = [self.categories[int(el)] for el in np.array(pred['labels'])]
        else:
            color_labels = [color_palette[int(l)] for l in pred['labels'].cpu().numpy()]
            pred['labels'] = [self.categories[int(el)] for el in pred['labels'].cpu().numpy()]

        color_labels = [matplotlib.colors.rgb2hex(color_labels[i]) for i in range(len(color_labels))]

        # DEBUG
        if Configuration.PRINT_OBJS_PREDICTIONS:
            goal_objs = Configuration.GOAL_OBJECTS
            print_pred_labels = [el for el in pred['labels'] if el.lower().strip() in goal_objs]
            print_pred_labels = [el for el in pred['labels']]
            printed_pred_boxes = [pred['boxes'][i].detach().numpy() for i in range(len(pred['boxes']))
                                  if pred['labels'][i].lower().strip() in goal_objs]
            printed_pred_boxes = [pred['boxes'][i].detach().numpy() for i in range(len(pred['boxes']))]

            pred_img = transforms.ToPILImage()(draw_bounding_boxes(rgb_img_tensor, torch.FloatTensor(printed_pred_boxes), colors=color_labels, labels=print_pred_labels))
            prev_preds = [img for img in os.listdir(Logger.LOG_DIR_PATH) if img.startswith("pred_")]
            pred_img.save('{}/pred_{}.jpg'.format(Logger.LOG_DIR_PATH, len(prev_preds)), "JPEG")

        # Return predicted bboxes with labels and scores
        pred['boxes'] = pred['boxes'].cpu().detach().numpy()
        pred['labels'] = np.array(pred['labels'])
        pred['scores'] = pred['scores'].cpu().detach().numpy()
        return pred


    def resize_boxes(self, boxes, original_size, new_size):
        # type: (Tensor, List[int], List[int]) -> Tensor
        ratios = [
            torch.tensor(s, dtype=torch.float32, device=boxes.device) /
            torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
            for s, s_orig in zip(new_size, original_size)
        ]
        ratio_height, ratio_width = ratios
        xmin, ymin, xmax, ymax = boxes.unbind(1)

        xmin = xmin * ratio_width
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height
        return torch.stack((xmin, ymin, xmax, ymax), dim=1)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output