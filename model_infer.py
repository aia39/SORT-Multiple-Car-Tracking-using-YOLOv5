import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from utils.general import (
    non_max_suppression, scale_coords)  
from utils.torch_utils import time_synchronized


##hyperparameters
conf_thres = 0.4
iou_thres = 0.5
classes = None
agnostic_nms = False
augment = False
webcam = False

class yolo_detector():
    def __init__(self, model, device, half):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.model = model
        self.device = device
        self.half = half

        if self.half:
            self.model.half()  # to FP16



    def infer_on_single_img(self, dataset, imgsz = 1024):
        for path, img, im0s in dataset:    ##img is RGB img of size(imgsz,imgsz)our desired size(padded) and img0s is BGR Raw image
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0   (image is normalized here)
            #if img.ndimension() == 3:
            img = img.unsqueeze(0)    ##make it to (3,1024,1024) to (1,3,1024,1024)

            # Inference
            #t1 = time_synchronized()
            with torch.no_grad():
                pred = self.model(img, augment= augment)[0]
                # Apply NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic= agnostic_nms)
                
            #t2 = time_synchronized()

            single_img_bbox_list = []  
            # Process detections
            for i, det in enumerate(pred):  # detections(on one image here) per image
                _, _, im0 = path, '', im0s    ##im0 is original image loaded by cv2.imread()
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()    ##convert all bbox to desired img size to original image size##
                    t3 = time_synchronized()                            
                    for *xyxy, conf, cls in det:                    
                        c1 = np.asarray([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), conf.detach().cpu().item()])   ##conf wasn't there at first time
                        single_img_bbox_list.append(c1)
                        
                    t0 = time_synchronized()    
                # Print time (inference + NMS)
                # print('Done. (%.3fs)' % (t2 - t1))
                # print('loop Done. (%.3fs)' % (t0 - t3))
                
        return single_img_bbox_list, im0
