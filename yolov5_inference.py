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

#from models.experimental import attempt_load
#from utils.datasets import LoadImages
from utils.general import (
    non_max_suppression, scale_coords)   #check_img_size, 
from utils.torch_utils import time_synchronized


##hyperparameters
conf_thres = 0.4
iou_thres = 0.5
classes = None
agnostic_nms = False
augment = True   ##was False by default for TTA off
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



    def infer_on_single_img(self, dataset, imgsz = 1024, save_txt = False):
        t7 = time_synchronized()    
        
        #imgsz = check_img_size(imgsz, s=self.model.stride.max())  # check img_size  ##returns new image size multiple of 32(model max stride number)
        
        #dataset = LoadImages(path, img_size=imgsz)   ##initialize dataset to read single image from it's directory


        ####need to clarify why one time this is run####
        #img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img, 3 for 3 channel and 1 for one image
        #_ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once   (((Don't know why need this)))

        t5 = time_synchronized()
        for path, img, im0s, vid_cap in dataset:    ##img is RGB img of size(imgsz,imgsz)our desired size(padded) and img0s is BGR Raw image
            #img = torch.from_numpy(frame).to(device)
            #print(f"img type is :{type(img)} and img0s type is {type(im0s)}")
            #print(f"img len is :{img.shape} and img0s len is {im0s.shape}")
            
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0   (image is normalized here)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)    ##make it to (3,1024,1024) to (1,3,1024,1024)

            # Inference
            t1 = time_synchronized()
            pred = self.model(img, augment= augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic= agnostic_nms)
            t2 = time_synchronized()
            
            single_img_bbox_list = []
            ## (t2-t1) is the total time of (model_prediction + NMS) 
                    
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
                        #print(f'{xyxy} and length is {len(xyxy)} and type is {type(xyxy)}')
                    t0 = time_synchronized()    
                # Print time (inference + NMS)
                print('Done. (%.3fs)' % (t2 - t1))
                print('loop Done. (%.3fs)' % (t0 - t3))
                #print(single_img_bbox_list)
            # t4 = time_synchronized()
            # print('Done2. (%.3fs)' % (t4 - t3))

        #t6 = time_synchronized()
        #torch.cuda.empty_cache()
        #print('Model takes. (%.3fs)' % (t6 - t7))
        #print('Done3. (%.3fs)' % (t6 - t5))
        #print('Done2. (%.3fs)' % (t6 - t7))
        #print(single_img_bbox_list)
        return single_img_bbox_list, im0