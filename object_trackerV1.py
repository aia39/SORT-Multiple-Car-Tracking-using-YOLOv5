# USAGE
# python object_trackerV1.py --weights best.pt

from centroidtrackerV1 import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from pathlib import Path
import pandas as pd
import glob
from tqdm import tqdm
import os
from yolov5_inference import yolov5_infer_on_single_img
from utils.torch_utils import time_synchronized, select_device
from models.experimental import attempt_load


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", type=str, default='yolov5s.pt', help='model.pt path(s)')
ap.add_argument("-i", "--img", type=int, default= 1024, help='image size to prediction')
ap.add_argument("-d", "--inp", type=str, default='input/images/', help='Input directory')
ap.add_argument("-s", "--savevid", type=str, default='True', help='Whether video will be saved or not')
args = vars(ap.parse_args())


def draw_bb(img, label):
    # Show boundingbox on Image
    #height, width = img.shape[0], img.shape[1]

    for car_no in range(len(label)):     ###i need to make it read from list
        #x = int(label['x'][car_no])
        #y = int(label['y'][car_no])
        #w = int(label['w'][car_no])
        #h = int(label['h'][car_no])

        x1 = label[car_no][0]
        y1 = label[car_no][1]
        x2 = label[car_no][2]
        y2 = label[car_no][3]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img


## All Image reading from one folder
dataset_dir = "input"
images_dir = dataset_dir + "/images"
img_name = glob.glob(images_dir + "/*.jpg")
img_list = []
for name in img_name:
    name = (Path(name).stem)+'.jpg'
    img_list.append(name)

list.sort(img_list)
print("Total images: " + str(len(img_list)))



# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)


#print("starting video ...")

####Input
##for video 
#video = cv2.VideoCapture("videoplayback.mp4")

# loop over the frames from the video stream
#while True:

if args["savevid"]:
    ##to save into a video
    image_file = os.path.join(args["inp"],img_list[0])
    frame0 = cv2.imread(image_file)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    height,width,layers= frame0.shape
    video=cv2.VideoWriter('video.avi', fourcc, 30, (width,height))


weights = args["weights"]
device = select_device('')
half = device.type != 'cpu'  # half precision only supported on CUDA
model = attempt_load(weights, map_location=device)  # load FP32 model   ##here model checks whether one model or ensemble of models will be loaded


for i in tqdm(range(0, len(img_list))):
    #t0 = time.time()
    t0 = time_synchronized()
    # read the next frame from the video stream
    #frame = vs.read()  ##for video
    parent_dir = args["inp"]
    image_file = os.path.join(parent_dir,img_list[i])  ##path of an image
    #frame = cv2.imread(image_file)
    
    ###prediction part of YOLOV5
    
    image_size = args["img"]
    #path = args["inp"]

    # obtain our output predictions, and initialize the list of bounding box rectangles
    rects , frame = yolov5_infer_on_single_img(path = image_file, model = model, device = device, half = half, imgsz = image_size)
    t2 = time_synchronized()
    print('Model takes %.3fs' % (t2 - t0))
    frame = draw_bb(frame, rects)


    ##for video
    #ok, frame = video.read()
    
    # if not ok:
    #     break
    #frame = imutils.resize(frame, width=400)

    
    # if the frame dimensions are None, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    
    
    ###sayeed part
    objects, trajectory = ct.update(rects)
    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        object_trajectory = trajectory[objectID]
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        pts = len(object_trajectory)
        if pts > 1:
            cv2.polylines(frame, [np.array(object_trajectory)], False, (0, 255, 255), 5)
    t1 = time.time()

    print('Tracking takes %.3fs' % (t1 - t2))
    # show the output frame
    #cv2.imshow("Frame", frame)
    if args["savevid"]:
        video.write(frame)
    key = cv2.waitKey(30) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
if args["savevid"]:
    video.release()