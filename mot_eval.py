##usage : python mot_eval.py

from __future__ import print_function
from scipy.spatial import distance as dist
from collections import OrderedDict
#import imutils
import time
import cv2
import numpy as np
import glob
from tqdm import tqdm
import pandas as pd
import argparse

from yolov5_inference import yolo_detector
from utils.torch_utils import time_synchronized, select_device
from models.experimental import attempt_load

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", type=str, default='best.pt', help='model.pt path(s)')
ap.add_argument("-i", "--img", type=int, default= 1024, help='image size to prediction')
ap.add_argument("-d", "--inp", type=str, default='input/images1', help='Input directory')
ap.add_argument("-l", "--lab", type=str, default='input/labels1', help='Labels directory')
args = vars(ap.parse_args())

########################
####### ALERT ############
########################
###SELECT THIS FIRST

#tracker = "euclidean"
tracker = "sort"


class EuclideanTracker():
    
    def __init__(self, maxDisappeared=50):      ##by default it was 50
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.boundingBoxes = OrderedDict()
        self.trajectory = OrderedDict()
        self.disappeared = OrderedDict()
        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared
        
        
    def register(self, centroid, boundingBox):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.boundingBoxes[self.nextObjectID] = boundingBox
        self.trajectory[self.nextObjectID] = list([centroid])
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
        
        
    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.boundingBoxes[objectID]
        del self.trajectory[objectID]
        del self.disappeared[objectID]
        
        
    def update(self, rects):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            # return early as there are no centroids or tracking info
            # to update
            return self.objects
        
        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
            
        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], rects[i])
                
        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()
            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]
            
            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue
                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.boundingBoxes[objectID] = rects[col]
                self.trajectory[objectID].append(inputCentroids[col])
                self.disappeared[objectID] = 0
                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)
                
            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            
            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
                        
            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], rects[col])
        # return the set of trackable objects
        return self.objects, self.trajectory, self.boundingBoxes




from numba import jit
import os.path

from scipy.optimize import linear_sum_assignment
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter
from collections import OrderedDict

from tqdm import tqdm
import pandas as pd


def iou(bb_test,bb_gt):
    """
    Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return(o)

def colinearity(det,hist):
    #det - current detection
    #hist - last 2 mean detections
    
    dims = det[2:4] - det[:2]
    diag = np.sqrt(sum(dims**2))
    a = det[:2] + dims/2 - hist[-2]
    b = hist[-1] - hist[-2]
    len1 = np.sqrt(sum(a*a))
    len2 = np.sqrt(sum(b*b))
    ratio = len2/float(len1)
    maxdist = diag*(min(dims)/max(dims)+1)
    maxval = b.dot(b)
    a *= ratio
    return a.dot(b)/float(maxval) if maxval and maxdist > len1 else 0

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
    """
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    x = bbox[0]+w/2.
    y = bbox[1]+h/2.
    s = w*h    #scale is just area
    r = w/float(h)
    return np.array([x,y,s,r]).reshape((4,1))

def convert_x_to_bbox(x,score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2]*x[3])
    h = x[2]/w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))
    


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0
    
    def __init__(self,bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.cthist = [self.kf.x[:2].ravel()]

    def update(self, bbox, n):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.cthist.append(bbox[:2] + (bbox[2:4] - bbox[:2]) / 2)
        self.cthist = self.cthist[-n:]

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
            self.kf.P *= 1.2 # we may be lost, increase uncertainty and responsiveness
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)
    

    
    
def associate_detections_to_trackers(detections, trackers, cost_fn = iou, threshold = 0.3):   ##less th gives better score
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    lendet = len(detections)
    lentrk = len(trackers)

    if(lentrk==0):
        return np.empty((0,2),dtype=int), np.arange(lendet), np.array([],dtype=int)
    cost_matrix = np.zeros((lendet,lentrk),dtype=np.float32)

    for d,det in enumerate(detections):
        for t,trk in enumerate(trackers):
            cost_matrix[d,t] = cost_fn(det,trk)
    cost_matrix[cost_matrix < threshold] = 0.
    matched_indices = linear_sum_assignment(-cost_matrix)   
    matched_indices = np.asarray(matched_indices)
    matched_indices = np.transpose(matched_indices)

    costs = cost_matrix[tuple(matched_indices.T)] # select values from cost matrix by matched indices
    matches = matched_indices[np.where(costs)[0]] # remove zero values from matches
    unmatched_detections = np.where(np.in1d(range(lendet), matches[:,0], invert=True))[0]
    unmatched_trackers = np.where(np.in1d(range(lentrk), matches[:,1], invert=True))[0]

    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)

    return matches, unmatched_detections, unmatched_trackers



class SortTracker(object):
    def __init__(self,max_age=10,min_hits=0):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.trajectory = OrderedDict()

    def update(self, rects, cnum = 3):
        dets = np.array(rects)
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
          cnum - number of center positions to average
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        #get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers),5))
        ctmean = []
        to_del = []
        ret = []
        
        for t,trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if(np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        #for t in reversed(to_del):
        for t in iter(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)

        for t in unmatched_trks:
            cnt = np.array(self.trackers[t].cthist)
            cnt = np.array([np.convolve(cnt[:,i], np.ones((cnum,))/cnum, mode='valid') for i in (0,1)]).T
            if cnt.shape[0] == 1: # fix same len
                cnt = np.concatenate((cnt,cnt),axis=0)
            ctmean.append(cnt)

        rematch, new_dets, lost_trks = associate_detections_to_trackers(dets[unmatched_dets],ctmean,colinearity,0.6)
        rematch = np.array([unmatched_dets[rematch[:,0]], unmatched_trks[rematch[:,1]]]).T
        matched = np.concatenate((matched, rematch.reshape(-1,2)))
        unmatched_dets = unmatched_dets[new_dets]
        unmatched_trks = unmatched_trks[lost_trks]

        #update matched trackers with assigned detections
        for t,trk in enumerate(self.trackers):
            if(t not in unmatched_trks):
                d = matched[np.where(matched[:,1]==t)[0],0]
                trk.update(dets[d,:][0], cnum+1)

        #create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:]) 
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
        #for trk in iter(self.trackers):  #don't work
            d = trk.get_state()[0]
            if((trk.time_since_update < self.max_age) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((d,[trk.id+1],[trk.time_since_update])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            i -= 1
            #remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        
        
        return_trackers = np.concatenate(ret)
        current_objects = OrderedDict()
        current_trajectory = OrderedDict()
        current_boundingBoxes = OrderedDict()
        
        for d in return_trackers:
            d = d.astype(np.int32)
            centroid = [int((d[0] + d[2]) / 2), int((d[1] + d[3]) / 2)]
            ID = d[4]
            current_objects[ID] = centroid
            current_boundingBoxes[ID] = d[0:4]
            
            try:
                self.trajectory[ID].append(centroid)
            except:
                self.trajectory[ID] = list([centroid])
            current_trajectory[ID] = self.trajectory[ID]
                
        return current_objects, current_trajectory, current_boundingBoxes   


# Get the bounding boxes for an image file
def get_bb(image_labels, image_file):
    label = pd.read_table(image_labels[image_file], delim_whitespace=True,
                        names=('~', 'x', 'y', 'w', 'h'),
                        dtype={'~': np.uint8, 'x': np.float32, 'y': np.float32, 'w': np.float32,
                               'h': np.float32})
    label = label.drop('~', axis=1)
    return label

# Convert centroid and height-width bb format to initial and endpoint format
def cvt_bb(label):
    rect = []
    for car_no in range(len(label)):
        startX = label['x'][car_no] - label['w'][car_no]/2
        startY = label['y'][car_no] - label['h'][car_no]/2
        endX = label['x'][car_no] + label['w'][car_no]/2
        endY = label['y'][car_no] + label['h'][car_no]/2
        
        rect.append((startX, startY, endX, endY))
    return rect

# Draw all the boundingboxes on Image
def draw_bb(img, label):
    height, width = img.shape[0], img.shape[1]

    for car_no in range(len(label)):
        x = int(label['x'][car_no] * width)
        y = int(label['y'][car_no] * height)
        w = int(label['w'][car_no] * width / 2)
        h = int(label['h'][car_no] * height / 2)

        cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)

    return img




#dataset_dir = "input"

#images_dir = dataset_dir + "/images1"
#labels_dir = dataset_dir + "/labels1"

images_dir = args["inp"]
labels_dir = args["lab"]

image_labels = {}


for img in glob.glob(images_dir + "/*.jpg", recursive=True):
    label = img.replace(images_dir, labels_dir)
    label = label.replace(".jpg", ".txt")
    image_labels[img] = label

images = list(image_labels.keys())
list.sort(images)

print("Total images: " + str(len(image_labels)))





'''
# dataset_dir = "../../Dataset/fisheye-day-30072020"

# images_dir = dataset_dir + "/images"
# labels_dir = dataset_dir + "/labels"

# sequence = "01_fisheye_day"

# image_labels = {}


# for img in glob.glob(images_dir + "/*.jpg", recursive=True):
#     label = img.replace(images_dir, labels_dir)
#     label = label.replace(".jpg", ".txt")
#     if sequence in img:
#         image_labels[img] = label

# images = list(image_labels.keys())
# list.sort(images)

# print("Total images: " + str(len(image_labels)))
'''

def gt_generator(tracker, tracking_GT_dir):
    if tracker == "euclidean":
        mot_tracker = EuclideanTracker()
    elif tracker == "sort":
        mot_tracker = SortTracker(max_age = 10, min_hits=6) #create instance of the SORT tracker
    
    ###Getting GT
    #tracking_GT_dir = "input/fisheye_day_tracking"
    
    if not os.path.exists(tracking_GT_dir):
        os.mkdir(tracking_GT_dir)
    
    
    frame = cv2.imread(images[0])
    (H, W) = frame.shape[:2]
    
    # loop over the frames from the video stream
    for i in tqdm(range(len(image_labels))):
        # read the next frame from the video stream and resize it
        image_file = images[i]
        label = get_bb(image_labels, image_file)
        rects = cvt_bb(label) * np.array([W, H, W, H])
          
        objects, _ , boundingBoxes = mot_tracker.update(rects)
    
        output_table = np.zeros((len(objects), 5), dtype=int)
        idx = 0
        
        for (objectID, centroid) in objects.items():
            output_table[idx, 0] = objectID
            output_table[idx, 1:5] = boundingBoxes[objectID].astype(int)
            idx += 1
    
        filename = image_labels[image_file].replace(labels_dir, tracking_GT_dir)
        np.savetxt(filename, output_table, fmt='%d')
        
    del mot_tracker
    print("Done with GT")


def det_generator(tracker, tracking_detection_dir):
    if tracker == "euclidean":
        mot_tracker1 = EuclideanTracker()
    elif tracker == "sort":
        mot_tracker1 = SortTracker(max_age = 10, min_hits=6) #create instance of the SORT tracker
    
    ####detection
    
    #tracking_detection_dir = "input/fisheye_day_detection"
    
    if not os.path.exists(tracking_detection_dir):
        os.mkdir(tracking_detection_dir)
        
    #mot_tracker1 = SortTracker() #create instance of the tracker
        
        
    frame = cv2.imread(images[0])
    (H, W) = frame.shape[:2]
    
    
    weights = args["weights"]
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = attempt_load(weights, map_location=device)  # load FP32 model   ##here model checks whether one model or ensemble of models will be loaded
    
    yolo = yolo_detector(model, device, half)
    
    
    # loop over the frames from the video stream
    for i in tqdm(range(len(image_labels))):
        
        image_file = images[i]
        #label = get_bb(image_labels, image_file)
        #rects = cvt_bb(label) * np.array([W, H, W, H])
        
        image_size = args["img"]
        rects , frame = yolo.infer_on_single_img(image_file, image_size)
        if tracker == "sort":
            rects = np.array(rects)
        
        objects, _ , boundingBoxes = mot_tracker1.update(rects)
    
        output_table = np.zeros((len(objects), 5), dtype=int)
        idx = 0
        
        for (objectID, centroid) in objects.items():
            output_table[idx, 0] = objectID
            output_table[idx, 1:5] = boundingBoxes[objectID].astype(int)
            idx += 1
    
        filename = image_labels[image_file].replace(labels_dir, tracking_detection_dir)
        np.savetxt(filename, output_table, fmt='%d')
        
    del mot_tracker1
    print("Done Detection")


tracking_detection_dir = "input/fisheye_day_detection"
tracking_GT_dir = "input/fisheye_day_tracking"

gt_generator(tracker, tracking_GT_dir)
det_generator(tracker, tracking_detection_dir)


import motmetrics as mm

def get_bb_with_ID(filename):
    lb = pd.read_table(filename, delim_whitespace=True,
                        names=('ID', 'x1', 'y1', 'x2', 'y2'),
                        dtype={'ID': np.uint16, 'x1': np.uint16, 'y1': np.uint16,
                               'x2': np.uint16, 'y2': np.uint16})
    
    BBs = np.zeros((len(lb), 4), dtype=int)
    

    IDs = np.array(lb['ID'][:])
    BBs[:, 0] = lb['x1'][:]
    BBs[:, 1] = lb['y1'][:]
    BBs[:, 2] = lb['x2'][:]
    BBs[:, 3] = lb['y2'][:]
        
    return IDs, BBs


def evaluate_tracking(tracking_GT_dir, tracking_detection_dir):
    detection_files = glob.glob(tracking_detection_dir + "/*.txt", recursive=True)
    list.sort(detection_files)
    
    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)
    
    for i in tqdm(range(len(detection_files))):
        det_file = detection_files[i]
        gt_file = det_file.replace(tracking_detection_dir, tracking_GT_dir)
        
        detector_hypotheses, detector_bbs = get_bb_with_ID(det_file)
        gt_objects, gt_bbs = get_bb_with_ID(gt_file)
                
        distances = mm.distances.iou_matrix(gt_bbs, detector_bbs, max_iou=0.5)
        
        # Call update once for per frame. For now, assume distances between
        # frame objects / hypotheses are given.
        acc.update(
            gt_objects,                     # Ground truth objects in this frame
            detector_hypotheses,            # Detector hypotheses in this frame
            distances                       # Distances from GT to hypotheses
        )

        
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=mm.metrics.motchallenge_metrics,   #metrics=['num_frames', 'mota', 'motp']
        name='overall'
        )

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    
    return strsummary

#tracking_detection_dir = "input/fisheye_day_detection"
#tracking_GT_dir = "input/fisheye_day_tracking"
summary = evaluate_tracking(tracking_GT_dir, tracking_detection_dir)

##saving result into a txt file
text_file = open("mot_result.txt", "w")
n = text_file.write(summary)
text_file.close()