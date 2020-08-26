#Code to extract frames from a video

# Importing all necessary libraries 
import cv2 
import os 

# Read the video from specified path 
cam = cv2.VideoCapture("video3.avi")   ##input video name

try: 
    
    # creating a folder named data 
    if not os.path.exists('video3'): 
        os.makedirs('video3') 

# if not created then raise error 
except OSError: 
    print ('Error: Creating directory of data') 

# frame 
currentframe = 0

while(True): 
    
    # reading from frame 
    ret,frame = cam.read() 

    if ret: 
        # if video is still left continue creating images 
        name = './video3/03_defish_' + str(currentframe).zfill(5) + '.jpg'   ##give appropriate name you like
        print ('Creating...' + name) 

        # writing the extracted images 
        cv2.imwrite(name, frame) 

        # increasing counter so that it will 
        # show how many frames are created 
        currentframe += 1
    else: 
        break

# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows() 
