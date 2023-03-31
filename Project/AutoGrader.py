import sys
import os
import imutils
import pickle
import time
import cv2 as cv
import torch
import numpy as np
import glob
from torchvision import datasets, transforms
from torch import nn, optim
from imutils.video import VideoStream
from imutils.video import FPS

# Model class must be defined somewhere
model = torch.load('handwriting_autograder_byron.pt')
model.eval()
print(model)

"""
    0 = First Camera detected by System
    1 = Next Camera detected by System

        etc. etc. etc.

    At home on my dock, USB Webcam is 0, Front Facing is 1
    But off my dock, Front Facing is 0, so will depend on your setup
"""
camera = cv.VideoCapture(0)
# camera = cv.VideoCapture(1)

start = (200,100)
end = (1050,900)
color = (252,3,202)
thickness = 3

imageCount = 0

if not camera.isOpened():
    print("Can't open the camera.")
    exit()

# loop over frames from the video file stream
while True:
    ret, frame = camera.read()

    frame = imutils.resize(frame,height=400,width=1250)

    if not ret:
        print("Shit went wrong fam")
        break
    cv.rectangle(frame, start, end, color, thickness)
    cv.imshow('frame', frame)

    keyPressed = cv.waitKey(1)

    ## Space bar to do image capture
    if keyPressed == 32:
        print("You took a picture!")
        file = f'\Captures\{imageCount}.jpg'
        cv.imwrite(file, frame)
        imageCount += 1
    
    ## Press q to quit the application
    elif keyPressed == ord('q'):
        break

"""
    Once all images are taken, we start the image prediction

    Need to do image masking to extract what we are wanting to look for
"""

i = 0

files = glob.glob("GriffinGrading\Captures\*.jpg")

print(files)

for file in files:
    img = cv.imread(file)
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[100:250, 150:450] = 255
    maskImage = cv.bitwise_and(img,img, mask=mask)
    cv.imwrite(f'Masks\{i}.jpg', maskImage)
    i += 1

# Close Camera Feeds
camera.release()
cv.destroyAllWindows()
