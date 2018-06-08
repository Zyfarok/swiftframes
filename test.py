from util.png_init import init
from util.png_util import *
from motion import computeMotion, getMotionFeatures
from interpolation import gen_inter_frame
from functools import reduce
import cv2 as cv
import time

## This "framesName" will be used to find frames with the name "input/framesName (i).png" where i is the frame id.
framesName = init()
## Change this n to the starting frame id
n = 1
## You can change how much frames will be read/used here
shape, imgs = readFrames(framesName, n, 5)
print(len(imgs))

motion_matchess = computeMotion(imgs, degree=4, dist_thresh=1.0, better_thresh=1.05)[0]
#print(list(map(lambda m: m.final_distance(), motion_matchess[0])))
#print(reduce(lambda x, y: x + (0.01 / (0.01 + y.final_distance())), motion_matchess[0],0))
#print(len(motion_matchess[1]))
print(len(motion_matchess))
print(len(motion_matchess[0]))
## Generate motionFeature for a frame in between the first and the second original frames (n + 0.5)
motion_features = getMotionFeatures(0, 0.5, motion_matchess[0], verbose=True)
print(len(motion_matchess))
print(len(motion_matchess[0]))
## Generate the frame n + 0.5
img, seg = gen_inter_frame(imgs[0], imgs[1], motion_features)
writeFrame(framesName, n + 0.5, img)
## This "seg" image is a image where you can see the areas where each feature was used for the interpolation
writeFrame(framesName + "-seg", n + 0.5, seg)
## Frames are written in the output/ folder