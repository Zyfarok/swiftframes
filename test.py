from util.png_init import init
from util.png_util import *
from motion import computeMotion
from functools import reduce
import cv2 as cv
import time

framesName = init()
shape, imgs = readFrames(framesName, 1, 11)
print(len(imgs))
motion_matchess, appearing_matchess, dispappearing_matchess = computeMotion(imgs, degree=1)
print(list(map(lambda m: m.match_distance, motion_matchess[0])))
print(reduce(lambda x, y: x + (0.01 / (0.01 + y.match_distance)), motion_matchess[0],0))
print(len(motion_matchess[0]))
print(len(motion_matchess[1]))