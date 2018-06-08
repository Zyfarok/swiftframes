from util.video_init import init
from util.video_util import *
from motion import computeMotion, getMotionFeatures
from interpolation import gen_inter_frame
from functools import reduce
import cv2 as cv
import time
import pickle

fileName = init()
degree = 4
half_degree = degree // 2

#### Compute motion
if(True):
    imgs, shape, fps, length = readFrames(fileName)
    motion_matchess = computeMotion(imgs, degree=degree, dist_thresh=1.0, better_thresh=1.05)[0]
    #print(list(map(lambda m: m.final_distance(), motion_matchess[0])))
    #print(reduce(lambda x, y: x + (0.01 / (0.01 + y.final_distance())), motion_matchess[0],0))
    #print(len(motion_matchess[1]))
    print(len(motion_matchess))
    print(len(motion_matchess[0]))
    pickle.dump(motion_matchess, open(fileName + "-motion_matchess.p", "wb"))

#### Interpolate from motion (doubling framerate)
if(True):
    motion_matchess = pickle.load(open(fileName + "-motion_matchess.p", "rb"))
    imgs, shape, fps, length = readFrames(fileName)

    def outImgs():
        previous = next(imgs)
        yield previous
        i = 0
        for img in imgs:
            print ("Generating frame :", i + 0.5, "/", length - 1)
            index = max(0, min(len(motion_matchess) - 1, i - (degree // 2)))
            motion_features = getMotionFeatures(i, 0.5, motion_matchess[index], verbose=False)
            nimg, seg = gen_inter_frame(previous, img, motion_features)
            yield nimg
            yield img
            previous = img
            i += 1
        
        # yield the last image one more time to have exactly 2 times more frames.
        yield previous

    writeFrames(fileName + "-out.avi", shape, outImgs(), fps * 2)
