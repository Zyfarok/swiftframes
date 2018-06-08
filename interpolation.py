import numpy as np
import cv2 as cv
from motion_feature import MotionFeature
from typing import List
from util.collection_util import groupbydefault
import math

import sys

def gen_inter_frame(img_query: np.ndarray, img_train: np.ndarray, motionFeatures: List[MotionFeature]):
    if(len(motionFeatures) < 10):
        return cv.addWeighted(img_query, 0.5, img_train, 0.5, 0.0), None
    
    colomnWidth = max(4, min(120, 720 / (len(motionFeatures) / 8)))
    def key(pt: np.ndarray):
        return int(pt[0] / colomnWidth)
    height, width  = img_query.shape[0], img_query.shape[1]
    colomnCount = math.ceil(width / colomnWidth)


    mf = motionFeatures[0]
    groupedMotionFeatures = groupbydefault(motionFeatures, lambda mf: key(mf.pt))
    
    outImg = np.zeros(img_query.shape, np.uint8)
    featureSeg = np.zeros(img_query.shape, np.uint8)

    def findBestAt(pos: np.ndarray, mf: MotionFeature):
        startKey = key(pos)
        rangeSize = colomnCount - startKey
        bestDist = mf.sq_dist(pos)
        for i in range(rangeSize):
            if((((i -1) * colomnWidth) ** 2) > bestDist):
                break
            mfs = groupedMotionFeatures[startKey + i]
            for nmf in mfs:
                ndist = nmf.sq_dist(pos)
                if(ndist < bestDist):
                    bestDist = ndist
                    mf = nmf
        return mf
    
    for y, x in np.ndindex(outImg.shape[:2]):
        if (x == 0):
            s = "progress: " + str(y) + " / " + str(height)
            sys.stdout.write(s)
            sys.stdout.flush()
            sys.stdout.write("\b" * (len(s))) # return to start of line
            #print("progress: ", y, " / ", height)
            
        pos = np.array((x, y))
        mf = findBestAt(pos, mf)
        
        p_q = mf.query_pixel_at(pos)
        q_x, q_y = p_q[0], p_q[1]
        p_t = mf.train_pixel_at(pos)
        t_x, t_y = p_t[0], p_t[1]
        q_valid = q_x > 0 and q_x < width and q_y > 0 and q_y < height
        t_valid = t_x > 0 and t_x < width and t_y > 0 and t_y < height

        #print(p_q, p_t, q_x, q_y, t_x, t_y)
        if (q_valid):
            if(t_valid):
                inpix = np.round((img_query[q_y, q_x] / 2) + (img_train[t_y, t_x] / 2))
                outImg[y][x] = inpix
            else:
                # print((x,y), " BVector : ", (x - pxb, y - pyb), " (Feature : ", (int(xf),int(yf)),"), last = ", currentColomn + i)
                outImg[y][x] = img_query[q_y, q_x]
        elif (t_valid):
            # print((x,y), " AVector : ", (pxa - x, pya - y), " (Feature : ", (int(xf),int(yf)),"), last = ", currentColomn + i)
            outImg[y][x] = img_train[t_y, t_x]
        else:
            outImg[y][x] = img_query[y, x]
        featureSeg[y][x] = img_query[int(mf.query_pt[1]), int(mf.query_pt[0])]

    return outImg, featureSeg
    #cv.imshow("Final", outImg)
    #cv.waitKey()
    #writeFrame(framesName, position, outImg)
    #cv.imshow("Features", featureSeg)
    #cv.waitKey()
    #writeFrame(framesName + "-fseg", position, featureSeg)