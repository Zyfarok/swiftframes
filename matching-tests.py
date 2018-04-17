from util.png_init import init
from util.png_util import *

from functools import reduce
import time
import cv2 as cv

framesName = init()
shape, imgs = readFrames(framesName, 4, 2)

img1 = imgs[0]
img2 = imgs[1]

bfmatcher = cv.BFMatcher_create(crossCheck=False)

def testMatching(detector: cv.Feature2D, descriptor: cv.Feature2D, name, matcher=bfmatcher):
    kps1 = detector.detect(img1, None)
    start_time = time.time()
    kps1 = detector.detect(img1, None)
    print(name, "detect 1 exec time :", time.time() - start_time)

    print(name, "feature 1 count :", len(kps1))
    nimg = cv.drawKeypoints(img1, kps1, None, color=(0,255,0), flags=cv.DrawMatchesFlags_DEFAULT)
    cv2.imshow(name + ' features 1',nimg)
    cv2.imwrite("matching-tests/"+framesName+"-"+name+"-fp1.png", nimg)
    #nimg = cv.drawKeypoints(img, kps1, None, color=(0,255,0), flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    #cv2.imshow(name + ' rich features',nimg)
    cv2.waitKey(0)

    start_time = time.time()
    kps1, descs1 = detector.compute(img1, kps1)
    print(name, "compute 1 exec time :", time.time() - start_time)

    start_time = time.time()
    kps2 = detector.detect(img2, None)
    print(name, "detect 2 exec time :", time.time() - start_time)

    print(name, "feature 2 count :", len(kps2))
    nimg = cv.drawKeypoints(img2, kps2, None, color=(0,255,0), flags=cv.DrawMatchesFlags_DEFAULT)
    cv2.imshow(name + ' features 2',nimg)
    cv2.imwrite("matching-tests/"+framesName+"-"+name+"-fp2.png", nimg)
    #nimg = cv.drawKeypoints(img2, kps2, None, color=(0,255,0), flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    #cv2.imshow(name + ' rich features',nimg)
    cv2.waitKey(0)

    start_time = time.time()
    kps2, descs2 = detector.compute(img2, kps2)
    print(name, "compute 2 exec time :", time.time() - start_time)

    start_time = time.time()
    matches = matcher.match(descs1, descs2)
    print(name, "matching exec time :", time.time() - start_time)

    #print(name, "match count :", reduce(lambda x,y: x + len(y), matches, 0))
    print(name, "match count :", len(matches))
    matches = sorted(matches,key=lambda x: x.distance)
    nimg = cv.drawMatches(img1, kps1, img2, kps2, matches[:2000], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite("matching-tests/"+framesName+"-"+name+"-matches2k.png", nimg)
    cv2.imshow(name + ' matches',nimg)
    nimg = cv.drawMatches(img1, kps1, img2, kps2, matches[:1000], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite("matching-tests/"+framesName+"-"+name+"-matches1k.png", nimg)
    nimg = cv.drawMatches(img1, kps1, img2, kps2, matches[:500], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite("matching-tests/"+framesName+"-"+name+"-matches500.png", nimg)
    cv2.waitKey(0)

    print(list(map(lambda m: m.distance, matches)))


#### Feature detector(s)
## SIFT Defaults : nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10.0, sigma=1.6
sift = cv.xfeatures2d.SIFT_create(nOctaveLayers=3, contrastThreshold=0.01, edgeThreshold=100.0, sigma=1.6)
## SURF Defaults : hessianThreshold=100, nOctaves=4, nOctaveLayers=3, extended=False, upright=False
surf = cv.xfeatures2d.SURF_create(hessianThreshold=25, nOctaves=4, nOctaveLayers=3, extended=True, upright=True)
## Agast Defaults : threshold=10,nonmaxSuppression=True,type=cv.AgastFeatureDetector_OAST_9_16
#agast = cv.AgastFeatureDetector_create(threshold=5,nonmaxSuppression=True,type=cv.AgastFeatureDetector_OAST_9_16)
## Fast Defaults : threshold=10, nonmaxSuppression=True, type=cv.FastFeatureDetector_TYPE_9_16
#fast = cv.FastFeatureDetector_create(threshold=4, nonmaxSuppression=True, type=cv.FastFeatureDetector_TYPE_9_16)
## Star Defaults : maxSize=45, responseThreshold=30, lineThresholdProjected=10, lineThresholdBinarized=8, suppressNonmaxSize=5
#star = cv.xfeatures2d.StarDetector_create(maxSize=15, responseThreshold=1, lineThresholdProjected=10, lineThresholdBinarized=8, suppressNonmaxSize=3)

parameters = list((
    (sift,sift,"sift"),
    (surf, surf,"surf"),
    #(agast,"agast"),
    #(fast,"fast"),
    #(star,"star"),
))

for param in parameters:
    testMatching(param[0], param[1], param[2])