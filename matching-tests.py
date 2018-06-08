from util.png_init import init
from util.png_util import *

from typing import List, Tuple
from functools import reduce
import time
import cv2 as cv

framesName = init()
shape, imgs = readFrames(framesName, 4, 2)

img1 = imgs[0]
img2 = imgs[1]

bfmatcherl1 = (cv.BFMatcher_create(crossCheck=True,normType=cv.NORM_L1),        "l1",
    cv.BFMatcher_create(crossCheck=False,normType=cv.NORM_L1))
bfmatcherl2 = (cv.BFMatcher_create(crossCheck=True,normType=cv.NORM_L2),        "l2",
    cv.BFMatcher_create(crossCheck=False,normType=cv.NORM_L2))
bfmatcherH1 = (cv.BFMatcher_create(crossCheck=True,normType=cv.NORM_HAMMING),   "H1",
    cv.BFMatcher_create(crossCheck=False,normType=cv.NORM_HAMMING))
bfmatcherH2 = (cv.BFMatcher_create(crossCheck=True,normType=cv.NORM_HAMMING2),  "H2",
    cv.BFMatcher_create(crossCheck=False,normType=cv.NORM_HAMMING2))

def testMatching(detector: cv.Feature2D, detectorName: str, descriptorsAndMatchers: List[Tuple[cv.Feature2D,str,cv.DescriptorMatcher]]):
    print("##################################### Detector :",detectorName,"#####################################")
    kps1 = detector.detect(img1, None)
    total_detector_time = time.time()
#    start_time = time.time()
#    kps1 = detector.detect(img1, None)
#    print(detectorName, "detect 1 exec time :", time.time() - start_time)
#
#    start_time = time.time()
#    kps2 = detector.detect(img2, None)
#    
#    print(detectorName, "detect 2 exec time :", time.time() - start_time)
    kpss = detector.detect(list((img1, img2)), None)
    total_detector_time = time.time() - total_detector_time
    kps1, kps2 = kpss[0], kpss[1]

    print(detectorName, "feature 1 count :", len(kps1))
    kps1_count = len(kps1)
    print(detectorName, "feature 2 count :", len(kps2))

    #nimg = cv.drawKeypoints(img1, kps1, None, color=(0,255,0), flags=cv.DrawMatchesFlags_DEFAULT)
    #cv2.imshow('features of ' + detectorName + ' 1',nimg)
    #cv2.imwrite("matching-tests/"+framesName+"-features-"+detectorName+"-1.png", nimg)
    #nimg = cv.drawKeypoints(img, kps1, None, color=(0,255,0), flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    #cv2.imshow(name + ' rich features',nimg)
    #cv2.waitKey(0)

    #nimg = cv.drawKeypoints(img2, kps2, None, color=(0,255,0), flags=cv.DrawMatchesFlags_DEFAULT)
    #cv2.imshow('features of ' + detectorName + ' 2',nimg)
    #cv2.imwrite("matching-tests/"+framesName+"-features-"+detectorName+"-2.png", nimg)
    #nimg = cv.drawKeypoints(img2, kps2, None, color=(0,255,0), flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    #cv2.imshow(name + ' rich features',nimg)
    #cv2.waitKey(0)

    
    for descriptor, descriptorName, namedMatcher in descriptorsAndMatchers:
        matcher, matcherName, speedMatcher = namedMatcher
        full_name = detectorName + "+" + descriptorName + "(" + matcherName+")"
        print("--------------------------------------", full_name, "--------------------------------------")

        try:
            descriptor.compute(img1, kps1)
        except Exception:
            print("#-#-# Error with",full_name)
            continue
        
#        total_descriptor_time = time.time()
#        start_time = time.time()
#        nkps1, descs1 = descriptor.compute(img1, kps1)
#        print(full_name, "compute 1 exec time :", time.time() - start_time)
#
#        start_time = time.time()
#        nkps2, descs2 = descriptor.compute(img2, kps2)
#        total_descriptor_time = (time.time() - total_descriptor_time) + total_detector_time
#        print(full_name, "compute 2 exec time :", time.time() - start_time)
        total_descriptor_time = time.time()
        nkpss, descss = descriptor.compute(list((img1, img2)), kpss)
        nkps1, descs1 = nkpss[0], descss[0]
        nkps2, descs2 = nkpss[1], descss[1]
        total_descriptor_time = (time.time() - total_descriptor_time) + total_detector_time

        start_time = time.time()
        matches = list(map(lambda x: x[0], filter(lambda x: len(x)>0, matcher.knnMatch(descs1, descs2,1))))
        print(full_name, "matching exec time :", time.time() - start_time)

        start_time = time.time()
        speedMatcher.match(descs1[:3000], descs2[:3000:9])
        matching_time = time.time() - start_time
        print(full_name, "matching expected time :", matching_time)

        #print(name, "match count :", reduce(lambda x,y: x + len(y), matches, 0))
        matches = sorted(matches, key=lambda x: x.distance)
        print(full_name,"detect+desc time :", total_descriptor_time)
        print(full_name,"future time :", total_descriptor_time + 0.1*matching_time)
        print(full_name,"desc : {:%}, match : {:%}, final : {:%}".format(((len(descs1) / kps1_count)), ((len(matches) / len(descs1))), ((len(matches) / kps1_count))))
        #print("####",full_name,"match :", str((len(matches) / len(descs1))*100) + "%")
        print(full_name,"avg match distance :", sum(map(lambda x: x.distance, matches))/len(matches))
        match_distances = map(lambda z: ((((z[0][0] - z[1][0])**2) + ((z[0][1] - z[1][1])**2)) / (60 ** 2), z[0], z[1]), map(lambda x: (nkps1[x.queryIdx].pt,nkps2[x.trainIdx].pt), matches))
        probably_good_matches = list(filter(lambda x: x[0] < 1, match_distances))
        print(full_name,"probably good matches (<60px) : {}, {:%} of matches, {:%} of keypoints".format(len(probably_good_matches), len(probably_good_matches)/len(matches), len(probably_good_matches)/kps1_count))
        #print(probably_good_matches)
        #
        nimg = cv.drawMatches(img1, nkps1, img2, nkps2, matches[:2000:5], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite("matching-tests/"+framesName+"-matches-400r-"+detectorName+"+"+descriptorName+"-"+matcherName+".png", nimg)
        cv2.imwrite("matching-tests/"+framesName+"-matchesByDescriptor-400r-"+descriptorName+"+"+detectorName+"-"+matcherName+".png", nimg)
        ##cv2.imshow(descriptorName + ' matches',nimg)
        nimg = cv.drawMatches(img1, nkps1, img2, nkps2, matches[:3000], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite("matching-tests/"+framesName+"-matches-3kb-"+detectorName+"+"+descriptorName+"-"+matcherName+".png", nimg)
        cv2.imwrite("matching-tests/"+framesName+"-matchesByDescriptor-3kb-"+descriptorName+"+"+detectorName+"-"+matcherName+".png", nimg)
        #
        nimg = cv.drawMatches(img1, nkps1, img2, nkps2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite("matching-tests/"+framesName+"-matches-all-"+detectorName+"+"+descriptorName+"-"+matcherName+".png", nimg)
        cv2.imwrite("matching-tests/"+framesName+"-matchesByDescriptor-all-"+descriptorName+"+"+detectorName+"-"+matcherName+".png", nimg)
        nimg = cv.drawMatches(img1, nkps1, img2, nkps2, matches, None, flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
        cv2.imwrite("matching-tests/"+framesName+"-matches-single-"+detectorName+"+"+descriptorName+"-"+matcherName+".png", nimg)
        cv2.imwrite("matching-tests/"+framesName+"-matchesByDescriptor-single-"+descriptorName+"+"+detectorName+"-"+matcherName+".png", nimg)
        nimg = cv.drawMatches(img1, nkps1, img2, nkps2, matches, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite("matching-tests/"+framesName+"-matches-rich-"+detectorName+"+"+descriptorName+"-"+matcherName+".png", nimg)
        cv2.imwrite("matching-tests/"+framesName+"-matchesByDescriptor-rich-"+descriptorName+"+"+detectorName+"-"+matcherName+".png", nimg)
        nimg = cv.drawMatches(img1, nkps1, img2, nkps2, matches[::25], None, flags=(cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS | cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS))
        cv2.imwrite("matching-tests/"+framesName+"-matches-both-"+detectorName+"+"+descriptorName+"-"+matcherName+".png", nimg)
        cv2.imwrite("matching-tests/"+framesName+"-matchesByDescriptor-both-"+descriptorName+"+"+detectorName+"-"+matcherName+".png", nimg)
        #
        nkps1 = [nkps1[x.queryIdx] for x in matches]
        nimg = cv.drawKeypoints(img1, nkps1, None, color=(0,255,0), flags=cv.DrawMatchesFlags_DEFAULT)
        cv2.imwrite("matching-tests/"+framesName+"-matches-kps-"+detectorName+"+"+descriptorName+"-"+matcherName+"-1.png", nimg)
        cv2.imwrite("matching-tests/"+framesName+"-matchesByDescriptor-kps-"+descriptorName+"+"+detectorName+"-"+matcherName+"-1.png", nimg)
        nimg = cv.drawKeypoints(img1, nkps1, None, color=(0,255,0), flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        cv2.imwrite("matching-tests/"+framesName+"-matches-rkps-"+detectorName+"+"+descriptorName+"-"+matcherName+"-1.png", nimg)
        cv2.imwrite("matching-tests/"+framesName+"-matchesByDescriptor-rkps-"+descriptorName+"+"+detectorName+"-"+matcherName+"-1.png", nimg)
        nkps2 = [nkps2[x.trainIdx] for x in matches]
        nimg = cv.drawKeypoints(img2, nkps2, None, color=(0,255,0), flags=cv.DrawMatchesFlags_DEFAULT)
        cv2.imwrite("matching-tests/"+framesName+"-matches-kps-"+detectorName+"+"+descriptorName+"-"+matcherName+"-2.png", nimg)
        cv2.imwrite("matching-tests/"+framesName+"-matchesByDescriptor-kps-"+descriptorName+"+"+detectorName+"-"+matcherName+"-2.png", nimg)
        nimg = cv.drawKeypoints(img2, nkps2, None, color=(0,255,0), flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        cv2.imwrite("matching-tests/"+framesName+"-matches-rkps-"+detectorName+"+"+descriptorName+"-"+matcherName+"-2.png", nimg)
        cv2.imwrite("matching-tests/"+framesName+"-matchesByDescriptor-rkps-"+descriptorName+"+"+detectorName+"-"+matcherName+"-2.png", nimg)

        #cv2.waitKey(0)7

sift = cv.xfeatures2d.SIFT_create(nOctaveLayers=3, contrastThreshold=0.01, edgeThreshold=100.0, sigma=1.6)
surf = cv.xfeatures2d.SURF_create(hessianThreshold=10,nOctaves=4,nOctaveLayers=3,extended=False,upright=True)
surfExtended = cv.xfeatures2d.SURF_create(hessianThreshold=10,nOctaves=4,nOctaveLayers=3,extended=True,upright=True)

agast = cv.AgastFeatureDetector_create(threshold=5,nonmaxSuppression=True,type=cv.AgastFeatureDetector_OAST_9_16)
#agast2 = cv.AgastFeatureDetector_create(threshold=5,nonmaxSuppression=True,type=cv.AgastFeatureDetector_AGAST_7_12d)
#akaze = cv.AKAZE_create(descriptor_type=cv.AKAZE_DESCRIPTOR_MLDB_UPRIGHT, descriptor_size=0, descriptor_channels=3, threshold=0.00005, nOctaves=4, nOctaveLayers=4, diffusivity=cv.KAZE_DIFF_PM_G2)
brisk = cv.BRISK_create(thresh=10, octaves=8, patternScale=1.0)
fast = cv.FastFeatureDetector_create(threshold=5, nonmaxSuppression=True, type=cv.FastFeatureDetector_TYPE_9_16)
gftt = cv.GFTTDetector_create(maxCorners=20000, qualityLevel=0.002, minDistance=1.0, blockSize=3, useHarrisDetector=False, k=0.04)
#kaze = cv.KAZE_create(extended=False, upright=True, threshold=0.00005,  nOctaves=4, nOctaveLayers=4, diffusivity=cv.KAZE_DIFF_PM_G2)
#mser = cv.MSER_create(_delta=1, _min_area=30, _max_area=1440, _max_variation=0.025, _min_diversity=0.8, _max_evolution=200, _area_threshold=1.01, _min_margin=0.003, _edge_blur_size=3)
orb = cv.ORB_create(edgeThreshold=24, patchSize=31, nlevels=8, fastThreshold=14, scaleFactor=1.2, WTA_K=2, scoreType=cv.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=6000)
orb2 = cv.ORB_create(edgeThreshold=36, patchSize=47, nlevels=8, fastThreshold=14, scaleFactor=1.2, WTA_K=2, scoreType=cv.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=6000)
#sbd = cv.SimpleBlobDetector_create() # See params for more
#boost = cv.xfeatures2d.BoostDesc_create(use_scale_orientation=False, scale_factor=6.25)
brief = cv.xfeatures2d.BriefDescriptorExtractor_create(bytes=32, use_orientation=False)
daisy = cv.xfeatures2d.DAISY_create(radius=15.0, q_radius=3, q_theta=8, q_hist=8, norm=cv.xfeatures2d.DAISY_NRM_NONE, interpolation=True, use_orientation=False)
freak = cv.xfeatures2d.FREAK_create(orientationNormalized=False,scaleNormalized=False,patternScale=22.0,nOctaves=4)
#harris = cv.xfeatures2d.HarrisLaplaceFeatureDetector_create(numOctaves=6, corn_thresh=0.01, DOG_thresh=0.01, maxCorners=20000, num_layers=4)
#latch = cv.xfeatures2d.LATCH_create(bytes=32,rotationInvariance=False,half_ssd_size=3,sigma=2.0)
lucid = cv.xfeatures2d.LUCID_create(lucid_kernel=1,blur_kernel=2)
#pct = cv.xfeatures2d.PCTSignatures_create(initSampleCount=2000,initSeedCount=400,pointDistribution=0)
#star = cv.xfeatures2d.StarDetector_create(maxSize=15, responseThreshold=1, lineThresholdProjected=10, lineThresholdBinarized=8, suppressNonmaxSize=3)
#vgg = cv.xfeatures2d.VGG_create(isigma=1.4, img_normalize=True, use_scale_orientation=False, scale_factor=6.25, dsc_normalize=False)

detectors = list(
    (
#        (sift, "sift"),
#        (surf, "surf"),     # 5/10 # 8/10  0.08675765991210938 0.03822040557861328
        (agast,"agast"),    # /10 # 4/10  0.005762577056884766 0.011286735534667969
#        (brisk,"brisk"),    # /10 # 2/10  0.036965131759643555 0.07984471321105957
#        (fast,"fast"),
#        (gftt,"gftt"),
# bad   (mser,"mser"), # Blob ?
        (orb,"orb"),
        (orb2,"orb2"),
    )
)

descriptorsAndMatchers = list(
    (
#        (sift, "sift", bfmatcherl1), # Crashes with brisk
#        (surf, "surf", bfmatcherl1),
#        (surfExtended,"surfext", bfmatcherl1),
#        (brisk,"brisk", bfmatcherH1),
        (orb,"orb", bfmatcherH1),
        (orb2,"orb2", bfmatcherH2),
        (brief,"brief", bfmatcherH1),
#        (daisy,"daisy", bfmatcherl1),
#        (freak,"freak", bfmatcherH1),
#        (lucid,"lucid", bfmatcherH2),
    )
)

for detector in detectors:
    testMatching(detector[0], detector[1], descriptorsAndMatchers)