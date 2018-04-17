from util.png_init import init
from util.png_util import *

import time
import cv2 as cv

framesName = init()
shape, imgs = readFrames(framesName, 4, 1)

img = imgs[0]

def testDetector(detector, name):
    kps = detector.detect(img, None)
    start_time = time.time()
    kps = detector.detect(img, None)
    print(name, "exec time :", time.time() - start_time)

    print(name, "feature count :", len(kps))
    nimg = cv.drawKeypoints(img, kps, None, color=(0,255,0), flags=cv.DrawMatchesFlags_DEFAULT)
    cv.imshow(name + ' features',nimg)
    cv.imwrite("detector-tests/" + framesName + "-" + name + "-features.png", nimg)
    cv.waitKey(0)
    nimg = cv.drawKeypoints(img, kps, None, color=(0,255,0), flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    cv.imwrite("detector-tests/" + framesName + "-" + name + "-richfeatures.png", nimg)
    #cv.imshow(name + ' rich features',nimg)

##### Feature detector(s)
### SIFT Defaults : nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10.0, sigma=1.6
#sift = cv.xfeatures2d.SIFT_create(nOctaveLayers=3, contrastThreshold=0.01, edgeThreshold=100.0, sigma=1.6)
### SURF Defaults : hessianThreshold=100, nOctaves=4, nOctaveLayers=3, extended=False, upright=False
#surf = cv.xfeatures2d.SURF_create(hessianThreshold=25, nOctaves=4, nOctaveLayers=3, extended=False, upright=True)
### Agast Defaults : threshold=10,nonmaxSuppression=True,type=cv.AgastFeatureDetector_OAST_9_16
#agast = cv.AgastFeatureDetector_create(threshold=5,nonmaxSuppression=True,type=cv.AgastFeatureDetector_OAST_9_16)
### Fast Defaults : threshold=10, nonmaxSuppression=True, type=cv.FastFeatureDetector_TYPE_9_16
#fast = cv.FastFeatureDetector_create(threshold=4, nonmaxSuppression=True, type=cv.FastFeatureDetector_TYPE_9_16)
### Star Defaults : maxSize=45, responseThreshold=30, lineThresholdProjected=10, lineThresholdBinarized=8, suppressNonmaxSize=5
#star = cv.xfeatures2d.StarDetector_create(maxSize=15, responseThreshold=1, lineThresholdProjected=10, lineThresholdBinarized=8, suppressNonmaxSize=3)
#
#detectors = list((
#    (sift,"sift"),
#    (surf,"surf"),
#    (agast,"agast"),
#    (fast,"fast"),
#    (star,"star"),
#))


sift = cv.xfeatures2d.SIFT_create(nOctaveLayers=3, contrastThreshold=0.01, edgeThreshold=100.0, sigma=1.6)
surf = cv.xfeatures2d.SURF_create(hessianThreshold=10,nOctaves=4,nOctaveLayers=3,extended=False,upright=True)

agast = cv.AgastFeatureDetector_create(threshold=5,nonmaxSuppression=True,type=cv.AgastFeatureDetector_OAST_9_16)
akaze = cv.AKAZE_create(descriptor_type=cv.AKAZE_DESCRIPTOR_MLDB_UPRIGHT, descriptor_size=0, descriptor_channels=3, threshold=0.00005, nOctaves=4, nOctaveLayers=4, diffusivity=cv.KAZE_DIFF_PM_G2)
brisk = cv.BRISK_create(thresh=10, octaves=8, patternScale=1.0)
fast = cv.FastFeatureDetector_create(threshold=4, nonmaxSuppression=True, type=cv.FastFeatureDetector_TYPE_9_16)
gftt = cv.GFTTDetector_create(maxCorners=20000, qualityLevel=0.002, minDistance=1.0, blockSize=3, useHarrisDetector=False, k=0.04)
kaze = cv.KAZE_create(extended=False, upright=True, threshold=0.00005,  nOctaves=4, nOctaveLayers=4, diffusivity=cv.KAZE_DIFF_PM_G2)
mser = cv.MSER_create(_delta=1, _min_area=30, _max_area=1440, _max_variation=0.025, _min_diversity=0.8, _max_evolution=200, _area_threshold=1.01, _min_margin=0.003, _edge_blur_size=3)
orb = cv.ORB_create(edgeThreshold=31, patchSize=31, nlevels=8, fastThreshold=12, scaleFactor=1.2, WTA_K=2, scoreType=cv.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=500000)
sbd = cv.SimpleBlobDetector_create() # See params for more
boost = cv.xfeatures2d.BoostDesc_create(use_scale_orientation=False, scale_factor=6.25)
brief = cv.xfeatures2d.BriefDescriptorExtractor_create(bytes=32, use_orientation=False)
daisy = cv.xfeatures2d.DAISY_create(radius=15.0, q_radius=3, q_theta=8, q_hist=8, norm=cv.xfeatures2d.DAISY_NRM_NONE, interpolation=True, use_orientation=False)
freak = cv.xfeatures2d.FREAK_create(orientationNormalized=False,scaleNormalized=False,patternScale=22.0,nOctaves=4)
harris = cv.xfeatures2d.HarrisLaplaceFeatureDetector_create(numOctaves=6, corn_thresh=0.01, DOG_thresh=0.01, maxCorners=20000, num_layers=4)
latch = cv.xfeatures2d.LATCH_create(bytes=32,rotationInvariance=False,half_ssd_size=3,sigma=2.0)
lucid = cv.xfeatures2d.LUCID_create(lucid_kernel=1,blur_kernel=2)
pct = cv.xfeatures2d.PCTSignatures_create(initSampleCount=2000,initSeedCount=400,pointDistribution=0)
star = cv.xfeatures2d.StarDetector_create(maxSize=15, responseThreshold=1, lineThresholdProjected=10, lineThresholdBinarized=8, suppressNonmaxSize=3)
vgg = cv.xfeatures2d.VGG_create(isigma=1.4, img_normalize=True, use_scale_orientation=False, scale_factor=6.25, dsc_normalize=False)

detectors = list(
    (
        (sift, "sift"),  # To slow     # 5/10  0.4838266372680664 0.20328330993652344
        (surf, "surf"),     # 8/10  0.08675765991210938 0.03822040557861328
        (agast,"agast"),    # 4/10  0.005762577056884766 0.011286735534667969
        (akaze,"akaze"), # To slow     # 5/10  0.23596453666687012 0.09460330009460449
        (brisk,"brisk"),    # 2/10  0.036965131759643555 0.07984471321105957
        (fast,"fast"),      # 4/10  0.0025038719177246094 0.0035512447357177734
        (gftt,"gftt"),      # 3/10  0.0755758285522461 0.029958724975585938
        (kaze,"kaze"), # To slow     # 5/10  1.1794540882110596 0.6536948680877686
        (mser,"mser"), # Blob ?
        (orb,"orb"),        # 2/10  0.011170148849487305 0.013445377349853516
        #(sbd,"sbd"), # WTF ? 1 Blob ?
        ##(boost,"boost"), # Not a detector
        ##(brief,"brief"), # Not a detector
        ##(daisy,"daisy"), # Not a detector
        ##(freak,"freak"), # Not a detector
        (harris,"harris"), # WTF ?
        ##(latch,"latch"), # Not a detector
        ##(lucid,"lucid"), # Not a detector
        ##(pct,"pct"), # Not a detector
        (star,"star"),      # 6/10  0.06461644172668457 0.022640228271484375
        ##(vgg,"vgg"), # Not a detector
    )
)

for detector in detectors:
    testDetector(detector[0], detector[1])